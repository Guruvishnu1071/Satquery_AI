"""
agent/orchestrator.py
=======================
The Agent Controller — the single entry point that turns
(raster file paths, natural-language query) into an evidence-grounded,
auditable answer.

Pipeline
--------
1. **Load & validate** every supplied image (`core.geo_io`, `core.validator`).
2. **Classify input configuration**: single image / bi-temporal pair /
   cross-modal pair, from image count + modality classification.
3. **Classify intent**: map the natural-language query to a task type
   from `core.config.ALL_TASKS` using a deterministic keyword/pattern
   classifier (`IntentClassifier`). This keeps routing auditable —
   no opaque LLM call decides which specialist runs.
4. **Select tool(s)** from `agent.tool_registry` matching
   (task_type, input_config).
5. **Execute** the selected tool(s), passing only permitted parameters.
6. **Aggregate** outputs (text, boxes, masks) and estimate a blended
   confidence score.
7. **Emit** the final answer plus a full `ExecutionTracer` JSON trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core import config
from core.geo_io import RasterImage, read_image, GeoIOError, resample_to_match
from core.validator import (
    validate_single_image, validate_pair, classify_modality,
)
from agent.tool_registry import find_tools, ToolResult
from agent.execution_tracer import ExecutionTracer

# Import specialist modules for their side-effecting @register_tool calls.
# (Imported here, not at package top, to avoid import-order issues when
# `models` needs `agent.tool_registry` which is defined above.)
def _ensure_tools_registered() -> None:
    from models import single_vqa_caption   # noqa: F401
    from models import visual_grounding     # noqa: F401
    from models import change_analyzer      # noqa: F401
    from models import optical_sar_fusion   # noqa: F401


class OrchestratorError(Exception):
    """Raised when the query cannot be routed/executed at all
    (invalid input, no matching tool, etc.). Always caught at the
    top-level `run_query` and turned into a failed trace rather than
    propagated raw to the GUI."""


# --------------------------------------------------------------------------
# Deterministic intent classification
# --------------------------------------------------------------------------
_INTENT_PATTERNS: List[Tuple[str, List[str]]] = [
    (config.TASK_CHANGE_VQA, [
        r"\bincrease[d]?\b", r"\bdecrease[d]?\b", r"\bchanged?\b.*\bmuch\b",
        r"\bhas\b.*\bchanged\b", r"\bhow much\b.*\bchange", r"\bgrown\b", r"\bshrunk\b",
    ]),
    (config.TASK_CHANGE_DETECTION, [
        r"\bwhat changed\b", r"\bchange[s]?\b", r"\bdifference[s]?\b.*\bdates?\b",
        r"\bbefore and after\b", r"\bcompare\b.*\bimages?\b", r"\bwhere did\b.*\boccur",
    ]),
    (config.TASK_GROUNDING, [
        r"\bhighlight\b", r"\blocate\b", r"\bwhere is\b", r"\bpoint(?:ed)? out\b",
        r"\bfind the\b", r"\bbounding box\b", r"\bgrounding\b", r"\bmark the\b",
        r"\breferred to\b",
    ]),
    (config.TASK_FUSION, [
        r"\boptical and sar\b", r"\bsar and optical\b", r"\btogether\b.*\bidentify\b",
        r"\bfus(?:e|ion)\b", r"\bcombine\b.*\b(sar|optical|radar)\b", r"\bcloud[- ]?free\b",
    ]),
    (config.TASK_CAPTION, [
        r"\bdescribe\b", r"\bcaption\b", r"\bscene description\b", r"\bsummari[sz]e the image\b",
        r"\bwhat does this image show\b", r"\boverview of\b",
    ]),
    (config.TASK_VQA, [
        r"\bwhat\b", r"\bhow many\b", r"\bis there\b", r"\bare there\b",
        r"\bwhich\b", r"\bdoes\b", r"\bcan you tell\b",
    ]),
]


@dataclass
class IntentResult:
    task_type: str
    matched_pattern: Optional[str]
    all_scores: Dict[str, int]


class IntentClassifier:
    """Deterministic, regex/keyword-based query -> task_type classifier.

    Order in `_INTENT_PATTERNS` encodes priority: more specific intents
    (change-VQA, grounding, fusion) are checked before the generic VQA
    catch-all, so e.g. "Has the built-up area increased?" resolves to
    CHANGE_VQA rather than plain VQA.
    """

    def classify(self, query: str) -> IntentResult:
        q = query.lower().strip()
        scores: Dict[str, int] = {t: 0 for t, _ in _INTENT_PATTERNS}
        first_match: Optional[Tuple[str, str]] = None
        for task, patterns in _INTENT_PATTERNS:
            for pat in patterns:
                if re.search(pat, q):
                    scores[task] += 1
                    if first_match is None:
                        first_match = (task, pat)
        if first_match is None:
            # Default: treat as open VQA — mandatory baseline task.
            return IntentResult(task_type=config.TASK_VQA, matched_pattern=None, all_scores=scores)
        return IntentResult(task_type=first_match[0], matched_pattern=first_match[1], all_scores=scores)


# --------------------------------------------------------------------------
# Input configuration classification
# --------------------------------------------------------------------------
def classify_input_config(images: List[RasterImage], declared_pair_type: Optional[str] = None) -> str:
    """
    Determine INPUT_CONFIG_* from the number of supplied images and their
    modalities. If the caller (GUI) already knows the pair semantics
    (e.g. user explicitly uploaded "Optical" + "SAR" into labelled slots),
    `declared_pair_type` overrides inference.
    """
    if declared_pair_type in (config.INPUT_CONFIG_CROSS_MODAL, config.INPUT_CONFIG_BI_TEMPORAL):
        return declared_pair_type

    if len(images) == 1:
        return config.INPUT_CONFIG_SINGLE
    if len(images) == 2:
        mods = [classify_modality(im) for im in images]
        sar_present = "sar" in mods
        optical_present = any(m in ("optical", "multispectral") for m in mods)
        if sar_present and optical_present:
            return config.INPUT_CONFIG_CROSS_MODAL
        return config.INPUT_CONFIG_BI_TEMPORAL
    raise OrchestratorError(f"Unsupported number of input images: {len(images)} "
                             f"(SatQuery AI supports 1 or 2 images per query).")


# --------------------------------------------------------------------------
# Agent Controller
# --------------------------------------------------------------------------
class AgentController:
    """The orchestrator entry point used by the web app / evaluation harness."""

    def __init__(self):
        _ensure_tools_registered()
        self.intent_classifier = IntentClassifier()

    # ------------------------------------------------------------
    def run_query(self, image_paths: List[str], query: str,
                  declared_pair_type: Optional[str] = None) -> Dict:
        """
        Full pipeline entry point.

        Parameters
        ----------
        image_paths : list of 1 or 2 file paths.
        query : natural-language query string.
        declared_pair_type : optional explicit input-config override.

        Returns
        -------
        dict — the final JSON execution trace (see `ExecutionTracer.to_json`).
        This is the sole return type regardless of success/failure, so the
        GUI and evaluation harness only ever have to handle one shape.
        """
        tracer = ExecutionTracer(query=query)
        try:
            images, meta = self._load_and_validate(image_paths, tracer)
            input_config = classify_input_config(images, declared_pair_type)
            intent = self.intent_classifier.classify(query)
            tracer.set_intent(intent.task_type, input_config)

            images = self._align_pair_if_needed(images, input_config, tracer)

            candidates = find_tools(intent.task_type, input_config)
            if not candidates:
                # Fall back to VQA if the specific task has no tool for this
                # input configuration (keeps the mandatory single-image VQA
                # baseline always reachable).
                tracer.add_warning(
                    f"No tool registered for task='{intent.task_type}' under "
                    f"input_config='{input_config}'; falling back to VQA.")
                intent = IntentResult(config.TASK_VQA, None, {})
                tracer.selected_task = config.TASK_VQA
                candidates = find_tools(config.TASK_VQA, input_config)
            if not candidates:
                raise OrchestratorError(
                    f"No specialist tool available for input_config='{input_config}'.")

            tool_spec = candidates[0]
            params = {"input_config": input_config, "task_type": intent.task_type}
            tracer.log_tool_dispatch(tool_spec.name, params)

            result: ToolResult = tool_spec.fn(images=images, query=query, **params)

            tracer.log_tool_result(tool_spec.name, result.confidence,
                                    len(result.bounding_boxes), len(result.masks))
            tracer.add_evidence(result.bounding_boxes, result.masks)
            for w in result.warnings:
                tracer.add_warning(w)

            geometric_validity = 1.0 if all(im.crs for im in images) or len(images) == 1 else 0.7
            modality_agreement = 1.0 if input_config != config.INPUT_CONFIG_CROSS_MODAL else \
                (1.0 if result.evidence.get("modality_consistent", True) else 0.5)
            tracer.set_confidence(
                evidence_strength=result.confidence,
                modality_agreement=modality_agreement,
                geometric_validity=geometric_validity,
            )
            return tracer.finalize(result.text_answer, status="success")

        except (OrchestratorError, GeoIOError) as exc:
            tracer.add_warning(str(exc))
            return tracer.finalize(f"Query could not be completed: {exc}", status="error")
        except Exception as exc:  
            import traceback
            error_details = traceback.format_exc()
            print("\n=== THE REAL AGENT CRASH ===")
            print(error_details)
            print("============================\n")
            
            tracer.add_warning(f"Unexpected internal error: {exc}")
            # We are forcing the dashboard to show the actual Python error!
            return tracer.finalize(f"CRASH REPORT: {str(exc)}", status="error")

    # ------------------------------------------------------------
    def _load_and_validate(self, image_paths: List[str],
                            tracer: ExecutionTracer) -> Tuple[List[RasterImage], List[dict]]:
        if not image_paths or len(image_paths) not in (1, 2):
            raise OrchestratorError("Provide exactly 1 image (single) or 2 images "
                                     "(bi-temporal / cross-modal pair).")
        images: List[RasterImage] = []
        metas: List[dict] = []
        for p in image_paths:
            img = read_image(p)
            vres = validate_single_image(img, p)
            if not vres.is_valid:
                raise OrchestratorError(f"Input validation failed for '{p}': "
                                         f"{'; '.join(vres.messages)}")
            img.modality = vres.modality
            images.append(img)
            metas.append(vres.metadata)
            for m in vres.messages:
                tracer.add_warning(f"[{p}] {m}")

        if len(images) == 2:
            declared = classify_input_config(images)
            pres = validate_pair(images[0], images[1], declared)
            if not pres.is_valid:
                raise OrchestratorError(f"Pair validation failed: {'; '.join(pres.messages)}")
            for m in pres.messages:
                tracer.add_warning(m)
            metas.append({"pair_validation": pres.metadata})

        tracer.log_input_metadata(metas)
        return images, metas

    def _align_pair_if_needed(self, images: List[RasterImage], input_config: str,
                               tracer: ExecutionTracer) -> List[RasterImage]:
        if len(images) != 2:
            return images
        a, b = images
        if a.height != b.height or a.width != b.width:
            tracer.log_step("resample", reference=a.source_path, target=b.source_path)
            b = resample_to_match(b, a)
        return [a, b]

    # ------------------------------------------------------------
    def pipeline_graph(self) -> List[dict]:
        """Static description of tool registry, for the GUI's live pipeline
        graph view."""
        from agent.tool_registry import registry_summary
        return registry_summary()
