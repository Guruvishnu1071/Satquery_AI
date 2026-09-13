"""
agent/execution_tracer.py
============================
Builds the auditable, ISRO/SAC-audit-spec-compatible execution trace for
every query processed by the Agent Controller.

The trace is a structured JSON object recording exactly:
  * the parsed intent / selected task,
  * the input configuration and per-image metadata,
  * every model/tool that was dispatched and its parameters,
  * spatial evidence (bounding boxes / masks),
  * a blended confidence score,
  * the final evidence-grounded natural-language answer,
  * step-level timestamps for full auditability.

Only the observable execution trace is surfaced — no hidden internal
"reasoning" text is included, per the ISRO/SAC evaluation spec ("internal
reasoning text is neither required nor evaluated").
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import config


@dataclass
class TraceStep:
    step_index: int
    name: str
    detail: Dict[str, Any]
    timestamp: float


class ExecutionTracer:
    """Accumulates steps for a single query and emits the final JSON trace."""

    def __init__(self, query: str):
        self.query_id: str = str(uuid.uuid4())
        self.query: str = query
        self.start_time: float = time.time()
        self.steps: List[TraceStep] = []
        self.selected_task: Optional[str] = None
        self.input_config: Optional[str] = None
        self.input_metadata: List[dict] = []
        self.dispatched_tools: List[dict] = []
        self.bounding_boxes: List[dict] = []
        self.masks_summary: List[dict] = []
        self.confidence_components: Dict[str, float] = {}
        self.final_confidence: Optional[float] = None
        self.text_answer: Optional[str] = None
        self.warnings: List[str] = []
        self.status: str = "in_progress"

    # ---------------------------------------------------------------
    def log_step(self, name: str, **detail: Any) -> None:
        self.steps.append(TraceStep(
            step_index=len(self.steps), name=name, detail=detail,
            timestamp=round(time.time() - self.start_time, 4),
        ))

    def set_intent(self, task: str, input_config: str) -> None:
        self.selected_task = task
        self.input_config = input_config
        self.log_step("intent_classification", selected_task=task, input_config=input_config)

    def log_input_metadata(self, metadata_list: List[dict]) -> None:
        self.input_metadata = metadata_list
        self.log_step("input_validation", images=metadata_list)

    def log_tool_dispatch(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        entry = {"tool": tool_name, "parameters": parameters}
        self.dispatched_tools.append(entry)
        self.log_step("tool_dispatch", **entry)

    def log_tool_result(self, tool_name: str, confidence: float,
                         n_boxes: int, n_masks: int) -> None:
        self.log_step("tool_result", tool=tool_name, confidence=confidence,
                       n_boxes=n_boxes, n_masks=n_masks)

    def add_evidence(self, bounding_boxes: List[dict], masks: List[dict]) -> None:
        self.bounding_boxes.extend(bounding_boxes)
        for m in masks:
            arr = m.get("array")
            area = int(arr.sum()) if arr is not None else None
            self.masks_summary.append({
                "label": m.get("label"), "score": m.get("score"), "area_px": area,
            })

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        self.log_step("warning", message=msg)

    def set_confidence(self, evidence_strength: float, modality_agreement: float,
                        geometric_validity: float) -> float:
        w = config.THRESHOLDS["confidence_weights"]
        blended = (
            w["evidence_strength"] * evidence_strength +
            w["modality_agreement"] * modality_agreement +
            w["geometric_validity"] * geometric_validity
        )
        blended = float(max(0.0, min(1.0, blended)))
        self.confidence_components = {
            "evidence_strength": round(evidence_strength, 4),
            "modality_agreement": round(modality_agreement, 4),
            "geometric_validity": round(geometric_validity, 4),
        }
        self.final_confidence = round(blended, 4)
        self.log_step("confidence_estimation", **self.confidence_components,
                       final_confidence=self.final_confidence)
        return blended

    def finalize(self, text_answer: str, status: str = "success") -> Dict[str, Any]:
        self.text_answer = text_answer
        self.status = status
        self.log_step("finalize", status=status)
        return self.to_json()

    # ---------------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "status": self.status,
            "selected_task": self.selected_task,
            "input_configuration": self.input_config,
            "input_metadata": self.input_metadata,
            "dispatched_tools": self.dispatched_tools,
            "spatial_evidence": {
                "bounding_boxes": self.bounding_boxes,
                "masks": self.masks_summary,
            },
            "confidence": {
                "components": self.confidence_components,
                "final_score": self.final_confidence,
            },
            "evidence_grounded_answer": self.text_answer,
            "warnings": self.warnings,
            "execution_trace": [
                {"step": s.step_index, "name": s.name, "t_sec": s.timestamp, **s.detail}
                for s in self.steps
            ],
            "total_latency_sec": round(time.time() - self.start_time, 4),
        }

    def to_json_string(self, indent: int = 2) -> str:
        def _default(o):
            if hasattr(o, "tolist"):
                return "array<%s>" % str(getattr(o, "shape", "?"))
            return str(o)
        return json.dumps(self.to_json(), indent=indent, default=_default)
