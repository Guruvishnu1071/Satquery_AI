"""
models/single_vqa_caption.py
==============================
Module 1 — Remote-Sensing VQA & Scene Captioning.

Domain adaptation lineage: intended production backbone is a
Qwen2-VL-2B (or GeoChat) LoRA adapter fine-tuned on BigEarthNet
image-text pairs and evaluated against RSVQA / VRSBench (see
`core.config.MODEL_REGISTRY["rs_vqa_caption"]`). `RSFeatureBackbone`
(`models/adaptation_base.py`) transparently swaps between that learned
path and the classical evidence-grounded path implemented here.

Classical implementation strategy
----------------------------------
* **Captioning**: a deterministic template generator conditions on the
  computed evidence dict (dominant land-cover fractions, brightness,
  texture/edge density, modality) to produce a grounded scene
  description — every noun phrase in the caption is backed by a
  specific numeric threshold crossing, which is exactly what makes the
  output *evidence-grounded* rather than a hallucinated free-text
  caption.
* **VQA**: the natural-language question is parsed for a small set of
  question archetypes (presence/absence, quantity, dominant-class,
  comparison) and answered directly from the same evidence dict, with a
  confidence derived from how decisively the underlying statistic
  crosses its threshold.

This module registers itself for both `TASK_VQA` and `TASK_CAPTION`
under `INPUT_CONFIG_SINGLE` (the mandatory baseline) and additionally
under `INPUT_CONFIG_CROSS_MODAL` (VQA over the optical member of a
pair, when no fusion-specific question is detected upstream).
"""

from __future__ import annotations

import re
from typing import List

import numpy as np

from agent.tool_registry import register_tool, ToolResult
from core import config
from core.geo_io import RasterImage
from models.adaptation_base import RSFeatureBackbone

_backbone = RSFeatureBackbone("rs_vqa_caption")


# --------------------------------------------------------------------------
# Captioning
# --------------------------------------------------------------------------
def _generate_caption(evidence: dict) -> str:
    modality = evidence.get("modality", "unknown")
    clauses: List[str] = []

    if modality in ("optical", "multispectral"):
        veg = evidence.get("vegetation_fraction", 0.0)
        water = evidence.get("water_fraction", 0.0)
        builtup = evidence.get("builtup_fraction", 0.0)
        bright = evidence.get("brightness_mean", 0.0)
        edge = evidence.get("edge_density", 0.0)

        cover_terms = []
        if veg > 0.4:
            cover_terms.append("predominantly vegetated land cover")
        elif veg > 0.15:
            cover_terms.append("mixed vegetation interspersed with other land cover")
        if water > 0.1:
            cover_terms.append(f"water bodies covering roughly {water:.0%} of the scene")
        if builtup > 0.15:
            cover_terms.append(f"built-up / impervious surfaces covering roughly {builtup:.0%} of the scene")
        if not cover_terms:
            cover_terms.append("sparse vegetation over predominantly bare or fallow ground")
        clauses.append("The image shows " + "; ".join(cover_terms) + ".")

        if edge > 0.08:
            clauses.append("A high density of linear/edge features suggests structured or "
                            "urban infrastructure (roads, buildings, field boundaries).")
        else:
            clauses.append("Low edge density suggests relatively homogeneous, natural terrain.")

        clauses.append(f"Overall scene brightness is {'high' if bright > 0.6 else 'moderate' if bright > 0.35 else 'low'} "
                        f"(mean normalised reflectance {bright:.2f}).")

    elif modality == "sar":
        low = evidence.get("low_backscatter_fraction", 0.0)
        high = evidence.get("high_backscatter_fraction", 0.0)
        edge = evidence.get("edge_density", 0.0)
        clauses.append(
            f"SAR backscatter analysis indicates approximately {low:.0%} of the scene has "
            f"low backscatter (consistent with smooth surfaces such as calm water or bare "
            f"soil) and {high:.0%} has high backscatter (consistent with rough or "
            f"double-bounce urban/vegetated structures).")
        clauses.append("Speckle-consistent texture confirms SAR imaging modality." if edge > 0.05
                        else "Relatively smooth backscatter texture across the scene.")
    else:
        clauses.append("Modality could not be confidently determined; description limited to "
                        "generic brightness and texture statistics.")
        clauses.append(f"Mean normalised brightness is {evidence.get('brightness_mean', 0):.2f} "
                        f"with edge density {evidence.get('edge_density', 0):.2f}.")

    return " ".join(clauses)


# --------------------------------------------------------------------------
# VQA
# --------------------------------------------------------------------------
_YES_NO_PATTERN = re.compile(r"\b(is there|are there|does|is|are)\b", re.I)
_HOWMANY_PATTERN = re.compile(r"\bhow many\b", re.I)
_WHICH_DOMINANT_PATTERN = re.compile(r"\b(which|what).*(dominant|majority|most)\b", re.I)

_TOPIC_KEYWORDS = {
    "water": ["water", "river", "lake", "pond", "sea", "flood"],
    "vegetation": ["vegetation", "forest", "tree", "crop", "green", "plant", "farmland"],
    "builtup": ["built", "urban", "building", "road", "settlement", "impervious", "infrastructure"],
    "bare": ["bare", "barren", "soil", "fallow", "sand"],
}


def _detect_topic(query: str) -> str:
    q = query.lower()
    for topic, kws in _TOPIC_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return topic
    return "generic"


def _answer_vqa(query: str, evidence: dict) -> tuple[str, float]:
    topic = _detect_topic(query)
    modality = evidence.get("modality", "unknown")

    frac_map = {
        "water": evidence.get("water_fraction"),
        "vegetation": evidence.get("vegetation_fraction"),
        "builtup": evidence.get("builtup_fraction"),
    }

    if _WHICH_DOMINANT_PATTERN.search(query):
        candidates = {k: v for k, v in frac_map.items() if v is not None}
        if not candidates:
            return ("Dominant land-cover class cannot be determined for this modality.", 0.3)
        dominant = max(candidates, key=candidates.get)
        conf = min(0.95, 0.5 + candidates[dominant])
        return (f"The dominant land-cover class is '{dominant}', covering approximately "
                f"{candidates[dominant]:.0%} of the scene (evidence: spectral-index fraction).", conf)

    if _YES_NO_PATTERN.search(query) and topic in frac_map and frac_map[topic] is not None:
        frac = frac_map[topic]
        threshold = 0.05
        present = frac > threshold
        conf = float(min(0.97, 0.55 + abs(frac - threshold) * 2))
        answer = (f"Yes, {topic} is present, covering approximately {frac:.0%} of the image."
                   if present else
                   f"No significant {topic} was detected (estimated coverage {frac:.0%}, below "
                   f"the {threshold:.0%} detection threshold).")
        return (answer, conf)

    if _HOWMANY_PATTERN.search(query):
        return ("Object counting requires the text-guided grounding tool to enumerate discrete "
                "instances; based on scene-level statistics only, a precise count cannot be "
                "returned from single-image VQA.", 0.35)

    if topic in frac_map and frac_map[topic] is not None:
        frac = frac_map[topic]
        return (f"Estimated {topic} coverage in the scene is approximately {frac:.0%}, derived "
                f"from spectral-index thresholding (evidence-grounded).", float(min(0.9, 0.5 + frac)))

    # Generic fallback answer grounded in overall scene statistics.
    bright = evidence.get("brightness_mean", 0.0)
    edge = evidence.get("edge_density", 0.0)
    return (f"Based on {modality} imagery statistics (mean brightness {bright:.2f}, edge "
            f"density {edge:.2f}), no specific object/class matched the query terms; "
            f"a general scene summary is provided instead of a targeted answer.", 0.4)


# --------------------------------------------------------------------------
# Registered tool entry point
# --------------------------------------------------------------------------
@register_tool(
    "rs_vqa_caption",
    task_types=[config.TASK_VQA, config.TASK_CAPTION],
    input_configs=[config.INPUT_CONFIG_SINGLE, config.INPUT_CONFIG_CROSS_MODAL],
    description="Remote-sensing VQA and scene captioning (BigEarthNet/RSVQA/VRSBench-adapted).",
)
def run(images: List[RasterImage], query: str, task_type: str, **kwargs) -> ToolResult:
    """Entry point dispatched by the Agent Controller."""
    primary = images[0]
    if len(images) == 2:
        # For a cross-modal pair routed here (fusion keywords absent),
        # prefer the optical/multispectral member for VQA/captioning.
        primary = images[0] if images[0].modality in ("optical", "multispectral") else images[1]

    evidence = _backbone.describe(primary)
    warnings = []
    if evidence.get("modality") == "unknown":
        warnings.append("Modality unknown; VQA/captioning confidence reduced.")

    if task_type == config.TASK_CAPTION:
        text = _generate_caption(evidence)
        conf = 0.85 if evidence.get("modality") != "unknown" else 0.4
    else:
        text, conf = _answer_vqa(query, evidence)

    numeric_evidence = {k: v for k, v in evidence.items() if not k.endswith("_map") and not isinstance(v, np.ndarray)}
    numeric_evidence["backbone_mode"] = _backbone.mode

    return ToolResult(
        task_type=task_type,
        text_answer=text,
        confidence=float(conf),
        bounding_boxes=[],
        masks=[],
        evidence=numeric_evidence,
        model_used=f"rs_vqa_caption[{_backbone.mode}]",
        warnings=warnings,
    )
