"""
models/visual_grounding.py
=============================
Module 2 — Text-Guided Visual Grounding.

Domain adaptation lineage: intended production backbone is a
GroundingDINO variant fine-tuned on remote-sensing imagery, or a
RemoteCLIP-guided region-proposal ranker (`core.config.MODEL_REGISTRY
["rs_grounding"]`).

Classical implementation strategy
----------------------------------
1. Parse the query for a target class among {water, vegetation,
   built-up, bare-soil, road} using keyword matching (same taxonomy as
   `single_vqa_caption._detect_topic`, kept separate here to allow
   grounding-specific synonyms such as "road"/"highway").
2. Build a binary evidence mask for that class via the same spectral
   index / SAR backscatter thresholds used elsewhere in the system
   (NDWI for water, NDVI for vegetation, NDBI/backscatter for built-up,
   inverse-NDVI+low-edge for bare soil, high edge-density ridge
   detection for roads).
3. Extract connected components from the mask (`cv2.connectedComponents
   WithStats`), filter by minimum area, and rank by area — this gives
   deterministic, reproducible bounding boxes with a per-box confidence
   derived from the fraction of the box covered by the evidence mask.

This keeps every returned bounding box traceable to a concrete
thresholded raster region, satisfying the "evidence-grounded" and
"auditable execution trace" requirements even without a learned
detector.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from agent.tool_registry import register_tool, ToolResult
from core import config
from core.geo_io import RasterImage
from models.adaptation_base import RSFeatureBackbone

_backbone = RSFeatureBackbone("rs_grounding")

_GROUNDING_KEYWORDS = {
    "water": ["water", "river", "lake", "pond", "sea", "reservoir", "flood", "canal"],
    "vegetation": ["vegetation", "forest", "tree", "crop", "green area", "farmland", "field"],
    "builtup": ["built", "urban", "building", "settlement", "structure", "houses"],
    "road": ["road", "highway", "street", "track"],
    "bare": ["bare", "barren", "soil", "sand", "fallow"],
}


def _detect_target_class(query: str) -> str:
    q = query.lower()
    for cls, kws in _GROUNDING_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return cls
    return "salient"  # generic saliency fallback if no class keyword found


def _build_class_mask(image: RasterImage, evidence: dict, target_class: str) -> np.ndarray:
    modality = evidence.get("modality", "unknown")
    h, w = image.height, image.width

    if modality in ("optical", "multispectral"):
        ndvi = evidence.get("ndvi_map")
        ndwi = evidence.get("ndwi_map")
        ndbi = evidence.get("ndbi_map")
        if target_class == "water" and ndwi is not None:
            return ndwi > config.THRESHOLDS["ndwi_water"]
        if target_class == "vegetation" and ndvi is not None:
            return ndvi > config.THRESHOLDS["ndvi_vegetation"]
        if target_class == "builtup" and ndbi is not None:
            return ndbi > config.THRESHOLDS["ndbi_builtup"]
        if target_class == "bare" and ndvi is not None and ndwi is not None:
            return (ndvi < 0.1) & (ndwi < 0.0)
        if target_class == "road":
            return _ridge_mask(image)
    elif modality == "sar":
        norm = _sar_norm01(image)
        vv = norm[0]
        if target_class == "water":
            return vv < 0.25
        if target_class == "builtup":
            return vv > 0.6
        if target_class == "road":
            return _ridge_mask(image)

    # Generic saliency fallback: high local contrast regions.
    return _saliency_mask(image)


def _sar_norm01(image: RasterImage) -> np.ndarray:
    from core.geo_io import normalize_sar
    return normalize_sar(image, config.THRESHOLDS["sar_db_min"], config.THRESHOLDS["sar_db_max"])


def _ridge_mask(image: RasterImage) -> np.ndarray:
    import cv2
    gray = image.array.mean(axis=0)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
    g8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(g8, 40, 120)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return dilated > 0


def _saliency_mask(image: RasterImage) -> np.ndarray:
    import cv2
    gray = image.array.mean(axis=0)
    g8 = np.clip((gray - gray.min()) / (gray.max() - gray.min() + 1e-6) * 255, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(g8, (9, 9), 0)
    diff = cv2.absdiff(g8, blur)
    thresh = diff > np.percentile(diff, 90)
    return thresh


def _extract_boxes(mask: np.ndarray, target_class: str,
                    max_boxes: int, min_area: int) -> List[dict]:
    import cv2
    mask_u8 = (mask.astype(np.uint8)) * 255
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes = []
    for label in range(1, n_labels):  # skip background label 0
        x, y, bw, bh, area = stats[label]
        if area < min_area:
            continue
        region_mask = labels == label
        coverage = float(region_mask.sum()) / float(bw * bh + 1e-6)
        score = float(min(0.98, 0.5 + 0.5 * coverage))
        boxes.append({
            "xmin": int(x), "ymin": int(y), "xmax": int(x + bw), "ymax": int(y + bh),
            "label": target_class, "score": round(score, 4), "area_px": int(area),
        })
    boxes.sort(key=lambda b: b["area_px"], reverse=True)
    return boxes[:max_boxes]


@register_tool(
    "rs_grounding",
    task_types=[config.TASK_GROUNDING],
    input_configs=[config.INPUT_CONFIG_SINGLE, config.INPUT_CONFIG_CROSS_MODAL],
    description="Text-guided open-vocabulary bounding-box grounding over remote-sensing imagery.",
)
def run(images: List[RasterImage], query: str, **kwargs) -> ToolResult:
    """Entry point dispatched by the Agent Controller."""
    primary = images[0]
    if len(images) == 2:
        primary = images[0] if images[0].modality in ("optical", "multispectral", "sar") else images[1]

    evidence = _backbone.describe(primary)
    target_class = _detect_target_class(query)
    mask = _build_class_mask(primary, evidence, target_class)

    min_area = config.THRESHOLDS["grounding_min_region_area_px"]
    max_boxes = int(config.THRESHOLDS["grounding_max_boxes"])
    boxes = _extract_boxes(mask, target_class, max_boxes, min_area)

    warnings = []
    if not boxes:
        warnings.append(f"No connected regions of class '{target_class}' exceeded the "
                         f"minimum area threshold ({min_area}px); returning empty result.")
        text = (f"No '{target_class}' region satisfying the minimum area threshold could be "
                f"localised in the image.")
        conf = 0.2
    else:
        total_area = int(mask.sum())
        frac = total_area / (primary.height * primary.width)
        text = (f"Localised {len(boxes)} candidate region(s) matching '{target_class}', "
                f"covering approximately {frac:.1%} of the image. The largest region spans "
                f"{boxes[0]['xmax'] - boxes[0]['xmin']}x{boxes[0]['ymax'] - boxes[0]['ymin']} px "
                f"at ({boxes[0]['xmin']}, {boxes[0]['ymin']}).")
        conf = float(np.mean([b["score"] for b in boxes]))

    return ToolResult(
        task_type=config.TASK_GROUNDING,
        text_answer=text,
        confidence=conf,
        bounding_boxes=boxes,
        masks=[{"label": target_class, "array": mask, "score": conf}],
        evidence={"target_class": target_class, "mask_coverage": float(mask.mean()),
                  "backbone_mode": _backbone.mode},
        model_used=f"rs_grounding[{_backbone.mode}]",
        warnings=warnings,
    )
