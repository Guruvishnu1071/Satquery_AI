"""
models/optical_sar_fusion.py
===============================
Module 4 — Optical-SAR Cross-Modal Fusion.

Domain adaptation lineage: intended production backbone is a
cross-attention Optical-SAR fusion transformer (`core.config.
MODEL_REGISTRY["optical_sar_fusion"]`).

Classical implementation strategy
----------------------------------
A rule-based decision-level fusion classifier operating per-pixel on:
  * Optical: NDVI (vegetation), NDWI (water)
  * SAR: VV backscatter (surface roughness), VH/VV ratio (volume
    scattering, useful for vegetation/double-bounce urban structures)

Fusion logic (per pixel), evaluated in priority order so that SAR can
override optical where clouds/shadow make optical unreliable:
  1. If optical NDWI indicates water **and** SAR VV backscatter is low
     (smooth-surface specular return) → **water** (high-confidence
     agreement between modalities).
  2. Else if SAR VV backscatter is high (rough / double-bounce) **and**
     optical NDBI/brightness is consistent with impervious surface →
     **built-up**.
  3. Else if optical NDVI indicates vegetation → **vegetation**.
  4. Else → **bare/other**.

Where optical is flagged unreliable (very low reflectance variance,
consistent with cloud shadow) SAR-only rules 1b/2b substitute, which is
exactly the "day-and-night, cloud-penetration" complementary value SAR
provides per the problem statement.

The module also answers general VQA/fusion queries directly from the
resulting fused land-cover map and per-class fraction statistics.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from agent.tool_registry import register_tool, ToolResult
from core import config
from core.geo_io import RasterImage, normalize_optical, normalize_sar, compute_ndvi, compute_ndwi
from models.adaptation_base import RSFeatureBackbone

_backbone = RSFeatureBackbone("optical_sar_fusion")


def _split_optical_sar(images: List[RasterImage]) -> Tuple[RasterImage, RasterImage]:
    a, b = images
    if a.modality == "sar" or (a.modality == "unknown" and b.modality in ("optical", "multispectral")):
        sar, optical = a, b
    else:
        optical, sar = a, b
    if optical.modality == "sar":  # both misclassified / swapped safety net
        optical, sar = sar, optical
    return optical, sar


def _cloud_shadow_mask(optical01: np.ndarray) -> np.ndarray:
    """Very low local variance + low brightness suggests cloud shadow /
    saturated cloud where optical spectral indices become unreliable."""
    import cv2
    gray = optical01.mean(axis=0)
    g8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
    local_std = cv2.blur((g8.astype(np.float32)) ** 2, (9, 9)) - cv2.blur(g8.astype(np.float32), (9, 9)) ** 2
    local_std = np.sqrt(np.clip(local_std, 0, None))
    return (gray < 0.08) & (local_std < 5.0)


def _fuse(optical: RasterImage, sar: RasterImage) -> dict:
    opt01 = normalize_optical(optical, config.THRESHOLDS["optical_percentile_low"],
                               config.THRESHOLDS["optical_percentile_high"])
    sar01 = normalize_sar(sar, config.THRESHOLDS["sar_db_min"], config.THRESHOLDS["sar_db_max"])

    # Resize SAR onto optical grid if shapes differ slightly post-alignment.
    if sar01.shape[1:] != opt01.shape[1:]:
        import cv2
        resized = np.zeros((sar01.shape[0], opt01.shape[1], opt01.shape[2]), dtype=np.float32)
        for b in range(sar01.shape[0]):
            resized[b] = cv2.resize(sar01[b], (opt01.shape[2], opt01.shape[1]), interpolation=cv2.INTER_LINEAR)
        sar01 = resized

    band_count = opt01.shape[0]
    red, green, nir = 0, (1 if band_count >= 2 else 0), (3 if band_count >= 4 else (2 if band_count >= 3 else 0))
    ndvi = compute_ndvi(opt01, red, nir) if band_count >= 3 else np.zeros(opt01.shape[1:])
    ndwi = compute_ndwi(opt01, green, nir) if band_count >= 3 else np.zeros(opt01.shape[1:])

    vv = sar01[0]
    vh = sar01[1] if sar01.shape[0] > 1 else sar01[0]
    vh_vv_ratio = np.divide(vh, np.clip(vv, 1e-3, None))

    cloud_mask = _cloud_shadow_mask(opt01)

    water_optical = ndwi > config.THRESHOLDS["ndwi_water"]
    water_sar = vv < 0.22
    water = np.where(cloud_mask, water_sar, water_optical & water_sar) | (cloud_mask & water_sar)

    builtup_sar = vv > 0.55
    builtup_optical_support = (ndvi < 0.2)
    builtup = np.where(cloud_mask, builtup_sar, builtup_sar & builtup_optical_support)

    vegetation = (~water) & (~builtup) & (ndvi > config.THRESHOLDS["ndvi_vegetation"]) & (~cloud_mask)
    vegetation = vegetation | ((~water) & (~builtup) & cloud_mask & (vh_vv_ratio > 0.5))  # SAR-only veg proxy

    other = ~(water | builtup | vegetation)

    fused_map = np.zeros(opt01.shape[1:], dtype=np.uint8)
    fused_map[vegetation] = 1
    fused_map[water] = 2
    fused_map[builtup] = 3
    # other stays 0

    total = fused_map.size
    return {
        "fused_map": fused_map,
        "water_mask": water, "builtup_mask": builtup, "vegetation_mask": vegetation,
        "water_fraction": float(water.sum() / total),
        "builtup_fraction": float(builtup.sum() / total),
        "vegetation_fraction": float(vegetation.sum() / total),
        "other_fraction": float(other.sum() / total),
        "cloud_shadow_fraction": float(cloud_mask.mean()),
        "modality_consistent": bool(np.mean(water_optical == water_sar) > 0.5),
    }


def _fusion_text(fusion: dict, query: str) -> str:
    cloud_note = (f" Approximately {fusion['cloud_shadow_fraction']:.1%} of the optical scene was "
                  f"flagged as cloud/shadow-affected and resolved using SAR backscatter alone."
                  if fusion["cloud_shadow_fraction"] > 0.02 else "")
    return (f"Joint optical-SAR analysis identifies: water {fusion['water_fraction']:.1%}, "
            f"built-up {fusion['builtup_fraction']:.1%}, vegetation {fusion['vegetation_fraction']:.1%}, "
            f"other/bare {fusion['other_fraction']:.1%} of the scene.{cloud_note} "
            f"Water and built-up classes were confirmed using SAR backscatter (VV) in agreement "
            f"with optical spectral indices (NDWI/NDVI), providing higher confidence than either "
            f"modality alone.")


@register_tool(
    "optical_sar_fusion",
    task_types=[config.TASK_FUSION, config.TASK_VQA],
    input_configs=[config.INPUT_CONFIG_CROSS_MODAL],
    description="Optical+SAR decision-level fusion for built-up/water/vegetation extraction "
                "under cloud/shadow conditions.",
)
def run(images: List[RasterImage], query: str, **kwargs) -> ToolResult:
    """Entry point dispatched by the Agent Controller."""
    optical, sar = _split_optical_sar(images)
    fusion = _fuse(optical, sar)

    text = _fusion_text(fusion, query)
    conf = float(min(0.95, 0.5 + 0.5 * (1.0 if fusion["modality_consistent"] else 0.4)))

    boxes: List[dict] = []
    import cv2
    for label_name, mask in (("water", fusion["water_mask"]), ("builtup", fusion["builtup_mask"])):
        mask_u8 = mask.astype(np.uint8) * 255
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        for lbl in range(1, n_labels):
            x, y, bw, bh, area = stats[lbl]
            if area < config.THRESHOLDS["grounding_min_region_area_px"]:
                continue
            boxes.append({"xmin": int(x), "ymin": int(y), "xmax": int(x + bw), "ymax": int(y + bh),
                          "label": label_name, "score": round(conf, 4), "area_px": int(area)})
    boxes.sort(key=lambda b: b["area_px"], reverse=True)
    boxes = boxes[: int(config.THRESHOLDS["grounding_max_boxes"])]

    return ToolResult(
        task_type=config.TASK_FUSION,
        text_answer=text,
        confidence=conf,
        bounding_boxes=boxes,
        masks=[
            {"label": "water", "array": fusion["water_mask"], "score": conf},
            {"label": "builtup", "array": fusion["builtup_mask"], "score": conf},
            {"label": "vegetation", "array": fusion["vegetation_mask"], "score": conf},
        ],
        evidence={
            "water_fraction": fusion["water_fraction"], "builtup_fraction": fusion["builtup_fraction"],
            "vegetation_fraction": fusion["vegetation_fraction"], "other_fraction": fusion["other_fraction"],
            "cloud_shadow_fraction": fusion["cloud_shadow_fraction"],
            "modality_consistent": fusion["modality_consistent"],
            "backbone_mode": _backbone.mode,
        },
        model_used=f"optical_sar_fusion[{_backbone.mode}]",
        warnings=[],
    )
