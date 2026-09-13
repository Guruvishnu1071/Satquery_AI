"""
models/change_analyzer.py
============================
Module 3 — Bi-Temporal Change Detection & Change-VQA.

Domain adaptation lineage: intended production backbone is a Siamese
ResNet feature-differencing network with a BiT-CD-style change head,
evaluated on CDVQA (`core.config.MODEL_REGISTRY["change_analyzer"]`).

Classical implementation strategy
----------------------------------
* **Change Vector Analysis (CVA)**: per-pixel Euclidean distance between
  radiometrically normalised T1/T2 band vectors.
* **PCA + k-means thresholding** (Celik, 2009, "Unsupervised Change
  Detection in Satellite Images Using Principal Component Analysis and
  k-Means Clustering"): the CVA difference image is decomposed into
  local PCA blocks and clustered with k=2 (change / no-change), giving
  an unsupervised, deterministic binary change mask without requiring
  any labelled training data — appropriate for a general-purpose
  auditable pipeline.
* **Change-VQA**: built on top of class-specific index deltas (ΔNDVI,
  ΔNDBI, ΔNDWI for optical; Δbackscatter for SAR) to answer
  increase/decrease/unchanged questions about a specific land-cover
  class, and a natural-language change-description summary.

This module is only registered for `INPUT_CONFIG_BI_TEMPORAL`.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np

from agent.tool_registry import register_tool, ToolResult
from core import config
from core.geo_io import RasterImage, normalize_optical, normalize_sar, compute_ndvi, compute_ndwi, compute_ndbi
from models.adaptation_base import RSFeatureBackbone

_backbone = RSFeatureBackbone("change_analyzer")


# --------------------------------------------------------------------------
# Change Vector Analysis + PCA-kmeans thresholding
# --------------------------------------------------------------------------
def _normalize_pair(t1: RasterImage, t2: RasterImage) -> Tuple[np.ndarray, np.ndarray, str]:
    modality = t1.modality if t1.modality != "unknown" else t2.modality
    if modality == "sar":
        n1 = normalize_sar(t1, config.THRESHOLDS["sar_db_min"], config.THRESHOLDS["sar_db_max"])
        n2 = normalize_sar(t2, config.THRESHOLDS["sar_db_min"], config.THRESHOLDS["sar_db_max"])
    else:
        n1 = normalize_optical(t1, config.THRESHOLDS["optical_percentile_low"],
                                config.THRESHOLDS["optical_percentile_high"])
        n2 = normalize_optical(t2, config.THRESHOLDS["optical_percentile_low"],
                                config.THRESHOLDS["optical_percentile_high"])
        modality = "optical"
    return n1, n2, modality


def _cva_magnitude(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance across bands between the two dates."""
    common_bands = min(n1.shape[0], n2.shape[0])
    diff = n1[:common_bands] - n2[:common_bands]
    return np.sqrt(np.sum(diff ** 2, axis=0))


def _pca_kmeans_change_mask(cva: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Celik (2009)-style unsupervised change mask:
    1. Extract local blocks around the CVA magnitude map as feature
       vectors, reduce with PCA.
    2. Cluster PCA-projected features with k=2 (change / no-change).
    3. The cluster with the higher mean CVA magnitude is labelled 'change'.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    h, w = cva.shape
    block = 3
    pad = block // 2
    padded = np.pad(cva, pad, mode="reflect")
    patches = np.lib.stride_tricks.sliding_window_view(padded, (block, block))
    patches = patches.reshape(h * w, block * block)

    n_components = min(4, patches.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    projected = pca.fit_transform(patches)

    k = int(config.THRESHOLDS["change_kmeans_clusters"])
    km = KMeans(n_clusters=k, n_init=4, random_state=0)
    labels = km.fit_predict(projected)

    cva_flat = cva.ravel()
    cluster_means = [cva_flat[labels == c].mean() if np.any(labels == c) else -np.inf
                      for c in range(k)]
    change_cluster = int(np.argmax(cluster_means))
    change_mask = (labels == change_cluster).reshape(h, w)

    # Morphological cleanup to remove speckle-scale false positives.
    import cv2
    mask_u8 = change_mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    change_mask = mask_u8 > 0

    stats = {
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "cluster_mean_cva": [round(float(c), 4) for c in cluster_means],
        "change_cluster_index": change_cluster,
    }
    return change_mask, stats


# --------------------------------------------------------------------------
# Change-VQA: class-specific index deltas
# --------------------------------------------------------------------------
_CHANGE_TOPIC_KEYWORDS = {
    "builtup": ["built", "urban", "settlement", "construction", "infrastructure"],
    "vegetation": ["vegetation", "forest", "green", "crop", "tree"],
    "water": ["water", "river", "lake", "flood"],
}


def _detect_change_topic(query: str) -> str:
    q = query.lower()
    for topic, kws in _CHANGE_TOPIC_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return topic
    return "generic"


def _index_delta_evidence(n1: np.ndarray, n2: np.ndarray, modality: str) -> dict:
    if modality != "optical" or n1.shape[0] < 3:
        return {}
    band_count = min(n1.shape[0], n2.shape[0])
    red, green, nir = 0, 1, (3 if band_count >= 4 else 2)
    swir = 4 if band_count >= 5 else nir

    ndvi1, ndvi2 = compute_ndvi(n1, red, nir), compute_ndvi(n2, red, nir)
    ndwi1, ndwi2 = compute_ndwi(n1, green, nir), compute_ndwi(n2, green, nir)
    ndbi1, ndbi2 = compute_ndbi(n1, swir, nir), compute_ndbi(n2, swir, nir)

    return {
        "ndvi_frac_t1": float(np.mean(ndvi1 > config.THRESHOLDS["ndvi_vegetation"])),
        "ndvi_frac_t2": float(np.mean(ndvi2 > config.THRESHOLDS["ndvi_vegetation"])),
        "ndwi_frac_t1": float(np.mean(ndwi1 > config.THRESHOLDS["ndwi_water"])),
        "ndwi_frac_t2": float(np.mean(ndwi2 > config.THRESHOLDS["ndwi_water"])),
        "ndbi_frac_t1": float(np.mean(ndbi1 > config.THRESHOLDS["ndbi_builtup"])),
        "ndbi_frac_t2": float(np.mean(ndbi2 > config.THRESHOLDS["ndbi_builtup"])),
    }


def _answer_change_vqa(query: str, delta_evidence: dict, change_frac: float) -> Tuple[str, float]:
    topic = _detect_change_topic(query)
    key_map = {"builtup": "ndbi_frac", "vegetation": "ndvi_frac", "water": "ndwi_frac"}

    if topic in key_map and f"{key_map[topic]}_t1" in delta_evidence:
        t1 = delta_evidence[f"{key_map[topic]}_t1"]
        t2 = delta_evidence[f"{key_map[topic]}_t2"]
        delta = t2 - t1
        direction = "increased" if delta > 0.02 else "decreased" if delta < -0.02 else "remained approximately unchanged"
        conf = float(min(0.95, 0.5 + abs(delta) * 3))
        text = (f"The {topic} area has {direction} between the two dates "
                f"(from {t1:.1%} to {t2:.1%} of scene coverage, Δ={delta:+.1%}), based on "
                f"spectral-index fraction comparison.")
        return text, conf

    direction_word = "significant" if change_frac > 0.15 else "moderate" if change_frac > 0.05 else "minimal"
    text = (f"Overall {direction_word} change was detected between the two acquisition dates, "
            f"affecting approximately {change_frac:.1%} of the scene (unsupervised CVA + "
            f"PCA-kmeans change mask). No specific land-cover class matched the query terms, "
            f"so an overall change magnitude is reported instead.")
    conf = float(min(0.85, 0.4 + change_frac))
    return text, conf


def _change_description(change_frac: float, delta_evidence: dict) -> str:
    parts = [f"Change detection between the two dates identified changed regions covering "
             f"approximately {change_frac:.1%} of the scene."]
    if delta_evidence:
        d_veg = delta_evidence.get("ndvi_frac_t2", 0) - delta_evidence.get("ndvi_frac_t1", 0)
        d_bu = delta_evidence.get("ndbi_frac_t2", 0) - delta_evidence.get("ndbi_frac_t1", 0)
        d_water = delta_evidence.get("ndwi_frac_t2", 0) - delta_evidence.get("ndwi_frac_t1", 0)
        if abs(d_bu) > 0.02:
            parts.append(f"Built-up extent {'increased' if d_bu > 0 else 'decreased'} by "
                         f"{abs(d_bu):.1%} of scene area.")
        if abs(d_veg) > 0.02:
            parts.append(f"Vegetated extent {'increased' if d_veg > 0 else 'decreased'} by "
                         f"{abs(d_veg):.1%} of scene area.")
        if abs(d_water) > 0.02:
            parts.append(f"Water extent {'increased' if d_water > 0 else 'decreased'} by "
                         f"{abs(d_water):.1%} of scene area.")
    return " ".join(parts)


def _change_boxes(change_mask: np.ndarray, min_area: int, max_boxes: int) -> List[dict]:
    import cv2
    mask_u8 = change_mask.astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes = []
    for label in range(1, n_labels):
        x, y, bw, bh, area = stats[label]
        if area < min_area:
            continue
        boxes.append({"xmin": int(x), "ymin": int(y), "xmax": int(x + bw), "ymax": int(y + bh),
                      "label": "change", "score": round(min(0.95, 0.5 + area / (bw * bh + 1e-6) * 0.5), 4),
                      "area_px": int(area)})
    boxes.sort(key=lambda b: b["area_px"], reverse=True)
    return boxes[:max_boxes]


@register_tool(
    "change_analyzer",
    task_types=[config.TASK_CHANGE_DETECTION, config.TASK_CHANGE_VQA],
    input_configs=[config.INPUT_CONFIG_BI_TEMPORAL],
    description="Bi-temporal change detection (CVA + PCA-kmeans) and change-VQA (CDVQA-style).",
)
def run(images: List[RasterImage], query: str, task_type: str, **kwargs) -> ToolResult:
    """Entry point dispatched by the Agent Controller."""
    t1, t2 = images[0], images[1]
    n1, n2, modality = _normalize_pair(t1, t2)

    cva = _cva_magnitude(n1, n2)
    change_mask, pca_stats = _pca_kmeans_change_mask(cva)
    change_frac = float(change_mask.mean())

    delta_evidence = _index_delta_evidence(n1, n2, modality)

    min_area = config.THRESHOLDS["change_min_area_px"]
    boxes = _change_boxes(change_mask, min_area, int(config.THRESHOLDS["grounding_max_boxes"]))

    if task_type == config.TASK_CHANGE_VQA:
        text, conf = _answer_change_vqa(query, delta_evidence, change_frac)
    else:
        text = _change_description(change_frac, delta_evidence)
        conf = float(min(0.9, 0.45 + change_frac))

    warnings = []
    if change_frac < 0.005:
        warnings.append("Detected change fraction is near-zero; results may reflect "
                         "radiometric noise rather than true land-cover change.")

    return ToolResult(
        task_type=task_type,
        text_answer=text,
        confidence=conf,
        bounding_boxes=boxes,
        masks=[{"label": "change_mask", "array": change_mask, "score": conf}],
        evidence={"modality": modality, "change_fraction": change_frac,
                  "pca_kmeans_stats": pca_stats, **delta_evidence,
                  "backbone_mode": _backbone.mode},
        model_used=f"change_analyzer[{_backbone.mode}]",
        warnings=warnings,
    )
