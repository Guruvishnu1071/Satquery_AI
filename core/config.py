"""
core/config.py
================
Central configuration for SatQuery AI: filesystem paths, supported data
formats, the static specialist-model registry (weights + metadata used by
`agent.tool_registry`), and numeric thresholds shared across modules.

Nothing in this file performs I/O on import; all paths are created lazily
by `ensure_dirs()` so the module is safe to import in read-only / test
environments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
UPLOAD_DIR: Path = DATA_DIR / "uploads"
CACHE_DIR: Path = DATA_DIR / "cache"
REPORT_DIR: Path = DATA_DIR / "reports"
MODEL_WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
LOG_DIR: Path = PROJECT_ROOT / "logs"
BENCHMARK_DIR: Path = DATA_DIR / "benchmarks"


def ensure_dirs() -> None:
    """Create all runtime directories if they do not already exist."""
    for d in (DATA_DIR, UPLOAD_DIR, CACHE_DIR, REPORT_DIR,
              MODEL_WEIGHTS_DIR, LOG_DIR, BENCHMARK_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Supported data formats
# --------------------------------------------------------------------------
GEOSPATIAL_EXTENSIONS: List[str] = [".tif", ".tiff", ".geotiff"]
BENCHMARK_ONLY_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg"]
ALL_SUPPORTED_EXTENSIONS: List[str] = GEOSPATIAL_EXTENSIONS + BENCHMARK_ONLY_EXTENSIONS

# Sensor / band heuristics used by validator.classify_modality()
SAR_BAND_COUNT_RANGE = (1, 2)          # single/dual-pol SAR: VV, VV+VH
OPTICAL_MIN_BANDS = 3                   # RGB or more (multispectral)
MULTISPECTRAL_MIN_BANDS = 4             # e.g. Sentinel-2 B,G,R,NIR

# --------------------------------------------------------------------------
# Input configuration types (must match agent.orchestrator.InputConfig)
# --------------------------------------------------------------------------
INPUT_CONFIG_SINGLE = "single_image"
INPUT_CONFIG_CROSS_MODAL = "cross_modal_pair"   # optical + SAR, same time
INPUT_CONFIG_BI_TEMPORAL = "bi_temporal_pair"   # same sensor, two dates

# --------------------------------------------------------------------------
# Task taxonomy (must match agent.tool_registry task_type strings)
# --------------------------------------------------------------------------
TASK_VQA = "visual_question_answering"
TASK_CAPTION = "scene_captioning"
TASK_GROUNDING = "text_guided_grounding"
TASK_CHANGE_DETECTION = "change_detection"
TASK_CHANGE_VQA = "change_vqa"
TASK_FUSION = "optical_sar_fusion"

ALL_TASKS: List[str] = [
    TASK_VQA, TASK_CAPTION, TASK_GROUNDING,
    TASK_CHANGE_DETECTION, TASK_CHANGE_VQA, TASK_FUSION,
]

# --------------------------------------------------------------------------
# Specialist model / tool registry metadata
# --------------------------------------------------------------------------
# Each entry documents the intended production backbone (for the deep
# learning upgrade path) as well as the classical fallback that ships by
# default. `weights_path` is only consulted if the optional torch stack is
# installed (see models/adaptation_base.py::RSFeatureBackbone).
MODEL_REGISTRY: Dict[str, dict] = {
    "rs_vqa_caption": {
        "task_types": [TASK_VQA, TASK_CAPTION],
        "backbone": "Qwen2-VL-2B-LoRA (BigEarthNet/RSVQA/VRSBench adapted)",
        "fallback": "spectral-statistics + rule-grounded template generator",
        "weights_path": str(MODEL_WEIGHTS_DIR / "rs_vqa_caption_lora"),
        "input_configs": [INPUT_CONFIG_SINGLE, INPUT_CONFIG_CROSS_MODAL],
    },
    "rs_grounding": {
        "task_types": [TASK_GROUNDING],
        "backbone": "GroundingDINO-RS / RemoteCLIP-guided proposal ranking",
        "fallback": "spectral-index thresholding + connected-component proposals",
        "weights_path": str(MODEL_WEIGHTS_DIR / "rs_grounding_dino"),
        "input_configs": [INPUT_CONFIG_SINGLE, INPUT_CONFIG_CROSS_MODAL],
    },
    "change_analyzer": {
        "task_types": [TASK_CHANGE_DETECTION, TASK_CHANGE_VQA],
        "backbone": "Siamese ResNet feature-differencing + BiT-CD head",
        "fallback": "Change Vector Analysis (CVA) + PCA-kmeans (Celik 2009)",
        "weights_path": str(MODEL_WEIGHTS_DIR / "change_siamese"),
        "input_configs": [INPUT_CONFIG_BI_TEMPORAL],
    },
    "optical_sar_fusion": {
        "task_types": [TASK_FUSION, TASK_VQA],
        "backbone": "Cross-attention Optical-SAR fusion transformer",
        "fallback": "NDVI/NDWI + VV/VH backscatter rule-based fusion classifier",
        "weights_path": str(MODEL_WEIGHTS_DIR / "optical_sar_fusion"),
        "input_configs": [INPUT_CONFIG_CROSS_MODAL],
    },
}

# --------------------------------------------------------------------------
# Numeric thresholds shared across modules
# --------------------------------------------------------------------------
THRESHOLDS: Dict[str, float] = {
    # Radiometric
    "optical_percentile_low": 2.0,
    "optical_percentile_high": 98.0,
    "sar_db_min": -25.0,
    "sar_db_max": 5.0,
    # Spectral indices
    "ndvi_vegetation": 0.30,
    "ndwi_water": 0.10,
    "ndbi_builtup": 0.05,
    # Change detection
    "change_kmeans_clusters": 2,
    "change_min_area_px": 25,
    # Grounding
    "grounding_min_region_area_px": 30,
    "grounding_max_boxes": 8,
    # CRS / geometry
    "max_pixel_shape_mismatch_pct": 2.0,   # allowed % diff before resample required
    "min_iou_georef_overlap": 0.30,
    # Confidence blending weights
    "confidence_weights": {
        "evidence_strength": 0.5,
        "modality_agreement": 0.3,
        "geometric_validity": 0.2,
    },
}

# --------------------------------------------------------------------------
# Benchmark dataset registry (used by evaluation/benchmark_eval.py)
# --------------------------------------------------------------------------
BENCHMARK_REGISTRY: Dict[str, dict] = {
    "BigEarthNet": {"role": "domain_adaptation_pretraining", "task": None},
    "VRSBench": {"role": "evaluation", "task": [TASK_CAPTION, TASK_GROUNDING, TASK_VQA]},
    "RSVQA": {"role": "evaluation", "task": [TASK_VQA]},
    "CDVQA": {"role": "evaluation", "task": [TASK_CHANGE_VQA]},
}

ISRO_SAC_SENSOR_PAIR = ("Cartosat-2S", "RISAT")

ENV_DEBUG = os.environ.get("SATQUERY_DEBUG", "0") == "1"
