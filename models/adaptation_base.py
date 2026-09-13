"""
models/adaptation_base.py
============================
`RSFeatureBackbone` is the single pluggable interface every specialist
module uses to obtain domain-adapted image embeddings and, optionally,
free-text generation.

Two execution modes, auto-selected at construction time:

1. **Learned mode** (`torch` + `transformers` [+ `peft`] installed and a
   checkpoint directory present under `core.config.MODEL_REGISTRY[...]
   ["weights_path"]`): loads a RemoteCLIP/GeoChat/Qwen2-VL-style backbone,
   optionally with LoRA adapters fine-tuned on BigEarthNet/RSVQA/VRSBench,
   and produces learned embeddings / generated captions.

2. **Classical mode** (default, dependency-light, always available):
   produces a deterministic, physically-interpretable embedding from
   band statistics, spectral indices, texture (GLCM-lite via local
   variance), and colour histograms — i.e. a hand-engineered analogue of
   what a domain-adapted encoder would capture (vegetation vigor, water
   presence, built-up density, structural texture). This keeps every
   number in the embedding traceable to raster evidence, which is
   valuable for the auditable-evaluation requirement even when the deep
   backbone *is* available.

Both modes expose the same `encode(image) -> np.ndarray` and
`describe(image) -> dict` methods, so `models/*.py` never needs to
branch on which mode is active.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.geo_io import RasterImage, normalize_optical, normalize_sar, compute_ndvi, compute_ndwi, compute_ndbi
from core import config

logger = logging.getLogger("satquery_ai.adaptation_base")

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    _HAS_TORCH_STACK = True
except Exception:
    _HAS_TORCH_STACK = False


class RSFeatureBackbone:
    """
    Domain-adapted feature extractor with automatic learned/classical
    mode selection.

    Parameters
    ----------
    registry_key : str
        Key into `core.config.MODEL_REGISTRY`, used to look up the
        intended checkpoint path and backbone description.
    """

    def __init__(self, registry_key: str):
        self.registry_key = registry_key
        self.spec = config.MODEL_REGISTRY.get(registry_key, {})
        self.mode = "classical"
        self._model = None
        if _HAS_TORCH_STACK:
            try:
                self._try_load_learned_backbone()
                self.mode = "learned"
            except Exception as exc:
                logger.info("Learned backbone unavailable for '%s' (%s); "
                             "using classical fallback.", registry_key, exc)
                self.mode = "classical"

    # ------------------------------------------------------------
    def _try_load_learned_backbone(self) -> None:
        """
        Attempts to load the configured checkpoint. Raises if the
        checkpoint directory does not exist — this is expected and
        caught by `__init__` in offline/no-weights environments, which
        is the default state of this repository.
        """
        from pathlib import Path
        weights_path = Path(self.spec.get("weights_path", ""))
        
        if not weights_path.exists():
            raise FileNotFoundError(f"No checkpoint at {weights_path}")
            
       # Load the actual deep learning model
        from transformers import AutoModel, AutoProcessor
        self._model = AutoModel.from_pretrained(weights_path)
        
        if (weights_path / "adapter_config.json").exists():
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, weights_path)

    # ------------------------------------------------------------
    def encode(self, image: RasterImage) -> np.ndarray:
        """Return a fixed-length embedding vector for `image`."""
        if self.mode == "learned":  # pragma: no cover - requires real weights
            return self._encode_learned(image)
        return self._encode_classical(image)

    def describe(self, image: RasterImage) -> dict:
        """
        Return an interpretable evidence dictionary: spectral index
        summary statistics, texture/edge density, dominant colour, and
        (for SAR) backscatter statistics. Used by both the classical
        embedding and, directly, by the VQA/captioning template engine.
        """
        modality = image.modality if image.modality != "unknown" else self._infer_modality(image)
        evidence: dict = {"modality": modality}

        if modality in ("optical", "multispectral"):
            norm = normalize_optical(image, config.THRESHOLDS["optical_percentile_low"],
                                      config.THRESHOLDS["optical_percentile_high"])
            evidence.update(self._optical_evidence(norm, image.band_count))
        elif modality == "sar":
            norm = normalize_sar(image, config.THRESHOLDS["sar_db_min"], config.THRESHOLDS["sar_db_max"])
            evidence.update(self._sar_evidence(norm))
        else:
            evidence["note"] = "modality unknown; only generic texture statistics computed."
            evidence.update(self._generic_evidence(image.array))
        return evidence

    # ------------------------------------------------------------
    def _encode_learned(self, image: RasterImage) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("Learned backbone not shipped in this build.")

    def _encode_classical(self, image: RasterImage) -> np.ndarray:
        evidence = self.describe(image)
        feats = []
        for key in ("mean_r", "mean_g", "mean_b", "ndvi_mean", "ndwi_mean", "ndbi_mean",
                    "texture_energy", "edge_density", "brightness_mean", "brightness_std"):
            v = evidence.get(key, 0.0)
            feats.append(float(v) if v is not None else 0.0)
        return np.asarray(feats, dtype=np.float32)

    # ------------------------------------------------------------
    def _infer_modality(self, image: RasterImage) -> str:
        from core.validator import classify_modality
        return classify_modality(image)

    def _optical_evidence(self, norm01: np.ndarray, band_count: int) -> dict:
        bands = norm01
        
        # 1. Create a boolean mask to ignore pitch-black no-data borders
        # If the maximum value across all bands for a pixel is 0, it is outside the swath.
        valid_mask = np.max(bands, axis=0) > 0
        valid_count = np.sum(valid_mask)
        
        # Prevent division by zero if an empty array is passed
        if valid_count == 0:
            valid_count = 1

        red_idx = 0 if band_count >= 3 else 0
        green_idx = 1 if band_count >= 3 else 0
        blue_idx = 2 if band_count >= 3 else 0
        nir_idx = 3 if band_count >= 4 else (2 if band_count >= 3 else 0)
        swir_idx = 4 if band_count >= 5 else nir_idx

        ndvi = compute_ndvi(bands, red_idx, nir_idx) if band_count >= 3 else np.zeros(bands.shape[1:])
        ndwi = compute_ndwi(bands, green_idx, nir_idx) if band_count >= 3 else np.zeros(bands.shape[1:])
        ndbi = compute_ndbi(bands, swir_idx, nir_idx) if band_count >= 3 else np.zeros(bands.shape[1:])

        gray = bands.mean(axis=0)
        texture_energy, edge_density = _texture_stats(gray)

        # 2. Helper functions to compute statistics ONLY on valid pixels
        def masked_mean(arr):
            # Replaces masked out pixels with NaN, then computes the mean ignoring NaNs
            return float(np.nanmean(np.where(valid_mask, arr, np.nan)))

        def masked_std(arr):
            return float(np.nanstd(np.where(valid_mask, arr, np.nan)))

        def masked_fraction(arr, threshold):
            # Only count pixels that pass the threshold AND are inside the valid mask
            return float(np.sum((arr > threshold) & valid_mask) / valid_count)

        return {
            "mean_r": masked_mean(bands[red_idx]), 
            "mean_g": masked_mean(bands[green_idx]),
            "mean_b": masked_mean(bands[blue_idx]),
            "ndvi_mean": masked_mean(ndvi), 
            "ndvi_std": masked_std(ndvi),
            "ndwi_mean": masked_mean(ndwi),
            "ndbi_mean": masked_mean(ndbi),
            "vegetation_fraction": masked_fraction(ndvi, config.THRESHOLDS["ndvi_vegetation"]),
            "water_fraction": masked_fraction(ndwi, config.THRESHOLDS["ndwi_water"]),
            "builtup_fraction": masked_fraction(ndbi, config.THRESHOLDS["ndbi_builtup"]),
            "brightness_mean": masked_mean(gray), 
            "brightness_std": masked_std(gray),
            "texture_energy": texture_energy, 
            "edge_density": edge_density,
            "ndvi_map": ndvi, "ndwi_map": ndwi, "ndbi_map": ndbi,
        }

    def _sar_evidence(self, norm01: np.ndarray) -> dict:
        vv = norm01[0]
        vh = norm01[1] if norm01.shape[0] > 1 else norm01[0]
        ratio = np.divide(vh, np.clip(vv, 1e-3, None))
        texture_energy, edge_density = _texture_stats(vv)
        return {
            "vv_mean": float(vv.mean()), "vv_std": float(vv.std()),
            "vh_mean": float(vh.mean()), "vh_std": float(vh.std()),
            "vh_vv_ratio_mean": float(np.nanmean(ratio)),
            "low_backscatter_fraction": float(np.mean(vv < 0.25)),  # smooth surfaces / water proxy
            "high_backscatter_fraction": float(np.mean(vv > 0.6)),  # rough / urban proxy
            "texture_energy": texture_energy, "edge_density": edge_density,
            "brightness_mean": float(vv.mean()), "brightness_std": float(vv.std()),
        }

    def _generic_evidence(self, arr: np.ndarray) -> dict:
        gray = arr.mean(axis=0)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
        texture_energy, edge_density = _texture_stats(gray)
        return {
            "brightness_mean": float(gray.mean()), "brightness_std": float(gray.std()),
            "texture_energy": texture_energy, "edge_density": edge_density,
        }


def _texture_stats(gray: np.ndarray):
    """Cheap, dependency-light texture descriptors: local-variance
    'energy' proxy and Canny edge density — used across modalities as a
    structural-complexity signal (urban/rough terrain vs smooth water/
    bare ground)."""
    import cv2
    g8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(g8, 50, 150)
    edge_density = float(np.mean(edges > 0))
    lap = cv2.Laplacian(g8, cv2.CV_64F)
    texture_energy = float(np.var(lap)) / 10000.0  # scaled to a friendlier range
    return texture_energy, edge_density
