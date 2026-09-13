"""
core/validator.py
===================
Validates and characterises raster inputs before they are handed to the
agent orchestrator: modality classification (optical vs SAR), format
compatibility, and cross-image alignment checks for cross-modal /
bi-temporal pairs.

Every check returns a `ValidationResult` (never raises for *expected*
data problems) so the orchestrator can build a rich, auditable error
message instead of crashing. Only truly unrecoverable I/O errors raise
`GeoIOError` (propagated from `core.geo_io`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core import config
from core.geo_io import RasterImage


@dataclass
class ValidationResult:
    is_valid: bool
    modality: str = "unknown"          # "optical" | "sar" | "multispectral"
    messages: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add(self, msg: str) -> None:
        self.messages.append(msg)


@dataclass
class PairValidationResult:
    is_valid: bool
    requires_resample: bool = False
    messages: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Single-image validation & modality classification
# --------------------------------------------------------------------------
def classify_modality(img: RasterImage) -> str:
    """
    Heuristic modality classifier.

    Rules
    -----
    * band_count in SAR_BAND_COUNT_RANGE (1-2) AND high dynamic range /
      speckle-like local variance -> "sar"
    * band_count >= MULTISPECTRAL_MIN_BANDS -> "multispectral"
    * band_count in [3, MULTISPECTRAL_MIN_BANDS) -> "optical" (RGB)
    * otherwise -> "unknown"
    """
    bc = img.band_count
    if config.SAR_BAND_COUNT_RANGE[0] <= bc <= config.SAR_BAND_COUNT_RANGE[1]:
        # Disambiguate 1-band optical (panchromatic/grayscale) from SAR using
        # local speckle statistics: SAR exhibits high-frequency multiplicative
        # noise -> high coefficient-of-variation in small windows.
        band = img.array[0]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            return "unknown"
        cv = _local_coefficient_of_variation(band)
        return "sar" if cv > 0.35 else "optical"
    if bc >= config.MULTISPECTRAL_MIN_BANDS:
        return "multispectral"
    if bc >= config.OPTICAL_MIN_BANDS:
        return "optical"
    return "unknown"


def _local_coefficient_of_variation(band: np.ndarray, win: int = 7) -> float:
    """Mean local (std/mean) over sliding windows — a cheap, dependency-free
    proxy for SAR speckle detection."""
    import cv2
    b = band.astype(np.float32)
    b = np.nan_to_num(b)
    mean = cv2.blur(b, (win, win))
    sq_mean = cv2.blur(b * b, (win, win))
    var = np.clip(sq_mean - mean ** 2, 0, None)
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_map = np.where(np.abs(mean) > 1e-6, std / np.abs(mean), 0.0)
    return float(np.nanmedian(cv_map))


def validate_single_image(img: RasterImage, filename: str) -> ValidationResult:
    """Validate a single raster: format, band sanity, NoData coverage."""
    result = ValidationResult(is_valid=True)

    ext_ok = any(filename.lower().endswith(e) for e in config.ALL_SUPPORTED_EXTENSIONS)
    if not ext_ok:
        result.is_valid = False
        result.add(f"Unsupported file extension for '{filename}'. "
                    f"Supported: {config.ALL_SUPPORTED_EXTENSIONS}")
        return result

    if img.band_count < 1:
        result.is_valid = False
        result.add("Image has zero bands.")
        return result

    if img.height < 8 or img.width < 8:
        result.is_valid = False
        result.add(f"Image too small ({img.height}x{img.width}); minimum 8x8 px.")
        return result

    valid_frac = float(img.valid_mask().mean())
    if valid_frac < 0.05:
        result.is_valid = False
        result.add(f"Image is >{95}% NoData ({valid_frac:.1%} valid pixels).")
        return result
    if valid_frac < 0.5:
        result.add(f"Warning: {1 - valid_frac:.1%} of pixels are NoData; "
                    f"results may have reduced confidence.")

    modality = classify_modality(img)
    result.modality = modality
    if modality == "unknown":
        result.add("Could not confidently classify modality (optical vs SAR); "
                    "proceeding with 'unknown' — some tasks may be unavailable.")

    is_geo = img.crs is not None and img.transform is not None
    ext_is_tif = any(filename.lower().endswith(e) for e in config.GEOSPATIAL_EXTENSIONS)
    if ext_is_tif and not is_geo:
        result.add("TIFF has no embedded CRS/affine transform; "
                    "geospatial outputs (GeoJSON export) will be pixel-only.")

    result.metadata = {
        "filename": filename,
        "band_count": img.band_count,
        "height": img.height,
        "width": img.width,
        "dtype": img.dtype,
        "crs": img.crs,
        "georeferenced": is_geo,
        "valid_pixel_fraction": round(valid_frac, 4),
        "modality": modality,
    }
    return result


# --------------------------------------------------------------------------
# Pair validation (cross-modal or bi-temporal)
# --------------------------------------------------------------------------
def validate_pair(img_a: RasterImage, img_b: RasterImage,
                   pair_type: str) -> PairValidationResult:
    """
    Validate compatibility of two co-registered (or nominally co-registered)
    rasters for either a `cross_modal_pair` or `bi_temporal_pair`.

    Checks
    ------
    * Shape compatibility (flags resample requirement, does not fail).
    * CRS match (if both georeferenced).
    * For cross-modal pairs: one image should classify as optical/
      multispectral and the other as SAR.
    * For bi-temporal pairs: both images should share the same modality.
    """
    result = PairValidationResult(is_valid=True)

    shape_a = (img_a.height, img_a.width)
    shape_b = (img_b.height, img_b.width)
    if shape_a != shape_b:
        diff_pct = 100.0 * abs(shape_a[0] * shape_a[1] - shape_b[0] * shape_b[1]) / \
            max(shape_a[0] * shape_a[1], 1)
        result.requires_resample = True
        result.add(f"Shape mismatch {shape_a} vs {shape_b} "
                    f"({diff_pct:.1f}% area difference); will resample "
                    f"second image onto the first image's grid.")

    if img_a.crs and img_b.crs and img_a.crs != img_b.crs:
        result.add(f"CRS mismatch: '{img_a.crs}' vs '{img_b.crs}'. "
                    f"Reprojection will be applied during resampling.")
    if (img_a.crs is None) != (img_b.crs is None):
        result.add("One image is georeferenced and the other is not; "
                    "pixel-grid alignment only (no CRS-based warping) will be used.")

    mod_a = classify_modality(img_a)
    mod_b = classify_modality(img_b)

    if pair_type == config.INPUT_CONFIG_CROSS_MODAL:
        sar_present = "sar" in (mod_a, mod_b)
        optical_present = any(m in ("optical", "multispectral") for m in (mod_a, mod_b))
        if not (sar_present and optical_present):
            result.is_valid = False
            result.add(f"Cross-modal pair requires one optical/multispectral and one "
                        f"SAR image; classified as ({mod_a}, {mod_b}).")
    elif pair_type == config.INPUT_CONFIG_BI_TEMPORAL:
        if mod_a != mod_b and "unknown" not in (mod_a, mod_b):
            result.add(f"Warning: bi-temporal images classified with different "
                        f"modalities ({mod_a} vs {mod_b}); change statistics may "
                        f"be less reliable across sensor types.")
    else:
        result.is_valid = False
        result.add(f"Unknown pair_type '{pair_type}'.")

    result.metadata = {
        "modality_a": mod_a, "modality_b": mod_b,
        "shape_a": shape_a, "shape_b": shape_b,
        "crs_a": img_a.crs, "crs_b": img_b.crs,
        "pair_type": pair_type,
    }
    return result
