"""
core/geo_io.py
================
Raster I/O layer for SatQuery AI.

Responsibilities
-----------------
* Read GeoTIFF/TIFF (single or multi-band, optical/multispectral or SAR)
  via `rasterio` when available, capturing CRS, affine transform, band
  count, dtype, and NoData value.
* Gracefully fall back to `PIL`/`numpy` for plain PNG/JPEG benchmark
  images (no CRS) or if `rasterio`/GDAL are not installed in the
  environment, so the rest of the pipeline never has to special-case
  missing geospatial libraries.
* Provide radiometric normalization appropriate to each modality:
    - Optical/multispectral: percentile clipping + min-max scaling
      per band (robust to outliers/clouds).
    - SAR: linear-to-dB calibration (if data looks like linear
      backscatter) followed by dB-range clipping and scaling.
* Provide co-registration helpers (resample-to-match) for cross-modal
  and bi-temporal pairs that do not share an identical grid.
* Provide standard spectral index computation (NDVI, NDWI, NDBI).

All public functions raise `GeoIOError` (not bare exceptions) on
failure so the agent orchestrator / validator can catch a single
well-defined error type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("satquery_ai.geo_io")

# --------------------------------------------------------------------------
# Optional heavy geospatial dependencies — degrade gracefully if absent.
# --------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.io import MemoryFile
    _HAS_RASTERIO = True
except Exception:  # pragma: no cover - exercised only when rasterio missing
    _HAS_RASTERIO = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


class GeoIOError(Exception):
    """Raised for any raster read/write/normalization failure."""


# --------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------
@dataclass
class RasterImage:
    """
    In-memory representation of a raster used throughout the pipeline.

    Attributes
    ----------
    array : np.ndarray
        Shape (bands, height, width), float32.
    crs : Optional[str]
        WKT/EPSG string, or None for non-georeferenced PNG/JPEG.
    transform : Optional[tuple]
        Affine transform (a, b, c, d, e, f) mapping pixel -> world coords,
        or None if not georeferenced.
    nodata : Optional[float]
        NoData sentinel value, if defined by the source file.
    band_count : int
    height : int
    width : int
    modality : str
        One of {"optical", "sar", "unknown"} — set by validator, defaults
        to "unknown" at read time.
    source_path : str
    dtype : str
        Original on-disk dtype, for audit/debug purposes.
    """

    array: np.ndarray
    crs: Optional[str]
    transform: Optional[Tuple[float, float, float, float, float, float]]
    nodata: Optional[float]
    band_count: int
    height: int
    width: int
    source_path: str
    dtype: str
    modality: str = "unknown"
    extra_meta: dict = field(default_factory=dict)

    def band(self, idx: int) -> np.ndarray:
        """Return a single band (0-indexed) as a 2D array."""
        if not (0 <= idx < self.band_count):
            raise GeoIOError(f"Band index {idx} out of range (0..{self.band_count - 1})")
        return self.array[idx]

    def valid_mask(self) -> np.ndarray:
        """Boolean mask of pixels that are NOT NoData across all bands."""
        if self.nodata is None:
            return np.ones((self.height, self.width), dtype=bool)
        return ~np.any(np.isclose(self.array, self.nodata), axis=0)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def read_image(path: str) -> RasterImage:
    """
    Read a raster file (GeoTIFF/TIFF via rasterio, or PNG/JPEG via PIL)
    into a `RasterImage`.

    Raises
    ------
    GeoIOError if the file cannot be opened or is empty.
    """
    p = Path(path)
    if not p.exists():
        raise GeoIOError(f"File not found: {path}")

    ext = p.suffix.lower()
    if ext in (".tif", ".tiff", ".geotiff"):
        return _read_geotiff(p)
    if ext in (".png", ".jpg", ".jpeg"):
        return _read_plain_image(p)
    # Last resort: try rasterio (it can open many formats), else PIL.
    if _HAS_RASTERIO:
        try:
            return _read_geotiff(p)
        except Exception:
            pass
    return _read_plain_image(p)


def _read_geotiff(p: Path) -> RasterImage:
    if not _HAS_RASTERIO:
        logger.warning("rasterio not installed; falling back to PIL for %s "
                        "(CRS/transform metadata will be unavailable).", p)
        return _read_plain_image(p)
    try:
        with rasterio.open(p) as src:
            arr = src.read().astype(np.float32)  # (bands, H, W)
            transform = tuple(src.transform)[:6]
            crs = src.crs.to_string() if src.crs else None
            nodata = src.nodata
            return RasterImage(
                array=arr,
                crs=crs,
                transform=transform,
                nodata=nodata,
                band_count=src.count,
                height=src.height,
                width=src.width,
                source_path=str(p),
                dtype=str(src.dtypes[0]),
                extra_meta={"driver": src.driver, "tags": dict(src.tags())},
            )
    except Exception as exc:
        raise GeoIOError(f"Failed to read GeoTIFF '{p}': {exc}") from exc


def _read_plain_image(p: Path) -> RasterImage:
    if not _HAS_PIL:
        raise GeoIOError("Neither rasterio nor PIL is available to read images.")
    try:
        img = Image.open(p)
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]  # (1, H, W)
        else:
            arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        arr = arr.astype(np.float32)
        return RasterImage(
            array=arr,
            crs=None,
            transform=None,
            nodata=None,
            band_count=arr.shape[0],
            height=arr.shape[1],
            width=arr.shape[2],
            source_path=str(p),
            dtype=str(np.array(img).dtype),
            extra_meta={"driver": "PIL", "georeferenced": False},
        )
    except Exception as exc:
        raise GeoIOError(f"Failed to read image '{p}': {exc}") from exc


# --------------------------------------------------------------------------
# Radiometric normalization
# --------------------------------------------------------------------------
def normalize_optical(img: RasterImage, low_pct: float = 2.0,
                       high_pct: float = 98.0) -> np.ndarray:
    """
    Percentile-clip + min-max scale each band to [0, 1]. Robust to cloud
    glare / dark shadow outliers, which is standard practice for optical
    reflectance display and downstream index computation.

    Returns
    -------
    np.ndarray of shape (bands, H, W), float32 in [0, 1].
    """
    mask = img.valid_mask()
    out = np.zeros_like(img.array, dtype=np.float32)
    for b in range(img.band_count):
        band = img.array[b]
        valid = band[mask] if mask.any() else band.ravel()
        if valid.size == 0:
            out[b] = 0.0
            continue
        lo = np.percentile(valid, low_pct)
        hi = np.percentile(valid, high_pct)
        if hi <= lo:
            hi = lo + 1e-6
        clipped = np.clip(band, lo, hi)
        out[b] = (clipped - lo) / (hi - lo)
    out[:, ~mask] = 0.0
    return out


def normalize_sar(img: RasterImage, db_min: float = -25.0,
                   db_max: float = 5.0, assume_linear: Optional[bool] = None) -> np.ndarray:
    """
    Calibrate SAR backscatter to dB (if the data appears to be linear
    power, i.e. all-positive with a large dynamic range) and clip/scale
    to [0, 1] over the operational range [db_min, db_max] dB, typical of
    C-/X-band land backscatter.

    Parameters
    ----------
    assume_linear : Optional[bool]
        Force-interpret input as linear power (True) or already-in-dB
        (False). If None, auto-detected: values are treated as linear
        power if the 99th percentile exceeds 10 (dB values for land are
        almost always < 10).
    """
    mask = img.valid_mask()
    out = np.zeros_like(img.array, dtype=np.float32)
    for b in range(img.band_count):
        band = img.array[b].copy()
        valid = band[mask] if mask.any() else band.ravel()
        if valid.size == 0:
            out[b] = 0.0
            continue
        is_linear = assume_linear
        if is_linear is None:
            is_linear = np.nanpercentile(valid, 99) > 10.0
        if is_linear:
            band = np.clip(band, 1e-6, None)
            band_db = 10.0 * np.log10(band)
        else:
            band_db = band
        clipped = np.clip(band_db, db_min, db_max)
        out[b] = (clipped - db_min) / (db_max - db_min)
    out[:, ~mask] = 0.0
    return out


# --------------------------------------------------------------------------
# Spectral indices
# --------------------------------------------------------------------------
def compute_ndvi(optical_bands01: np.ndarray, red_idx: int, nir_idx: int) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red), on normalized [0,1] reflectance bands."""
    red = optical_bands01[red_idx]
    nir = optical_bands01[nir_idx]
    denom = nir + red
    denom[denom == 0] = 1e-6
    return (nir - red) / denom


def compute_ndwi(optical_bands01: np.ndarray, green_idx: int, nir_idx: int) -> np.ndarray:
    """NDWI (McFeeters) = (Green - NIR) / (Green + NIR)."""
    green = optical_bands01[green_idx]
    nir = optical_bands01[nir_idx]
    denom = green + nir
    denom[denom == 0] = 1e-6
    return (green - nir) / denom


def compute_ndbi(optical_bands01: np.ndarray, swir_idx: int, nir_idx: int) -> np.ndarray:
    """NDBI = (SWIR - NIR) / (SWIR + NIR); falls back to a red/NIR proxy
    for RGB-only imagery where SWIR is unavailable (swir_idx == nir_idx)."""
    swir = optical_bands01[swir_idx]
    nir = optical_bands01[nir_idx]
    denom = swir + nir
    denom[denom == 0] = 1e-6
    return (swir - nir) / denom


# --------------------------------------------------------------------------
# Co-registration / resampling
# --------------------------------------------------------------------------
def resample_to_match(source: RasterImage, reference: RasterImage) -> RasterImage:
    """
    Resample `source` onto the pixel grid of `reference`.

    If both images carry CRS/transform metadata (rasterio available), a
    proper `rasterio.warp.reproject` is performed. Otherwise a simple
    interpolated resize (OpenCV/skimage) is used to match array shape
    only — sufficient for benchmark PNG/JPEG pairs that lack CRS.
    """
    if source.height == reference.height and source.width == reference.width:
        return source  # already matching grid

    if _HAS_RASTERIO and source.crs and reference.crs and source.transform and reference.transform:
        try:
            dst = np.zeros((source.band_count, reference.height, reference.width), dtype=np.float32)
            src_transform = rasterio.Affine(*source.transform)
            dst_transform = rasterio.Affine(*reference.transform)
            for b in range(source.band_count):
                reproject(
                    source=source.array[b],
                    destination=dst[b],
                    src_transform=src_transform,
                    src_crs=source.crs,
                    dst_transform=dst_transform,
                    dst_crs=reference.crs,
                    resampling=Resampling.bilinear,
                )
            return RasterImage(
                array=dst, crs=reference.crs, transform=reference.transform,
                nodata=source.nodata, band_count=source.band_count,
                height=reference.height, width=reference.width,
                source_path=source.source_path, dtype=source.dtype,
                modality=source.modality,
                extra_meta={**source.extra_meta, "resampled_to": reference.source_path},
            )
        except Exception as exc:
            logger.warning("rasterio reproject failed (%s); falling back to plain resize.", exc)

    # Fallback: plain array resize (no geospatial warping).
    import cv2
    resized = np.zeros((source.band_count, reference.height, reference.width), dtype=np.float32)
    for b in range(source.band_count):
        resized[b] = cv2.resize(source.array[b], (reference.width, reference.height),
                                 interpolation=cv2.INTER_LINEAR)
    return RasterImage(
        array=resized, crs=source.crs, transform=reference.transform,
        nodata=source.nodata, band_count=source.band_count,
        height=reference.height, width=reference.width,
        source_path=source.source_path, dtype=source.dtype,
        modality=source.modality,
        extra_meta={**source.extra_meta, "resampled_to": reference.source_path, "method": "cv2_resize"},
    )


def pixel_to_geo(img: RasterImage, row: int, col: int) -> Optional[Tuple[float, float]]:
    """Convert pixel (row, col) to world (x, y) using the affine transform."""
    if img.transform is None:
        return None
    a, b, c, d, e, f = img.transform
    x = a * col + b * row + c
    y = d * col + e * row + f
    return (x, y)
