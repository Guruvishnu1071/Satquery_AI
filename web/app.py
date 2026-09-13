"""
web/app.py
===========
Interactive Streamlit GUI for SatQuery AI.

Features
--------
* Four input configurations: Single Image, Bi-Temporal Pair (Change
  Detection), Cross-Modal Pair (Optical + SAR), and an Interactive 3D
  Map Picker that fetches fresh imagery straight from the Copernicus
  Data Space Ecosystem Process API.
* A single, cached fine-tuned EuroSAT classifier (`load_ai_brain`) used
  for scene classification, crop-health (NDVI) scoring, and bi-temporal
  change classification.
* Gemini Vision (`analyze_images_with_gemini`) as the single report-writing
  engine for every mode: it receives the actual image(s) *and* the local
  PyTorch/telemetry evidence, so the final intelligence report is both
  visually and numerically grounded. (The previous local-Ollama LLM path
  has been fully removed.)
* A dedicated, prominently-styled "Changes" panel for Bi-Temporal mode:
  area-altered metric, stable/moderate/critical breakdown, a discrete
  3-colour change heatmap, geospatial impact coordinates, and a Gemini
  Vision verification pass over both T1 and T2 images.

Run with:  streamlit run web/app.py

Required environment variables
-------------------------------
  OPENAI_API_KEY      — for Gemini Vision report generation.
  CDSE_CLIENT_ID       — Copernicus Data Space Ecosystem OAuth client id.
  CDSE_CLIENT_SECRET   — Copernicus Data Space Ecosystem OAuth client secret.
Set these in your shell or a `.env` file loaded before `streamlit run`.
Never hardcode real credentials in this file.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import os

# Force Python to look outside the 'web' folder to find 'core' and 'agent'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Your normal imports (like import streamlit as st) go below this line! ---

import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm

# --------------------------------------------------------------------------
# Project root / import path setup
# --------------------------------------------------------------------------
# Making this absolute (rather than relying on the process's working
# directory) is what fixes the BatchNorm/relative-path divergence between
# `python test.py` and `streamlit run web/app.py` invocations.
root_dir = Path(__file__).resolve().parent.parent
os.chdir(str(root_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from train import setup_model_for_finetuning  # noqa: E402
from core import config  # noqa: E402
from core.geo_io import read_image, normalize_optical, normalize_sar  # noqa: E402
from core.validator import classify_modality  # noqa: E402
from agent.orchestrator import AgentController  # noqa: E402
from web.report_generator import generate_pdf_report, generate_geojson_report  # noqa: E402

config.ensure_dirs()
st.set_page_config(page_title="SatQuery AI", layout="wide", page_icon="🛰️")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
EUROSAT_CLASSES: List[str] = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Infrastructure", "Pasture", "PermanentCrop", "Residential", "SeaLake",
]

EXAMPLE_QUERIES = [
    "Describe the land-cover and major objects visible in this image.",
    "Highlight the water body referred to in the query.",
    "What changed between these two dates, and where did the change occur?",
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "Has the built-up area increased, decreased, or remained unchanged?",
]

INPUT_MODES = [
    "Single Image",
    "Bi-Temporal Pair (Change Detection)",
    "Cross-Modal Pair (Optical + SAR)",
    "Interactive 3D Map Picker",
]

CHANGE_THRESHOLDS = {"moderate": 0.15, "severe": 0.35}

_OPENAI_PLACEHOLDER = "sk-REPLACE_WITH_YOUR_OPENAI_API_KEY"


# --------------------------------------------------------------------------
# Model loading (single, consolidated, cached loader)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading SatQuery AI model...")
def load_ai_brain():
    """
    Loads the fine-tuned EuroSAT (10-class) classifier exactly once per
    Streamlit session. Tries `satquery_custom_model.pth` first and falls
    back to `satquery_v6.pth` if the custom checkpoint isn't present.

    Returns
    -------
    (model, classes, device, checkpoint_name)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_model_for_finetuning(len(EUROSAT_CLASSES))

    candidates = [root_dir / "satquery_custom_model.pth", root_dir / "satquery_v6.pth"]
    loaded_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            checkpoint = torch.load(str(candidate), map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            loaded_path = candidate
            break

    if loaded_path is None:
        raise FileNotFoundError(
            f"No model checkpoint found. Looked for: {[str(c) for c in candidates]}. "
            f"Place one of these files in the project root: {root_dir}"
        )

    model.eval()
    model = model.to(device)
    return model, EUROSAT_CLASSES, device, loaded_path.name


@st.cache_resource
def get_controller() -> AgentController:
    return AgentController()


# --------------------------------------------------------------------------
# Core analytics
# --------------------------------------------------------------------------
def predict_image(image_path: str) -> Tuple[str, float]:
    """Classify a single image into one of the 10 EuroSAT classes."""
    model, classes, device, _ = load_ai_brain()

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = img_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, predicted = torch.max(probabilities, 0)

    predicted_class = classes[predicted.item()]
    confidence = conf.item() * 100
    return predicted_class, confidence


def analyze_crop_health(image_path: str) -> Tuple[str, float]:
    """Calculates crop health using NDVI (multispectral) or a greenness
    index fallback (RGB-only imagery)."""
    import rasterio

    with rasterio.open(image_path) as src:
        image = src.read()

    red = image[0].astype(np.float32)

    if image.shape[0] >= 4:
        nir = image[3].astype(np.float32)
        threshold = 0.4
    else:
        nir = image[1].astype(np.float32)  # green-band fallback for RGB-only imagery
        threshold = 0.05

    health_score = (nir - red) / (nir + red + 1e-8)
    mean_health_score = float(np.mean(health_score))

    status = "🟢 Healthy" if mean_health_score > threshold else "🔴 Needs Attention"
    return status, mean_health_score

def get_current_weather_for_image(image_path: str) -> str:
    """Extracts GPS coordinates from a GeoTIFF and fetches live weather for the AI telemetry."""
    try:
        import rasterio
        import requests
        
        with rasterio.open(image_path) as src:
            # Check if the image has GPS data embedded
            if not (src.transform and src.crs):
                return "Meteorological Data: N/A (Image lacks georeferencing)"
            
            # Find the exact center pixel of the image and convert to Lat/Lon
            center_x, center_y = src.width // 2, src.height // 2
            lon, lat = src.transform * (center_x, center_y)
            
            # Ping Open-Meteo API
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            res = requests.get(url, timeout=3).json()
            cw = res.get("current_weather", {})
            temp = cw.get("temperature", "Unknown")
            wind = cw.get("windspeed", "Unknown")
            
            # Map the weather codes to English for Gemini
            code = cw.get('weathercode', 0)
            if code <= 3: cond = "Clear / Partly Cloudy"
            elif code <= 49: cond = "Fog / Overcast"
            elif code <= 69: cond = "Rain / Drizzle"
            elif code <= 79: cond = "Snow"
            else: cond = "Heavy Storms"
            
            return f"Live Meteorological Context (Lat {lat:.2f}, Lon {lon:.2f}): {temp}°C, Wind {wind} km/h, Conditions: {cond}"
            
    except Exception:
        return "Meteorological Data: N/A (Could not fetch network weather)"

def compute_sar_optical_fusion(optical_path: str, sar_path: str) -> Dict:
    """
    Fuses an Optical image with a SAR image. Extracts low-backscatter
    (dark) regions from SAR (typically water/floods) and projects them
    as a high-visibility cyan overlay on the Optical layer.
    """
    import cv2

    opt_img = cv2.imread(optical_path)
    opt_img = cv2.cvtColor(opt_img, cv2.COLOR_BGR2RGB)

    sar_img = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
    sar_img = cv2.resize(sar_img, (opt_img.shape[1], opt_img.shape[0]))

    _, sar_water_mask = cv2.threshold(sar_img, 50, 255, cv2.THRESH_BINARY_INV)

    cyan_overlay = np.zeros_like(opt_img)
    cyan_overlay[:, :] = [0, 255, 255]

    full_blend = cv2.addWeighted(opt_img, 0.5, cyan_overlay, 0.5, 0)
    mask_3d = sar_water_mask[:, :, None] == 255
    fusion_visual = np.where(mask_3d, full_blend, opt_img)

    total_px = sar_img.size
    water_px = int(np.sum(sar_water_mask == 255))
    water_pct = (water_px / total_px) * 100

    return {
        "fused_image": fusion_visual,
        "water_pct": water_pct,
        "sar_water_mask": sar_water_mask,
    }


def extract_impact_coordinates(image_path: str, anomaly_mask: np.ndarray,
                                threshold: float = 0.35) -> Dict[str, str]:
    """
    Finds the bounding box of affected pixels and maps them to real-world
    GPS coordinates when the source file is a georeferenced GeoTIFF, with
    a pixel-grid fallback for plain PNG/JPEG imagery.
    """
    y_indices, x_indices = np.where(anomaly_mask >= threshold)

    if len(x_indices) == 0:
        return {"center": "N/A", "bounding_box": "No structural anomalies detected", "crs": "N/A"}

    min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
    min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
    center_x, center_y = int(np.mean(x_indices)), int(np.mean(y_indices))

    try:
        import rasterio
        with rasterio.open(image_path) as src:
            if src.transform and src.crs:
                lon_center, lat_center = src.transform * (center_x, center_y)
                lon_min, lat_max = src.transform * (min_x, min_y)
                lon_max, lat_min = src.transform * (max_x, max_y)
                return {
                    "center": f"{lat_center:.5f}° N, {lon_center:.5f}° E",
                    "bounding_box": f"[{lat_min:.5f}, {lon_min:.5f}] to [{lat_max:.5f}, {lon_max:.5f}]",
                    "crs": str(src.crs),
                }
    except Exception:
        pass  # Not a GeoTIFF / no CRS — fall through to pixel-grid coordinates.

    return {
        "center": f"Pixel ({center_x}, {center_y})",
        "bounding_box": f"X:[{min_x}-{max_x}], Y:[{min_y}-{max_y}]",
        "crs": "Pixel Grid (Non-georeferenced)",
    }


def compute_bitemporal_change(path_t1: str, path_t2: str) -> Dict:
    """Computes semantic transition and pixel-level difference heatmap
    between T1 and T2."""
    import rasterio

    class_t1, conf_t1 = predict_image(path_t1)
    class_t2, conf_t2 = predict_image(path_t2)

    def _read_and_normalize(p: str) -> np.ndarray:
        with rasterio.open(p) as src:
            arr = src.read()
        if arr.shape[0] >= 3:
            rgb = arr[:3].astype(np.float32)
        else:
            rgb = np.repeat(arr[0:1], 3, axis=0).astype(np.float32)
        max_val = np.max(rgb)
        if max_val > 0:
            rgb = rgb / max_val
        return np.transpose(rgb, (1, 2, 0))

    img1 = _read_and_normalize(path_t1)
    img2 = _read_and_normalize(path_t2)

    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1, img2 = img1[:h, :w], img2[:h, :w]

    diff = np.abs(img2 - img1)
    diff_magnitude = np.mean(diff, axis=2)

    change_mask = diff_magnitude > CHANGE_THRESHOLDS["moderate"]
    change_percentage = float((np.sum(change_mask) / (h * w)) * 100.0)

    return {
        "class_t1": class_t1, "conf_t1": conf_t1,
        "class_t2": class_t2, "conf_t2": conf_t2,
        "heatmap": diff_magnitude,
        "change_pct": change_percentage,
    }

def _render_change_heatmap(heatmap_arr: np.ndarray) -> None:
    """Renders a discrete 3-colour (Green/Yellow/Red) change heatmap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = ListedColormap(["#2ecc40", "#ffdc00", "#ff4136"])
    max_val = max(1.0, float(np.max(heatmap_arr)))
    bounds = [0.0, CHANGE_THRESHOLDS["moderate"], CHANGE_THRESHOLDS["severe"], max_val]
    norm = BoundaryNorm(bounds, cmap.N)
    
    ax.imshow(heatmap_arr, cmap=cmap, norm=norm)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Gemini Vision — single universal report-generation engine
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Gemini Vision — Free multimodal report-generation engine
# --------------------------------------------------------------------------
def _load_image_for_gemini(image_path: str, max_size: int = 1024):
    """
    Loads and converts any supported input (PNG/JPEG or GeoTIFF/SAR) into a
    standard RGB PIL Image for Gemini to physically 'look' at.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        # Fallback for GeoTIFF/SAR using your existing geo_io logic
        from core.geo_io import read_image, normalize_optical, normalize_sar
        from core.validator import classify_modality
        raster = read_image(image_path)
        if raster.modality == "unknown":
            raster.modality = classify_modality(raster)
        if raster.modality == "sar":
            norm = normalize_sar(raster, config.THRESHOLDS["sar_db_min"], config.THRESHOLDS["sar_db_max"])
            gray = norm[0]
            rgb = np.stack([gray, gray, gray], axis=-1)
        else:
            norm = normalize_optical(raster, config.THRESHOLDS["optical_percentile_low"],
                                      config.THRESHOLDS["optical_percentile_high"])
            rgb = np.transpose(norm[:3], (1, 2, 0)) if norm.shape[0] >= 3 else \
                np.stack([norm[0]] * 3, axis=-1)
        arr = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    img.thumbnail((max_size, max_size))
    return img


def analyze_images_with_gemini(image_paths: List[str], user_query: str, telemetry: str) -> str:
    """
    Universal multimodal reporting function using Google's free Gemini API.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "⚠️ Gemini API is not configured. Set the `GEMINI_API_KEY` environment "
            "variable.\n\n**Raw telemetry evidence (unformatted):**\n{telemetry}"
        )

    try:
        from google import genai
    except ImportError:
        return (
            "⚠️ The `google-genai` package is not installed. Run `pip install google-genai`.\n\n"
            f"**Raw telemetry evidence:**\n{telemetry}"
        )

    system_instruction = (
        "You are SatQuery AI, an expert satellite intelligence analyst. "
        "Cross-check the visual imagery against the numeric telemetry provided. "
        "Treat the telemetry as ground truth. Write a professional, concise intelligence report."
    )

    # Gemini is brilliant: it accepts a simple list of text and raw PIL Images!
    contents = [
        f"{system_instruction}\n\nAnalyst query: {user_query}\n\nTelemetry Evidence:\n{telemetry}"
    ]
    
    for path in image_paths[:2]:
        try:
            img = _load_image_for_gemini(path)
            contents.append(img)
        except Exception as exc:
            contents[0] += f"\n\n(Note: could not attach image '{path}': {exc})"

    try:
        # Connect to Google and use the high-speed 3.5 Flash model
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=contents,
        )
        return response.text
    except Exception as exc:
        return f"⚠️ Gemini API request failed: {exc}\n\n**Raw telemetry:**\n{telemetry}"


# --------------------------------------------------------------------------
# Streamlit display helpers
# --------------------------------------------------------------------------
def _save_upload(uploaded_file) -> str:
    dest = config.UPLOAD_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)


def _display_array_for_plot(path: str) -> np.ndarray:
    """Load and return an RGB-displayable (H, W, 3) uint8 array from any
    supported raster, for quick preview / overlay rendering."""
    img = read_image(path)
    if img.modality == "unknown":
        img.modality = classify_modality(img)
    if img.modality == "sar":
        norm = normalize_sar(img, config.THRESHOLDS["sar_db_min"], config.THRESHOLDS["sar_db_max"])
        gray = norm[0]
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        norm = normalize_optical(img, config.THRESHOLDS["optical_percentile_low"],
                                  config.THRESHOLDS["optical_percentile_high"])
        if norm.shape[0] >= 3:
            rgb = np.transpose(norm[:3], (1, 2, 0))
        else:
            gray = norm[0]
            rgb = np.stack([gray, gray, gray], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def _render_overlay(path: str, boxes: list, title: str) -> None:
    arr = _display_array_for_plot(path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    colors = {"water": "#1f77ff", "builtup": "#ff4136", "vegetation": "#2ecc40",
              "change": "#ffdc00", "road": "#b10dc9", "salient": "#ff851b"}
    for b in boxes:
        c = colors.get(b.get("label"), "#ffffff")
        rect = patches.Rectangle((b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                                  linewidth=2, edgecolor=c, facecolor="none")
        ax.add_patch(rect)
        ax.text(b["xmin"], max(b["ymin"] - 4, 0), f"{b.get('label','')} {b.get('score',0):.2f}",
                color=c, fontsize=8, weight="bold")
    st.pyplot(fig)
    plt.close(fig)


def _render_map_picker():
    st.subheader("🌍 SatQuery AI — 3D Global Imagery & Daily NRT Picker")
    st.markdown("Use the 3D globe below to navigate, draw your Area of Interest (AOI), "
                "check daily feeds, and fetch data.")

    import streamlit.components.v1 as components
    import os

    component_dir = os.path.join(os.path.dirname(__file__), "map_component")
    if not os.path.exists(component_dir):
        st.error("⚠️ Could not find the 'web/map_component' folder. Did you create it and "
                  "rename the HTML to index.html?")
        return

    map_picker = components.declare_component("map_picker", path=component_dir)
    map_data = map_picker(key="map_picker_widget")

    if not map_data:
        return

    bbox = map_data.get("bbox")
    date = map_data.get("date")
    
    # 1. Calculate the exact Center Coordinates from the drawn Bounding Box
    center_lon = (bbox[0] + bbox[2]) / 2.0
    center_lat = (bbox[1] + bbox[3]) / 2.0

    st.success(f"🎯 Coordinates Acquired: [W: {bbox[0]:.2f}, S: {bbox[1]:.2f}, E: {bbox[2]:.2f}, N: {bbox[3]:.2f}]")
    
    # 2. THE NEW WEATHER DASHBOARD
    st.divider()
    st.subheader(f"🌤️ Live Meteorological Report (Lat: {center_lat:.2f}, Lon: {center_lon:.2f})")
    try:
        # Ping the free Open-Meteo API using the map coordinates
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={center_lat}&longitude={center_lon}&current_weather=true"
        w_res = requests.get(weather_url, timeout=5).json()
        current_weather = w_res.get("current_weather", {})
        
        # Interpret the WMO Weather Code for the UI
        code = current_weather.get('weathercode', 0)
        if code <= 3: condition = "Clear / Partly Cloudy 🌤️"
        elif code <= 49: condition = "Fog / Overcast 🌫️"
        elif code <= 69: condition = "Rain / Drizzle 🌧️"
        elif code <= 79: condition = "Snow ❄️"
        else: condition = "Heavy Storms ⛈️"

        # Display the metrics beautifully in 3 columns
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.metric(label="🌡️ Temperature", value=f"{current_weather.get('temperature')} °C")
        with col_w2:
            st.metric(label="💨 Wind Speed", value=f"{current_weather.get('windspeed')} km/h")
        with col_w3:
            st.metric(label="☁️ Conditions", value=condition)
            
    except Exception as e:
        st.warning(f"Could not fetch live weather data: {e}")
        
    st.divider()

    # 3. Resume the normal image download process
    st.info(f"📅 Target Date: {date} | Initiating Sentinel-2 download...")
    with st.spinner("Fetching high-res GeoTIFF from Copernicus Space API. This takes ~15 seconds..."):
        try:
            # Saving directly to the standard upload directory
            out_filename = str(config.UPLOAD_DIR / f"satquery_download_{date}.tif")
            saved_path = fetch_geotiff_from_copernicus(
                bbox=bbox, date_from=date, date_to=date, output_filename=out_filename,
            )
            st.success(f"✅ Image successfully downloaded and saved to: `{saved_path}`")
            
            # --- NEW BROWSER DOWNLOAD BUTTON ---
            # We open the saved file in "rb" (read-binary) mode so Streamlit can send it
            with open(saved_path, "rb") as file_data:
                st.download_button(
                    label="📥 Download GeoTIFF to your PC",
                    data=file_data,
                    file_name=f"satquery_{date}.tif",
                    mime="image/tiff"
                )
            # -----------------------------------

            st.markdown("You can now switch to **Single Image** or **Bi-Temporal Pair** mode "
                        "and upload this file for AI analysis!")
        except Exception as exc:
            st.error(f"Download failed. Check your API credentials. Error: {exc}")


def _handle_single_image(image_paths: List[str], query: str):
    try:
        predicted_class, confidence = predict_image(image_paths[0])

        st.divider()
        st.success(f"### 🎯 Deep Learning Classification: {predicted_class}")
        st.info(f"**Neural Network Confidence:** {confidence:.2f}%")

        reliability = ("High Confidence" if confidence >= 75 else
                       "Low Confidence - visual signature is highly anomalous or distorted "
                       "(potential disaster/flood zone).")
        if confidence < 75:
            st.warning("⚠️ Low confidence detected. Terrain may be experiencing severe "
                       "environmental disturbance.")

        ndvi_info = "N/A (Terrain is non-agricultural)"
        if any(k in predicted_class for k in ("Crop", "Forest", "Vegetation", "Pasture")):
            st.divider()
            st.subheader("🌾 Agricultural Intelligence")
            with st.spinner("Calculating NDVI crop health metrics..."):
                health_status, ndvi_score = analyze_crop_health(image_paths[0])
                ndvi_info = f"Status: {health_status}, NDVI Score: {ndvi_score:.3f}"

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("NDVI Score", f"{ndvi_score:.3f}")
                with col2:
                    if "Healthy" in health_status:
                        st.success(f"**Status:** {health_status}")
                    else:
                        st.warning(f"**Status:** {health_status} (Possible water stress or disease)")
                st.caption("NDVI (Normalized Difference Vegetation Index) calculated using spectral bands.")

        # 1. Ask the helper function to calculate the weather for this specific image
        weather_context = get_current_weather_for_image(image_paths[0])

        # 2. Add the new weather variable to Gemini's cheat sheet
        telemetry = (
            f"- Primary Land Classification: {predicted_class}\n"
            f"- Model Confidence: {confidence:.2f}% ({reliability})\n"
            f"- Multispectral / NDVI Health: {ndvi_info}\n"
            f"- {weather_context}"  # <--- WE ADDED THIS NEW LINE!
        )

        st.divider()
        st.subheader("🧠 Satquery Vision Intelligence Report")
        with st.spinner("Satquery is inspecting the imagery and drafting a report..."):
            report = analyze_images_with_gemini([image_paths[0]], query, telemetry)
            st.success(report)

    except Exception as exc:
        st.error(f"The AI Brain encountered an error: {exc}")


def _handle_bitemporal(image_paths: List[str], query: str):
    try:
        st.divider()
        st.subheader("🛰️ Bi-Temporal Land Change Analysis")

        with st.spinner("Analyzing temporal delta between T1 and T2..."):
            results = compute_bitemporal_change(image_paths[0], image_paths[1])

        col_t1, col_arrow, col_t2 = st.columns([4, 1, 4])
        with col_t1:
            st.markdown(f"**Time 1 (Initial):** {results['class_t1']}")
            st.caption(f"Confidence: {results['conf_t1']:.1f}%")
        with col_arrow:
            st.markdown("### ➡️")
        with col_t2:
            st.markdown(f"**Time 2 (Current):** {results['class_t2']}")
            st.caption(f"Confidence: {results['conf_t2']:.1f}%")

        if results["class_t1"] != results["class_t2"]:
            st.warning(f"🚨 **Land-Cover Shift Detected:** Transition from "
                      f"`{results['class_t1']}` to `{results['class_t2']}`.")
        else:
            st.success(f"✅ **Land-Cover Stable:** Classified as `{results['class_t1']}` "
                      f"across both timestamps.")

        # -------------------- "Changes" panel --------------------
        st.divider()
        with st.container(border=True):
            st.markdown("## 🔍 Changes")

            heatmap_arr = results["heatmap"]
            total_px = heatmap_arr.size
            green_pct = float(np.sum(heatmap_arr < CHANGE_THRESHOLDS["moderate"]) / total_px * 100)
            yellow_pct = float(np.sum((heatmap_arr >= CHANGE_THRESHOLDS["moderate"]) &
                                       (heatmap_arr < CHANGE_THRESHOLDS["severe"])) / total_px * 100)
            red_pct = float(np.sum(heatmap_arr >= CHANGE_THRESHOLDS["severe"]) / total_px * 100)

            spatial_data = extract_impact_coordinates(image_paths[1], heatmap_arr,
                                                       threshold=CHANGE_THRESHOLDS["severe"])

            col_m1, col_m2 = st.columns([1, 3])
            with col_m1:
                st.metric(
                    label="Area Altered",
                    value=f"{results['change_pct']:.2f}%",
                    delta=f"{results['change_pct']:.2f}% Change",
                    delta_color="inverse",
                )
                progress_val = min(results["change_pct"] / 100.0, 1.0)
                if results["change_pct"] >= CHANGE_THRESHOLDS["severe"] * 100:
                    st.error("🚨 Critical morphological impact detected.")
                elif results["change_pct"] >= CHANGE_THRESHOLDS["moderate"] * 100:
                    st.warning("⚠️ Moderate environmental disturbance.")
                else:
                    st.info("✅ Negligible physical variance.")
                st.progress(progress_val)

                st.caption(f"🟢 **Stable Surface Area:** {green_pct:.1f}%")
                st.caption(f"🟡 **Moderate Disturbance:** {yellow_pct:.1f}%")
                st.caption(f"🔴 **Severe / Critical Transformation:** {red_pct:.1f}%")

                st.markdown("**📍 Spatial Impact Coordinates**")
                st.write(f"Target Center: {spatial_data['center']}")
                st.write(f"Bounding Box: {spatial_data['bounding_box']}")
                st.write(f"CRS: {spatial_data['crs']}")

            with col_m2:
                st.markdown("**Change Magnitude Heatmap**")
                _render_change_heatmap(heatmap_arr)

            st.markdown("**Side-by-Side Transition (T1 → T2)**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(_display_array_for_plot(image_paths[0]), caption="T1 (Before)", use_container_width=True)
            with col_b:
                st.image(_display_array_for_plot(image_paths[1]), caption="T2 (After)", use_container_width=True)

            telemetry = (
                f"- Previous Land Cover: {results['class_t1']}\n"
                f"- Current Land Cover: {results['class_t2']}\n"
                f"- Total Area Altered: {results['change_pct']:.2f}%\n"
                f"- Stable / Moderate / Critical breakdown: {green_pct:.1f}% / {yellow_pct:.1f}% / {red_pct:.1f}%\n"
                f"- Threat Level: {'Critical' if results['change_pct'] >= 35 else 'Moderate' if results['change_pct'] >= 15 else 'Negligible'}\n"
                f"- Target Center Coordinates: {spatial_data['center']}\n"
                f"- Impact Bounding Zone: {spatial_data['bounding_box']}\n"
                f"- Coordinate Reference System: {spatial_data['crs']}"
            )

            st.markdown("**🧠 Satquery Vision Verification**")
            with st.spinner("Satquery is visually comparing T1 and T2..."):
                report = analyze_images_with_gemini(
                    [image_paths[0], image_paths[1]], query, telemetry,
                )
                st.success(report)

    except Exception as exc:
        st.error(f"Bi-temporal pipeline error: {exc}")


def _handle_cross_modal(image_paths: List[str], query: str):
    st.divider()
    st.subheader("Cross-Modal (Optical + SAR) Fusion")

    try:
        fusion_results = compute_sar_optical_fusion(image_paths[0], image_paths[1])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_paths[0], caption="Optical (Clouded/Context)", use_container_width=True)
        with col2:
            st.image(image_paths[1], caption="SAR (Cloud-Penetrating)", use_container_width=True)
        with col3:
            st.image(fusion_results["fused_image"], caption="Fused Threat Map", use_container_width=True)

        threat_level = ("Critical (Widespread Inundation)" if fusion_results["water_pct"] > 15
                        else "Moderate (Localized Pooling)")

        telemetry = (
            "- Sensor 1 (Optical): Degraded visibility / Context baseline\n"
            "- Sensor 2 (SAR): Active microwave penetration successful\n"
            "- Cross-Modal Verification: Positive for low-backscatter anomalies (Water)\n"
            f"- Area Submerged: {fusion_results['water_pct']:.1f}%\n"
            f"- Tactical Threat Level: {threat_level}"
        )

        st.subheader("🧠 Satquery Multi-Sensor Intelligence Report")
        with st.spinner("Satquery is cross-referencing optical and SAR imagery..."):
            report = analyze_images_with_gemini([image_paths[0], image_paths[1]], query, telemetry)
            st.success(report)

    except Exception as exc:
        st.error(f"Cross-modal fusion pipeline error: {exc}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.title("🛰️ SatQuery AI")
    st.caption("Agentic Vision-Language Assistant for Multimodal Remote Sensing Image Analysis")

    with st.sidebar:
        st.header("1. Input Configuration")
        input_mode = st.radio(
            "Select input configuration", INPUT_MODES, key="main_input_mode_selector",
        )
        st.divider()
        st.header("Tool Registry")
        try:
            controller_preview = get_controller()
            for tool in controller_preview.pipeline_graph():
                with st.expander(tool["name"]):
                    st.write(f"**Tasks:** {', '.join(tool['task_types'])}")
                    st.write(f"**Input configs:** {', '.join(tool['input_configs'])}")
                    st.caption(tool["description"])
        except Exception as exc:
            st.warning(f"Tool registry unavailable: {exc}")

    image_paths: List[str] = []

    if input_mode == "Interactive 3D Map Picker":
        _render_map_picker()

    elif input_mode == "Single Image":
        f = st.file_uploader("Upload GeoTIFF/TIFF/PNG/JPEG", type=["tif", "tiff", "png", "jpg", "jpeg"])
        if f:
            image_paths = [_save_upload(f)]

    elif input_mode == "Bi-Temporal Pair (Change Detection)":
        col1, col2 = st.columns(2)
        with col1:
            f1 = st.file_uploader("Time T1 image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="t1")
        with col2:
            f2 = st.file_uploader("Time T2 image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="t2")
        if f1 and f2:
            image_paths = [_save_upload(f1), _save_upload(f2)]

    else:  # Cross-Modal Pair
        col1, col2 = st.columns(2)
        with col1:
            f1 = st.file_uploader("Optical / Multispectral image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="opt")
        with col2:
            f2 = st.file_uploader("SAR image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="sar")
        if f1 and f2:
            image_paths = [_save_upload(f1), _save_upload(f2)]

    # Natural-language query is shown for every mode except the map picker.
    query = ""
    run = False
    if input_mode != "Interactive 3D Map Picker":
        st.header("2. Natural-Language Query")
        example = st.selectbox("Example queries (optional)", ["(type your own below)"] + EXAMPLE_QUERIES)
        default_text = "" if example == "(type your own below)" else example
        query = st.text_area("Query", value=default_text, height=80)
        run = st.button("🚀 Run SatQuery AI", type="primary", disabled=not (image_paths and query.strip()))

    if run:
        with st.spinner("Initializing Deep Learning inference..."):
            if input_mode == "Single Image":
                _handle_single_image(image_paths, query)
            elif input_mode == "Bi-Temporal Pair (Change Detection)":
                _handle_bitemporal(image_paths, query)
            else:
                _handle_cross_modal(image_paths, query)


# --------------------------------------------------------------------------
# Copernicus Data Space Ecosystem — Process API fetch
# --------------------------------------------------------------------------
def fetch_geotiff_from_copernicus(bbox: List[float], date_from: str, date_to: str,
                                   output_filename: str = "downloaded_scene.tif") -> str:
    """
    Sends a request to the Copernicus Data Space Ecosystem (CDSE) Sentinel
    Hub Process API for a True-Color RGB (B04/B03/B02) GeoTIFF over the
    given `bbox`.

    Credentials are read from environment variables — set
    `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` before running the app.
    Register an OAuth client at
    https://shapps.dataspace.copernicus.eu/dashboard/ if you don't have one.

    Parameters
    ----------
    bbox : [west, south, east, north] in EPSG:4326.
    date_from, date_to : "YYYY-MM-DD" strings.
    output_filename : destination path for the downloaded GeoTIFF.
    """
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise EnvironmentError(
            "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET environment variables must be set. "
            "Register a client at https://shapps.dataspace.copernicus.eu/dashboard/ "
            "and never hardcode these values in source code."
        )

    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    auth_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }
    token_resp = requests.post(auth_url, data=auth_data, timeout=30)
    if token_resp.status_code != 200:
        raise RuntimeError(f"Authentication failed: {token_resp.text}")
    token = token_resp.json().get("access_token")

    process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    evalscript_code = """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3 }
        }
    }
    function evaluatePixel(sample) {
        return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
    }
    """

    payload = {
        "input": {
            "bounds": {"bbox": bbox},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date_from}T00:00:00Z",
                        "to": f"{date_to}T23:59:59Z",
                    }
                },
            }],
        },
        "evalscript": evalscript_code,
        "output": {
            "width": 512, "height": 512,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
    }

    response = requests.post(process_url, json=payload, headers=headers, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch GeoTIFF: {response.text}")

    with open(output_filename, "wb") as f:
        f.write(response.content)
    return output_filename


if __name__ == "__main__":
    main()
