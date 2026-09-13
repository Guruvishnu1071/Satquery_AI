# SatQuery AI

**An Interactive, Agentic Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Natural-Language Queries.**

SatQuery AI is a query-driven agentic system that analyses single, bi-temporal, and
cross-modal (Optical + SAR) remote-sensing imagery. A deterministic **Agent
Controller** parses a natural-language query, validates the supplied imagery,
selects the correct specialist tool(s) from a **Tool Registry**, executes the
pipeline, and returns an **evidence-grounded, auditable** answer (text +
bounding boxes / change masks + confidence + a structured JSON execution
trace suitable for ISRO/SAC-style audit).

## Why the implementation looks the way it does

This repository is built to run **fully offline, dependency-light, and
deterministically**, because:

1. Pretrained domain-adapted VLM checkpoints (GeoChat, RemoteCLIP,
   Qwen2-VL-LoRA fine-tuned on BigEarthNet/RSVQA/VRSBench) are multi-GB
   downloads that cannot be fetched inside an offline evaluation sandbox.
2. ISRO/SAC evaluation requires a **deterministic, auditable** execution
   trace — every module here is built primarily on transparent,
   physically-grounded remote-sensing algorithms (spectral indices, change
   vector analysis, SAR backscatter statistics, texture/segmentation-based
   grounding) so that every answer can be traced back to concrete raster
   evidence.
3. Every module in `models/` is written against a **pluggable backbone
   interface** (`core/adaptation_base.py` → `RSFeatureBackbone`). When
   `torch` + `transformers` + `peft` + real checkpoints (RemoteCLIP,
   GeoChat, Qwen2-VL) are available in the deployment environment, they are
   auto-detected and used to *augment* the deterministic evidence with
   learned embeddings / LoRA-adapted generation. When they are not
   available, the system **falls back to fully classical, still-correct**
   remote-sensing computer vision, rather than failing or returning
   placeholders.

This means the codebase is genuinely runnable today (`pip install -r
requirements.txt` covers the classical path; the deep-learning path is an
opt-in upgrade), while being architected exactly to the required module
breakdown so that swapping in fine-tuned weights is a drop-in change to
`models/adaptation_base.py`.

## Folder structure

```
satquery_ai/
├── core/
│   ├── config.py         # paths, model/tool registry config, formats, thresholds
│   ├── geo_io.py         # GeoTIFF/TIFF/SAR reader (rasterio + gdal, with PIL fallback)
│   └── validator.py      # input compatibility & metadata validation
├── agent/
│   ├── orchestrator.py       # Agentic controller (intent → routing → execution)
│   ├── tool_registry.py      # Registry decorator + tool interfaces
│   └── execution_tracer.py   # Auditable JSON execution trace + confidence
├── models/
│   ├── adaptation_base.py    # RemoteCLIP/LoRA backbone wrapper (pluggable)
│   ├── single_vqa_caption.py # RSVQA / captioning / grounding QA engine
│   ├── visual_grounding.py   # text-guided bounding-box grounding
│   ├── change_analyzer.py    # bi-temporal change detection & CD-VQA
│   └── optical_sar_fusion.py # optical+SAR cross-modal fusion
├── web/
│   ├── app.py                # Streamlit GUI
│   └── report_generator.py   # PDF / GeoJSON report export
└── evaluation/
    ├── benchmark_eval.py      # RSVQA / VRSBench / CDVQA / ISRO evaluator
    └── metrics.py             # BLEU, CIDEr, IoU, F1, OA, aggregation
```

## Quick start

```bash
pip install -r requirements.txt
streamlit run web/app.py
```

## Optional deep-learning upgrade path

```bash
pip install torch transformers peft open_clip_torch
# then place checkpoints under core/config.py -> MODEL_WEIGHTS_DIR
# RSFeatureBackbone will auto-detect and switch from classical to learned mode.
```
