"""
web/report_generator.py
=========================
Generates the "downloadable report" required by the problem statement:

* `generate_pdf_report`: a one-click PDF summarising the query, selected
  task, evidence-grounded answer, confidence breakdown, dispatched
  tools, and a spatial-evidence table — built with `reportlab` (pure
  Python, no external binary dependencies).
* `generate_geojson_report`: exports bounding boxes / change regions as
  a GeoJSON `FeatureCollection`. If the source image carries a CRS/
  affine transform, pixel coordinates are converted to world
  coordinates via `core.geo_io.pixel_to_geo`; otherwise pixel
  coordinates are exported directly with a `"pixel_coordinates": true`
  property flag so downstream GIS tools do not silently misinterpret
  them as geographic coordinates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from core import config
from core.geo_io import read_image, pixel_to_geo


def generate_pdf_report(trace: dict, image_paths: List[str]) -> str:
    """Build a PDF audit report for `trace` and return the output file path."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )
    from reportlab.lib.units import cm

    config.ensure_dirs()
    out_path = config.REPORT_DIR / f"satquery_report_{trace['query_id'][:8]}.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18)
    h2 = styles["Heading2"]
    body = styles["BodyText"]

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                             leftMargin=2 * cm, rightMargin=2 * cm,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = []

    story.append(Paragraph("SatQuery AI — Analytical Report", title_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Query ID: {trace['query_id']}", body))
    story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", body))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("1. Query", h2))
    story.append(Paragraph(trace["query"], body))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("2. Task Routing", h2))
    route_table = Table([
        ["Selected task", trace["selected_task"]],
        ["Input configuration", trace["input_configuration"]],
        ["Status", trace["status"]],
        ["Dispatched tools", ", ".join(t["tool"] for t in trace["dispatched_tools"]) or "—"],
    ], colWidths=[5 * cm, 10 * cm])
    route_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(route_table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("3. Evidence-Grounded Answer", h2))
    story.append(Paragraph(trace["evidence_grounded_answer"] or "—", body))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("4. Confidence Breakdown", h2))
    comp = trace["confidence"]["components"] or {}
    conf_rows = [["Component", "Score"]] + [[k, f"{v:.2f}"] for k, v in comp.items()]
    conf_rows.append(["Final confidence", f"{trace['confidence']['final_score']:.2f}"
                       if trace["confidence"]["final_score"] is not None else "—"])
    conf_table = Table(conf_rows, colWidths=[8 * cm, 4 * cm])
    conf_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceefb")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(conf_table)
    story.append(Spacer(1, 0.4 * cm))

    boxes = trace["spatial_evidence"]["bounding_boxes"]
    story.append(Paragraph("5. Spatial Evidence (Bounding Boxes)", h2))
    if boxes:
        rows = [["Label", "Score", "xmin", "ymin", "xmax", "ymax"]]
        for b in boxes[:20]:
            rows.append([b.get("label", ""), f"{b.get('score', 0):.2f}",
                         str(b.get("xmin")), str(b.get("ymin")), str(b.get("xmax")), str(b.get("ymax"))])
        box_table = Table(rows, colWidths=[3 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm])
        box_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceefb")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(box_table)
    else:
        story.append(Paragraph("No bounding-box evidence produced for this task.", body))
    story.append(Spacer(1, 0.4 * cm))

    if trace["warnings"]:
        story.append(Paragraph("6. Warnings", h2))
        for w in trace["warnings"]:
            story.append(Paragraph(f"• {w}", body))
        story.append(Spacer(1, 0.4 * cm))

    story.append(PageBreak())
    story.append(Paragraph("Appendix — Full Execution Trace (JSON)", h2))
    trace_str = json.dumps(trace, indent=2, default=str)
    for chunk_start in range(0, len(trace_str), 3000):
        chunk = trace_str[chunk_start:chunk_start + 3000]
        story.append(Paragraph(f"<font face='Courier' size=6>{_escape(chunk)}</font>", body))

    doc.build(story)
    return str(out_path)


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\n", "<br/>"))


def generate_geojson_report(trace: dict, image_paths: List[str]) -> str:
    """Return a GeoJSON `FeatureCollection` string for the bounding-box /
    change-region evidence in `trace`."""
    boxes = trace["spatial_evidence"]["bounding_boxes"]
    features = []

    ref_img = None
    if image_paths:
        try:
            ref_img = read_image(image_paths[0])
        except Exception:
            ref_img = None

    is_geo = ref_img is not None and ref_img.crs is not None and ref_img.transform is not None

    for b in boxes:
        corners_px = [(b["xmin"], b["ymin"]), (b["xmax"], b["ymin"]),
                      (b["xmax"], b["ymax"]), (b["xmin"], b["ymax"]), (b["xmin"], b["ymin"])]
        if is_geo:
            coords = [list(pixel_to_geo(ref_img, row=y, col=x)) for x, y in corners_px]
        else:
            coords = [[x, y] for x, y in corners_px]

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "label": b.get("label"),
                "score": b.get("score"),
                "area_px": b.get("area_px"),
                "task": trace.get("selected_task"),
                "pixel_coordinates": not is_geo,
            },
        })

    fc = {
        "type": "FeatureCollection",
        "properties": {
            "query_id": trace.get("query_id"),
            "query": trace.get("query"),
            "final_confidence": trace["confidence"].get("final_score"),
            "crs": ref_img.crs if is_geo else None,
        },
        "features": features,
    }
    return json.dumps(fc, indent=2)
