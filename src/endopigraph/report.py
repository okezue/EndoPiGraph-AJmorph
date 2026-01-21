from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def write_html_report(
    out_dir: str | Path,
    study_accession: Optional[str],
    image_items: List[Dict[str, str]],
    title: str = "EndoPiGraph-AJmorph report",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts: List[str] = []
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append(f"<title>{title}</title>")
    parts.append("<style>body{font-family:Arial,sans-serif;} code{background:#f0f0f0;padding:2px 4px;} .item{margin-bottom:18px;} img{max-width:480px;border:1px solid #ddd;margin-right:8px;} .small{color:#666;font-size:12px;}</style>")
    parts.append("</head><body>")
    parts.append(f"<h1>{title}</h1>")
    if study_accession:
        parts.append(f"<p class='small'>Study: {study_accession}</p>")

    parts.append("<h2>Images processed</h2>")
    parts.append("<ul>")
    for it in image_items:
        image_id = it.get("image_id", "(unknown)")
        parts.append(f"<li><a href='#{image_id}'>{image_id}</a></li>")
    parts.append("</ul>")

    for it in image_items:
        image_id = it.get("image_id", "(unknown)")
        parts.append(f"<hr><div class='item' id='{image_id}'>")
        parts.append(f"<h3>{image_id}</h3>")
        parts.append(f"<p class='small'><code>{it.get('path','')}</code></p>")

        qc_seg = it.get("qc_seg", "")
        qc_graph = it.get("qc_graph", "")
        qc_feat = it.get("qc_feat", "")

        parts.append("<div>")
        for p in (qc_seg, qc_graph, qc_feat):
            if p:
                parts.append(f"<a href='{p}'><img src='{p}'></a>")
        parts.append("</div>")

        edges_csv = it.get("edges_csv", "")
        cells_csv = it.get("cells_csv", "")
        graph_json = it.get("graph_json", "")

        parts.append("<p>")
        if cells_csv:
            parts.append(f"<a href='{cells_csv}'>cells.csv</a> ")
        if edges_csv:
            parts.append(f"<a href='{edges_csv}'>edges.csv</a> ")
        if graph_json:
            parts.append(f"<a href='{graph_json}'>graph.json</a> ")
        parts.append("</p>")

        parts.append("</div>")

    parts.append("</body></html>")

    out_path = out_dir / "report.html"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
