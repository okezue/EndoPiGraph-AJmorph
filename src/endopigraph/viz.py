from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from skimage.segmentation import find_boundaries


def save_segmentation_qc(
    img: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    title: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Path:
    from PIL import Image as PILImage
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = img.astype(float)
    if vmin is None:
        vmin = np.percentile(base, 1)
    if vmax is None:
        vmax = np.percentile(base, 99)
    norm = np.clip((base - vmin) / (vmax - vmin + 1e-8), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    b = find_boundaries(labels, mode="inner")
    rgb[b] = [255, 50, 50]
    PILImage.fromarray(rgb).save(str(out_path))
    raw_path = out_path.parent / out_path.name.replace("qc_seg", "raw_display").replace("qc_cells", "raw_display")
    PILImage.fromarray(gray).save(str(raw_path))
    return out_path


MORPH_COLORS = {
    "straight": "#4CAF50",
    "thick": "#2196F3",
    "thick_to_reticular": "#00BCD4",
    "reticular": "#FF9800",
    "fingers": "#E91E63",
    "discontinuous": "#9C27B0",
    "punctate": "#FFC107",
    "minimal": "#607D8B",
    "other": "#795548",
    "unknown": "#9E9E9E",
}

def save_graph_plot(
    G: nx.Graph,
    cells: pd.DataFrame,
    out_path: str | Path,
    title: str = "",
    node_x: str = "centroid_x",
    node_y: str = "centroid_y",
    bg_img: np.ndarray = None,
    vmin: float = None,
    vmax: float = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos = {}
    if node_x in cells.columns and node_y in cells.columns:
        for _, r in cells.iterrows():
            pos[int(r["cell_id"])] = (float(r[node_x]), float(r[node_y]))

    edge_colors = []
    for u, v, d in G.edges(data=True):
        m = d.get("aj_morph", d.get("AJ_morph_label", "unknown"))
        edge_colors.append(MORPH_COLORS.get(str(m), "#9E9E9E"))

    fig, ax = plt.subplots(figsize=(8, 8))

    if bg_img is not None:
        if vmin is None:
            vmin = np.percentile(bg_img, 1)
        if vmax is None:
            vmax = np.percentile(bg_img, 99)
        ax.imshow(bg_img, cmap="gray", vmin=vmin, vmax=vmax)

    if pos:
        nx.draw_networkx_nodes(G, pos=pos, node_size=15, node_color="#6897bb", ax=ax)
        nx.draw_networkx_edges(G, pos=pos, edge_color=edge_colors, width=1.5, alpha=0.85, ax=ax)
    else:
        nx.draw(G, node_size=15, edge_color=edge_colors, width=1.5, alpha=0.85, ax=ax)

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_feature_distributions(
    edges: pd.DataFrame,
    out_path: str | Path,
    title: str = "",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [c for c in edges.columns if any(s in c for s in ("_mean", "_occupancy", "_clusters"))]
    cols = cols[:6]

    n = len(cols)
    if n == 0:
        return out_path

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        vals = edges[c].dropna().values
        ax.hist(vals, bins=30)
        ax.set_title(c)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
