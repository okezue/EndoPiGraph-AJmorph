from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = img.astype(float)
    if vmin is None:
        vmin = np.percentile(base, 1)
    if vmax is None:
        vmax = np.percentile(base, 99)

    b = find_boundaries(labels, mode="inner")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(base, cmap="gray", vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.array(b, mask=~b), cmap="autumn", alpha=0.8)
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_graph_plot(
    G: nx.Graph,
    cells: pd.DataFrame,
    out_path: str | Path,
    title: str = "",
    node_x: str = "centroid_x",
    node_y: str = "centroid_y",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos = {}
    if node_x in cells.columns and node_y in cells.columns:
        for _, r in cells.iterrows():
            pos[int(r["cell_id"])] = (float(r[node_x]), float(r[node_y]))

    fig, ax = plt.subplots(figsize=(8, 8))

    if pos:
        nx.draw(G, pos=pos, node_size=20, with_labels=False, ax=ax)
    else:
        nx.draw(G, node_size=20, with_labels=False, ax=ax)

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
