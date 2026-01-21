from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tifffile


def export_interface_patches(
    image_id: str,
    marker: np.ndarray,
    edges: pd.DataFrame,
    boundary_coords: Dict[Tuple[int, int], np.ndarray],
    out_dir: str | Path,
    patch_size: int = 96,
    max_patches: int = 300,
) -> Path:
    """Export cropped patches around cell-cell interfaces for manual labeling.

    This writes OME-TIFF-like 2D TIFFs (single-channel) named:
        <image_id>__i-<cell_i>__j-<cell_j>.tif

    It also writes a CSV manifest with patch paths and edge metadata.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    half = patch_size // 2
    rows = []

    # basic deterministic order: highest-contact first
    edges_sorted = edges.sort_values("contact_px", ascending=False).head(max_patches)

    for _, erow in edges_sorted.iterrows():
        i = int(erow["cell_i"])
        j = int(erow["cell_j"])
        coords = boundary_coords.get((min(i, j), max(i, j)))
        if coords is None or len(coords) == 0:
            continue

        cy = int(np.round(coords[:, 0].mean()))
        cx = int(np.round(coords[:, 1].mean()))

        y0 = max(0, cy - half)
        y1 = min(marker.shape[0], cy + half)
        x0 = max(0, cx - half)
        x1 = min(marker.shape[1], cx + half)

        patch = marker[y0:y1, x0:x1].astype(np.float32)

        patch_name = f"{image_id}__i-{i}__j-{j}.tif"
        patch_path = out_dir / patch_name
        tifffile.imwrite(patch_path, patch)

        row = {
            "image_id": image_id,
            "cell_i": i,
            "cell_j": j,
            "contact_px": int(erow.get("contact_px", 0)),
            "patch_path": str(patch_path),
            "center_y": cy,
            "center_x": cx,
        }
        rows.append(row)

    manifest_path = out_dir / f"{image_id}__patch_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path
