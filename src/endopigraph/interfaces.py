from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from skimage.morphology import binary_dilation, disk


@dataclass
class InterfaceData:
    edges: pd.DataFrame
    boundary_coords: Dict[Tuple[int, int], np.ndarray]
    all_boundary_mask: np.ndarray


def _pair_code(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Encode (i,j) into a single int64 code, assuming i,j fit in 32 bits."""
    return (i.astype(np.int64) << 32) | j.astype(np.int64)


def compute_interfaces(labels: np.ndarray, min_contact_px: int = 10) -> InterfaceData:
    """Compute undirected cell-cell interfaces.

    Returns
    -------
    InterfaceData
        edges: DataFrame with columns [cell_i, cell_j, contact_px]
        boundary_coords: dict mapping (cell_i, cell_j) -> (N,2) array of (row,col) boundary pixel coords
        all_boundary_mask: boolean mask of all cell-cell boundaries
    """
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D; got {labels.shape}")

    H, W = labels.shape

    # Horizontal comparisons
    a_h = labels[:, :-1]
    b_h = labels[:, 1:]
    mask_h = (a_h != b_h) & (a_h > 0) & (b_h > 0)
    r_h, c_h = np.nonzero(mask_h)
    u_h = a_h[mask_h]
    v_h = b_h[mask_h]

    i_h = np.minimum(u_h, v_h).astype(np.int32)
    j_h = np.maximum(u_h, v_h).astype(np.int32)
    code_h = _pair_code(i_h, j_h)

    # Vertical comparisons
    a_v = labels[:-1, :]
    b_v = labels[1:, :]
    mask_v = (a_v != b_v) & (a_v > 0) & (b_v > 0)
    r_v, c_v = np.nonzero(mask_v)
    u_v = a_v[mask_v]
    v_v = b_v[mask_v]

    i_v = np.minimum(u_v, v_v).astype(np.int32)
    j_v = np.maximum(u_v, v_v).astype(np.int32)
    code_v = _pair_code(i_v, j_v)

    # Contact counts (length proxy)
    codes = np.concatenate([code_h, code_v])
    if codes.size == 0:
        edges_df = pd.DataFrame(columns=["cell_i", "cell_j", "contact_px"])
        return InterfaceData(edges=edges_df, boundary_coords={}, all_boundary_mask=np.zeros_like(labels, dtype=bool))

    uniq_codes, counts = np.unique(codes, return_counts=True)

    # Decode codes back to (i,j)
    cell_i = (uniq_codes >> 32).astype(np.int32)
    cell_j = (uniq_codes & 0xFFFFFFFF).astype(np.int32)

    edges_df = pd.DataFrame({"cell_i": cell_i, "cell_j": cell_j, "contact_px": counts.astype(np.int32)})

    # Filter tiny contacts
    edges_df = edges_df[edges_df["contact_px"] >= int(min_contact_px)].reset_index(drop=True)
    keep_set = set(_pair_code(edges_df["cell_i"].values, edges_df["cell_j"].values).tolist())

    # Build boundary coord lists (pixels on both sides of the interface)
    boundary_coords: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def _add_coords(code_arr: np.ndarray, r: np.ndarray, c: np.ndarray, dc: int, dr: int) -> None:
        for code, rr, cc in zip(code_arr.tolist(), r.tolist(), c.tolist()):
            if code not in keep_set:
                continue
            i = int(code >> 32)
            j = int(code & 0xFFFFFFFF)
            key = (i, j)
            boundary_coords.setdefault(key, []).append((rr, cc))
            boundary_coords.setdefault(key, []).append((rr + dr, cc + dc))

    # Horizontal: neighbor at (r, c+1)
    _add_coords(code_h, r_h, c_h, dc=1, dr=0)
    # Vertical: neighbor at (r+1, c)
    _add_coords(code_v, r_v, c_v, dc=0, dr=1)

    # Convert lists to arrays and also create a global boundary mask
    all_boundary_mask = np.zeros_like(labels, dtype=bool)
    boundary_arrays: Dict[Tuple[int, int], np.ndarray] = {}
    for key, coords in boundary_coords.items():
        arr = np.array(coords, dtype=np.int32)
        # Unique boundary pixels (avoid duplicates)
        if arr.size > 0:
            arr = np.unique(arr, axis=0)
            all_boundary_mask[arr[:, 0], arr[:, 1]] = True
        boundary_arrays[key] = arr

    return InterfaceData(edges=edges_df, boundary_coords=boundary_arrays, all_boundary_mask=all_boundary_mask)


def build_interface_mask_from_coords(shape: Tuple[int, int], coords: np.ndarray, dilate_px: int = 2) -> np.ndarray:
    """Create a boolean mask for an interface given boundary pixel coordinates.

    For performance, this returns a full-size mask. For per-edge feature extraction,
    prefer cropping to a bounding box (see `local_interface_mask`).
    """
    mask = np.zeros(shape, dtype=bool)
    if coords.size == 0:
        return mask
    mask[coords[:, 0], coords[:, 1]] = True
    if dilate_px > 0:
        mask = binary_dilation(mask, footprint=disk(int(dilate_px)))
    return mask


def local_interface_mask(coords: np.ndarray, margin: int, shape: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """Create a cropped interface mask around coords.

    Returns
    -------
    bbox: (r0,r1,c0,c1) inclusive-exclusive
    mask_local: boolean mask of shape (r1-r0, c1-c0) with boundary pixels set True.
    """
    if coords.size == 0:
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=bool)

    H, W = shape
    r0 = max(int(coords[:, 0].min()) - margin, 0)
    r1 = min(int(coords[:, 0].max()) + margin + 1, H)
    c0 = max(int(coords[:, 1].min()) - margin, 0)
    c1 = min(int(coords[:, 1].max()) + margin + 1, W)

    local = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    local[rr, cc] = True
    return (r0, r1, c0, c1), local

# Alias used by the CLI.
extract_interfaces = compute_interfaces


# Wrapper used by the pipeline: coords-first signature.
def interface_mask_from_coords(coords: np.ndarray, shape: tuple[int, int], dilate_px: int = 2) -> np.ndarray:
    (r0, r1, c0, c1), local = build_interface_mask_from_coords(shape=shape, coords=coords, dilate_px=dilate_px)
    if dilate_px and dilate_px > 0:
        local = binary_dilation(local, disk(dilate_px))
    out = np.zeros(shape, dtype=bool)
    out[r0:r1, c0:c1] = local
    return out

