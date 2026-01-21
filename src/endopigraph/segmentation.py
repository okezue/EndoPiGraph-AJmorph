from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage import exposure
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt


def _get_channel(arr: np.ndarray, channel_names: List[str], channel_spec: dict) -> np.ndarray:
    """Return a single-channel image (H,W) from (C,H,W) given a spec.

    channel_spec can contain:
      - channel_index: int
      - channel_name: str

    Raises ValueError if not resolvable.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected arr with shape (C,H,W); got {arr.shape}")

    if "channel_index" in channel_spec and channel_spec["channel_index"] is not None:
        idx = int(channel_spec["channel_index"])
        if idx < 0 or idx >= arr.shape[0]:
            raise ValueError(f"channel_index {idx} out of bounds for C={arr.shape[0]}")
        return arr[idx]

    name = channel_spec.get("channel_name")
    if name:
        try:
            idx = channel_names.index(name)
        except ValueError as e:
            raise ValueError(f"channel_name '{name}' not found in channel_names={channel_names}") from e
        return arr[idx]

    raise ValueError("channel_spec must include channel_index or channel_name")


def segment_cells(arr: np.ndarray, channel_names: List[str], seg_cfg: Dict) -> np.ndarray:
    method = seg_cfg.get("method", "cellpose")
    if method == "cellpose":
        return segment_cells_cellpose(arr, channel_names, seg_cfg.get("cellpose", {}))
    if method == "watershed":
        return segment_cells_watershed(arr, channel_names, seg_cfg.get("watershed", {}))
    raise ValueError(f"Unknown segmentation method: {method}")


def segment_cells_cellpose(arr: np.ndarray, channel_names: List[str], cfg: Dict) -> np.ndarray:
    """Cellpose segmentation.

    Requires `cellpose` to be installed.
    """
    try:
        from cellpose import models
    except Exception as e:
        raise ImportError(
            "cellpose is not installed. Install with: pip install -e \".[cellpose]\""
        ) from e

    model_type = cfg.get("model_type", "cyto2")
    diameter = cfg.get("diameter", 30)

    # Resolve channels
    ch_cfg = cfg.get("channels", {})
    cyto = _get_channel(arr, channel_names, ch_cfg.get("cyto", {"channel_index": 0}))
    nuclei_spec = ch_cfg.get("nuclei", None)

    if nuclei_spec:
        nuc = _get_channel(arr, channel_names, nuclei_spec)
        img = np.stack([cyto, nuc], axis=-1)  # (H,W,2)
        channels = [1, 2]
    else:
        img = cyto[..., None]
        channels = [0, 0]

    # Basic normalization for stability
    img = exposure.rescale_intensity(img, in_range="image", out_range=(0, 1))

    model = models.Cellpose(model_type=model_type)
    masks, *_ = model.eval(img, diameter=diameter, channels=channels)

    # Ensure integer labels with background=0
    return masks.astype(np.int32)


def segment_cells_watershed(arr: np.ndarray, channel_names: List[str], cfg: Dict) -> np.ndarray:
    """Watershed fallback segmentation using nuclei markers.

    This is a pragmatic (not SOTA) approach meant to be a no-GPU baseline.
    """
    nuclei = _get_channel(arr, channel_names, cfg.get("nuclei", {"channel_index": 0}))
    membrane = _get_channel(arr, channel_names, cfg.get("membrane", {"channel_index": 0}))

    nuclei_s = gaussian(nuclei, sigma=cfg.get("nuclei_sigma", 1.0))
    thr = threshold_otsu(nuclei_s)
    nuclei_mask = nuclei_s > thr
    nuclei_mask = remove_small_objects(nuclei_mask, min_size=int(cfg.get("min_nuclei_area_px", 50)))
    nuclei_mask = remove_small_holes(nuclei_mask, area_threshold=int(cfg.get("max_nuclei_hole_px", 50)))

    # Create seed markers from nuclei
    coords = peak_local_max(nuclei_s, labels=nuclei_mask, min_distance=int(cfg.get("min_peak_dist_px", 5)))
    seeds = np.zeros_like(nuclei_s, dtype=np.int32)
    for k, (r, c) in enumerate(coords, start=1):
        seeds[r, c] = k
    if seeds.max() == 0:
        # Fallback: label connected components of nuclei mask
        seeds = label(nuclei_mask)

    # Elevation map from membrane (high gradient at borders)
    mem = exposure.rescale_intensity(membrane, in_range="image", out_range=(0, 1))
    elevation = gaussian(1.0 - mem, sigma=cfg.get("membrane_sigma", 1.0))

    labels = watershed(elevation, markers=seeds, mask=None)

    # Optional: remove tiny segments
    labels = _relabel_and_filter(labels, min_cell_area_px=int(cfg.get("min_cell_area_px", 200)))
    return labels


def _relabel_and_filter(labels: np.ndarray, min_cell_area_px: int = 200) -> np.ndarray:
    labels = labels.astype(np.int32)
    if labels.max() == 0:
        return labels

    # Remove small objects by setting them to 0
    keep = np.zeros(labels.max() + 1, dtype=bool)
    keep[0] = True
    vals, counts = np.unique(labels, return_counts=True)
    for v, c in zip(vals, counts):
        if v == 0:
            continue
        if c >= min_cell_area_px:
            keep[v] = True
    out = labels.copy()
    out[~keep[labels]] = 0

    # Re-label to 1..N
    vals = np.unique(out)
    vals = vals[vals != 0]
    mapping = {int(v): i + 1 for i, v in enumerate(vals)}
    rel = np.zeros_like(out)
    for v, newv in mapping.items():
        rel[out == v] = newv
    return rel
