from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile


def read_image(path: str | Path) -> Tuple[np.ndarray, List[str]]:
    """Read an image and return (array, channel_names).

    Returns
    -------
    arr:
        Numpy array with shape (C, H, W) for multichannel images, or (1, H, W) for single-channel.
    channel_names:
        List of channel names (length C) if present, else generic names.

    Notes
    -----
    - Supports OME-TIFF channel names when OME-XML is present.
    - For non-OME TIFFs, attempts to infer whether the first dimension is channels.
    """
    path = Path(path)
    with tifffile.TiffFile(str(path)) as tf:
        arr = tf.asarray()
        channel_names = _get_ome_channel_names(tf) or []

    arr = np.asarray(arr)

    # Handle common axis layouts.
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        # Heuristic: if first axis looks like channels (small), treat as (C,H,W).
        if arr.shape[0] <= 6 and arr.shape[1] > 16 and arr.shape[2] > 16:
            pass  # (C,H,W)
        elif arr.shape[2] <= 6 and arr.shape[0] > 16 and arr.shape[1] > 16:
            arr = np.transpose(arr, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        else:
            # Assume single-channel volume; collapse to max projection.
            arr = arr.max(axis=0, keepdims=True)
    elif arr.ndim == 4:
        # Could be (Z,C,H,W) or (C,Z,H,W) or (T,C,H,W). We max-project over non-spatial dims.
        # Keep the last two dims as (H,W).
        # Strategy: identify the two largest dims as spatial.
        shape = arr.shape
        spatial = sorted(range(len(shape)), key=lambda i: shape[i], reverse=True)[:2]
        spatial = sorted(spatial)
        # Move spatial dims to end
        perm = [i for i in range(arr.ndim) if i not in spatial] + spatial
        arr2 = np.transpose(arr, perm)
        # Now arr2 has shape (..., H, W)
        hw = arr2.shape[-2:]
        rest = int(np.prod(arr2.shape[:-2]))
        arr2 = arr2.reshape((rest,) + hw)
        # treat rest as "channels" after projection of time/z if needed
        arr = arr2
    else:
        raise ValueError(f"Unsupported image ndim={arr.ndim} for {path}")

    c = arr.shape[0]
    if not channel_names or len(channel_names) != c:
        channel_names = [f"ch{idx}" for idx in range(c)]

    return arr.astype(np.float32), channel_names


def _get_ome_channel_names(tf: tifffile.TiffFile) -> Optional[List[str]]:
    ome = tf.ome_metadata
    if ome is None:
        return None
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(ome)
        # OME namespace can vary; match by suffix.
        channels = []
        for ch in root.iter():
            if ch.tag.endswith("Channel"):
                name = ch.attrib.get("Name") or ch.attrib.get("ID")
                if name:
                    channels.append(name)
        if channels:
            return channels
    except Exception:
        return None
    return None


def channel_index(channel_spec, channel_names: List[str]) -> int:
    """Resolve a channel spec into an index.

    channel_spec may be:
    - int
    - a string channel name (case-insensitive exact match)
    """
    if isinstance(channel_spec, int):
        return int(channel_spec)
    if channel_spec is None:
        raise ValueError("channel_spec is None")
    s = str(channel_spec).strip().lower()
    for i, name in enumerate(channel_names):
        if name.strip().lower() == s:
            return i
    raise ValueError(f"Channel '{channel_spec}' not found. Available: {channel_names}")
