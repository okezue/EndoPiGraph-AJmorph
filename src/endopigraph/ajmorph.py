from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


def compute_threshold(values: np.ndarray, method: str) -> float:
    """Compute a scalar threshold from a 1D array of intensities.

    Supported methods
    -----------------
    - "otsu"
    - "percentile:<p>"  (e.g. percentile:90)
    - numeric string (e.g. "123.4")
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")

    m = method.strip().lower()
    if m == "otsu":
        return float(threshold_otsu(v))
    if m.startswith("percentile:"):
        p = float(m.split(":", 1)[1])
        return float(np.percentile(v, p))
    # numeric literal
    try:
        return float(method)
    except ValueError as e:
        raise ValueError(f"Unknown threshold method: {method!r}") from e


def interface_marker_features(
    marker: np.ndarray,
    interface_mask: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute intensity and morphology features for a marker on an interface region."""
    vals = marker[interface_mask]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "occupancy": float("nan"),
            "cluster_count": 0,
            "cluster_area_mean": float("nan"),
            "skeleton_len": 0,
            "thickness_proxy": float("nan"),
        }

    mean = float(np.mean(vals))
    median = float(np.median(vals))
    vmax = float(np.max(vals))
    std = float(np.std(vals))

    bin_mask = (marker > threshold) & interface_mask
    occ = float(bin_mask.sum() / max(int(interface_mask.sum()), 1))

    lab = label(bin_mask)
    props = regionprops(lab)
    cluster_count = int(len(props))
    cluster_area_mean = float(np.mean([p.area for p in props])) if props else float("nan")

    # Skeleton length as a proxy for filamentous / reticular structure
    skel = skeletonize(bin_mask)
    skeleton_len = int(skel.sum())

    # Simple thickness proxy: area / skeleton_len (larger = thicker, smaller = thin/filament)
    if skeleton_len > 0:
        thickness_proxy = float(bin_mask.sum() / skeleton_len)
    else:
        thickness_proxy = float("nan")

    return {
        "mean": mean,
        "median": median,
        "max": vmax,
        "std": std,
        "occupancy": occ,
        "cluster_count": cluster_count,
        "cluster_area_mean": cluster_area_mean,
        "skeleton_len": skeleton_len,
        "thickness_proxy": thickness_proxy,
    }


AJMORPH_CLASSES = ["straight", "thick", "thick_to_reticular", "reticular", "fingers", "unknown"]


def heuristic_ajmorph_class(features: Dict[str, Any]) -> str:
    """A very rough heuristic classifier for AJ morphology.

    This is *not* a validated model. It is intended as a placeholder to enable
    end-to-end figure generation until you train a supervised classifier.

    The morphology labels follow common categories used in junction-morphology
    analysis (e.g. straight, thick, reticular, fingers).
    """
    occ = features.get("occupancy", float("nan"))
    ncl = features.get("cluster_count", 0)
    th = features.get("thickness_proxy", float("nan"))
    sk = features.get("skeleton_len", 0)

    if not np.isfinite(occ):
        return "unknown"

    # Fingers: sparse occupancy, few but elongated/skeletal elements
    if occ < 0.15 and sk > 30 and ncl <= 3:
        return "fingers"

    # Reticular: many clusters and lots of skeletonization / fragmentation
    if occ >= 0.15 and ncl >= 8:
        return "reticular"

    # Thick: high occupancy + thick proxy
    if occ > 0.6 and np.isfinite(th) and th > 3.0:
        return "thick"

    # Straight: high occupancy, low clustering, relatively thin
    if occ > 0.6 and ncl <= 4:
        return "straight"

    # Intermediate transition
    if 0.25 <= occ <= 0.6 and ncl >= 4:
        return "thick_to_reticular"

    return "unknown"


def add_ajmorph_columns(df: pd.DataFrame, prefix: str = "aj_") -> pd.DataFrame:
    """Convenience function: infer an AJ morphology class from AJ feature columns.

    Expects columns:
      - f"{prefix}occupancy"
      - f"{prefix}cluster_count"
      - f"{prefix}thickness_proxy"
      - f"{prefix}skeleton_len"
    """
    out = df.copy()

    def _row_class(row: pd.Series) -> str:
        feats = {
            "occupancy": float(row.get(f"{prefix}occupancy", np.nan)),
            "cluster_count": int(row.get(f"{prefix}cluster_count", 0)),
            "thickness_proxy": float(row.get(f"{prefix}thickness_proxy", np.nan)),
            "skeleton_len": int(row.get(f"{prefix}skeleton_len", 0)),
        }
        return heuristic_ajmorph_class(feats)

    out[f"{prefix}class"] = out.apply(_row_class, axis=1)
    return out

# Aliases used by the pipeline (v0.1 naming).
compute_interface_features = interface_marker_features
infer_ajmorph_label_heuristic = heuristic_ajmorph_class

