from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, h_maxima


def count_clusters_robust(
    intensity: np.ndarray,
    binary_mask: np.ndarray,
    h_threshold: float = 0.1,
) -> int:
    """Count clusters using regional maxima detection - robust to blur.

    Standard connected component counting fails when blur merges adjacent clusters.
    This method finds intensity peaks within the masked region, which persist even
    when boundaries are blurred together.

    Parameters
    ----------
    intensity : np.ndarray
        The marker intensity image (not thresholded).
    binary_mask : np.ndarray
        Binary mask of the region of interest.
    h_threshold : float
        H-maxima threshold as fraction of intensity range. Higher values = fewer
        spurious peaks from noise, but may miss weak clusters.

    Returns
    -------
    int
        Number of detected clusters (regional maxima).
    """
    if not binary_mask.any():
        return 0

    # Apply mask and normalize intensity
    masked = intensity.astype(float) * binary_mask

    # Get intensity range within mask
    vals = masked[binary_mask]
    if vals.size == 0:
        return 0

    val_range = float(vals.max() - vals.min())
    if val_range == 0:
        # Uniform intensity - count connected components as fallback
        lab = label(binary_mask)
        return int(lab.max())

    # Find h-maxima (regional maxima that are at least h above surroundings)
    # This filters out noise peaks while preserving real clusters
    h_value = h_threshold * val_range

    try:
        # h_maxima returns a binary image of preserved maxima
        maxima = h_maxima(masked, h=h_value)
        # Restrict to within our mask
        maxima = maxima & binary_mask
        # Count separate maxima regions
        labeled_maxima = label(maxima)
        n_maxima = int(labeled_maxima.max())
    except Exception:
        # Fallback to simple connected components
        lab = label(binary_mask)
        return int(lab.max())

    # If no maxima found, fall back to connected components
    if n_maxima == 0:
        lab = label(binary_mask)
        return int(lab.max())

    return n_maxima


def count_skeleton_components(binary_mask: np.ndarray) -> int:
    """Count skeleton components - more robust to blur than CC counting.

    The skeleton represents the "core" structure of the mask. When blur merges
    adjacent regions, the skeleton may still show separate branches that can
    be counted. This method counts the number of separate skeleton components.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask to analyze.

    Returns
    -------
    int
        Number of skeleton connected components.
    """
    if not binary_mask.any():
        return 0

    skel = skeletonize(binary_mask)
    skel_labeled = label(skel)
    return int(skel_labeled.max())


def compute_skeleton_complexity(binary_mask: np.ndarray) -> dict:
    """Compute skeleton-based complexity metrics that are more blur-robust.

    Instead of counting separate clusters (which merge under blur), this
    computes topological features of the skeleton:
    - endpoints: terminal points (degree 1)
    - branch_points: junction points (degree >= 3)
    - skeleton_components: separate skeleton pieces

    These metrics capture the "complexity" of the junction pattern in a way
    that degrades more gracefully under blur.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask to analyze.

    Returns
    -------
    dict
        Dictionary with skeleton complexity metrics.
    """
    if not binary_mask.any():
        return {
            "skeleton_components": 0,
            "endpoints": 0,
            "branch_points": 0,
            "complexity_score": 0,
        }

    skel = skeletonize(binary_mask)
    if not skel.any():
        return {
            "skeleton_components": 0,
            "endpoints": 0,
            "branch_points": 0,
            "complexity_score": 0,
        }

    # Count skeleton components
    skel_labeled = label(skel)
    n_components = int(skel_labeled.max())

    # Find endpoints and branch points using hit-or-miss with neighbor counting
    # Convolve with a 3x3 kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    neighbor_count = ndimage.convolve(skel.astype(int), kernel, mode='constant', cval=0)

    # Endpoints have exactly 1 neighbor
    endpoints = ((neighbor_count == 1) & skel).sum()

    # Branch points have 3+ neighbors
    branch_points = ((neighbor_count >= 3) & skel).sum()

    # Complexity score: weighted combination that captures "fragmentation"
    # More components = more fragments, more branch points = more complex structure
    complexity_score = n_components + 0.5 * branch_points

    return {
        "skeleton_components": n_components,
        "endpoints": int(endpoints),
        "branch_points": int(branch_points),
        "complexity_score": float(complexity_score),
    }


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
    use_robust_clustering: bool = True,
) -> Dict[str, Any]:
    """Compute intensity and morphology features for a marker on an interface region.

    Parameters
    ----------
    marker : np.ndarray
        Marker intensity image.
    interface_mask : np.ndarray
        Binary mask of the interface region.
    threshold : float
        Intensity threshold for binarization.
    use_robust_clustering : bool
        If True, use h-maxima based cluster counting which is robust to blur.
        If False, use simple connected component counting (original behavior).

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed features.
    """
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
            "cluster_count_cc": 0,  # Connected component count (original)
            "cluster_area_mean": float("nan"),
            "skeleton_len": 0,
            "skeleton_components": 0,
            "skeleton_endpoints": 0,
            "skeleton_branch_points": 0,
            "complexity_score": 0.0,
            "thickness_proxy": float("nan"),
        }

    mean = float(np.mean(vals))
    median = float(np.median(vals))
    vmax = float(np.max(vals))
    std = float(np.std(vals))

    bin_mask = (marker > threshold) & interface_mask
    occ = float(bin_mask.sum() / max(int(interface_mask.sum()), 1))

    # Connected component counting (original method - sensitive to blur)
    lab = label(bin_mask)
    props = regionprops(lab)
    cluster_count_cc = int(len(props))
    cluster_area_mean = float(np.mean([p.area for p in props])) if props else float("nan")

    # Robust cluster counting using regional maxima (blur-resistant)
    if use_robust_clustering:
        cluster_count = count_clusters_robust(marker, bin_mask, h_threshold=0.1)
    else:
        cluster_count = cluster_count_cc

    # Skeleton length as a proxy for filamentous / reticular structure
    skel = skeletonize(bin_mask)
    skeleton_len = int(skel.sum())

    # Compute skeleton complexity metrics (more blur-robust than cluster_count)
    skel_complexity = compute_skeleton_complexity(bin_mask)

    # Simple thickness proxy: area / skeleton_len (larger = thicker, smaller = thin/filament)
    if skeleton_len > 0:
        thickness_proxy = float(bin_mask.sum() / skeleton_len)
    else:
        thickness_proxy = float("nan")

    # Compute cluster density: clusters per unit marker area
    # This normalizes for the fact that blur increases individual cluster size
    marker_area = bin_mask.sum()
    if marker_area > 0:
        cluster_density = float(cluster_count_cc) / (marker_area / 1000.0)  # per 1000 pixels
    else:
        cluster_density = 0.0

    return {
        "mean": mean,
        "median": median,
        "max": vmax,
        "std": std,
        "occupancy": occ,
        "cluster_count": cluster_count,
        "cluster_count_cc": cluster_count_cc,  # Keep original for comparison
        "cluster_density": cluster_density,  # Normalized by marker area
        "cluster_area_mean": cluster_area_mean,
        "skeleton_len": skeleton_len,
        "skeleton_components": skel_complexity["skeleton_components"],
        "skeleton_endpoints": skel_complexity["endpoints"],
        "skeleton_branch_points": skel_complexity["branch_points"],
        "complexity_score": skel_complexity["complexity_score"],
        "thickness_proxy": thickness_proxy,
    }


AJMORPH_CLASSES = [
    "straight",           # Linear, continuous junction (high occupancy, low clustering)
    "thick",              # Dense, wide junction (high occupancy, thick)
    "thick_to_reticular", # Transitional pattern
    "reticular",          # Fragmented, complex pattern (many clusters)
    "fingers",            # Sparse elongated elements
    "discontinuous",      # Multiple separated clusters
    "punctate",           # Few isolated spots
    "minimal",            # Almost no junction signal
    "unknown",            # Unclassified
]


def heuristic_ajmorph_class(features: Dict[str, Any], blur_robust: bool = False) -> str:
    """A very rough heuristic classifier for AJ morphology.

    This is *not* a validated model. It is intended as a placeholder to enable
    end-to-end figure generation until you train a supervised classifier.

    The morphology labels follow common categories used in junction-morphology
    analysis (e.g. straight, thick, reticular, fingers).

    Parameters
    ----------
    features : Dict[str, Any]
        Dictionary of computed features from interface_marker_features.
    blur_robust : bool
        If True, use skeleton-based metrics instead of cluster_count.
        This is more robust to image blur but may be less sensitive to
        fine junction fragmentation.

    Returns
    -------
    str
        Morphology class label.
    """
    occ = features.get("occupancy", float("nan"))
    th = features.get("thickness_proxy", float("nan"))
    sk = features.get("skeleton_len", 0)

    if not np.isfinite(occ):
        return "unknown"

    # Get cluster count (use skeleton-based if blur_robust)
    if blur_robust:
        skel_comp = features.get("skeleton_components", 0)
        complexity = features.get("complexity_score", 0)
        ncl = skel_comp
    else:
        ncl = features.get("cluster_count", features.get("cluster_count_cc", 0))
        complexity = features.get("complexity_score", 0)

    # === SPARSE/MINIMAL JUNCTIONS (occupancy < 0.15) ===
    if occ < 0.15:
        if sk > 30 and ncl <= 3:
            return "fingers"  # Elongated sparse elements
        elif sk > 0 and ncl >= 1:
            return "discontinuous"  # Some signal but sparse
        else:
            return "minimal"  # Almost no junction signal

    # === LOW-MEDIUM OCCUPANCY (0.15 - 0.4) ===
    if occ < 0.4:
        if ncl >= 6 or (blur_robust and complexity >= 8):
            return "reticular"  # Fragmented pattern
        elif ncl >= 3:
            return "discontinuous"  # Multiple separated clusters
        else:
            return "punctate"  # Few isolated spots

    # === MEDIUM OCCUPANCY (0.4 - 0.6) ===
    if occ < 0.6:
        if ncl >= 8 or (blur_robust and complexity >= 10):
            return "reticular"  # Many clusters = reticular
        elif ncl >= 4:
            return "thick_to_reticular"  # Transitional
        else:
            return "thick"  # Moderate coverage, few clusters

    # === HIGH OCCUPANCY (> 0.6) ===
    if occ > 0.6:
        if np.isfinite(th) and th > 3.0 and ncl <= 2:
            return "thick"  # Dense, wide junction
        elif ncl <= 4:
            return "straight"  # Continuous, linear junction
        elif ncl >= 8 or (blur_robust and complexity >= 10):
            return "reticular"  # High coverage but fragmented
        else:
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

