"""
Blur-robust metrics and utilities for EndoPiGraph.

This module provides:
1. Blur detection (Laplacian variance method)
2. Blur correction (unsharp masking)
3. Blur-robust metric subset
4. Blur-robust junction classifier

Based on stability analysis showing which metrics are sensitive to blur:
- STABLE (|Cohen's d| < 0.3): mean_intensity, occupancy
- MARGINAL (0.3-0.5): skeleton_len
- UNSTABLE (>0.5): cluster_count, thickness_proxy, cluster_density, etc.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy import ndimage
from skimage.filters import threshold_otsu, laplace, unsharp_mask
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


# =============================================================================
# Blur Detection
# =============================================================================

def estimate_blur_score(image: np.ndarray) -> float:
    """
    Estimate image blur using Laplacian variance.

    Higher values indicate sharper images. Lower values indicate blur.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D grayscale).

    Returns
    -------
    float
        Blur score (Laplacian variance). Typical ranges:
        - Sharp images: > 100
        - Moderate blur: 20-100
        - Heavy blur: < 20
    """
    if image.ndim != 2:
        raise ValueError("Image must be 2D grayscale")

    # Normalize to 0-1
    img = image.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Compute Laplacian
    lap = laplace(img)

    # Variance of Laplacian
    return float(np.var(lap) * 1e6)  # Scale for readability


def detect_blur(image: np.ndarray, threshold: float = 50.0) -> Tuple[bool, float]:
    """
    Detect if an image is blurry.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    threshold : float
        Blur score threshold. Images below this are considered blurry.
        Default 50.0 based on typical microscopy images.

    Returns
    -------
    Tuple[bool, float]
        (is_blurry, blur_score)
    """
    score = estimate_blur_score(image)
    return score < threshold, score


# =============================================================================
# Blur Correction
# =============================================================================

def correct_blur(
    image: np.ndarray,
    radius: float = 1.0,
    amount: float = 1.0,
) -> np.ndarray:
    """
    Apply unsharp masking to correct mild blur.

    This enhances edges and can partially compensate for blur,
    but cannot recover information lost to severe blur.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    radius : float
        Gaussian blur radius for unsharp mask.
    amount : float
        Sharpening strength (1.0 = moderate, 2.0 = strong).

    Returns
    -------
    np.ndarray
        Sharpened image.
    """
    # Normalize
    img = image.astype(float)
    img_min, img_max = img.min(), img.max()
    img_norm = (img - img_min) / (img_max - img_min + 1e-8)

    # Apply unsharp mask
    sharpened = unsharp_mask(img_norm, radius=radius, amount=amount)

    # Rescale back
    sharpened = sharpened * (img_max - img_min) + img_min

    return sharpened.astype(image.dtype)


# =============================================================================
# Blur-Robust Metrics
# =============================================================================

# Metrics ranked by blur stability (Cohen's d under 1-2px Gaussian blur)
STABLE_METRICS = [
    'mean_intensity',      # d = 0.000 (perfectly stable)
    'median_intensity',    # d ≈ 0 (stable)
    'occupancy',           # d = 0.214 (stable)
]

MARGINAL_METRICS = [
    'skeleton_len',        # d = 0.428 (marginally stable)
    'total_area',          # d ≈ 0.3 (relatively stable)
]

UNSTABLE_METRICS = [
    'cluster_count',       # d = 0.600 (unstable)
    'cluster_count_cc',    # d = 0.600 (unstable)
    'skeleton_endpoints',  # d = 0.531 (unstable)
    'skeleton_components', # d = 0.600 (unstable)
    'complexity_score',    # d = 0.669 (unstable)
    'skeleton_branch_points',  # d = 0.825 (very unstable)
    'cluster_density',     # d = 1.397 (very unstable)
    'mean_cluster_size',   # d = 3.351 (extremely unstable)
    'thickness_proxy',     # d = 4.039 (extremely unstable)
]


def compute_blur_robust_features(
    marker: np.ndarray,
    interface_mask: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Compute only blur-robust features for junction analysis.

    This function returns a subset of metrics that are stable under
    typical microscopy blur (1-2px Gaussian sigma).

    Parameters
    ----------
    marker : np.ndarray
        Marker intensity image.
    interface_mask : np.ndarray
        Binary mask of the interface region.
    threshold : float
        Intensity threshold for binarization.

    Returns
    -------
    Dict[str, Any]
        Dictionary with blur-robust features only.
    """
    vals = marker[interface_mask]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            'mean_intensity': float('nan'),
            'median_intensity': float('nan'),
            'std_intensity': float('nan'),
            'occupancy': float('nan'),
            'total_area': 0,
            'skeleton_len': 0,
            'blur_robust_score': float('nan'),
        }

    # Intensity metrics (very stable)
    mean_int = float(np.mean(vals))
    median_int = float(np.median(vals))
    std_int = float(np.std(vals))

    # Occupancy (stable)
    bin_mask = (marker > threshold) & interface_mask
    total_area = int(bin_mask.sum())
    interface_area = int(interface_mask.sum())
    occupancy = float(total_area / max(interface_area, 1))

    # Skeleton length (marginally stable - better than cluster_count)
    skel = skeletonize(bin_mask)
    skeleton_len = int(skel.sum())

    # Compute a blur-robust composite score
    # Weighted combination of stable metrics
    # Higher = more junction signal
    if np.isfinite(occupancy) and np.isfinite(mean_int):
        blur_robust_score = occupancy * 0.6 + (skeleton_len / max(interface_area, 1)) * 0.4
    else:
        blur_robust_score = float('nan')

    return {
        'mean_intensity': mean_int,
        'median_intensity': median_int,
        'std_intensity': std_int,
        'occupancy': occupancy,
        'total_area': total_area,
        'skeleton_len': skeleton_len,
        'blur_robust_score': blur_robust_score,
    }


def blur_robust_classifier(features: Dict[str, Any]) -> str:
    """
    Classify junction morphology using only blur-robust features.

    This classifier uses occupancy and skeleton_len (the two most
    blur-stable structural metrics) instead of cluster_count.

    Parameters
    ----------
    features : Dict[str, Any]
        Features from compute_blur_robust_features or interface_marker_features.

    Returns
    -------
    str
        Junction class label.
    """
    occ = features.get('occupancy', float('nan'))
    sk = features.get('skeleton_len', 0)
    total_area = features.get('total_area', 0)

    if not np.isfinite(occ):
        return 'unknown'

    # Compute skeleton density (more stable than cluster metrics)
    # This measures how "spread out" the junction signal is
    skel_density = sk / max(total_area, 1) if total_area > 0 else 0

    # Classification based on occupancy and skeleton density
    # (both are relatively blur-stable)

    if occ < 0.1:
        return 'minimal'

    if occ < 0.25:
        if skel_density > 0.3:
            return 'fingers'  # Sparse but elongated
        else:
            return 'punctate'  # Sparse, compact

    if occ < 0.5:
        if skel_density > 0.2:
            return 'discontinuous'  # Medium coverage, spread out
        else:
            return 'thick_to_reticular'  # Medium coverage, compact

    if occ < 0.7:
        if skel_density > 0.15:
            return 'reticular'  # High coverage, spread pattern
        else:
            return 'thick'  # High coverage, compact

    # High occupancy (> 0.7)
    if skel_density < 0.1:
        return 'straight'  # Very high coverage, linear
    else:
        return 'thick'  # Very high coverage, some structure


def get_recommended_metrics(blur_score: float) -> list:
    """
    Get recommended metrics based on image blur level.

    Parameters
    ----------
    blur_score : float
        Blur score from estimate_blur_score().

    Returns
    -------
    list
        List of recommended metric names.
    """
    if blur_score > 100:
        # Sharp image - all metrics OK
        return STABLE_METRICS + MARGINAL_METRICS + ['cluster_count', 'complexity_score']
    elif blur_score > 50:
        # Moderate blur - avoid most unstable
        return STABLE_METRICS + MARGINAL_METRICS
    else:
        # Heavy blur - only use stable metrics
        return STABLE_METRICS


# =============================================================================
# Adaptive Feature Computation
# =============================================================================

def compute_adaptive_features(
    marker: np.ndarray,
    interface_mask: np.ndarray,
    threshold: float,
    auto_correct_blur: bool = True,
    blur_threshold: float = 50.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute features with automatic blur detection and optional correction.

    Parameters
    ----------
    marker : np.ndarray
        Marker intensity image.
    interface_mask : np.ndarray
        Binary mask of the interface region.
    threshold : float
        Intensity threshold for binarization.
    auto_correct_blur : bool
        If True, apply unsharp masking when blur is detected.
    blur_threshold : float
        Blur score threshold for detection.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        (features, metadata) where metadata includes blur info.
    """
    # Detect blur
    is_blurry, blur_score = detect_blur(marker, blur_threshold)

    # Optionally correct blur
    if is_blurry and auto_correct_blur:
        marker_corrected = correct_blur(marker, radius=1.0, amount=1.5)
        blur_corrected = True
    else:
        marker_corrected = marker
        blur_corrected = False

    # Compute features
    features = compute_blur_robust_features(marker_corrected, interface_mask, threshold)

    # Add metadata
    metadata = {
        'blur_score': blur_score,
        'is_blurry': is_blurry,
        'blur_corrected': blur_corrected,
        'recommended_metrics': get_recommended_metrics(blur_score),
    }

    return features, metadata
