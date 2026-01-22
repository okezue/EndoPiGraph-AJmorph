#!/usr/bin/env python3
"""Test blur robustness of the new cluster counting method."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label
import tifffile

from endopigraph.ajmorph import count_clusters_robust, interface_marker_features, compute_skeleton_complexity


def test_blur_robustness():
    """Compare cluster counting methods under blur."""

    # Find a test image
    data_dir = Path(__file__).parent.parent / "data" / "S-BIAD1540" / "images_egm2"
    tiffs = list(data_dir.glob("*.tif"))

    if not tiffs:
        print("No test images found")
        return

    img_path = tiffs[0]
    print(f"Testing with: {img_path.name}")

    # Load image (use channel 0 for VE-cadherin)
    raw = tifffile.imread(img_path)
    if raw.ndim == 3:
        img = raw[0].astype(float)  # First channel (VE-cadherin)
    else:
        img = raw.astype(float)

    # Create a simple test interface mask (center region)
    h, w = img.shape
    interface_mask = np.zeros((h, w), dtype=bool)
    interface_mask[h//4:3*h//4, w//4:3*w//4] = True

    # Test at different blur levels
    blur_sigmas = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print("\n" + "="*70)
    print("BLUR ROBUSTNESS TEST")
    print("="*70)
    print(f"{'Blur σ':>8} | {'CC Count':>10} | {'Robust Count':>12} | {'CC Δ%':>8} | {'Robust Δ%':>10}")
    print("-"*70)

    baseline_cc = None
    baseline_robust = None

    for sigma in blur_sigmas:
        # Apply blur
        if sigma > 0:
            blurred = gaussian_filter(img, sigma=sigma)
        else:
            blurred = img.copy()

        # Compute threshold
        vals = blurred[interface_mask]
        thresh = threshold_otsu(vals)

        # Binary mask
        bin_mask = (blurred > thresh) & interface_mask

        # Method 1: Connected components (original)
        lab = label(bin_mask)
        cc_count = int(lab.max())

        # Method 2: Robust h-maxima based
        robust_count = count_clusters_robust(blurred, bin_mask, h_threshold=0.1)

        # Track baseline
        if baseline_cc is None:
            baseline_cc = cc_count
            baseline_robust = robust_count

        # Calculate percent change from baseline
        cc_delta = ((cc_count - baseline_cc) / max(baseline_cc, 1)) * 100
        robust_delta = ((robust_count - baseline_robust) / max(baseline_robust, 1)) * 100

        print(f"{sigma:>8.1f} | {cc_count:>10} | {robust_count:>12} | {cc_delta:>+7.1f}% | {robust_delta:>+9.1f}%")

    print("-"*70)
    print("\nInterpretation:")
    print("- CC Count: Connected component counting (original method)")
    print("- Robust Count: H-maxima based counting (new method)")
    print("- Lower |Δ%| means more robust to blur")
    print()

    # Also test with full feature extraction
    print("\n" + "="*70)
    print("FULL FEATURE EXTRACTION TEST")
    print("="*70)

    for sigma in [0, 1.0, 2.0]:
        if sigma > 0:
            blurred = gaussian_filter(img, sigma=sigma)
        else:
            blurred = img.copy()

        vals = blurred[interface_mask]
        thresh = threshold_otsu(vals)

        features = interface_marker_features(blurred, interface_mask, thresh, use_robust_clustering=True)

        print(f"\nBlur σ = {sigma}:")
        print(f"  cluster_count (robust): {features['cluster_count']}")
        print(f"  cluster_count_cc (orig): {features['cluster_count_cc']}")
        print(f"  occupancy: {features['occupancy']:.3f}")
        print(f"  skeleton_len: {features['skeleton_len']}")


def compute_cohens_d(baseline_vals, changed_vals):
    """Compute Cohen's d effect size."""
    n1, n2 = len(baseline_vals), len(changed_vals)
    var1, var2 = np.var(baseline_vals, ddof=1), np.var(changed_vals, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(baseline_vals) - np.mean(changed_vals)) / pooled_std


def test_stability_improvement():
    """Test if skeleton-based metrics improve stability under blur."""

    print("\n" + "="*70)
    print("COMPREHENSIVE STABILITY ANALYSIS (Cohen's d)")
    print("="*70)

    # Find test images
    data_dir = Path(__file__).parent.parent / "data" / "S-BIAD1540" / "images_egm2"
    tiffs = list(data_dir.glob("*.tif"))[:5]  # Use up to 5 images

    if len(tiffs) < 2:
        print("Need at least 2 images for stability analysis")
        return

    print(f"Using {len(tiffs)} images for analysis")

    # Metrics to track
    metrics = {
        "cluster_count_cc": {"baseline": [], "blur1": [], "blur2": []},
        "skeleton_components": {"baseline": [], "blur1": [], "blur2": []},
        "complexity_score": {"baseline": [], "blur1": [], "blur2": []},
        "skeleton_endpoints": {"baseline": [], "blur1": [], "blur2": []},
        "skeleton_branch_points": {"baseline": [], "blur1": [], "blur2": []},
        "skeleton_len": {"baseline": [], "blur1": [], "blur2": []},
    }

    for tiff_path in tiffs:
        raw = tifffile.imread(tiff_path)
        if raw.ndim == 3:
            img = raw[0].astype(float)  # First channel
        else:
            img = raw.astype(float)

        # Simple center mask
        h, w = img.shape
        interface_mask = np.zeros((h, w), dtype=bool)
        interface_mask[h//4:3*h//4, w//4:3*w//4] = True

        for sigma, condition in [(0, "baseline"), (1.0, "blur1"), (2.0, "blur2")]:
            if sigma > 0:
                blurred = gaussian_filter(img, sigma=sigma)
            else:
                blurred = img.copy()

            vals = blurred[interface_mask]
            thresh = threshold_otsu(vals)
            bin_mask = (blurred > thresh) & interface_mask

            # Connected components
            lab = label(bin_mask)
            metrics["cluster_count_cc"][condition].append(int(lab.max()))

            # Skeleton-based metrics
            skel_metrics = compute_skeleton_complexity(bin_mask)
            metrics["skeleton_components"][condition].append(skel_metrics["skeleton_components"])
            metrics["complexity_score"][condition].append(skel_metrics["complexity_score"])
            metrics["skeleton_endpoints"][condition].append(skel_metrics["endpoints"])
            metrics["skeleton_branch_points"][condition].append(skel_metrics["branch_points"])

            # Skeleton length
            from skimage.morphology import skeletonize
            skel = skeletonize(bin_mask)
            metrics["skeleton_len"][condition].append(int(skel.sum()))

    # Compute and display Cohen's d for each metric
    print(f"\n{'Metric':<25} | {'blur_1px d':>12} | {'blur_2px d':>12} | {'Status':>10}")
    print("-"*70)

    stable_count = 0
    total_tests = 0

    for metric_name, data in metrics.items():
        d_blur1 = compute_cohens_d(data["baseline"], data["blur1"])
        d_blur2 = compute_cohens_d(data["baseline"], data["blur2"])

        # Count stability
        blur1_stable = abs(d_blur1) < 0.5
        blur2_stable = abs(d_blur2) < 0.5

        if blur1_stable:
            stable_count += 1
        if blur2_stable:
            stable_count += 1
        total_tests += 2

        status1 = "stable" if blur1_stable else "UNSTABLE"
        status2 = "stable" if blur2_stable else "UNSTABLE"

        print(f"{metric_name:<25} | {d_blur1:>+12.3f} | {d_blur2:>+12.3f} | {status1}/{status2}")

    print("-"*70)
    print(f"\nOverall stability: {stable_count}/{total_tests} ({100*stable_count/total_tests:.1f}%)")
    print("\nKey insight: Lower |Cohen's d| means more robust to blur.")
    print("Target: |d| < 0.5 for stability.")

    # Highlight best metrics
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("""
For blur-robust junction quantification, consider using:

1. skeleton_len - Total skeleton pixels (relatively stable)
2. skeleton_components - Number of separate skeleton pieces
3. complexity_score - Combined metric (components + 0.5*branch_points)

These metrics degrade more gracefully under blur than raw cluster_count
because they capture the topological "core" of junction structures.

Note: Some blur sensitivity is EXPECTED and physically meaningful -
blur genuinely simplifies image structure. The goal is graceful
degradation, not perfect invariance.
""")


if __name__ == "__main__":
    test_blur_robustness()
    test_stability_improvement()
