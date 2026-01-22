#!/usr/bin/env python3
"""
Test blur-robust improvements for EndoPiGraph.

This script compares classification stability under blur for:
1. Original EndoPiGraph (heuristic classifier using cluster_count)
2. Blur-robust EndoPiGraph (classifier using occupancy + skeleton_len only)
3. Simple Baseline (intensity/occupancy only)
4. Junction Mapper equivalent (cluster-based)

Key metric: Classification consistency rate under blur (% edges keeping same class).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import tifffile
import pandas as pd
import json

from endopigraph.ajmorph import interface_marker_features, heuristic_ajmorph_class
from endopigraph.blur_robust import (
    compute_blur_robust_features,
    blur_robust_classifier,
    estimate_blur_score,
    detect_blur,
    STABLE_METRICS,
    MARGINAL_METRICS,
    UNSTABLE_METRICS,
)


def baseline_classifier(features):
    """
    Simple baseline classifier using ONLY intensity and occupancy.
    Most blur-stable approach (no cluster/skeleton analysis).
    """
    occ = features.get('occupancy', 0)
    mean_int = features.get('mean', features.get('mean_intensity', 0))
    std_int = features.get('std', features.get('std_intensity', 0))

    if not np.isfinite(occ):
        return 'unknown'

    cv = std_int / mean_int if mean_int > 0 else 0

    if occ > 0.6:
        if cv < 0.3:
            return 'straight'
        else:
            return 'thick'
    elif occ < 0.2:
        return 'fingers'
    elif occ < 0.4:
        return 'discontinuous'
    else:
        if cv > 0.4:
            return 'reticular'
        else:
            return 'thick_to_reticular'


def junction_mapper_classifier(features):
    """
    Junction Mapper equivalent classifier using occupancy + cluster_count.
    """
    occ = features.get('occupancy', 0)
    n_clusters = features.get('cluster_count_cc', features.get('cluster_count', 0))

    if not np.isfinite(occ):
        return 'unknown'

    if occ > 0.7 and n_clusters <= 2:
        return 'straight'
    if occ > 0.5 and n_clusters <= 4:
        return 'thick'
    if n_clusters >= 8:
        return 'reticular'
    if occ < 0.2:
        return 'fingers'
    if occ < 0.4 and n_clusters >= 3:
        return 'discontinuous'
    return 'thick_to_reticular'


def create_synthetic_interfaces(image, n_interfaces=20):
    """Create synthetic interface masks for testing."""
    h, w = image.shape
    masks = []

    # Create random horizontal and vertical stripe interfaces
    for i in range(n_interfaces):
        mask = np.zeros((h, w), dtype=bool)

        if i % 2 == 0:
            # Horizontal stripe
            y = int(h * (0.1 + 0.8 * (i / n_interfaces)))
            thickness = np.random.randint(5, 20)
            x_start = np.random.randint(0, w // 4)
            x_end = np.random.randint(3 * w // 4, w)
            mask[max(0, y-thickness):min(h, y+thickness), x_start:x_end] = True
        else:
            # Vertical stripe
            x = int(w * (0.1 + 0.8 * (i / n_interfaces)))
            thickness = np.random.randint(5, 20)
            y_start = np.random.randint(0, h // 4)
            y_end = np.random.randint(3 * h // 4, h)
            mask[y_start:y_end, max(0, x-thickness):min(w, x+thickness)] = True

        if mask.sum() > 100:  # Ensure meaningful size
            masks.append(mask)

    return masks


def run_blur_stability_test(image, interfaces, thresh, blur_sigma=1.5):
    """
    Run classification under baseline and blur, compare consistency.

    Returns dict with classification results for each method.
    """
    # Apply blur
    blurred = gaussian_filter(image, sigma=blur_sigma)

    results = {
        'epg_original': {'baseline': [], 'blurred': [], 'match': []},
        'epg_blur_robust': {'baseline': [], 'blurred': [], 'match': []},
        'baseline_simple': {'baseline': [], 'blurred': [], 'match': []},
        'junction_mapper': {'baseline': [], 'blurred': [], 'match': []},
    }

    for mask in interfaces:
        # Original image
        epg_feats = interface_marker_features(image, mask, thresh)
        blur_feats = compute_blur_robust_features(image, mask, thresh)

        # Blurred image
        epg_feats_blur = interface_marker_features(blurred, mask, thresh)
        blur_feats_blur = compute_blur_robust_features(blurred, mask, thresh)

        # EPG Original (uses cluster_count)
        cls_epg_base = heuristic_ajmorph_class(epg_feats, blur_robust=False)
        cls_epg_blur = heuristic_ajmorph_class(epg_feats_blur, blur_robust=False)
        results['epg_original']['baseline'].append(cls_epg_base)
        results['epg_original']['blurred'].append(cls_epg_blur)
        results['epg_original']['match'].append(cls_epg_base == cls_epg_blur)

        # EPG Blur-Robust (uses occupancy + skeleton_len)
        cls_robust_base = blur_robust_classifier(blur_feats)
        cls_robust_blur = blur_robust_classifier(blur_feats_blur)
        results['epg_blur_robust']['baseline'].append(cls_robust_base)
        results['epg_blur_robust']['blurred'].append(cls_robust_blur)
        results['epg_blur_robust']['match'].append(cls_robust_base == cls_robust_blur)

        # Simple Baseline (intensity/occupancy only)
        cls_base_base = baseline_classifier(epg_feats)
        cls_base_blur = baseline_classifier(epg_feats_blur)
        results['baseline_simple']['baseline'].append(cls_base_base)
        results['baseline_simple']['blurred'].append(cls_base_blur)
        results['baseline_simple']['match'].append(cls_base_base == cls_base_blur)

        # Junction Mapper equivalent
        cls_jm_base = junction_mapper_classifier(epg_feats)
        cls_jm_blur = junction_mapper_classifier(epg_feats_blur)
        results['junction_mapper']['baseline'].append(cls_jm_base)
        results['junction_mapper']['blurred'].append(cls_jm_blur)
        results['junction_mapper']['match'].append(cls_jm_base == cls_jm_blur)

    return results


def main():
    print("=" * 80)
    print("BLUR-ROBUST IMPROVEMENT TEST")
    print("=" * 80)

    # Find test images
    benchmark_dir = Path(__file__).parent.parent / "benchmark" / "images"
    data_dir = Path(__file__).parent.parent / "data" / "S-BIAD1540" / "images_egm2"

    tiffs = []
    if benchmark_dir.exists():
        tiffs = list(benchmark_dir.glob("*.tif"))[:10]
    if len(tiffs) < 3 and data_dir.exists():
        tiffs = list(data_dir.glob("*.tif"))[:10]

    if len(tiffs) < 1:
        print("No test images found")
        return

    print(f"\nUsing {len(tiffs)} images for stability testing")

    # Test multiple blur levels
    blur_levels = [1.0, 1.5, 2.0]

    all_results = []

    for tiff_path in tiffs:
        print(f"\nProcessing: {tiff_path.name}")

        raw = tifffile.imread(tiff_path)
        if raw.ndim == 3:
            img = raw[0].astype(float)
        else:
            img = raw.astype(float)

        # Check blur score of original
        blur_score = estimate_blur_score(img)
        is_blurry, _ = detect_blur(img)
        print(f"  Original blur score: {blur_score:.1f} ({'blurry' if is_blurry else 'sharp'})")

        # Create synthetic interfaces
        interfaces = create_synthetic_interfaces(img, n_interfaces=30)
        print(f"  Created {len(interfaces)} test interfaces")

        # Compute threshold
        vals = img[img > 0]
        if len(vals) > 0:
            thresh = threshold_otsu(vals)
        else:
            thresh = img.mean()

        for sigma in blur_levels:
            results = run_blur_stability_test(img, interfaces, thresh, blur_sigma=sigma)

            for method, data in results.items():
                matches = data['match']
                stability = sum(matches) / len(matches) if matches else 0

                all_results.append({
                    'image': tiff_path.name,
                    'blur_sigma': sigma,
                    'method': method,
                    'stability': stability,
                    'n_interfaces': len(matches),
                    'n_consistent': sum(matches),
                })

    # Aggregate results
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("CLASSIFICATION STABILITY RESULTS")
    print("=" * 80)

    # Summary by method
    print("\n### Stability by Method (averaged across all images and blur levels)")
    print("-" * 70)
    print(f"{'Method':<25} | {'Stability %':>12} | {'Rank':>8}")
    print("-" * 70)

    method_stats = df.groupby('method')['stability'].mean().sort_values(ascending=False)

    for rank, (method, stability) in enumerate(method_stats.items(), 1):
        method_display = method.replace('_', ' ').title()
        print(f"{method_display:<25} | {stability*100:>11.1f}% | {rank:>8}")

    # Breakdown by blur level
    print("\n### Stability by Blur Level")
    print("-" * 70)

    for sigma in blur_levels:
        sigma_df = df[df['blur_sigma'] == sigma]
        print(f"\nBlur σ = {sigma} px:")

        method_stability = sigma_df.groupby('method')['stability'].mean()
        for method, stability in method_stability.sort_values(ascending=False).items():
            method_display = method.replace('_', ' ').title()
            print(f"  {method_display:<25}: {stability*100:.1f}%")

    # Improvement calculation
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)

    epg_orig = method_stats.get('epg_original', 0)
    epg_robust = method_stats.get('epg_blur_robust', 0)
    baseline = method_stats.get('baseline_simple', 0)
    jm = method_stats.get('junction_mapper', 0)

    print(f"""
Original EndoPiGraph:     {epg_orig*100:.1f}%
Blur-Robust EndoPiGraph:  {epg_robust*100:.1f}%
Simple Baseline:          {baseline*100:.1f}%
Junction Mapper:          {jm*100:.1f}%

Improvement (blur-robust over original): {(epg_robust - epg_orig)*100:+.1f} percentage points
Comparison to baseline:                  {(epg_robust - baseline)*100:+.1f} percentage points
""")

    if epg_robust > epg_orig:
        print("✓ Blur-robust classifier IMPROVES stability!")
    else:
        print("⚠ Blur-robust classifier needs further tuning")

    if epg_robust >= baseline * 0.9:
        print("✓ Blur-robust EndoPiGraph matches baseline stability!")
    else:
        gap = (baseline - epg_robust) * 100
        print(f"⚠ Blur-robust EndoPiGraph still {gap:.1f}% below baseline")

    # Metric-level stability analysis
    print("\n" + "=" * 80)
    print("METRIC STABILITY REFERENCE")
    print("=" * 80)

    print("\nStable metrics (|Cohen's d| < 0.3 under blur):")
    for m in STABLE_METRICS:
        print(f"  ✓ {m}")

    print("\nMarginal metrics (|Cohen's d| 0.3-0.5):")
    for m in MARGINAL_METRICS:
        print(f"  ~ {m}")

    print("\nUnstable metrics (|Cohen's d| > 0.5):")
    for m in UNSTABLE_METRICS[:5]:
        print(f"  ✗ {m}")
    print("  ...")

    # Save results
    output_dir = Path(__file__).parent.parent / "runs" / "blur_robust_improvement"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "stability_results.csv", index=False)

    summary = {
        'overall_stability': method_stats.to_dict(),
        'by_blur_level': {
            f'sigma_{sigma}': df[df['blur_sigma'] == sigma].groupby('method')['stability'].mean().to_dict()
            for sigma in blur_levels
        },
        'improvement': {
            'epg_original': float(epg_orig),
            'epg_blur_robust': float(epg_robust),
            'baseline_simple': float(baseline),
            'junction_mapper': float(jm),
            'improvement_pp': float(epg_robust - epg_orig),
        }
    }

    with open(output_dir / "stability_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Generate report
    report = f"""# Blur-Robust Improvement Report

## Summary

| Method | Classification Stability |
|--------|--------------------------|
| Simple Baseline | {baseline*100:.1f}% |
| Blur-Robust EndoPiGraph | {epg_robust*100:.1f}% |
| Junction Mapper | {jm*100:.1f}% |
| Original EndoPiGraph | {epg_orig*100:.1f}% |

## Key Findings

1. **Improvement**: Blur-robust classifier achieves {epg_robust*100:.1f}% stability vs {epg_orig*100:.1f}% for original ({(epg_robust-epg_orig)*100:+.1f} pp)

2. **Comparison to Baseline**: {"Matches" if epg_robust >= baseline * 0.9 else "Below"} baseline stability ({baseline*100:.1f}%)

3. **Mechanism**: Blur-robust classifier uses only occupancy + skeleton_len (stable metrics), avoiding cluster_count (unstable under blur)

## Recommendations

- For sharp images (blur score > 100): Use full EndoPiGraph metrics
- For moderate blur (50-100): Use blur-robust classifier
- For heavy blur (< 50): Apply unsharp masking correction first

## Technical Details

The blur-robust classifier in `src/endopigraph/blur_robust.py` provides:
- `estimate_blur_score()`: Laplacian variance blur detection
- `correct_blur()`: Unsharp masking correction
- `blur_robust_classifier()`: Classification using only stable metrics
- `compute_adaptive_features()`: Auto-detection and correction
"""

    with open(output_dir / "BLUR_ROBUST_REPORT.md", 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_dir / 'BLUR_ROBUST_REPORT.md'}")

    return df, summary


if __name__ == "__main__":
    main()
