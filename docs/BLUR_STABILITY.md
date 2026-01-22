# Blur Stability in EndoPiGraph

This document describes how EndoPiGraph handles image blur and provides guidance on achieving reliable results with variable image quality.

## Overview

Image blur is common in microscopy due to:
- Out-of-focus planes in thick samples
- Motion blur during acquisition
- Optical aberrations
- Lower-quality objectives

Some EndoPiGraph metrics are sensitive to blur, while others remain stable. This document explains which metrics to use and how to detect/correct for blur.

## Metric Stability Reference

Metrics are categorized by their stability under 1-2 pixel Gaussian blur (simulating typical microscopy focus issues):

### Stable Metrics (|Cohen's d| < 0.3)

These metrics are highly reliable regardless of blur:

| Metric | Cohen's d | Description |
|--------|-----------|-------------|
| `mean_intensity` | 0.000 | Average marker intensity in interface |
| `median_intensity` | ~0 | Median marker intensity |
| `occupancy` | 0.214 | Fraction of interface with signal above threshold |

### Marginal Metrics (|Cohen's d| 0.3-0.5)

These metrics are moderately stable but may show some variation:

| Metric | Cohen's d | Description |
|--------|-----------|-------------|
| `skeleton_len` | 0.428 | Total skeleton length (pixels) |
| `total_area` | ~0.3 | Total area of thresholded signal |

### Unstable Metrics (|Cohen's d| > 0.5)

These metrics change significantly under blur and should be used cautiously:

| Metric | Cohen's d | Description |
|--------|-----------|-------------|
| `cluster_count` | 0.600 | Number of distinct clusters |
| `skeleton_endpoints` | 0.531 | Number of skeleton endpoints |
| `skeleton_components` | 0.600 | Number of separate skeletons |
| `complexity_score` | 0.669 | Topological complexity |
| `skeleton_branch_points` | 0.825 | Number of skeleton junctions |
| `cluster_density` | 1.397 | Clusters per unit length |
| `mean_cluster_size` | 3.351 | Average cluster area |
| `thickness_proxy` | 4.039 | Estimated junction thickness |

## Blur Detection

Use `estimate_blur_score()` to assess image quality:

```python
from endopigraph import estimate_blur_score, detect_blur

# Get blur score (Laplacian variance)
score = estimate_blur_score(image)

# Automatic detection
is_blurry, score = detect_blur(image, threshold=50.0)

# Interpretation:
# - Sharp images: > 100
# - Moderate blur: 20-100
# - Heavy blur: < 20
```

## Blur Correction

For mildly blurry images, unsharp masking can partially restore sharpness:

```python
from endopigraph import correct_blur

# Apply correction (works best for mild blur)
sharpened = correct_blur(image, radius=1.0, amount=1.5)

# Note: Cannot recover information lost to severe blur
```

## Blur-Robust Classification

Use the blur-robust classifier for reliable results on variable-quality data:

```python
from endopigraph import (
    compute_blur_robust_features,
    blur_robust_classifier,
    compute_adaptive_features,
)

# Option 1: Manual blur-robust analysis
features = compute_blur_robust_features(marker, interface_mask, threshold)
classification = blur_robust_classifier(features)

# Option 2: Automatic detection + correction
features, metadata = compute_adaptive_features(
    marker,
    interface_mask,
    threshold,
    auto_correct_blur=True,
    blur_threshold=50.0,
)

# Check what happened
print(f"Blur score: {metadata['blur_score']}")
print(f"Was blurry: {metadata['is_blurry']}")
print(f"Was corrected: {metadata['blur_corrected']}")
print(f"Recommended metrics: {metadata['recommended_metrics']}")
```

## Classification Stability Comparison

Benchmark results comparing classification consistency under blur:

| Method | Classification Stability |
|--------|--------------------------|
| Simple Baseline (intensity/occupancy only) | 99.9% |
| **Blur-Robust EndoPiGraph** | **93.3%** |
| Original EndoPiGraph (uses cluster_count) | 89.1% |
| Junction Mapper equivalent | 46.9% |

The blur-robust classifier achieves near-baseline stability while providing meaningful morphological classification.

## Recommended Workflow

### For Sharp Images (blur score > 100)

Use full EndoPiGraph metrics:
```python
from endopigraph import interface_marker_features, heuristic_ajmorph_class

features = interface_marker_features(marker, interface_mask, threshold)
classification = heuristic_ajmorph_class(features, blur_robust=False)
```

### For Variable/Unknown Quality

Use blur-robust mode:
```python
from endopigraph import interface_marker_features, heuristic_ajmorph_class

features = interface_marker_features(marker, interface_mask, threshold)
classification = heuristic_ajmorph_class(features, blur_robust=True)
```

### For Explicitly Blurry Images

Use adaptive features with correction:
```python
from endopigraph import compute_adaptive_features, blur_robust_classifier

features, metadata = compute_adaptive_features(
    marker, interface_mask, threshold,
    auto_correct_blur=True
)
classification = blur_robust_classifier(features)
```

## Technical Details

### How Blur Affects Metrics

Gaussian blur (simulating out-of-focus imaging):
1. **Reduces intensity gradients** → Makes thresholding less precise
2. **Merges adjacent structures** → Reduces cluster_count
3. **Expands boundaries** → Increases thickness_proxy
4. **Preserves totals** → mean_intensity, occupancy remain stable

### Why Cluster Count is Unstable

```
Sharp image:        Blurred image:
[●] [●] [●]    →    [●●●●●●]
3 clusters          1 cluster
```

Adjacent clusters merge when their boundaries become indistinct.

### Why Occupancy is Stable

```
Sharp image:        Blurred image:
■■□■■□■■     →     ■■■■■■■■
Occupancy: 75%      Occupancy: 75%
```

The total signal area remains approximately constant.

## References

- Cohen's d interpretation: |d| < 0.2 trivial, < 0.5 small, < 0.8 medium, ≥ 0.8 large
- Laplacian variance method: Pech-Pacheco et al. (2000)
- Unsharp masking: Classic image enhancement technique
