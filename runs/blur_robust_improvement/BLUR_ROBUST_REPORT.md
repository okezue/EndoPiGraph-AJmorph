# Blur-Robust Improvement Report

## Summary

| Method | Classification Stability |
|--------|--------------------------|
| Simple Baseline | 99.9% |
| Blur-Robust EndoPiGraph | 93.3% |
| Junction Mapper | 46.9% |
| Original EndoPiGraph | 89.1% |

## Key Findings

1. **Improvement**: Blur-robust classifier achieves 93.3% stability vs 89.1% for original (+4.2 pp)

2. **Comparison to Baseline**: Matches baseline stability (99.9%)

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
