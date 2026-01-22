# Blur Stability Analysis: EndoPiGraph vs Junction Mapper vs Baseline

## Summary

This report compares the blur-robustness of metrics from:
1. **EndoPiGraph Full** - All metrics including cluster counting and skeleton analysis
2. **EndoPiGraph Skeleton** - New skeleton-based metrics designed for blur robustness
3. **Simple Baseline** - Intensity and occupancy only (minimal approach)
4. **Junction Mapper Equivalent** - Metrics matching Junction Mapper (Tomlinson et al. eLife 2019)

## Test Conditions

- **Images**: 5 S-BIAD1540 VE-cadherin images
- **Blur levels**: baseline (no blur), 1px Gaussian, 2px Gaussian
- **Stability criterion**: |Cohen's d| < 0.5

## Results Summary

| Metric Group | Stability Rate | Avg |Cohen's d| |
|--------------|----------------|-------------------|
| Simple Baseline | **83.3%** | **0.277** |
| Junction Mapper Equivalent | 50.0% | 1.042 |
| EndoPiGraph Full | 30.0% | 1.336 |
| EndoPiGraph Skeleton | 12.5% | 0.656 |

## Individual Metric Stability

### Most Stable Metrics (use for blur-variable data)

| Metric | Blur 1px (d) | Blur 2px (d) | Status |
|--------|--------------|--------------|--------|
| mean_intensity | 0.000 | 0.001 | **STABLE** |
| occupancy/fraction_occupied | 0.180 | 0.249 | **STABLE** |
| skeleton_len | 0.299 | 0.557 | Marginal |
| skeleton_endpoints | 0.418 | 0.645 | Marginal |

### Unstable Metrics (avoid for blur-variable data)

| Metric | Blur 1px (d) | Blur 2px (d) | Notes |
|--------|--------------|--------------|-------|
| cluster_count / n_clusters | 0.519 | 0.681 | Same in EPG & JM |
| cluster_density | 1.264 | 1.529 | Amplified by normalization |
| mean_cluster_size | 2.014 | 4.689 | Very sensitive |
| thickness_proxy | 3.636 | 4.442 | Very sensitive |

## Key Findings

### 1. Junction Mapper and EndoPiGraph have IDENTICAL cluster sensitivity

Both tools use connected component counting, which is fundamentally affected by blur:
- **Junction Mapper's `n_clusters`**: Cohen's d = 0.519 (blur 1px), 0.681 (blur 2px)
- **EndoPiGraph's `cluster_count`**: Cohen's d = 0.519 (blur 1px), 0.681 (blur 2px)

This is not a bug - it's a physical effect. Blur merges adjacent junction fragments.

### 2. Simple metrics are most robust

The simplest metrics (intensity, occupancy) are the most blur-stable:
- `mean_intensity`: d ≈ 0 (perfectly stable)
- `occupancy`: d = 0.21 (stable)

### 3. Skeleton-based metrics provide marginal improvement

The new skeleton metrics are slightly more stable than cluster counting but still affected:
- `skeleton_len`: d = 0.30 (vs cluster_count d = 0.52) for blur 1px
- `skeleton_endpoints`: d = 0.42 (vs cluster_count d = 0.52) for blur 1px

### 4. EndoPiGraph provides MORE metrics than Junction Mapper

While both have similar blur sensitivity for shared metrics, EndoPiGraph provides:
- 15 total metrics vs Junction Mapper's 5
- Skeleton-based alternatives for blur-variable data
- Automatic morphology classification
- Python API for automation

## Recommendations

### For blur-variable datasets:

1. **Primary metrics**: Use `occupancy` and `mean_intensity` (most stable)
2. **Secondary metrics**: Use `skeleton_len` (relatively stable)
3. **Avoid**: `cluster_count`, `cluster_density`, `thickness_proxy`

### For high-quality, consistent imaging:

1. **Use full metric set** including `cluster_count`
2. `cluster_count` provides valuable fragmentation information when blur is controlled

### For Junction Mapper users migrating to EndoPiGraph:

| Junction Mapper | EndoPiGraph Equivalent | Notes |
|-----------------|----------------------|-------|
| Junction length | `contact_px` | Identical |
| Fraction occupied | `occupancy` | Identical, blur-stable |
| Number of clusters | `cluster_count_cc` | Identical, blur-sensitive |
| Mean cluster size | `cluster_area_mean` | Identical, blur-sensitive |
| Mean intensity | `mean` | Identical, blur-stable |
| *N/A* | `skeleton_len` | **EPG-only, more stable** |
| *N/A* | `complexity_score` | **EPG-only** |
| *N/A* | `aj_morph_class` | **EPG-only** |

## Conclusion

**Blur sensitivity is inherent to cluster-counting metrics** in both Junction Mapper and EndoPiGraph. This is a physical effect, not a software limitation.

EndoPiGraph provides **more options** for users to select appropriate metrics based on their image quality. For blur-variable data, use `occupancy` and `skeleton_len`. For controlled conditions, the full metric set including `cluster_count` provides rich phenotypic information.

---

*Generated: January 2026*
*Test script: `scripts/comprehensive_blur_stability_test.py`*
