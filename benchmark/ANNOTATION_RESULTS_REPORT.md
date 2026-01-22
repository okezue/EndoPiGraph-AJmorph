# Benchmark Annotation Results Report

## Summary

Three junction classification methods were run on 25,104 cell-cell edges:

1. **EndoPiGraph (Full)** - Novel tool with full feature set (occupancy, cluster_count, skeleton metrics)
2. **Baseline** - Simple classifier using only intensity and occupancy
3. **Junction Mapper Equivalent** - Emulates JM's available metrics (no skeleton features)

## Class Distributions

| Class | EndoPiGraph | EPG Blur-Robust | Baseline | Junction Mapper |
|-------|-------------|-----------------|----------|-----------------|
| unknown | 71.5% | 75.2% | - | - |
| reticular | 22.8% | 14.7% | 5.9% | 12.2% |
| fingers | - | 0.2% | 70.9% | 64.9% |
| discontinuous | - | - | 12.0% | 5.2% |
| thick_to_reticular | 4.0% | 6.3% | 2.1% | 6.7% |
| thick | 1.5% | 3.1% | 8.8% | 5.6% |
| straight | 0.2% | 0.5% | 0.4% | 5.5% |

## Method Agreement

| Comparison | Agreement |
|------------|-----------|
| EndoPiGraph vs Baseline | **6.3%** |
| EndoPiGraph vs Junction Mapper | **9.9%** |
| Baseline vs Junction Mapper | **75.8%** |
| All three methods | **1.5%** |

## Key Finding: Data Distribution Mismatch

The benchmark data has **very low junction occupancy**:
- Median occupancy: **0.011** (1.1%)
- 75th percentile: **0.254** (25.4%)
- Mean: **0.166** (16.6%)

This means most cell-cell contacts have minimal VE-cadherin at the junction.

### Why EndoPiGraph Shows 71.5% "unknown"

EndoPiGraph's heuristic classifier rules don't cover the low-occupancy regime well:

```
"unknown" class breakdown:
- 82% have occupancy < 0.15
- Mean occupancy: 0.044
- Mean cluster_count: 1.8
- Mean skeleton_len: 5.0
```

The heuristic rules assume higher occupancy typical of confluent monolayers.

### Why Baseline/JM Show ~70% "fingers"

Both classify low-occupancy edges as "fingers" (sparse junctions):
- Baseline: occupancy < 0.2 → "fingers"
- JM: occupancy < 0.2 + few clusters → "fingers"

## Feature Statistics by EndoPiGraph Class

| Class | Occupancy | Clusters | Skeleton Len | Complexity |
|-------|-----------|----------|--------------|------------|
| reticular | 0.453 | 6.9 | 39.6 | 12.8 |
| thick_to_reticular | 0.402 | 3.0 | 19.4 | 5.6 |
| thick | 0.890 | 1.2 | 16.4 | 3.6 |
| straight | 0.730 | 1.3 | 21.9 | 6.7 |

These classes show reasonable feature separation when data matches the expected distribution.

## Recommendations

### 1. Improve EndoPiGraph Heuristic Classifier

Add rules for low-occupancy edges:
```python
# Current gap: occupancy < 0.15 with low clustering
if occ < 0.15 and sk < 10:
    return "sparse"  # or "minimal"
```

### 2. Use Supervised Learning

The heuristic classifier is a placeholder. For production use:
- Collect manual annotations on a subset (50-100 edges)
- Train RandomForest or similar on the full feature set
- EndoPiGraph's rich features should outperform simpler baselines

### 3. Consider Data-Specific Thresholds

The current thresholds (occ > 0.6 for thick, etc.) were designed for different data.
Dataset-specific calibration may be needed.

## Files Generated

- `benchmark_annotations_all_methods.csv` - Full annotation results (25,104 rows)
- `benchmark_analysis.json` - Summary statistics
- `confusion_epg_vs_baseline.csv` - EPG vs Baseline confusion matrix
- `confusion_epg_vs_jm.csv` - EPG vs JM confusion matrix

## Conclusion

The low method agreement (6-10%) highlights that:
1. **Junction classification is ambiguous** without ground truth
2. **Different methods use different criteria** (occupancy vs clustering vs skeleton)
3. **EndoPiGraph provides richer features** but needs better heuristic rules or supervised training

The high Baseline-JM agreement (75.8%) occurs because both primarily use occupancy thresholds,
while EndoPiGraph uses multiple features that don't align with occupancy-only rules.

---

*Generated: January 2026*
*Total edges: 25,104*
*Images: 30 (S-BIAD1540 HUVEC)*
