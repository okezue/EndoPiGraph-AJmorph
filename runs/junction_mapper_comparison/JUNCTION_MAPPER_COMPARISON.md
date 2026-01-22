# EndoPiGraph-AJmorph vs Junction Mapper Comparison

## Summary

| Metric | EndoPiGraph | Junction Mapper |
|--------|-------------|------------------|
| Capabilities | 15/15 | 9/15 |
| Fully Automated | Yes | No (semi-automated) |
| Unique Features | 4 | - |
| Programming API | Python | None (Java GUI) |

## Capability Comparison

| Capability | EndoPiGraph | Junction Mapper | Notes |
|------------|-------------|-----------------|-------|
| Automated cell segmentation | ✓ | ✗ | EndoPiGraph uses Cellpose (deep learning); JM requires manual outline correction |
| Junction length measurement | ✓ | ✓ | Both tools measure junction length |
| Junction area measurement | ✓ | ✓ | Both tools measure junction area |
| Intensity quantification | ✓ | ✓ | Both tools provide intensity metrics |
| Fragmented junction analysis | ✓ | ✓ | JM paper highlights this as unique capability; EPG measures cluster_count, cluster_density |
| Junction morphology classification | ✓ | ✗ | EndoPiGraph: heuristic + ML classifier; JM: manual phenotype assignment |
| Multi-junction type support | ✓ | ✓ | EndoPiGraph builds typed pi-graphs with multiple marker types |
| Corner/vertex detection | ✓ | ✓ | JM auto-detects corners; EPG detects via graph triangles |
| Network/graph analysis | ✓ | ✗ | EndoPiGraph: full NetworkX graph with clustering, degree, etc. |
| Batch processing | ✓ | ✗ | EndoPiGraph: fully automated batches; JM: manual per-image |
| Per-junction output | ✓ | ✓ | Both provide per-junction data |
| Skeleton-based morphometry | ✓ | ✓ | Both use skeletonization; EPG adds skeleton_len, thickness_proxy |
| Multiple cell types | ✓ | ✓ | JM validated on epithelial, cardiomyocytes, endothelial |
| Python API | ✓ | ✗ | EndoPiGraph: Python package; JM: Java desktop application |
| Polarity/flow analysis | ✓ | ✗ | EndoPiGraph: Golgi-nucleus polarity vectors |

## EndoPiGraph Key Advantages

- Fully automated batch processing (no manual intervention)
- Deep learning segmentation (Cellpose)
- Graph-based network analysis (clustering, degree, triangles)
- Automatic junction morphology classification
- Python API for custom pipelines
- Polarity/flow analysis capability
- Multi-junction type Pi-graphs

## Feature Mapping

| Feature | EndoPiGraph | Junction Mapper |
|---------|-------------|------------------|
| Contact length | `contact_px` | Junction length |
| Junction occupancy | `occupancy` | Fraction occupied |
| Mean intensity | `mean_intensity` | Mean junction intensity |
| Max intensity | `max_intensity` | Peak intensity |
| Intensity variation | `std_intensity` | Intensity SD |
| Fragment count | `cluster_count` | Number of clusters |
| Fragment density | `cluster_density` | *Not available* |
| Skeleton length | `skeleton_len` | Skeleton length |
| Thickness proxy | `thickness_proxy` | Area/skeleton ratio |
| Junction type | `aj_morph_label` | *Not available* |
| Cell degree | `degree` | *Not available* |
| Clustering coefficient | `local_clustering` | *Not available* |

## Blur Stability Comparison

Both Junction Mapper and EndoPiGraph use connected component counting for cluster metrics.
This makes both equally sensitive to image blur:

| Metric | EndoPiGraph | Junction Mapper | Blur Stability |
|--------|-------------|-----------------|----------------|
| Occupancy/Fraction occupied | `occupancy` | Fraction occupied | **Stable** (d=0.21) |
| Mean intensity | `mean` | Mean intensity | **Stable** (d≈0) |
| Cluster count | `cluster_count_cc` | Number of clusters | Unstable (d=0.52-0.68) |
| Cluster size | `cluster_area_mean` | Mean cluster size | Unstable (d=2.0-4.7) |
| Skeleton length | `skeleton_len` | Skeleton length | Marginal (d=0.30-0.56) |

**EndoPiGraph advantage**: Provides additional skeleton-based metrics (`skeleton_endpoints`,
`complexity_score`) as alternatives when blur is a concern.

See: `runs/blur_stability_comparison/BLUR_STABILITY_REPORT.md` for full analysis.

## References

- Junction Mapper: Tomlinson et al., eLife 2019 (https://elifesciences.org/articles/45413)
- GitHub: https://github.com/ImperialCollegeLondon/Junction_Mapper
