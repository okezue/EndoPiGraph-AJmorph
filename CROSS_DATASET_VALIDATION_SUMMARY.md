# Cross-Dataset Validation Summary for EndoPiGraph-AJmorph

## Overview

This document summarizes the comprehensive validation framework created to test EndoPiGraph-AJmorph across multiple independent datasets, with cross-dataset transfer learning tests, stability analysis, and comparison against Junction Mapper.

## Datasets Integrated

### 1. S-BIAD1540 (Reference - Existing)
- **Source**: BioImage Archive
- **Cell Type**: HUVEC (EGM2-treated)
- **Conditions**: Static, 6dyne, 18-20dyne shear stress
- **Images**: 102 total (subset tested)
- **Channels**: VE-cadherin, DAPI, GM130

### 2. Lymphatic EC Junction Morphology (NEW - Zenodo 13880404)
- **Source**: Zenodo ([doi:10.5281/zenodo.13880404](https://doi.org/10.5281/zenodo.13880404))
- **Cell Type**: Dermal lymphatic capillary endothelial cells
- **Conditions**: 3w, 5w, 25w developmental timepoints
- **Markers**: VE-cadherin + LYVE1
- **Junction Types**: Button, curvilinear, double, zipper
- **Image Size**: 2048x2048 pixels
- **Notes**: Includes ROI annotations for junction classification

### 3. HUVEC High-Content Screen (NEW)
- **Source**: High-content imaging siRNA screen
- **Cell Type**: HUVEC
- **Replicates**: A1, A2 (condition A); G1, G2 (condition G)
- **Images**: ~1920 TIFFs total (480 per replicate plate)
- **Channels**: 4 channels including VE-cadherin
- **Image Size**: 993x1342 pixels

### 4. S-BIAD463 Vascular Remodeling (NEW)
- **Source**: BioImage Archive
- **Cell Type**: High endothelial venules (HEV) in lymph nodes
- **Format**: OME-Zarr
- **Images**: 13 samples
- **Channels**: 3 fluorescence channels
- **Context**: Tumor-draining lymph node vascular remodeling

## Validation Framework Created

### Cross-Dataset Transfer Tests
Script: `scripts/cross_dataset_validation.py`

Tests include:
1. **Train-Test Transfer**: Train classifier on dataset A, evaluate on dataset B
2. **Leave-One-Dataset-Out (LODO)**: With ≥3 datasets, train on all-but-one and test on held-out dataset
3. **Within-Dataset Cross-Validation**: Baseline performance within each dataset

### Feature Columns for Transfer Learning
```python
feature_cols = [
    "occupancy",        # Fraction of interface with marker
    "mean_intensity",   # Average intensity at junction
    "max_intensity",    # Peak intensity
    "std_intensity",    # Intensity variation
    "cluster_count",    # Number of junction fragments
    "cluster_density",  # Fragments per unit length
    "skeleton_len",     # Skeletonized junction length
    "thickness_proxy",  # Area / skeleton length
]
```

### Stability Analysis
The framework tests robustness to:

1. **Parameter Variations**:
   - Cell diameter: 20, 30, 40 pixels
   - Threshold method: Otsu, percentile:90, percentile:95
   - Dilation: 1, 2, 3 pixels

2. **Intensity Perturbations**:
   - Scaling: 0.8x, 1.2x
   - Gaussian noise: low (5%), high (10%)
   - Gaussian blur: 1px, 2px sigma

3. **Segmentation Methods**:
   - Cellpose (deep learning)
   - Watershed (classical)

### Effect Size Reporting
- **Cohen's d**: Standardized mean difference
- **Rank-biserial r**: Non-parametric effect size from Mann-Whitney U
- **Per-image replicate testing**: Avoids pseudo-replication

## Junction Mapper Comparison

Full comparison report: `runs/junction_mapper_comparison/JUNCTION_MAPPER_COMPARISON.md`

### Summary
| Metric | EndoPiGraph | Junction Mapper |
|--------|-------------|-----------------|
| Capabilities | 15/15 | 9/15 |
| Fully Automated | Yes | No (semi-automated) |
| Unique Features | 4 | 0 |
| Programming API | Python | None (Java GUI) |

### Key EndoPiGraph Advantages over Junction Mapper
1. **Fully automated batch processing** - No manual intervention required
2. **Deep learning segmentation** - Cellpose vs manual outline correction
3. **Graph-based network analysis** - Clustering, degree, triangles
4. **Automatic junction morphology classification** - Heuristic + ML
5. **Python API** - Programmatic access for custom pipelines
6. **Polarity/flow analysis** - Golgi-nucleus vectors

### Feature Mapping
| Feature | EndoPiGraph | Junction Mapper |
|---------|-------------|-----------------|
| Contact length | `contact_px` | Junction length |
| Junction occupancy | `occupancy` | Fraction occupied |
| Fragment count | `cluster_count` | Number of clusters |
| Fragment density | `cluster_density` | *Not available* |
| Junction type | `aj_morph_label` | *Not available* |
| Cell degree | `degree` | *Not available* |

## Processing Results (Partial)

Due to the large image sizes and thousands of cells/edges per image, full validation runs require significant compute time:

| Dataset | Images | Total Edges | Processing Status |
|---------|--------|-------------|-------------------|
| S-BIAD1540 | 3 | 1,742 | Complete |
| Lymphatic EC | 3 | 36,540 | Complete |
| HUVEC Screen | 3 | 4,376 | Complete |
| S-BIAD463 | 2 | In progress | Partial |

### Performance Notes
- Lymphatic EC images: 2048x2048 with ~4000 cells each → ~10,000+ edges per image
- Processing time dominated by per-edge feature extraction
- Watershed segmentation is faster than Cellpose for validation purposes

## Files Created

### Scripts
- `scripts/cross_dataset_validation.py` - Main validation framework
- `scripts/junction_mapper_comparison.py` - Junction Mapper comparison

### Data Directories
- `data/lymphatic_ec/` - Lymphatic EC dataset (3w, 5w, 25w timepoints)
- `data/huvec_screen/` - HUVEC A1, A2, G1, G2 replicates
- `data/sbiad463/` - OME-Zarr vascular remodeling images

### Output
- `runs/junction_mapper_comparison/junction_mapper_comparison.json`
- `runs/junction_mapper_comparison/JUNCTION_MAPPER_COMPARISON.md`
- `runs/cross_dataset_validation/` (output directory for validation results)

## Code Fixes Applied

1. **Cellpose API Update**: Updated `segmentation.py` to use `CellposeModel` (v4 API)
2. **Interface Mask Fix**: Fixed `interface_mask_from_coords` return value handling

## Usage

### Run Full Validation
```bash
python scripts/cross_dataset_validation.py --max-images 10
```

### With Cellpose (slower, more accurate)
```bash
python scripts/cross_dataset_validation.py --max-images 10 --use-cellpose
```

### Generate Junction Mapper Comparison
```bash
python scripts/junction_mapper_comparison.py
```

## References

### Datasets
- S-BIAD1540: BioImage Archive
- Lymphatic EC: Zenodo 13880404 - "Dynamic cytoskeletal regulation of cell shape supports resilience of lymphatic endothelium" (Nature)
- S-BIAD463: "Automated detection of vascular remodeling in tumor-draining lymph nodes by the deep learning tool HEV-finder"

### Junction Mapper
- Tomlinson et al., "Junction Mapper is a novel computer vision tool to decipher cell-cell contact phenotypes" eLife 2019
- https://elifesciences.org/articles/45413
- https://github.com/ImperialCollegeLondon/Junction_Mapper

## Next Steps

1. **Complete Full Validation**: Run with more images once compute resources allow
2. **Add Manual Annotations**: For true AJ morphology label validation (current labels are heuristic-derived)
3. **Nuclei-Based Segmentation**: Compare VE-cadherin-based vs nuclei-based cell boundaries
4. **Cross-Lab Validation**: Test on images from different microscopes/labs
