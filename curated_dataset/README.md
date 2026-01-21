# EndoPiGraph-AJmorph Curated Dataset

Validated pipeline outputs from S-BIAD1540 endothelial cell dataset.

## Dataset Summary

| Metric | Value |
|--------|-------|
| Images | 15 (stratified by shear stress) |
| Conditions | 5 static, 5 6dyne/cm², 5 18-20dyne/cm² |
| Total cells | 1,602 |
| Cell-cell edges | 3,520 |
| Average degree | 4.4 |
| QC passed | 11/15 images |

## Directory Structure

```
curated_dataset/
├── data/
│   └── manifest_subset.csv      # Image manifest with metadata
├── outputs/
│   ├── all_cells.csv            # Cell features (1,602 rows)
│   ├── all_edges.csv            # Edge features + AJ morph (3,520 rows)
│   ├── all_polarity.csv         # Cell polarity vectors
│   ├── polarity_per_image.csv   # Per-image polarity statistics
│   ├── qc_metrics.csv           # Per-image QC metrics
│   ├── qc_excluded.txt          # Images failing QC
│   └── *.png                    # Visualizations
├── models/
│   └── ajmorph_classifier_v2.joblib  # Classifier with metadata
├── scripts/
│   ├── qc_criteria_v2.py        # QC validation script
│   ├── analyze_polarity_proper.py  # Per-image polarity analysis
│   └── train_classifier_proper.py  # Proper classifier training
└── README.md
```

## CRITICAL WARNINGS

### AJ Morphology Labels Are PROVISIONAL

**The `aj_morph` labels in `all_edges.csv` are heuristic-derived, NOT manually annotated.**

They were generated from threshold rules on the same features used for analysis:
- `cluster_count >= 6 AND thickness_proxy < 6.0` → reticular
- `tort <= 1.25 AND cluster_count <= 3 AND thickness_proxy < 10.0` → straight
- etc.

**Implications:**
- The ~99% classifier accuracy reflects learning these rules, NOT validated biology
- Treat label distributions as **exploratory**, not definitive
- Manual annotation is required for biological validation

Per Polarity-JaM paper: *"advanced classifier translating junction features into the 5 manual AJ classes was not ready and requires manual training data."*

### Polarity Analysis Notes

- Polarity is computed per-image (images are the unit of analysis)
- The signed polarity index V requires correct flow direction (cue angle)
- Default cue_angle=0° may not match actual flow direction in all images
- Verify `polarity_per_image.csv` for per-image R and V values

## Data Files

### all_cells.csv
Per-cell features including:
- `cell_id`: Unique cell identifier
- `image_id`: Source image
- `area`, `perimeter`, `eccentricity`: Shape features
- `centroid_x`, `centroid_y`: Cell center

### all_edges.csv
Per-interface (cell-cell contact) features including:
- `cell_i`, `cell_j`: Adjacent cell IDs
- `interface_length`: Contact boundary length (pixels)
- `aj_mean_intensity`, `aj_occupancy`: VE-cadherin signal features
- `aj_cluster_count`, `aj_skeleton_len`: Junction morphology metrics
- `aj_morph`: **PROVISIONAL** classified junction type

### Continuous Features (recommended for quantitative analysis)
Per Polarity-JaM, these continuous features are the primary outputs:
- **Junction occupancy** (`aj_occupancy`)
- **Cluster density** (`aj_cluster_count`)
- **Intensity per interface** (`aj_intensity_per_interface`)

### polarity_per_image.csv
Per-image polarity statistics:
- `R`: Mean resultant length (alignment strength, 0=random, 1=perfect)
- `V`: Signed polarity index (positive=with cue, negative=against)
- `mean_angle_deg`: Circular mean angle

## Polarity Summary by Condition

| Condition | R_mean | V_mean | Interpretation |
|-----------|--------|--------|----------------|
| Static | 0.126 | 0.082 | Weakly random |
| 6 dyne | 0.205 | 0.168 | Moderate alignment |
| High shear | 0.231 | -0.050 | Aligned, direction varies |

## QC Results

4/15 images excluded due to segmentation quality issues:
- High fraction of "giant cells" (likely under-segmented)
- See `qc_excluded.txt` for list

## Classifier Model

Random Forest classifier for reference (NOT validated):
- **File:** `models/ajmorph_classifier_v2.joblib`
- **Method:** 5-fold GroupKFold by image_id
- **Warning:** High accuracy reflects heuristic rule learning

Usage:
```python
import joblib
model_data = joblib.load('models/ajmorph_classifier_v2.joblib')
clf = model_data['classifier']
features = model_data['feature_cols']
print(model_data['metadata']['WARNING'])  # READ THIS
```

## Source Data & Citation

### Dataset
**BioImage Archive S-BIAD1540**
- URL: https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1540
- Cell type: HUVECs (Human Umbilical Vein Endothelial Cells)
- Channels: DAPI (nuclei), GM130 (Golgi), VE-cadherin (junctions)

### Citation (Required)
If using this dataset, please cite:

Yin, Z., Weidenhammer, A.A., Bhupathiraju, S.G. et al. **Polarity-JaM: an image
analysis toolbox for cell polarity, junction and morphology quantification.**
*Nature Communications* 16, 615 (2025). https://doi.org/10.1038/s41467-025-56643-x

### License
Data from BioImage Archive. Per EMBL-EBI Terms of Use, data are freely available
for reuse with appropriate citation. See: https://www.ebi.ac.uk/bioimage-archive/help-faq/

## Methodology References

- Polarity index (R) and signed polarity index (V): Polarity-JaM paper
- AJ morphology classification criteria: Polarity-JaM paper, Section "Junction morphology"
- QC thresholds: Based on expectations for confluent endothelial monolayers

## Version Info

- Pipeline: EndoPiGraph-AJmorph v1
- Analysis date: 2026-01-19
- scikit-learn version: See model metadata
