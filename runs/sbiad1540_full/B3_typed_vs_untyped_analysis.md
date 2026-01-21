# B3: Typed vs Untyped Features Experiment - Initial Results

## Hypothesis
Typed edge features (AJ morphology, junction quantification) should improve condition classification compared to untyped graph statistics. This would demonstrate the value of the typed π-graph representation.

## Dataset
- **Images**: 15 (5 per condition)
- **Conditions**: static, 6dyne, high_shear
- **CV Method**: Leave-One-Out (appropriate for small N)

## Feature Sets

### Untyped (7 features)
Basic graph statistics that don't require AJ typing:
- n_cells, n_edges, mean_degree, edge_density
- mean_cell_area, std_cell_area, cv_cell_area

### Typed (18 features)
AJ-specific features from edge quantification:
- AJ occupancy: mean, std, median
- AJ clusters: mean, std, max
- AJ intensity: mean, std
- Linearity index: mean, std
- Thickness proxy: mean, std
- Skeleton length: mean, std
- Morphology proportions: reticular, straight, fingers, other

### Combined (25 features)
All features from both sets.

## Results

| Feature Set | N Features | RF Accuracy | RF F1 (macro) | LR Accuracy | LR F1 (macro) |
|-------------|------------|-------------|---------------|-------------|---------------|
| **Untyped** | 7          | **0.600**   | **0.600**     | **0.600**   | **0.603**     |
| Typed       | 18         | 0.467       | 0.470         | 0.333       | 0.329         |
| Combined    | 25         | 0.533       | 0.539         | 0.467       | 0.446         |

## Interpretation

**Unexpected result**: Untyped features outperformed typed features.

### Possible explanations:

1. **Small sample size (N=15)**
   - Leave-one-out CV on 15 samples provides unstable estimates
   - High variance in performance metrics
   - May flip with more data

2. **Feature set size disparity**
   - Typed has 18 features vs 7 untyped
   - With only 15 samples, 18 features risks overfitting
   - Dimensionality curse working against typed features

3. **Heuristic label quality**
   - AJ morph labels are derived from thresholds on the same features
   - Not externally validated biological categories
   - May not capture true biological variation

4. **Graph structure is informative**
   - n_cells, edge_density may genuinely correlate with shear conditions
   - Shear stress affects cell density and packing
   - Basic topology captures this effectively

## Next Steps

1. **Scale to larger dataset (B2)**
   - Currently downloading 102 EGM2 images
   - ~34 static, 30 6dyne, 38 high_shear
   - Will provide more statistical power

2. **Re-run experiment on larger dataset**
   - More stable CV estimates
   - Can use stratified k-fold instead of LOO
   - Better assessment of feature importance

3. **Feature selection/reduction for typed**
   - PCA or LASSO to reduce dimensionality
   - May improve typed performance

## Conclusion (Preliminary)

On the initial 15-image dataset, basic graph structure predicts shear condition better than AJ-typed features. However, this result should be validated on a larger dataset before concluding that typed features don't add value. The small sample size likely contributes to the counterintuitive finding.
