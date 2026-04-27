# Network-Level Findings in Endothelial Biology

## Summary

Using graph/network analysis on the EndoPiGraph-AJmorph pipeline results, we identified **three robust findings** about how 6 dyn cm⁻² shear stress reorganizes endothelial cell contact networks relative to static. Two originally claimed effects (raw clustering coefficient, degree-occupancy correlation) were **withdrawn** after hardened per-image statistical testing.

**Statistical validation:** All statistics use **per-image replicate testing** (n = number of images, not cells/edges) to avoid pseudo-replication. Effect sizes reported as rank-biserial correlation r.

**Regime caveat:** These findings compare static vs 6 dyn cm⁻². The 18-20 dyn cm⁻² regime is *not* a monotonic continuation of the 6 dyn cm⁻² phenotype — see "High-shear regime" below.

---

## ~~Discovery 1: Clustering Coefficient Increases Under Flow~~ (CONFOUNDED)

**Original finding:** Clustering coefficient appears to increase under flow.

| Condition | Median Clustering | Mean | Std | n |
|-----------|------------------|------|-----|---|
| Static    | 0.376            | 0.381| 0.073| 30 |
| 6 dyne    | 0.469            | 0.442| 0.055| 30 |
| high_shear| 0.331            | 0.309| 0.122| 30 |

**CRITICAL: Confounded by Graph Density**

Regression analysis controlling for mean degree:
```
clustering ~ condition + mean_degree + n_cells
R² = 0.937 (mean degree explains 94% of variance!)

is_6dyne coef: -0.039, p = 1.34e-04 (NEGATIVE after control)
is_high_shear coef: -0.037, p = 3.53e-05 (NEGATIVE after control)
mean_degree coef: 0.113, p = 7.69e-46
```

Normalized clustering (C/C_random): static vs 6dyne **p = 0.68** (NOT significant)

**Conclusion:** The raw clustering increase is driven by changes in graph density (more edges = higher clustering mechanically). After controlling for density, the effect disappears. This finding is **withdrawn**.

---

## Discovery 2: Reticular Junctions Increase Under Flow

**Finding:** The proportion of mature, reticular-type adherens junctions increases under flow.

| Condition | Median % Reticular | Mean | Std | n |
|-----------|-------------------|------|-----|---|
| Static    | 51.5%             | 51.0%| 5.2%| 30 |
| 6 dyne    | 61.1%             | 62.1%| 8.6%| 30 |

**Per-image replicate statistics:**
- Median difference: +9.5% [95% CI: 6.7%, 16.6%]
- Mann-Whitney U = 133.5, **p = 2.98e-06**
- Effect size r = 0.703 (large)

**Biological interpretation:** Flow promotes junction maturation. Reticular junctions indicate stronger cell-cell adhesion and barrier function.

---

## ~~Discovery 3: High-Degree Cells Have Stronger Junctions~~ (NOT CONFIRMED)

**Original claim:** Cells with more neighbors have higher AJ occupancy.

**Per-image replicate testing:**
- Median within-image Spearman r = 0.001
- Wilcoxon test: **p = 0.808** (not significant)

**Conclusion:** The original pooled analysis suffered from pseudo-replication. When properly tested at the image level, the degree-occupancy correlation does NOT hold. This finding is **withdrawn**.

---

## Discovery 3 (renumbered): Tricellular Vertices Are Junction Hotspots

**Finding:** "All-reticular" triangles (where all 3 edges are reticular-type) increase dramatically under flow.

| Condition | Median % All-Reticular | Mean | Std | n |
|-----------|----------------------|------|-----|---|
| Static    | 15.9%                | 15.9%| 4.4%| 30 |
| 6 dyne    | 25.4%                | 25.3%| 10.2%| 30 |

**Per-image replicate statistics:**
- Median difference: +9.6% [95% CI: 4.2%, 13.8%]
- Mann-Whitney U = 193.5, **p = 1.54e-04**
- Effect size r = 0.570 (large)

**Biological interpretation:** Tricellular junctions (where 3 cells meet) are known hotspots for permeability. Flow drives junction maturation specifically at these multi-cell vertices.

---

## Discovery 4 (renumbered): Area-Degree Correlation Strengthens Under Flow

**Finding:** Cell area positively correlates with degree (number of neighbors), and this correlation strengthens under flow.

| Condition | Median r | Mean r | Range | n |
|-----------|----------|--------|-------|---|
| Static    | 0.455    | 0.445  | [0.17, 0.65] | 30 |
| 6 dyne    | 0.642    | 0.624  | [0.31, 0.76] | 30 |

**Per-image replicate statistics:**
- Both conditions: correlations differ from 0, **p = 1.86e-09**
- Condition comparison: Median diff = +0.187 [95% CI: 0.144, 0.246]
- Mann-Whitney U = 101.0, **p = 2.57e-07**
- Effect size r = 0.78 (large)

**Biological interpretation:** Larger cells have more neighbors, and this relationship tightens under flow. The tissue becomes more geometrically ordered.

---

## Overall Conclusion

**At 6 dyn cm⁻² relative to static, the contact network shifts toward more reticular junctions concentrated at multi-cell vertices, with tighter geometric ordering of cell area and degree.**

Three findings survive hardened per-image statistical testing:
1. Reticular junction percentage increases
2. All-reticular triangles increase (concentration at tricellular vertices)
3. Area-degree correlation strengthens

Two original claims were **withdrawn** after proper statistical validation:
- Raw clustering coefficient: 94% of variance is explained by mean degree; coefficient flips sign after density control.
- Degree-occupancy correlation: pooled analysis suffered from pseudo-replication; per-image Wilcoxon p = 0.81.

## High-shear regime is distinct, not a continuation

Per-image medians on the 18-20 dyn cm⁻² subset (n = 30) are closer to static than to 6 dyn cm⁻²:

| Metric | Static | 6 dyn cm⁻² | 18-20 dyn cm⁻² |
|---|---:|---:|---:|
| Reticular fraction (median) | 50.7% | 61.1% | 52.7% |
| All-reticular triangles (median) | 15.9% | 25.4% | 15.5% |

Subject to batch and density caveats, this is consistent with **intermediate shear producing the strongest reticular-network organization** while higher shear shifts toward a distinct regime. This is reported as exploratory in the manuscript pending balanced-batch validation.

---

## Statistical Methods

- **Sampling unit:** Image (not individual cells/edges)
- **Between-condition test:** Mann-Whitney U (non-parametric)
- **Within-condition test:** Wilcoxon signed-rank (for correlations differing from 0)
- **Effect size:** Rank-biserial correlation r
  - |r| < 0.1: negligible
  - |r| 0.1-0.3: small
  - |r| 0.3-0.5: medium
  - |r| > 0.5: large
- **Confidence intervals:** Bootstrap (1000 resamples)
- **Dataset:** 95 EGM2-treated HUVEC images (32 static, 30 at 6 dyn cm⁻², 33 at 18-20 dyn cm⁻²; replicate testing on the balanced 30+30 static-vs-6 dyn cm⁻² subset)

---

## Key Files

- Hardened statistics script: `scripts/harden_network_stats.py`
- Per-image results: `runs/egm2_full/hardened_network_stats.json`
- Cell/edge data: `runs/egm2_full/*/cells.csv`, `edges.csv`
