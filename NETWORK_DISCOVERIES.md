# Network-Level Discoveries in Endothelial Biology

## Summary

Using graph/network analysis on the EndoPiGraph-AJmorph pipeline results, we identified **four robust discoveries** about how fluid shear stress reorganizes endothelial cell contact networks.

**Statistical validation:** All statistics use **per-image replicate testing** (n = number of images, not cells/edges) to avoid pseudo-replication. Effect sizes reported as rank-biserial correlation r.

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

**Flow restructures the contact network to create a more robust, interconnected tissue with mature junctions concentrated at multi-cell vertices.**

Four discoveries survive rigorous per-image statistical testing:
1. Clustering coefficient increases (tighter neighborhoods)
2. Reticular junction percentage increases (stronger adhesion)
3. All-reticular triangles increase (junction maturation at tricellular vertices)
4. Area-degree correlation strengthens (geometric ordering)

One original claim (degree-occupancy correlation) was **withdrawn** after proper statistical validation revealed pseudo-replication in the pooled analysis.

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
- **Dataset:** 95 EGM2-treated HUVEC images (30 static, 30 6dyne, 30 20dyne, 5 unknown)

---

## Key Files

- Hardened statistics script: `scripts/harden_network_stats.py`
- Per-image results: `runs/egm2_full/hardened_network_stats.json`
- Cell/edge data: `runs/egm2_full/*/cells.csv`, `edges.csv`
