# Validation Summary

## A. High-Shear Naming Mismatch - FIXED

**Issue:** Only 62/102 images matched manifest; high_shear naming differed.

**Fix:** Updated condition parsing in `scripts/harden_network_stats.py` to recognize `18-20dyn` pattern as `high_shear`.

**Result:** Now correctly identifies:
- 30 static images
- 30 6dyne images
- 30 high_shear images
- 5 unknown

---

## B. Polarity Direction Calibration - FIXED

**Issue:** Signed polarity V flips sign for high_shear (0.17 → -0.05).

**Resolution:**
- Added `|V|` (absolute polarity) as directionless metric
- Recommend using **R (alignment strength)** as primary metric
- Treat V sign as **exploratory only** unless flow direction is confirmed from experimental metadata

**Files:**
- `scripts/fix_polarity_reporting.py`
- `runs/sbiad1540_full/polarity_summary_fixed.csv`

---

## C. Clustering Confounded by Graph Density - CRITICAL FINDING

**Issue:** Clustering coefficient might be driven by graph density, not true topology changes.

**Analysis:** Ran regression: `clustering ~ condition + mean_degree + n_cells`

**Results:**
```
is_6dyne coef: -0.0386, p = 1.34e-04
is_high_shear coef: -0.0367, p = 3.53e-05
mean_degree coef: 0.1127, p = 7.69e-46
R-squared: 0.937
```

**Conclusion:**
- **Mean degree explains 94% of variance in clustering**
- After controlling for degree, clustering DECREASES under flow (negative coefficients)
- Normalized clustering (C/C_random): static vs 6dyne p = 0.68 (NOT significant)

**Discovery 1 (raw clustering increases) is CONFOUNDED.** After density control, the effect disappears or reverses.

---

## D. Generalization to Independent Dataset - LIMITATION IDENTIFIED

**Dataset tested:** VE-Strat (Control vs Histamine)

**Result:** Pipeline successfully processed 10 images but found **0 edges**.

**Reason:** VE-Strat images show **non-confluent cells** with gaps between them. The pipeline requires confluent monolayers with clear cell-cell junctions.

**Limitation documented:**
- Pipeline assumes confluent endothelial monolayers
- Non-confluent cultures (like VE-Strat) produce no edges
- Network analysis not applicable to non-confluent samples

---

## E. Nuclei-Based Segmentation Sanity Check - DOCUMENTED AS FUTURE WORK

**Concern:** VE-cadherin segmentation might bias apparent cell area under conditions that change VE-cadherin intensity/contrast.

**Status:** The S-BIAD1540/EGM2 images have 3 channels, but channel assignments are not clearly documented. Nuclei-based watershed comparison requires:
1. Confirmed nuclei channel identification
2. Watershed implementation for comparison
3. Side-by-side area/perimeter comparison

**Documented as future validation requirement.**

---

## Summary of Revised Findings

### Network Discoveries (Revised)

| Discovery | Original Claim | After Validation | Status |
|-----------|---------------|------------------|--------|
| 1. Clustering +24% | p < 0.001 | **Confounded by density** (R²=0.94) | ⚠️ REVISED |
| 2. Reticular +26% | p < 1e-63 | **p = 3.0e-06** (per-image) | ✅ Confirmed |
| 3. Degree-occupancy | p < 1e-17 | **p = 0.81** (per-image) | ❌ Withdrawn |
| 4. Triangles +91% | p < 1e-45 | **p = 1.5e-04** (per-image) | ✅ Confirmed |
| 5. Area-degree | r increases | **p = 2.6e-07** (per-image) | ✅ Confirmed |

### Polarity

| Metric | Recommendation |
|--------|---------------|
| R (alignment) | ✅ Primary metric (directionless) |
| V (signed) | ⚠️ Exploratory only |
| |V| (absolute) | ✅ Alternative directionless |

### Generalization

| Dataset | Confluent | Edges Found | Network Analysis |
|---------|-----------|-------------|------------------|
| S-BIAD1540 | Yes | Yes | ✅ Applicable |
| EGM2 | Yes | Yes | ✅ Applicable |
| VE-Strat | **No** | **0** | ❌ Not applicable |

---

## Files Updated

- `scripts/harden_network_stats.py` - Fixed high_shear parsing, added density control
- `scripts/fix_polarity_reporting.py` - Added |V| metric
- `runs/egm2_full/hardened_network_stats.json` - Updated results
- `runs/sbiad1540_full/polarity_summary_fixed.csv` - With |V|
- `NETWORK_DISCOVERIES.md` - Revised claims
- `CLOSEOUT_CHECKLIST.md` - Updated limitations
