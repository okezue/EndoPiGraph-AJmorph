# Changelog

## v1.0.0 — 2026-04-27

First Zenodo-archivable release accompanying the PiMorph manuscript submission.

### Pipeline
- Three input modes: mask-driven, segmentation (Cellpose), and hybrid (nuclei seeds + watershed).
- Conservative contact inference with contact-length filter; precision-first design.
- Per-interface VE-cadherin feature extraction (occupancy, intensity stats, fragmentation, skeleton topology, thickness proxy, complexity score).
- Optional Golgi-nucleus polarity quantification with per-image R and signed V.
- AJMORPH morphology-state labels via unsupervised GMM clustering with BIC-selected k; bootstrap-resampled Adjusted Rand Index for stability.
- Blur-robust mode using stable feature subset (occupancy, skeleton length, intensity summaries).

### Validation suite
- Adjacency benchmarks on three modalities: LIVECell (phase-contrast, F1 = 78.4%), NuInsSeg (H&E, F1 = 82.2%), cornea cells (specular, F1 = 93.1%).
- Junction Mapper capability comparison.
- Robustness suite: 28/30 metric-perturbation pairs stable; intensity scaling perfectly absorbed; blur destabilizes only fragmentation features.
- Shear-stress demonstration on S-BIAD1540 EGM2 (95 fields of view, 22,175 interfaces).
- Manual-annotation infrastructure for AJMORPH validation (`scripts/sample_ajmorph_annotation.py`, `scripts/analyze_ajmorph_annotations.py`).
- Mixed-effects sensitivity analysis for batch and density confounds using biological replicate (3 reps, fully crossed with condition) as random intercept (`scripts/mixed_effects_sensitivity.py`); all three confirmed shear effects survive at p<0.002.

### Reproducibility
- 107 unit tests; CI on GitHub Actions.
- All randomness seeded; run metadata exported.
- Validation data bundle (Desktop, ~19 MB) contains all CSVs, JSONs, GraphML, reports.

### Known limitations
- Pipeline assumes confluent monolayers; non-confluent cultures (e.g., VE-strat) produce zero edges by design.
- AJMORPH labels are feature-derived states, not expert-validated biological classes; expert annotation infrastructure shipped, annotations themselves not yet collected.
- Polarity comparison limited to 5 fields per condition; descriptive only.
- High-shear (18-20 dyn cm⁻²) regime appears distinct from 6 dyn cm⁻² rather than a monotonic continuation.

### Citation
See `CITATION.cff`. Software DOI: [10.5281/zenodo.19831621](https://doi.org/10.5281/zenodo.19831621).
