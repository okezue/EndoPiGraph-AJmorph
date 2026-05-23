# PiMorph publication figures

5 figures (PDF + PNG, each PNG at 400 DPI). All scripts in `scripts/publication_figures.py`.

| File | Content |
|---|---|
| `fig1_corpus.pdf/.png` | 10-dataset corpus. (top) per-dataset lineage composition stacked bars with platform/tissue labels above each. (bottom-left) cell & edge counts per dataset on log scale. (bottom-right) platform & tissue summary. (footer) cell-lineage legend. |
| `fig2_cross_platform.pdf/.png` | (A) Jensen-Shannon distance heatmap of edge-type composition across 10 datasets, clustered, with platform + tissue annotation strips. (B) JS distance distributions by pair class — same-tissue cross-platform vs different-tissue same/cross platform — with Wilcoxon p. (C) PERMANOVA R² bars partitioning JS variance by tissue / platform. |
| `fig3_boundary_validation.pdf/.png` | The boundary-vs-body biological validation. (A) Xenium colon — gene-rank ladder with epithelial / membrane / AJ genes color-coded by category. EPCAM 1, CD24 2, KRT8 3, PIGR 4, CDH1 14, CTNNB1 34 of 362. (B) Xenium mouse brain — Aqp4 mid-pack (30/247), Cldn5 low (138/247). (C) MIBI glioma protein-level boundary/body intensity ratio; CD45, CD40, CD133 atop; Chym_Tryp (granule) bottom. (D) Side-by-side percentile bars of biology-of-interest genes across both tissues. |
| `fig4_null_deflation.pdf/.png` | The withdrawn-claim panel. (left) Top whole-genome GO term per panel with significance, "(0 enriched)" annotation above each showing the panel-bg null result. (right) Schematic of the deflation logic. |
| `fig5_conserved_biology.pdf/.png` | (A) Universal edge-type classes across the 10-dataset corpus, colored by # datasets ≥ 0.5%. (B) Brain cross-platform gene rank scatter (Xenium MB vs Stereo-seq MB) with top-15 conserved genes labeled — Spearman ρ. (C) Same for human colon (Xenium ↔ CosMx). |

## Headline (what the figure set demonstrates)

1. **Tissue dominates platform** (fig 2): same-tissue cross-platform JS distance is the tightest pair class (n=5, median 0.62, Wilcoxon p = 0.009); PERMANOVA tissue R² = 0.71, p = 0.009.
2. **Direct boundary-vs-body validation in Xenium colon** (fig 3A): epithelial surface / AJ genes occupy the top of the full-panel ranking (EPCAM 1/362, CD24 2, KRT8 3, PIGR 4, MUC12 11, CDH1 14, REG4 24, CTNNB1 34).
3. **Brain is ambiguous at the RNA level** (fig 3B): Aqp4 modest (rank 30/247); Cldn5 low (138/247). Protein-level AQP4 validation is the proper next step.
4. **Protein-level signal corroborates direction** (fig 3C): in MIBI glioma, canonical membrane proteins (CD45, CD40, CD133, CD47, CD86) sit at the top of the boundary/body ratio; granule contents (Chym_Tryp) at the bottom.
5. **GO enrichment claim withdrawn** (fig 4): 730 whole-genome enriched terms across 6 panels collapse to 0 under the proper shared-panel background — the original "cell junction p = 5 × 10⁻¹⁴" was a circular consequence of panel design.
6. **Cross-platform gene rank conservation** (fig 5): brain Spearman ρ = 0.61 across 246 shared genes; colon Spearman ρ = 0.10 across 98 shared genes (concordance varies by panel overlap and platform chemistry).

PDF for vector use, PNG at 400 DPI for raster screens.
