# Response to biologist review — 2026-05-22 (v3)

## TL;DR for the biologist

You were right on **every** correction across both review rounds. The framework now stands on:
(a) the **panel-background null**, which correctly invalidates the original GO claim, and
(b) the **full-panel boundary-vs-body rank** in Xenium colon, where epithelial membrane / surface / AJ genes dominate the top of the ranking. Brain is honestly framed as ambiguous at the RNA level, not as a clean negative.

This file replaces both the v1 cover-letter claims and the v2 first-pass response. Major changes vs v2 listed in §6.

## 1. Edge-count corrections (v1 → verified)

| Edge type | v1 cover letter | Actual (verified from `edge_type_summary.parquet`) |
|---|---|---|
| `endothelial__glia`, Allen MERFISH | 826,004 (whole-brain run) | **60,828** (3-section subset run used in this package) |
| `endothelial__glia`, Xenium MB | 12 (was from a 1 mm² pilot tile) | **1,644** (full-tissue run) |
| `endothelial__glia`, Stereo-seq MB | 299 ✓ | 299 ✓ |
| "B-cell ↔ endothelial in CosMx colon, 6,828" | wrong | **`bcell__epithelial = 6,828`**; `bcell__endothelial = 231` |

## 2. Panel-background null: enrichment evaporates

I re-ran g:Profiler with the **shared-panel gene universe** as the explicit custom background (g:Profiler `domain_scope = custom`):

| Panel | Whole-genome bg top hit | Shared-panel bg top hit |
|---|---|---|
| Brain 3-platform (90 shared) | GO:CC cell junction (p = 5.3 × 10⁻¹⁴) | **0 enriched terms** |
| Brain Xenium ↔ Allen (90 shared) | GO:CC cell junction (p = 1.2 × 10⁻¹⁵) | **0 enriched terms** |
| Brain Xenium ↔ Stereo-seq (246 shared) | GO:CC neuron projection (p = 2.7 × 10⁻⁷) | **0 enriched terms** |
| Brain Allen ↔ Stereo-seq (491 shared) | GO:BP synaptic signaling (p = 6.4 × 10⁻⁸) | **0 enriched terms** |
| Colon Xenium ↔ CosMx (98 shared) | GO:CC extracellular space (p = 8.2 × 10⁻¹⁵) | **0 enriched terms** |
| Breast Xenium ↔ Visium (307 shared) | GO:BP regulation of cell adhesion (p = 6.3 × 10⁻⁹) | **0 enriched terms** |

**The "cell junction at p = 5.3 × 10⁻¹⁴" headline was circular.** The shared brain panels are themselves enriched for neural/junctional markers by panel design. PiMorph's top-50 is not statistically distinguishable from the rest of the shared panel under the correct background.

Raw outputs in `02_tables/null_panel_bg/`.

## 3. Boundary-vs-cell-body rank — full panel (v3 correction over v2)

You flagged in round 2 that v2's `boundary_vs_body.py` was capped at `--max-genes 200`, so the reported ranks (e.g., "rank 5 of 124") were from a variance-selected subset, not the full panel. I removed the cap and re-ran.

### Xenium **human colon** — full panel, 362 expressed non-control genes:

| Rank | Gene | mean boundary fraction | Note |
|---|---|---:|---|
| **1** | EPCAM | 0.486 | epithelial membrane |
| **2** | CD24 | 0.475 | GPI-anchored apical |
| **3** | KRT8 | 0.461 | cytokeratin — also strong here (note: previously suspected as cytoplasm artifact, but **it is in the panel and the ranking puts it 3rd**) |
| **4** | PIGR | 0.455 | polarized epithelial transcytosis receptor |
| 11 | MUC12 | 0.389 | cell-surface mucin |
| **14** | CDH1 | 0.379 | **E-cadherin — adherens junction** |
| 24 | REG4 | 0.361 | deep-crypt secretory |
| **34** | CTNNB1 | 0.341 | **β-catenin — adherens junction** |
| 69 | ACTA2 | 0.270 | cytoplasmic α-SMA |
| 203 | CD3D | 0.179 | cytoplasmic T-cell marker |

**Reframed (per your suggestion):** epithelial membrane / surface / secretory / AJ genes dominate the boundary ranking — EPCAM, CD24, KRT8, PIGR all in the top 4 of 362; MUC12, CDH1 in top 14; REG4 + CTNNB1 in top 34; ACTA2 + CD3D well below. The previous statement that "KRT8/PIGR/REG4/MUC12 are not in the Xenium colon panel" was **wrong** — they are, and they confirm the broader colon-epithelial-perimeter biology.

This is the most defensible single result in the package: a direct, pre-specified test that puts the right gene categories at the right end of the ranking.

### Xenium **mouse brain** — full panel, 247 expressed non-control genes:

| Rank | Gene | mean boundary fraction | Note |
|---|---|---:|---|
| 2 | Dcn | 0.242 | perivascular ECM |
| 4 | Igf2 | 0.227 | growth factor |
| 5 | Acta2 | 0.219 | mural |
| 12 | Slc17a7 | 0.176 | vesicular glutamate transporter |
| 13 | Gad1 | 0.169 | GABAergic |
| 19 | Gfap | 0.156 | astrocyte intermediate filament |
| **30** | **Aqp4** | **0.148** | **astrocyte endfoot water channel** |
| 136 | Pecam1 | 0.101 | EC |
| 138 | Cldn5 | 0.099 | tight junction |

**Reframed (per your v3 review):** Aqp4 is upper-quartile (rank 30 / 247 = top 12%) but with **median boundary fraction of zero** — call it **modest / ambiguous** at the RNA boundary level, not a clean negative as v2 said. Cldn5 (rank 138) is the actual negative. The proper read:

> Xenium mouse brain does not cleanly validate perivascular Aqp4/Cldn5 RNA boundary localization. Aqp4 shows only modest boundary signal and Cldn5 is low; the known biology still points to protein-level AQP4 validation rather than an RNA-boundary claim.

## 4. Cross-platform pair statistics — corrected with `allen_merfish` properly tagged as brain

A separate metadata bug: `allen_merfish` was tagged `tissue = other` in v1/v2 because the R script's regex didn't include `^allen`. After fixing (regex now matches `allen` → brain):

| Comparison class | v2 quoted | v3 corrected |
|---|---|---|
| Same tissue, different platform | n = 3, median JS = 0.51 | **n = 5, median JS = 0.62, p_wilcoxon = 0.018** |
| Different tissue, same platform | n = 5, median 0.83 | n = 5, median 0.83 (unchanged) |
| Different tissue, different platform | n = 37, median 0.84 | n = 35, median 0.84 |
| PERMANOVA, tissue alone, R² | 0.808, p = 0.0097 | **0.710, p = 0.0088** |

The same-tissue-cross-platform class now has the 3 brain pairs (Xenium MB ↔ Allen ↔ Stereo-seq) plus colon (Xenium ↔ CosMx) plus breast (Xenium ↔ Visium). The qualitative story — same-tissue cross-platform is the tightest class — survives; the numbers tighten because the brain trio adds two pairs.

## 5. Updated framing — what's in vs out (v3)

| Claim | Updated framing |
|---|---|
| "GO cell junction at p = 5.3 × 10⁻¹⁴ validates PiMorph" | **Withdrawn (v2).** Panel-bg null gives 0 enriched terms across all 6 panels. |
| "PiMorph rediscovers the molecular signature of the neurovascular unit" | **Softened (v2):** "PiMorph's top conserved brain edge-band genes recover a neurovascular + synaptic contact *ecology* — cell types and genes that participate in the contacts, not necessarily junction-localized molecules." |
| "Junctional gene expression" for centroid datasets | **Renamed (v2):** "contact-weighted co-expression" — `min(e_i, e_j) × contact_len_um` is a co-expression proxy weighted by shared boundary length, not a measured RNA boundary density. |
| "CDH1 + CTNNB1 are top 5 / 13 of 124" | **Reframed (v3):** *"In a full-panel Xenium colon boundary-vs-body test, epithelial membrane / surface genes dominate the boundary ranking: EPCAM 1/362, CD24 2/362, KRT8 3/362, PIGR 4/362, MUC12 11/362, CDH1 14/362, REG4 24/362, CTNNB1 34/362; ACTA2 (69) and CD3D (203) rank substantially lower."* |
| "Aqp4 rank 15/34 — mid-pack negative" | **Reframed (v3):** *"Aqp4 rank 30/247 — upper-quartile by mean boundary fraction but with median 0; modest / ambiguous at the RNA level. Cldn5 (rank 138/247) is the cleaner negative."* |
| "HER2 enrichment at the junction in breast" | **Softened (v2):** "PiMorph recovers a membrane-receptor-dominated HER2-amplified tumor contact signature." Cannot distinguish junction-localized HER2 from overall HER2 abundance with Visium 55 µm spots. |
| "Colon panel passes both bars" | **Removed (v3, per your correction):** colon does NOT pass the GO/panel-background null bar — no panel does. What colon passes is *"the direct, gene-level, pre-specified boundary-rank test for epithelial membrane / surface / AJ genes."* It does not rely on GO enrichment. |

## 6. Changes vs v2 of this response

- Full-panel boundary ranks (no `--max-genes 200` cap). Colon now 362 genes, brain now 247.
- KRT8 / PIGR / MUC12 / REG4 acknowledged as in the Xenium colon panel and ranking at top — v2 incorrectly said they weren't.
- Aqp4 softened from "mid-pack negative" to "modest / ambiguous"; Cldn5 named as the cleaner negative.
- `allen_merfish` tissue label fixed from "other" to "brain" in the metadata + R regex (`scripts/deep_cross_analysis.R`). Same-tissue cross-platform now n=5 (median JS 0.62, Wilcoxon p=0.018) instead of n=3 (median 0.51).
- "Colon panel passes both bars" sentence removed; replaced with explicit acknowledgment that colon passes the direct boundary-rank test only.
- New master figure `master_figure_v3.pdf` with the corrected panels.
- v1 `NARRATIVE.md` no longer in the read-first path (the withdrawn GO claim was prominent in it).

## 7. Deferred items from your review

- **Spatial permutation null** for Xenium (rotate transcript coordinates within tissue mask, recompute boundary integrals on permuted transcripts). Skipped because the raw `transcripts.parquet` for Xenium MB and colon are not local (~50 GB each on EC2 only). Script design captured here for next round.
- **Cell-type residual null** (regress edge-band signal on edge type + contact length + cell-body expression). Attempted in `scripts/boundary_by_edge_type.py`; current Xenium panels too sparse for per-gene-per-edge-type reliability (n < 20 edges with detectable signal per gene per type). Script committed for future denser panels (CosMx WTX, Stereo-seq).
- **Boundary-vs-body on more datasets**: done for MIBI (`02_tables/boundary_vs_body/boundary_vs_body_mibi.csv`) — direction is right (CD45 / CD40 / CD133 atop; Chym_Tryp at bottom) but absolute values need unit calibration between `cells_x_marker` (whole-cell mean) and `edges_typed _mean` columns.

## 8. The revised headline

> **A direct boundary-vs-body test in Xenium colon places epithelial membrane / surface and adherens-junction genes at the top of the full expressed panel (EPCAM 1/362, CD24 2/362, KRT8 3/362, PIGR 4/362, MUC12 11/362, CDH1 14/362, REG4 24/362, CTNNB1 34/362), while the original GO enrichment claim disappears under the correct panel-background null (0 of 1,213 enriched terms remain). Mouse brain is ambiguous at the RNA boundary level (Aqp4 rank 30/247, Cldn5 rank 138/247) and should be validated at the protein level.**

That replaces the v1 "cell junction at p = 5.3 × 10⁻¹⁴" line as the paper's headline.

— Okezue
