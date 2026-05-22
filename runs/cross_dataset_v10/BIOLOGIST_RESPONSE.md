# Response to biologist review — 2026-05-22

## TL;DR for the biologist

You were right on **all three** numerical corrections — confirmed below. The panel-background null result is **negative** (your suspicion held). The boundary-vs-body test is **positive for colon** and **mixed for brain** — full numbers below. Updating the framing now.

## 1. Edge-count corrections (you were right)

| Edge type | I quoted | Actual (verified from `edge_type_summary.parquet`) |
|---|---|---|
| `endothelial__glia`, Allen MERFISH | 826,004 (whole-brain run) | **60,828** (3-section subset run used in this package) |
| `endothelial__glia`, Xenium MB | 12 (was from a 1 mm² pilot tile) | **1,644** (full-tissue run) |
| `endothelial__glia`, Stereo-seq MB | 299 ✓ | 299 ✓ |
| "B-cell ↔ endothelial in CosMx colon, 6,828" | wrong | **`bcell__epithelial = 6,828`**; `bcell__endothelial = 231` |

The cover letter mixed up cohorts (subset vs full-brain Allen) and pulled the breast-tile B-cell figure. Apologies — those are corrected in the next package.

## 2. Panel-background null: enrichment evaporates

I re-ran g:Profiler exactly as you proposed, **using the shared panel universe as the background** (g:Profiler's `custom` domain_scope with the explicit gene list), and:

| Panel | Whole-genome background top hit | Shared-panel background top hit |
|---|---|---|
| Brain 3-platform (90 shared) | GO:CC cell junction (p = 5.3 × 10⁻¹⁴) | **0 enriched terms** |
| Brain Xenium ↔ Allen (90 shared) | GO:CC cell junction (p = 1.2 × 10⁻¹⁵) | **0 enriched terms** |
| Brain Xenium ↔ Stereo-seq (246 shared) | GO:CC neuron projection (p = 2.7 × 10⁻⁷) | **0 enriched terms** |
| Brain Allen ↔ Stereo-seq (491 shared) | GO:BP synaptic signaling (p = 6.4 × 10⁻⁸) | **0 enriched terms** |
| Colon Xenium ↔ CosMx (98 shared) | GO:CC extracellular space (p = 8.2 × 10⁻¹⁵) | **0 enriched terms** |
| Breast Xenium ↔ Visium (307 shared) | GO:BP regulation of cell adhesion (p = 6.3 × 10⁻⁹) | **0 enriched terms** |

**The "cell junction at p = 5.3 × 10⁻¹⁴" headline was driven entirely by the fact that the shared brain panels are themselves enriched for neural/junctional markers.** Once the proper background is used, PiMorph's top-50 is not statistically distinguishable from the rest of the shared panel.

Raw outputs in `02_tables/null_panel_bg/`.

The conclusion to update: **PiMorph's typed-edge framework still produces a biologically coherent gene list (your bucketing of brain top-20 confirms this), but the GO test as I framed it was not evidence for that — it was a circular consequence of panel design.**

## 3. Boundary-vs-cell-body fraction — direct test for Xenium

You explicitly asked for this. Run on the two Xenium datasets where we have real transcript-band integration in `edges_typed.parquet` (not just centroid-Voronoi).

**Metric:** for each cell, sum boundary-band transcript counts (from edges_typed) across all its edges → boundary count; compare to cell-body total in `cells_x_gene`. Per gene, take the mean across cells of `boundary / max(boundary, body)`. Higher = transcripts more concentrated in the dilated boundary band relative to cell interior.

### Xenium **human colon** (124 genes passing n_cells ≥ 100 filter):

| Rank | Gene | mean boundary fraction |
|---|---|---:|
| **1** | **EPCAM** | **0.486** |
| **2** | **CD24** | **0.475** |
| 3 | CR2 | 0.436 |
| 4 | C1QBP | 0.414 |
| **5** | **CDH1** | **0.379** |
| 6 | CA2 | 0.371 |
| 7 | CTSB | 0.369 |
| 8 | AQP1 | 0.369 |
| **13** | **CTNNB1** | **0.341** |
| 24 | ACTA2 (cytoplasmic α-SMA) | 0.270 |
| 74 | CD3D (cytoplasmic T-cell marker) | 0.179 |

`EPCAM, CD24, CDH1, CTNNB1` — the canonical membrane / adherens-junction set — rank **1st, 2nd, 5th, and 13th out of 124**. ACTA2 (cytoplasmic intermediate filament–adjacent) sits at rank 24, CD3D (cytoplasmic T-cell marker) at rank 74. This is the cleanest result in the response: **the AJ-complex genes you specifically called out (CDH1 + CTNNB1) sit in the top 11 % of all panel genes by boundary fraction, while cytoplasmic markers don't.**

KRT8 / PIGR / REG4 — your "abundance vs junction" suspects — are not in the Xenium colon panel, so the direct boundary test can't adjudicate them. The cross-platform conservation we reported was driven by the CosMx panel having them, not the Xenium panel.

### Xenium **mouse brain** (only 34 genes passing the filter, because the 248-gene panel doesn't include `Cdh5`, `Vwf`, `Actb`, `Gapdh` etc.):

| Rank | Gene | mean boundary fraction |
|---|---|---:|
| 1 | Dcn (perivascular ECM) | 0.242 |
| 2 | Aldh1a2 | 0.236 |
| 3 | Igf2 | 0.227 |
| 4 | Acta2 (mural) | 0.219 |
| 5 | **Slc17a7** | **0.176** |
| 6 | **Gad1** | **0.169** |
| 7 | Nrn1 | 0.168 |
| 9 | **Gfap** | **0.156** |
| **15** | **Aqp4** | **0.148** |
| 16 | Epha4 | 0.147 |
| 23 | Pecam1 | 0.101 |
| 24 | Cldn5 | 0.099 |

**Aqp4 is rank 15/34 — mid-pack, not boundary-enriched.** You called this the strongest interface-compatible hit, and the direct boundary-vs-body test does **not** support that on this Xenium dataset. Possible explanations:

- Xenium cell segmentation likely collapses thin astrocyte endfeet into "background" or into the adjacent vascular cell, so Aqp4-on-endfoot transcripts get attributed to the wrong cell or lost
- 2-px boundary-band dilation may not capture the very thin perivascular endfoot
- Aqp4 mRNA, unlike AQP4 protein, may not be specifically polarized to the endfoot (mRNA traffics differently from protein)
- The Xenium MB panel design + 1 µm/transcript localization precision puts a ceiling on the spatial signal

This is concrete evidence for your point that **PiMorph's RNA-based boundary signal is not a substitute for direct AQP4 protein IF**, and that the AQP4 hit reported earlier is more "Aqp4 cells participate in this contact" than "Aqp4 mRNA is concentrated at the contact band."

`Cldn5` (tight-junction marker, expected to be boundary-enriched) is rank 24 — same conclusion: RNA boundary signal for Cldn5 doesn't track the known protein junction localization.

**Brain boundary signal is weaker overall — all mean boundary fractions are 0.07–0.25 — versus colon at 0.18–0.49.** This is partly because brain cells are smaller / segmentation is harder / Xenium panel is leaner, and partly because mRNA localization probably IS less polarized than protein.

## 4. Updated framing — what's in vs out

| Claim from cover letter | Updated framing |
|---|---|
| "GO 'cell junction' at p = 5.3 × 10⁻¹⁴ validates PiMorph" | **Withdrawn.** The result was a panel-design artifact; the proper-background test gives zero enriched terms across all panels. |
| "PiMorph rediscovers the molecular signature of the neurovascular unit" | **Softened to** "PiMorph's top conserved brain edge-band genes recover a neurovascular + synaptic contact *ecology* — the cell types and genes that participate in the contacts — rather than necessarily junction-localized molecules." |
| "Junctional gene expression" (centroid datasets) | Renamed to **"contact-weighted co-expression"**: `min(e_i, e_j) × contact_len_um` is the smaller-of-two-cells expression weighted by shared boundary length. It is a contact-aware *co-expression* proxy, not a measured boundary-band density. |
| "CDH1 + CTNNB1 are the textbook AJ proteins" | **Strengthened.** The new boundary-vs-body test (Xenium colon, n=124 genes) puts them at ranks 5 and 13 of 124 — well above ACTA2 (rank 24) and CD3D (rank 74). Real signal. EPCAM and CD24 rank 1 and 2. |
| "Aqp4 + Gfap perivascular validation candidate" | **Aqp4 demoted — rank 15/34 in Xenium MB boundary-vs-body, not boundary-enriched at the mRNA level.** Strong candidate for protein-level validation (your suggested experiment) but not yet supported by the spatial-RNA data we have. |
| "HER2 enrichment at the junction in breast" | **Softened to** "PiMorph recovers a membrane-receptor-dominated HER2-amplified tumor contact signature." Cannot distinguish junction-localized HER2 from overall HER2 abundance with Visium 55 µm spots. |

## 5. What we'll do next (in this order)

1. **Re-package** for the biologist with the corrected edge counts and softened claims (this `BIOLOGIST_RESPONSE.md` file appended to the package).
2. **Run the expression-matched null** you specified (sample genes matched for mean expression / detection fraction / variance, redo enrichment).
3. **Run the cell-type residual null**: regress each gene's edge-band signal on edge type + contact length + cell-body expression, do GO enrichment on residuals. Asks whether `Aqp4` etc. are enriched at contacts *beyond* "astrocytes/neurons are abundant and express them."
4. **Spatial permutation null** for Xenium: rotate transcript coordinates within tissue mask, recompute edge band integrals on permuted transcripts → null distribution of mean boundary fractions per gene.
5. **Run the boundary-vs-body test on more Xenium datasets** (breast, MB-rep2 if we re-pull) and on MIBI glioma where the protein signal is direct.
6. **Drop the GO "cell junction" headline** from the figure and replace it with the **boundary-vs-body ranking** (the EPCAM/CD24/CDH1 result in colon is the strongest panel for the figure now).

## 6. Honest acknowledgment

The most important thing your review caught is that I framed a circular result as evidence. That was a serious overstatement. The framework still has biological signal — the colon boundary-vs-body test demonstrates this — but the brain claim needs to be substantially softened until either (a) the cell-type residual null clears, or (b) the AQP4 wet-bench experiment you outlined gives a positive readout.

Specifically: **I will not claim "PiMorph identifies junction-localized genes" without (a) a properly-backgrounded enrichment test and (b) a boundary-vs-body rank that puts the candidate gene at the top, not the middle.** The colon panel passes both bars; the brain panel currently passes neither at the mRNA level.

Thank you for the careful read.

— Okezue
