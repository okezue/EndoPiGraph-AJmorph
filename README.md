# EndoPiGraph-AJmorph v1

EndoPiGraph-AJmorph v1 is a reference implementation for building **typed endothelial contact graphs** ("pi-graphs" in the sense of the manuscript) and extracting **adherens junction (AJ) morphology features** from fluorescence microscopy.

It is designed to run on BioImage Archive datasets (e.g. **S-BIAD1540**) and to produce:

- instance segmentation masks (cells)
- a cell-cell contact graph (neighbors)
- edge attributes for junction markers (AJ/TJ/GJ/NJ if present)
- AJ morphology features per interface (occupancy, cluster density, etc.)
- publication-ready QC figures and a lightweight HTML report

This repo is intentionally conservative and reproducible: it does **not** assume any specific microscope vendor format beyond TIFF/OME-TIFF, and it makes all channel choices explicit in a YAML config.

---

## Installation

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2) Install EndoPiGraph-AJmorph

```bash
pip install -e .
```

### 3) (Optional) Install Cellpose for segmentation

If you have a CUDA GPU, install an appropriate `torch` first, then:

```bash
pip install -e ".[cellpose]"
```

If you skip Cellpose, the pipeline can fall back to a simple watershed-based segmenter (less accurate).

---

## Quickstart: run on S-BIAD1540

### A) Download

This project includes a downloader that uses the BioStudies/BioImage Archive API to fetch the dataset FTP link.

```bash
endopigraph download --accession S-BIAD1540 --out data/raw --method print
```

The command prints a download URL/path and suggested `wget` commands.

If you want the tool to execute a `wget` mirror (recommended only if you have a stable connection):

```bash
endopigraph download --accession S-BIAD1540 --out data/raw --method wget
```

### B) Build a manifest (scan for images)

```bash
endopigraph make-manifest \
  --input data/raw/S-BIAD1540 \
  --out data/manifest_sbiad1540.csv
```

This CSV is the *only* thing the pipeline needs to know what to process.

### C) Edit the config

Copy the example config:

```bash
cp examples/config_sbiad1540.yaml config.yaml
```

Open `config.yaml` and set:

- the path to the manifest CSV
- which channel(s) to use for segmentation (e.g. nuclei + membrane/junction)
- which channel(s) represent junction markers (e.g. VE-cadherin for AJ)

### D) Run the pipeline

```bash
endopigraph run --config config.yaml
```

Outputs are written under the `output_dir` specified in your config.

---

## Outputs

For each image, the pipeline writes:

- `masks/<image_id>_labels.tif` : instance labels
- `tables/<image_id>_cells.csv` : per-cell geometry
- `tables/<image_id>_interfaces.csv` : per-interface AJmorph and junction stats
- `graphs/<image_id>.graphml` : NetworkX-compatible graph
- `graphs/<image_id>.json` : portable graph format
- `figures/<image_id>_qc_segmentation.png`
- `figures/<image_id>_qc_junctions.png`
- `figures/<image_id>_graph.png`
- `reports/<image_id>.html`

---

## AJmorph v1 features (per interface)

Given a segmentation mask and an AJ marker channel, for each contacting pair of cells `(i, j)` we compute:

- `contact_px` : estimated shared boundary length (pixel units)
- `aj_mean`, `aj_median`, `aj_max` : AJ intensity statistics along the interface
- `aj_threshold` : threshold used for binarization
- `aj_occupancy` : fraction of interface pixels above threshold
- `aj_cluster_count` : number of connected components in the thresholded interface region
- `aj_cluster_density` : `aj_cluster_count / contact_px`

These features are intended to support downstream learning (e.g. mapping to qualitative categories like "reticular" vs "straight").

---

## Optional: export interface patches for manual annotation

```bash
endopigraph export-patches --config config.yaml --max-per-image 200
```

This creates a folder of small PNG crops of interfaces to label in any annotation tool. Once you have labels in a CSV, you can train a simple classifier:

```bash
endopigraph train-ajmorph --features output/tables/all_interfaces.csv --labels your_labels.csv --out output/models
```

---

## Citation and status

This is a research prototype (v0.1.0). It is intended to make the computational part of the manuscript executable and testable on public data.

