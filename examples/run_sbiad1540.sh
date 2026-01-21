#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end run on S-BIAD1540.
# You should edit examples/config_sbiad1540.yaml (channel names) before running.

ACC="S-BIAD1540"
RAW_DIR="data/raw/${ACC}"

# 1) Download (prints recommended commands + optional wget)
endopigraph download "${ACC}" --out "data/raw" --method print

# 2) Build a manifest (lists images + channel names)
endopigraph make-manifest "${RAW_DIR}" --out "data/manifest.csv"

# 3) Run the pipeline
endopigraph run --config examples/config_sbiad1540.yaml
