#!/bin/bash
# Run EndoPiGraph-AJmorph on polarityjam test data
# This script runs the full AJ morphology analysis pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

echo "Running EndoPiGraph-AJmorph pipeline on polarityjam test data..."
echo ""

# Run the all-in-one pipeline with Cellpose segmentation
# Channel mapping from polarityjam parameters_golgi_nuclei.yml:
#   Channel 0: Golgi (GM130)
#   Channel 2: Nuclei (DAPI)
#   Channel 3: Junction (VE-cadherin)
python src/EndoPiGraph_AJmorph_v1_allinone.py run \
    --manifest data/manifest.csv \
    --dataset-root data/raw \
    --out runs/polarityjam_test \
    --aj-channel "3" \
    --cell-channel "3" \
    --nuc-channel "2" \
    --golgi-channel "0" \
    --flow-dir-deg 0.0 \
    --use-cellpose 1 \
    --cellpose-diameter 50

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Outputs are in: runs/polarityjam_test/"
echo ""
echo "Generated files:"
ls -la runs/polarityjam_test/

echo ""
echo "Per-image outputs:"
ls -la runs/polarityjam_test/*/
