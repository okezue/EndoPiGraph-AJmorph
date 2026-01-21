#!/bin/bash
# Run EndoPiGraph pipeline on all 102 EGM2 images
# This will take approximately 18 hours on CPU
#
# Usage:
#   ./scripts/run_egm2_batch.sh
#
# To run in background with logging:
#   nohup ./scripts/run_egm2_batch.sh > runs/egm2_full/batch_log.txt 2>&1 &

set -e

cd "$(dirname "$0")/.."

echo "Starting EGM2 batch processing at $(date)"
echo "This will process 102 images and take approximately 18 hours."
echo ""

python3 src/EndoPiGraph_AJmorph_v1_allinone.py run \
    --manifest runs/egm2_full/manifest_egm2_local.csv \
    --dataset-root . \
    --out runs/egm2_full \
    --aj-channel "VE-cadherin" \
    --cell-channel "VE-cadherin" \
    --nuc-channel "DAPI" \
    --golgi-channel "GM130" \
    --use-cellpose 1

echo ""
echo "Pipeline completed at $(date)"
echo ""
echo "Now running typed vs untyped experiment..."

python3 scripts/typed_vs_untyped_experiment.py runs/egm2_full

echo ""
echo "B3 experiment complete!"
echo "Results saved to: runs/egm2_full/typed_vs_untyped_results.json"
