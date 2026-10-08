#!/bin/bash
# Quick run script for QVAE-Anomaly on public AD benchmarks
#
# Usage:
#   bash run_public_ad.sh                              # default: thyroid 20 epochs
#   DATASET=creditcard bash run_public_ad.sh
#   EPOCHS=5 bash run_public_ad.sh
#   PYTHON=/path/to/python bash run_public_ad.sh       # override interpreter
#   bash run_public_ad.sh --lr 1e-3 --lambda-anom 30   # extra args -> train

set -euo pipefail

# Python interpreter. Activate your venv first, or override via PYTHON=...
PYTHON="${PYTHON:-python}"

# Defaults
DATASET="${DATASET:-thyroid}"
EPOCHS="${EPOCHS:-20}"

echo "========================================="
echo "QVAE-Anomaly Public AD Benchmark"
echo "Python:  $($PYTHON --version 2>&1)"
echo "Dataset: $DATASET, Epochs: $EPOCHS"
echo "========================================="

echo ""
echo "[Step 1/2] Training $DATASET ($EPOCHS epochs)..."
"$PYTHON" train_public_ad.py --dataset "$DATASET" --epochs "$EPOCHS" "$@"

echo ""
echo "[Step 2/2] Evaluating $DATASET..."
"$PYTHON" evaluate_public_ad.py --dataset "$DATASET"

echo ""
echo "Done!"