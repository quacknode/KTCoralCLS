#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=artifacts/pipeline_logs
mkdir -p "$LOG_DIR" artifacts/runs_sourceaware artifacts/comparison_sourceaware

echo "=== Group hold-out train ==="
python -m ktcoralcls.train_suite \
  --epochs 15 \
  --num-workers 4 \
  --seed 42 \
  --skip-existing \
  --patches-dir datasets/ktcoral/patches \
  --splits-dir datasets/ktcoral/splits_sourceaware \
  --runs-dir artifacts/runs_sourceaware \
  2>&1 | tee "$LOG_DIR/sourceaware_full_train.log"

echo "=== Group hold-out compare ==="
python -m ktcoralcls.compare \
  --runs-dir artifacts/runs_sourceaware \
  --out-dir artifacts/comparison_sourceaware \
  --patches-dir datasets/ktcoral/patches \
  --splits-dir datasets/ktcoral/splits_sourceaware \
  2>&1 | tee "$LOG_DIR/sourceaware_full_compare.log"

echo "SOURCEAWARE_BAKEOFF_COMPLETE"
