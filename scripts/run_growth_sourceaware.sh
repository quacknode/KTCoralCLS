#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=artifacts/pipeline_logs
mkdir -p "$LOG_DIR" artifacts/runs_growth_sourceaware artifacts/comparison_growth_sourceaware

echo "=== Growth multi-task train (group hold-out) ==="
python -m ktcoralcls.train_multitask \
  --epochs 15 \
  --batch-size 48 \
  --num-workers 4 \
  --patches-dir datasets/ktcoral/patches \
  --splits-dir datasets/ktcoral/splits_growth_sourceaware \
  --runs-dir artifacts/runs_growth_sourceaware \
  2>&1 | tee "$LOG_DIR/growth_sourceaware_train.log"

echo "=== Growth multi-task report ==="
python -m ktcoralcls.report_multitask \
  --run-dir artifacts/runs_growth_sourceaware/KTCoralsCLS-ConvNeXt-MultiTask \
  --out-dir artifacts/comparison_growth_sourceaware \
  --patches-dir datasets/ktcoral/patches \
  --splits-dir datasets/ktcoral/splits_growth_sourceaware \
  2>&1 | tee "$LOG_DIR/growth_sourceaware_report.log"

echo "GROWTH_SOURCEAWARE_COMPLETE"
