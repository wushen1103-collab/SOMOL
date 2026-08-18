#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$ROOT/logs" "$ROOT/results/calibration_grid"

"$PY" "$ROOT/scripts/tune_calibration_grid.py" \
  --baseline-predictions "$ROOT/results/transport_baselines/transport_baseline_predictions.parquet" \
  --output-dir "$ROOT/results/calibration_grid" \
  --min-pairs-grid 2,3,5,10,20,30 \
  --shrink-grid 1,2,5,10,20,50,100 \
  2>&1 | tee "$ROOT/logs/calibration_grid.log"
