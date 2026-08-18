#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$ROOT/logs" "$ROOT/results/graph_gauge_calibration"

"$PY" "$ROOT/scripts/tune_graph_gauge_calibration.py" \
  --baseline-predictions "$ROOT/results/transport_baselines/transport_baseline_predictions.parquet" \
  --output-dir "$ROOT/results/graph_gauge_calibration" \
  --min-pairs-grid 2,3,5,10,20,30 \
  --shrink-grid 1,2,5,10,20,50,100 \
  --gauge-lambdas 0.01,0.1,1,10,100,1000 \
  --gauge-alphas 0.1,0.25,0.5,0.75,0.9 \
  --min-graph-val-count-grid 0,100,500,1000 \
  2>&1 | tee "$ROOT/logs/graph_gauge_calibration.log"
