#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

"$PY" "$ROOT/scripts/compare_transport_results.py" \
  --baseline-predictions "$ROOT/results/transport_baselines/transport_baseline_predictions.parquet" \
  --operator-dir "$ROOT/results/operator_affine" \
  --extra-prediction-dir "$ROOT/results/latent_affine_id" \
  --extra-prediction-dir "$ROOT/results/support_gated_selector" \
  --extra-prediction-dir "$ROOT/results/residual_xgb" \
  --extra-prediction-dir "$ROOT/results/calibration_grid" \
  --extra-prediction-dir "$ROOT/results/gauge_calibration" \
  --extra-prediction-dir "$ROOT/results/graph_gauge_calibration" \
  --extra-prediction-dir "$ROOT/results/assay_context_surrogates" \
  --output-dir "$ROOT/results/transport_compare" \
  2>&1 | tee "$ROOT/logs/transport_compare.log"
