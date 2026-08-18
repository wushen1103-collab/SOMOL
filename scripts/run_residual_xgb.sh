#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
NTHREAD="${NTHREAD:-4}"
FINGERPRINT_JOBS="${FINGERPRINT_JOBS:-4}"

mkdir -p "$ROOT/logs" "$ROOT/results/residual_xgb"

"$PY" "$ROOT/scripts/train_residual_xgb.py" \
  --baseline-predictions "$ROOT/results/transport_baselines/transport_baseline_predictions.parquet" \
  --prediction-dir "$ROOT/results/operator_affine" \
  --prediction-dir "$ROOT/results/latent_affine_id" \
  --prediction-dir "$ROOT/results/support_gated_selector" \
  --output-dir "$ROOT/results/residual_xgb" \
  --base-method support_gated_val_micro \
  --nthread "$NTHREAD" \
  --fingerprint-jobs "$FINGERPRINT_JOBS" \
  --num-round 800 \
  --patience 40 \
  --max-depth 4 \
  --eta 0.03 \
  2>&1 | tee "$ROOT/logs/residual_xgb.log"
