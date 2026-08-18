#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

mkdir -p "$ROOT/logs" "$ROOT/results/support_gated_selector"

"$PY" "$ROOT/scripts/run_support_gated_selector.py" \
  --baseline-predictions "$ROOT/results/transport_baselines/transport_baseline_predictions.parquet" \
  --prediction-dir "$ROOT/results/operator_affine" \
  --prediction-dir "$ROOT/results/latent_affine_id" \
  --output-dir "$ROOT/results/support_gated_selector" \
  2>&1 | tee "$ROOT/logs/support_gated_selector.log"
