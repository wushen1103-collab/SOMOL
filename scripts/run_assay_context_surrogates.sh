#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$ROOT/logs" "$ROOT/results/assay_context_surrogates"

"$PY" "$ROOT/scripts/run_assay_context_surrogates.py" \
  --baseline-predictions "$ROOT/results/transport_baselines/transport_baseline_predictions.parquet" \
  --assay-metadata "$ROOT/results/tables/assay_metadata_table.csv" \
  --output-dir "$ROOT/results/assay_context_surrogates" \
  --min-pair-support 2 \
  --svd-dim 128 \
  --max-text-features 2048 \
  --knn-list 1,5,20 \
  --ridge-alpha-list 0.1,1,10,100 \
  2>&1 | tee "$ROOT/logs/assay_context_surrogates.log"
