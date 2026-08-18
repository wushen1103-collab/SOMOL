#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$ROOT/logs" "$ROOT/results/transport_baselines"

"$PY" "$ROOT/scripts/build_transport_pairs.py" \
  --clean "$ROOT/data/processed/ic50_clean.parquet" \
  --splits "$ROOT/data/processed/splits/split_assignments.parquet" \
  --endpoint IC50 \
  --output "$ROOT/data/processed/transport_pairs.parquet" \
  --summary "$ROOT/data/processed/transport_pairs_summary.json" \
  --min-assay-size 20 \
  2>&1 | tee "$ROOT/logs/build_transport_pairs.log"

"$PY" "$ROOT/scripts/run_transport_baselines.py" \
  --pairs "$ROOT/data/processed/transport_pairs.parquet" \
  --output-dir "$ROOT/results/transport_baselines" \
  --min-affine-pairs 5 \
  --min-isotonic-pairs 10 \
  2>&1 | tee "$ROOT/logs/transport_baselines.log"

echo "[transport] complete"
