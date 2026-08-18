#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$ROOT/logs" "$ROOT/results/operator_affine"

"$PY" "$ROOT/scripts/train_operator_affine.py" \
  --pairs "$ROOT/data/processed/transport_pairs.parquet" \
  --assay-metadata "$ROOT/results/tables/assay_metadata_table.csv" \
  --output-dir "$ROOT/results/operator_affine" \
  --method-name somol_affine_meta \
  --seed 13 \
  --epochs 120 \
  --patience 15 \
  --batch-size 65536 \
  --lr 0.003 \
  2>&1 | tee "$ROOT/logs/operator_affine_meta.log"

"$PY" "$ROOT/scripts/train_operator_affine.py" \
  --pairs "$ROOT/data/processed/transport_pairs.parquet" \
  --assay-metadata "$ROOT/results/tables/assay_metadata_table.csv" \
  --output-dir "$ROOT/results/operator_affine" \
  --method-name somol_affine_meta_id \
  --use-assay-residual \
  --seed 13 \
  --epochs 120 \
  --patience 15 \
  --batch-size 65536 \
  --lr 0.003 \
  2>&1 | tee "$ROOT/logs/operator_affine_meta_id.log"

echo "[operator-affine] complete"
