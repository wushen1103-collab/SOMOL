#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$ROOT/logs" "$ROOT/results/latent_affine_id"

"$PY" "$ROOT/scripts/train_latent_affine_id.py" \
  --clean "$ROOT/data/processed/ic50_clean.parquet" \
  --splits "$ROOT/data/processed/splits/split_assignments.parquet" \
  --pairs "$ROOT/data/processed/transport_pairs.parquet" \
  --output-dir "$ROOT/results/latent_affine_id" \
  --seed 13 \
  --epochs 80 \
  --patience 10 \
  --batch-size 131072 \
  --lr 0.003 \
  2>&1 | tee "$ROOT/logs/latent_affine_id.log"

echo "[latent-affine-id] complete"
