#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$ROOT/logs" "$ROOT/results/metadata_transport"

if [[ ! -f "$ROOT/data/processed/transport_pairs.parquet" ]]; then
  "$PY" "$ROOT/scripts/build_transport_pairs.py" \
    --clean "$ROOT/data/processed/ic50_clean.parquet" \
    --splits "$ROOT/data/processed/splits/split_assignments.parquet" \
    --endpoint IC50 \
    --output "$ROOT/data/processed/transport_pairs.parquet" \
    --summary "$ROOT/data/processed/transport_pairs_summary.json" \
    --min-assay-size 20
fi

"$PY" "$ROOT/scripts/run_metadata_transport.py" \
  --pairs "$ROOT/data/processed/transport_pairs.parquet" \
  --assay-metadata "$ROOT/results/tables/assay_metadata_table.csv" \
  --output-dir "$ROOT/results/metadata_transport" \
  --alpha 10.0 \
  --max-text-features 2048 \
  2>&1 | tee "$ROOT/logs/metadata_transport.log"

echo "[metadata-transport] complete"
