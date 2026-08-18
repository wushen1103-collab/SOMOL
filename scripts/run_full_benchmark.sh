#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
SQLITE="${CHEMBL_SQLITE:-$ROOT/data/raw/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db}"
WORKERS="${WORKERS:-4}"

exec "$PY" "$ROOT/scripts/run_multidataset_benchmark.py" \
  --root "$ROOT" \
  --sqlite "$SQLITE" \
  --endpoints IC50,KI,KD,EC50 \
  --seeds 13,17,23,29,31 \
  --workers "$WORKERS" \
  "$@"
