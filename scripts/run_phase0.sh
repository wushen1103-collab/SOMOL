#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
ENDPOINT="${ENDPOINT:-IC50}"
RELEASE="${RELEASE:-chembl_37}"
RAW_DIR="${RAW_DIR:-$ROOT/data/raw/$RELEASE}"
PROCESSED_DIR="${PROCESSED_DIR:-$ROOT/data/processed}"
GATE_DIR="${GATE_DIR:-$ROOT/results/gate}"
TABLE_DIR="${TABLE_DIR:-$ROOT/results/tables}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
WORKERS="${WORKERS:-4}"
DOWNLOAD_CONNECTIONS="${DOWNLOAD_CONNECTIONS:-16}"

mkdir -p "$RAW_DIR" "$PROCESSED_DIR" "$GATE_DIR" "$TABLE_DIR" "$LOG_DIR"

echo "[phase0] root=$ROOT"
echo "[phase0] endpoint=$ENDPOINT release=$RELEASE workers=$WORKERS"

"$PY" "$ROOT/scripts/download_chembl.py" \
  --release "$RELEASE" \
  --output-dir "$RAW_DIR" \
  --connections "$DOWNLOAD_CONNECTIONS" \
  2>&1 | tee "$LOG_DIR/download_${RELEASE}.log"

SQLITE_PATH="$(find "$RAW_DIR" -type f \( -name '*.db' -o -name '*.sqlite' \) | head -n 1)"
if [[ -z "$SQLITE_PATH" ]]; then
  echo "[phase0] no SQLite file found in $RAW_DIR" >&2
  exit 2
fi
echo "[phase0] sqlite=$SQLITE_PATH"

"$PY" "$ROOT/scripts/extract_endpoint.py" \
  --sqlite "$SQLITE_PATH" \
  --endpoint "$ENDPOINT" \
  --output-dir "$PROCESSED_DIR" \
  --workers "$WORKERS" \
  2>&1 | tee "$LOG_DIR/extract_${ENDPOINT}.log"

LOWER_ENDPOINT="$(echo "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
CLEAN="$PROCESSED_DIR/${LOWER_ENDPOINT}_clean.parquet"
DUPLICATES="$PROCESSED_DIR/${LOWER_ENDPOINT}_duplicate_stats.parquet"

"$PY" "$ROOT/scripts/run_gate.py" \
  --clean "$CLEAN" \
  --duplicates "$DUPLICATES" \
  --endpoint "$ENDPOINT" \
  --output-dir "$GATE_DIR" \
  --table-dir "$TABLE_DIR" \
  2>&1 | tee "$LOG_DIR/gate_${ENDPOINT}.log"

"$PY" "$ROOT/scripts/make_splits.py" \
  --clean "$CLEAN" \
  --output-dir "$PROCESSED_DIR/splits" \
  --seed 13 \
  2>&1 | tee "$LOG_DIR/splits_${ENDPOINT}.log"

echo "[phase0] complete"
