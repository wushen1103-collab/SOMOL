#!/usr/bin/env python3
"""Audit DOI overlap between ChEMBL endpoint tables and BindingDB rows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ENDPOINT_PATHS = {
    "IC50": "data/processed/ic50_clean.parquet",
    "EC50": "data/benchmark/ec50/processed/ec50_clean.parquet",
    "KD": "data/benchmark/kd/processed/kd_clean.parquet",
    "KI": "data/benchmark/ki/processed/ki_clean.parquet",
}


def normalize_doi(value: object) -> str | None:
    if pd.isna(value):
        return None
    doi = str(value).strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    doi = doi.rstrip(".,; ")
    return doi or None


def is_numeric_measurement(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(re.search(r"\d", str(value)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--bindingdb",
        type=Path,
        default=Path("data/external/bindingdb/BindingDB_BindingDB_Articles_202608_tsv.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ac_experiments/tables/bindingdb_publication_overlap_audit.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bindingdb_path = args.bindingdb if args.bindingdb.is_absolute() else args.root / args.bindingdb
    output_path = args.output if args.output.is_absolute() else args.root / args.output

    value_columns = ["IC50 (nM)", "EC50 (nM)", "Kd (nM)", "Ki (nM)"]
    bindingdb = pd.read_csv(
        bindingdb_path,
        sep="\t",
        compression="zip",
        usecols=["Article DOI", *value_columns],
        dtype=str,
        low_memory=False,
    )
    bindingdb["normalized_doi"] = bindingdb["Article DOI"].map(normalize_doi)

    rows: list[dict[str, int | float | str]] = []
    value_by_endpoint = dict(zip(ENDPOINT_PATHS, value_columns))
    for endpoint, relative_path in ENDPOINT_PATHS.items():
        chembl = pd.read_parquet(args.root / relative_path, columns=["doi"])
        chembl_dois = {doi for doi in chembl["doi"].map(normalize_doi) if doi}

        endpoint_rows = bindingdb[bindingdb[value_by_endpoint[endpoint]].map(is_numeric_measurement)].copy()
        doi_rows = endpoint_rows[endpoint_rows["normalized_doi"].notna()].copy()
        overlap = doi_rows[doi_rows["normalized_doi"].isin(chembl_dois)]
        bindingdb_dois = set(doi_rows["normalized_doi"])
        overlapping_dois = bindingdb_dois & chembl_dois

        rows.append(
            {
                "endpoint": endpoint,
                "chembl_unique_dois": len(chembl_dois),
                "bindingdb_numeric_rows": len(endpoint_rows),
                "bindingdb_rows_with_doi": len(doi_rows),
                "overlapping_bindingdb_rows": len(overlap),
                "row_overlap_pct": 100.0 * len(overlap) / len(doi_rows),
                "bindingdb_unique_dois": len(bindingdb_dois),
                "overlapping_unique_dois": len(overlapping_dois),
                "unique_doi_overlap_pct": 100.0 * len(overlapping_dois) / len(bindingdb_dois),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.6f")
    print(output_path)


if __name__ == "__main__":
    main()
