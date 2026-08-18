#!/usr/bin/env python
"""Extract and clean one ChEMBL bioactivity endpoint for SOMOL Phase 0."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

METAL_ATOMIC_NUMBERS = {
    3,
    4,
    11,
    12,
    13,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    55,
    56,
    57,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
}


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def existing_expr(conn: sqlite3.Connection, table: str, options: list[str], alias: str) -> str:
    cols = table_columns(conn, table)
    for option in options:
        if option in cols:
            return f"{table}.{option} AS {alias}"
    return f"NULL AS {alias}"


def standardize_one(smiles: str) -> tuple[str | None, float | None, str | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, "empty_smiles"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, "invalid_smiles"
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
        if any(atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS for atom in mol.GetAtoms()):
            return None, None, "metal_complex"
        mw = float(Descriptors.MolWt(mol))
        if mw < 100 or mw > 900:
            return None, mw, "mw_out_of_range"
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return can, mw, None
    except Exception as exc:  # RDKit can throw many chemistry-specific errors.
        return None, None, f"rdkit_error:{type(exc).__name__}"


def standardize_many(smiles_values: list[str], workers: int) -> dict[str, tuple[str | None, float | None, str | None]]:
    unique = sorted({s for s in smiles_values if isinstance(s, str) and s.strip()})
    if not unique:
        return {}
    if workers <= 1:
        values = [standardize_one(s) for s in tqdm(unique, desc="standardize")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            values = list(tqdm(pool.map(standardize_one, unique, chunksize=256), total=len(unique), desc="standardize"))
    return dict(zip(unique, values))


def unit_to_nm(value: float, units: str) -> float | None:
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    if units is None or (isinstance(units, float) and np.isnan(units)):
        return None
    unit = str(units).strip().replace("µ", "u").lower()
    if unit in {"nm", "nanom", "nanomolar"}:
        return float(value)
    if unit in {"um", "µm", "microm", "micromolar"}:
        return float(value) * 1_000.0
    if unit in {"mm", "millim", "millimolar"}:
        return float(value) * 1_000_000.0
    if unit in {"m", "molar"}:
        return float(value) * 1_000_000_000.0
    if unit in {"pm", "picom", "picomolar"}:
        return float(value) / 1_000.0
    return None


def build_query(conn: sqlite3.Connection, endpoint: str) -> str:
    assay_doc = existing_expr(conn, "assays", ["doc_id"], "assay_doc_id")
    act_doc = existing_expr(conn, "activities", ["doc_id"], "activity_doc_id")
    assay_chembl = existing_expr(conn, "assays", ["chembl_id"], "assay_chembl_id")
    molecule_chembl = existing_expr(conn, "molecule_dictionary", ["chembl_id"], "molecule_chembl_id")
    target_chembl = existing_expr(conn, "target_dictionary", ["chembl_id"], "target_chembl_id")
    mw_expr = existing_expr(conn, "compound_properties", ["mw_freebase", "full_mwt"], "chembl_mw")
    year_expr = existing_expr(conn, "docs", ["year"], "pub_year")
    journal_expr = existing_expr(conn, "docs", ["journal"], "journal")
    doi_expr = existing_expr(conn, "docs", ["doi"], "doi")
    pubmed_expr = existing_expr(conn, "docs", ["pubmed_id"], "pubmed_id")
    doc_type_expr = existing_expr(conn, "docs", ["doc_type"], "doc_type")
    title_expr = existing_expr(conn, "docs", ["title"], "doc_title")
    assay_strain = existing_expr(conn, "assays", ["assay_strain"], "assay_strain")
    assay_tissue = existing_expr(conn, "assays", ["assay_tissue"], "assay_tissue")
    assay_cell_type = existing_expr(conn, "assays", ["assay_cell_type"], "assay_cell_type")
    assay_subcellular_fraction = existing_expr(conn, "assays", ["assay_subcellular_fraction"], "assay_subcellular_fraction")
    bao_format = existing_expr(conn, "assays", ["bao_format"], "bao_format")
    relationship_type = existing_expr(conn, "assays", ["relationship_type"], "relationship_type")
    assay_tax_id = existing_expr(conn, "assays", ["assay_tax_id"], "assay_tax_id")
    potential_duplicate = existing_expr(conn, "activities", ["potential_duplicate"], "potential_duplicate")
    data_validity_comment = existing_expr(conn, "activities", ["data_validity_comment"], "data_validity_comment")

    endpoint_sql = endpoint.replace("'", "''").upper()
    return f"""
    SELECT
      activities.activity_id,
      activities.assay_id,
      {assay_chembl},
      activities.molregno,
      {molecule_chembl},
      compound_structures.canonical_smiles AS source_smiles,
      {mw_expr},
      activities.standard_type,
      activities.standard_relation,
      activities.standard_value,
      activities.standard_units,
      activities.pchembl_value,
      {data_validity_comment},
      {potential_duplicate},
      {act_doc},
      {assay_doc},
      assays.description,
      assays.assay_type,
      assays.assay_organism,
      {assay_tax_id},
      {assay_strain},
      {assay_tissue},
      {assay_cell_type},
      {assay_subcellular_fraction},
      {bao_format},
      assays.confidence_score,
      assays.src_id,
      {relationship_type},
      target_dictionary.tid,
      {target_chembl},
      target_dictionary.pref_name AS target_name,
      target_dictionary.target_type,
      target_dictionary.organism AS target_organism,
      component_sequences.accession,
      component_sequences.sequence,
      {year_expr},
      {journal_expr},
      {doi_expr},
      {pubmed_expr},
      {doc_type_expr},
      {title_expr}
    FROM activities
    JOIN assays ON activities.assay_id = assays.assay_id
    JOIN target_dictionary ON assays.tid = target_dictionary.tid
    JOIN target_components ON target_dictionary.tid = target_components.tid
    JOIN component_sequences ON target_components.component_id = component_sequences.component_id
    JOIN molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
    JOIN compound_structures ON activities.molregno = compound_structures.molregno
    LEFT JOIN compound_properties ON activities.molregno = compound_properties.molregno
    LEFT JOIN docs ON assays.doc_id = docs.doc_id
    WHERE UPPER(activities.standard_type) = '{endpoint_sql}'
      AND activities.standard_relation = '='
      AND activities.standard_value > 0
      AND target_dictionary.target_type = 'SINGLE PROTEIN'
      AND assays.confidence_score = 9
      AND compound_structures.canonical_smiles IS NOT NULL
    """


def prepare_chunk(chunk: pd.DataFrame, endpoint: str, workers: int) -> tuple[pd.DataFrame, dict[str, int]]:
    counters: dict[str, int] = {"input_rows": len(chunk)}
    chunk["standard_value"] = pd.to_numeric(chunk["standard_value"], errors="coerce")
    chunk["pchembl_value"] = pd.to_numeric(chunk["pchembl_value"], errors="coerce")
    chunk["value_nm"] = [unit_to_nm(v, u) for v, u in zip(chunk["standard_value"], chunk["standard_units"])]
    chunk = chunk.dropna(subset=["value_nm"]).copy()
    counters["supported_unit_rows"] = len(chunk)
    chunk[f"p{endpoint.upper()}"] = 9.0 - np.log10(chunk["value_nm"].astype(float))
    chunk["pchembl_delta"] = chunk["pchembl_value"] - chunk[f"p{endpoint.upper()}"]

    std_map = standardize_many(chunk["source_smiles"].tolist(), workers)
    std_values = chunk["source_smiles"].map(std_map)
    chunk["canonical_smiles_somol"] = [x[0] if isinstance(x, tuple) else None for x in std_values]
    chunk["rdkit_mw"] = [x[1] if isinstance(x, tuple) else None for x in std_values]
    chunk["standardization_error"] = [x[2] if isinstance(x, tuple) else "missing_map" for x in std_values]
    counters["standardized_rows"] = int(chunk["canonical_smiles_somol"].notna().sum())
    chunk = chunk[chunk["canonical_smiles_somol"].notna()].copy()

    component_counts = chunk.groupby("tid")["accession"].transform("nunique")
    chunk = chunk[component_counts == 1].copy()
    counters["unique_component_rows"] = len(chunk)
    chunk["endpoint"] = endpoint.upper()
    chunk["compound_target_key"] = chunk["canonical_smiles_somol"] + "||" + chunk["target_chembl_id"].astype(str)
    return chunk, counters


def aggregate_duplicates(df: pd.DataFrame, endpoint: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_col = f"p{endpoint.upper()}"
    group_cols = ["assay_id", "canonical_smiles_somol", "target_chembl_id"]
    stats = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_repeats=(y_col, "size"),
            min_p=(y_col, "min"),
            max_p=(y_col, "max"),
            median_p=(y_col, "median"),
            mean_p=(y_col, "mean"),
            std_p=(y_col, "std"),
            first_activity_id=("activity_id", "first"),
        )
        .reset_index()
    )
    stats["range_p"] = stats["max_p"] - stats["min_p"]
    stats["high_conflict"] = stats["range_p"] > 0.5

    keep_keys = stats.loc[~stats["high_conflict"], group_cols + ["median_p", "n_repeats", "range_p"]]
    first_cols = [
        c
        for c in df.columns
        if c not in {y_col, "value_nm", "standard_value", "pchembl_delta"}
    ]
    meta = df.sort_values("activity_id").drop_duplicates(group_cols)[first_cols]
    clean = meta.merge(keep_keys, on=group_cols, how="inner")
    clean[y_col] = clean["median_p"]
    clean["compound_target_key"] = clean["canonical_smiles_somol"] + "||" + clean["target_chembl_id"].astype(str)

    high_conflict = stats[stats["high_conflict"]].copy()
    return clean, stats, high_conflict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--endpoint", default="IC50")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 32) - 30))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = args.endpoint.upper()
    y_col = f"p{endpoint}"

    conn = sqlite3.connect(args.sqlite)
    query = build_query(conn, endpoint)
    frames: list[pd.DataFrame] = []
    counters: dict[str, int] = {"chunks": 0, "input_rows": 0, "supported_unit_rows": 0, "standardized_rows": 0, "unique_component_rows": 0}

    for chunk in pd.read_sql_query(query, conn, chunksize=args.chunksize):
        counters["chunks"] += 1
        prepared, partial = prepare_chunk(chunk, endpoint, args.workers)
        for key, value in partial.items():
            counters[key] = counters.get(key, 0) + int(value)
        frames.append(prepared)
        print(f"[extract] chunk {counters['chunks']} retained {len(prepared):,} rows")

    conn.close()
    if not frames:
        raise RuntimeError("No rows extracted; check endpoint filters and SQLite path.")

    df = pd.concat(frames, ignore_index=True)
    clean, duplicate_stats, high_conflict = aggregate_duplicates(df, endpoint)

    prefix = endpoint.lower()
    raw_filtered_path = output_dir / f"{prefix}_filtered_preduplicate.parquet"
    clean_path = output_dir / f"{prefix}_clean.parquet"
    duplicate_path = output_dir / f"{prefix}_duplicate_stats.parquet"
    conflict_path = output_dir / f"{prefix}_high_conflict.parquet"
    summary_path = output_dir / f"{prefix}_extraction_summary.json"

    df.to_parquet(raw_filtered_path, index=False)
    clean.to_parquet(clean_path, index=False)
    duplicate_stats.to_parquet(duplicate_path, index=False)
    high_conflict.to_parquet(conflict_path, index=False)

    summary = {
        "endpoint": endpoint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sqlite": args.sqlite,
        "workers": args.workers,
        "input_rows": counters["input_rows"],
        "supported_unit_rows": counters["supported_unit_rows"],
        "standardized_rows": counters["standardized_rows"],
        "unique_component_rows": counters["unique_component_rows"],
        "clean_rows": int(len(clean)),
        "clean_assays": int(clean["assay_id"].nunique()),
        "clean_targets": int(clean["target_chembl_id"].nunique()),
        "clean_compounds": int(clean["canonical_smiles_somol"].nunique()),
        "high_conflict_groups": int(len(high_conflict)),
        "duplicate_groups": int((duplicate_stats["n_repeats"] > 1).sum()),
        "y_min": float(clean[y_col].min()) if len(clean) else math.nan,
        "y_max": float(clean[y_col].max()) if len(clean) else math.nan,
        "outputs": {
            "raw_filtered": str(raw_filtered_path),
            "clean": str(clean_path),
            "duplicate_stats": str(duplicate_path),
            "high_conflict": str(conflict_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
