#!/usr/bin/env python
"""Create frozen SOMOL split manifests for Phase 0 and later modeling."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024 * 16), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_groups(groups: list[str], seed: int, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    groups = list(groups)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    assignment = {}
    for i, group in enumerate(groups):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        assignment[str(group)] = split
    return assignment


def scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "__INVALID__"
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return "__NO_SCAFFOLD__"
        return Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True)
    except Exception:
        return "__INVALID__"


def summarize_split(df: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for split, sub in df.groupby(column):
        rows[str(split)] = {
            "rows": int(len(sub)),
            "assays": int(sub["assay_id"].nunique()),
            "targets": int(sub["target_chembl_id"].nunique()),
            "compound_target_keys": int(sub["compound_target_key"].nunique()),
            "scaffolds": int(sub["scaffold"].nunique()) if "scaffold" in sub else 0,
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--output-dir", default="data/processed/splits")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.clean).reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df["compound_target_key"] = df["compound_target_key"].astype(str)
    df["assay_id_str"] = df["assay_id"].astype(str)
    df["scaffold"] = [scaffold(s) for s in df["canonical_smiles_somol"].astype(str)]

    s1_map = assign_groups(sorted(df["compound_target_key"].unique()), args.seed)
    df["S1_iid_grouped"] = df["compound_target_key"].map(s1_map)

    s2_map = assign_groups(sorted(df["assay_id_str"].unique()), args.seed + 1)
    df["S2_assay_disjoint"] = df["assay_id_str"].map(s2_map)

    train_scaffolds = set(df.loc[df["S2_assay_disjoint"] == "train", "scaffold"])
    df["S3_scaffold_assay_double_ood"] = df["S2_assay_disjoint"]
    seen_scaffold_test = (df["S2_assay_disjoint"] == "test") & (df["scaffold"].isin(train_scaffolds))
    df.loc[seen_scaffold_test, "S3_scaffold_assay_double_ood"] = "excluded_seen_scaffold"

    s6_map = assign_groups(sorted(df["compound_target_key"].unique()), args.seed + 5)
    df["S6_transport_pair"] = df["compound_target_key"].map(s6_map)
    ct_assay_counts = df.groupby("compound_target_key")["assay_id"].nunique()
    transport_keys = set(ct_assay_counts[ct_assay_counts >= 2].index)
    df["is_transport_candidate"] = df["compound_target_key"].isin(transport_keys)

    columns = [
        "row_id",
        "assay_id",
        "assay_chembl_id",
        "canonical_smiles_somol",
        "target_chembl_id",
        "compound_target_key",
        "scaffold",
        "S1_iid_grouped",
        "S2_assay_disjoint",
        "S3_scaffold_assay_double_ood",
        "S6_transport_pair",
        "is_transport_candidate",
    ]
    columns = [c for c in columns if c in df.columns]
    assignments = df[columns].copy()
    assignment_path = output_dir / "split_assignments.parquet"
    assignments.to_parquet(assignment_path, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "clean_input": str(args.clean),
        "assignment_path": str(assignment_path),
        "assignment_sha256": sha256_file(assignment_path),
        "n_rows": int(len(assignments)),
        "summaries": {
            "S1_iid_grouped": summarize_split(df, "S1_iid_grouped"),
            "S2_assay_disjoint": summarize_split(df, "S2_assay_disjoint"),
            "S3_scaffold_assay_double_ood": summarize_split(df, "S3_scaffold_assay_double_ood"),
            "S6_transport_pair": summarize_split(df, "S6_transport_pair"),
        },
        "transport_candidate_rows": int(df["is_transport_candidate"].sum()),
        "transport_candidate_compound_target_keys": int(len(transport_keys)),
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
