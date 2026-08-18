#!/usr/bin/env python
"""Build held-out directed cross-assay transport pairs.

Rows are paired only within the same exact compound-target key. The split is
the frozen S6 split, so a compound-target key belongs wholly to train, val, or
test and target observations cannot leak across transport splits.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--endpoint", default="IC50")
    parser.add_argument("--output", default="data/processed/transport_pairs.parquet")
    parser.add_argument("--summary", default="data/processed/transport_pairs_summary.json")
    parser.add_argument("--min-assay-size", type=int, default=20)
    args = parser.parse_args()

    endpoint = args.endpoint.upper()
    y_col = f"p{endpoint}"

    clean = pd.read_parquet(args.clean).reset_index(drop=True)
    clean["row_id"] = range(len(clean))
    split = pd.read_parquet(args.splits)[["row_id", "S6_transport_pair", "is_transport_candidate"]]
    df = clean.merge(split, on="row_id", how="inner")

    assay_sizes = df.groupby("assay_id").size()
    keep_assays = set(assay_sizes[assay_sizes >= args.min_assay_size].index)
    df = df[df["assay_id"].isin(keep_assays)].copy()
    df = df[df["is_transport_candidate"]].copy()
    df = df.dropna(subset=[y_col, "assay_id", "compound_target_key", "target_chembl_id"])

    rows: list[dict[str, object]] = []
    keep_cols = [
        "row_id",
        "assay_id",
        "assay_chembl_id",
        y_col,
        "compound_target_key",
        "target_chembl_id",
        "target_name",
        "S6_transport_pair",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    for _, group in df[keep_cols].groupby("compound_target_key", sort=False):
        if group["assay_id"].nunique() < 2:
            continue
        group = group.sort_values(["assay_id", "row_id"])
        records = group.to_dict("records")
        for left, right in combinations(records, 2):
            if left["assay_id"] == right["assay_id"]:
                continue
            for src, dst in ((left, right), (right, left)):
                rows.append(
                    {
                        "split": src["S6_transport_pair"],
                        "compound_target_key": src["compound_target_key"],
                        "target_chembl_id": src["target_chembl_id"],
                        "target_name": src.get("target_name"),
                        "src_row_id": src["row_id"],
                        "dst_row_id": dst["row_id"],
                        "assay_src": int(src["assay_id"]),
                        "assay_dst": int(dst["assay_id"]),
                        "assay_src_chembl": src.get("assay_chembl_id"),
                        "assay_dst_chembl": dst.get("assay_chembl_id"),
                        "y_src": float(src[y_col]),
                        "y_dst": float(dst[y_col]),
                    }
                )

    pairs = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "clean": args.clean,
        "splits": args.splits,
        "min_assay_size": args.min_assay_size,
        "n_pairs_directed": int(len(pairs)),
        "n_pairs_undirected": int(len(pairs) // 2),
        "n_compound_target_keys": int(pairs["compound_target_key"].nunique()) if len(pairs) else 0,
        "n_targets": int(pairs["target_chembl_id"].nunique()) if len(pairs) else 0,
        "n_assay_pairs_directed": int(pairs[["assay_src", "assay_dst"]].drop_duplicates().shape[0]) if len(pairs) else 0,
        "by_split": {
            str(split_name): {
                "directed_pairs": int(len(sub)),
                "compound_target_keys": int(sub["compound_target_key"].nunique()),
                "targets": int(sub["target_chembl_id"].nunique()),
                "assay_pairs_directed": int(sub[["assay_src", "assay_dst"]].drop_duplicates().shape[0]),
            }
            for split_name, sub in pairs.groupby("split")
        }
        if len(pairs)
        else {},
        "output": str(output),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
