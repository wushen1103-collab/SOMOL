#!/usr/bin/env python
"""Compare transport methods by support strata and evaluation scope."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def rmse(err: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(err.to_numpy(float)))))


def support_bucket(n: int) -> str:
    if n <= 0:
        return "00_zero"
    if n < 5:
        return "01_1-4"
    if n < 10:
        return "02_5-9"
    if n < 30:
        return "03_10-29"
    return "04_30plus"


def metric_rows(df: pd.DataFrame, methods: dict[str, str], group_name: str, group_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, col in methods.items():
        work = df.copy()
        work["err"] = work[col] - work["y_dst"]
        for bucket, sub in work.groupby(group_col):
            rows.append(
                {
                    "method": method,
                    "grouping": group_name,
                    "bucket": bucket,
                    "scope": "micro",
                    "n": int(len(sub)),
                    "groups": 1,
                    "mae": float(sub["err"].abs().mean()),
                    "rmse": rmse(sub["err"]),
                    "bias": float(sub["err"].mean()),
                }
            )
            grouped = (
                sub.groupby(["assay_src", "assay_dst"])
                .apply(
                    lambda g: pd.Series(
                        {
                            "mae": g["err"].abs().mean(),
                            "rmse": rmse(g["err"]),
                            "bias": g["err"].mean(),
                            "n": len(g),
                        }
                    ),
                    include_groups=False,
                )
                .reset_index()
            )
            rows.append(
                {
                    "method": method,
                    "grouping": group_name,
                    "bucket": bucket,
                    "scope": "macro_assay_pair",
                    "n": int(grouped["n"].sum()) if len(grouped) else 0,
                    "groups": int(len(grouped)),
                    "mae": float(grouped["mae"].mean()) if len(grouped) else np.nan,
                    "rmse": float(grouped["rmse"].mean()) if len(grouped) else np.nan,
                    "bias": float(grouped["bias"].mean()) if len(grouped) else np.nan,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--operator-dir", required=True)
    parser.add_argument("--extra-prediction-dir", action="append", default=[])
    parser.add_argument("--output-dir", default="results/transport_compare")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(args.baseline_predictions).reset_index(drop=True)
    methods = {col.replace("pred_", ""): col for col in base.columns if col.startswith("pred_")}

    prediction_dirs = [Path(args.operator_dir)] + [Path(p) for p in args.extra_prediction_dir]
    for pred_dir in prediction_dirs:
        if not pred_dir.exists():
            continue
        prediction_paths = sorted(pred_dir.glob("*_predictions.parquet"))
        if not prediction_paths:
            prediction_paths = sorted(pred_dir.glob("*predictions.parquet"))
        for path in prediction_paths:
            op = pd.read_parquet(path).reset_index(drop=True)
            pred_cols = [c for c in op.columns if c.startswith("pred_")]
            for col in pred_cols:
                if len(op) != len(base):
                    raise ValueError(f"Prediction length mismatch for {path}")
                base[col] = op[col].to_numpy()
                methods[col.replace("pred_", "")] = col

    train = base[base["split"] == "train"].copy()
    pair_support = train.groupby(["assay_src", "assay_dst"]).size().to_dict()
    target_pair_support = train.groupby(["target_chembl_id", "assay_src", "assay_dst"]).size().to_dict()

    test = base[base["split"] == "test"].copy()
    test["pair_train_support"] = [int(pair_support.get((a, b), 0)) for a, b in zip(test["assay_src"], test["assay_dst"])]
    test["target_pair_train_support"] = [
        int(target_pair_support.get((t, a, b), 0))
        for t, a, b in zip(test["target_chembl_id"], test["assay_src"], test["assay_dst"])
    ]
    test["pair_support_bucket"] = test["pair_train_support"].map(support_bucket)
    test["target_pair_support_bucket"] = test["target_pair_train_support"].map(support_bucket)

    rows = []
    rows.extend(metric_rows(test, methods, "assay_pair_train_support", "pair_support_bucket"))
    rows.extend(metric_rows(test, methods, "target_assay_pair_train_support", "target_pair_support_bucket"))
    metrics = pd.DataFrame(rows).sort_values(["grouping", "bucket", "scope", "mae", "method"]).reset_index(drop=True)
    metrics.to_csv(out_dir / "support_stratified_metrics.csv", index=False)
    support_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_test": int(len(test)),
        "pair_support_buckets": test["pair_support_bucket"].value_counts().sort_index().astype(int).to_dict(),
        "target_pair_support_buckets": test["target_pair_support_bucket"].value_counts().sort_index().astype(int).to_dict(),
        "methods": list(methods.keys()),
    }
    (out_dir / "support_stratified_summary.json").write_text(json.dumps(support_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(support_summary, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
