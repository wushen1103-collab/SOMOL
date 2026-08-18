#!/usr/bin/env python
"""Validation-tuned support-gated transport selector."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLS = [
    "split",
    "compound_target_key",
    "target_chembl_id",
    "assay_src",
    "assay_dst",
    "y_src",
    "y_dst",
]


def rmse(err: pd.Series | np.ndarray) -> float:
    values = np.asarray(err, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


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


def load_predictions(baseline_path: str, prediction_dirs: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    base = pd.read_parquet(baseline_path).reset_index(drop=True)
    methods = {col.replace("pred_", ""): col for col in base.columns if col.startswith("pred_")}
    for pred_dir in [Path(p) for p in prediction_dirs]:
        if not pred_dir.exists():
            continue
        paths = sorted(pred_dir.glob("*_predictions.parquet"))
        if not paths:
            paths = sorted(pred_dir.glob("*predictions.parquet"))
        for path in paths:
            frame = pd.read_parquet(path).reset_index(drop=True)
            if len(frame) != len(base):
                raise ValueError(f"Prediction length mismatch for {path}: {len(frame)} != {len(base)}")
            for col in [c for c in frame.columns if c.startswith("pred_")]:
                base[col] = frame[col].to_numpy()
                methods[col.replace("pred_", "")] = col
    return base, methods


def metric_rows(frame: pd.DataFrame, pred_col: str, method: str, split: str) -> list[dict[str, object]]:
    sub = frame[frame["split"] == split].copy()
    sub["err"] = sub[pred_col] - sub["y_dst"]
    rows = [
        {
            "method": method,
            "split": split,
            "scope": "micro",
            "bucket": "all",
            "n": int(len(sub)),
            "groups": 1,
            "mae": float(sub["err"].abs().mean()),
            "rmse": rmse(sub["err"]),
            "bias": float(sub["err"].mean()),
        }
    ]
    for bucket, bucket_df in sub.groupby("pair_support_bucket"):
        rows.append(
            {
                "method": method,
                "split": split,
                "scope": "micro_by_support",
                "bucket": bucket,
                "n": int(len(bucket_df)),
                "groups": 1,
                "mae": float(bucket_df["err"].abs().mean()),
                "rmse": rmse(bucket_df["err"]),
                "bias": float(bucket_df["err"].mean()),
            }
        )
    for scope, cols in {"macro_assay_pair": ["assay_src", "assay_dst"], "macro_target": ["target_chembl_id"]}.items():
        grouped = (
            sub.groupby(cols)
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
                "split": split,
                "scope": scope,
                "bucket": "all",
                "n": int(grouped["n"].sum()),
                "groups": int(len(grouped)),
                "mae": float(grouped["mae"].mean()),
                "rmse": float(grouped["rmse"].mean()),
                "bias": float(grouped["bias"].mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--prediction-dir", action="append", default=[])
    parser.add_argument("--output-dir", default="results/support_gated_selector")
    parser.add_argument("--candidate-method", action="append", default=[])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, methods = load_predictions(args.baseline_predictions, args.prediction_dir)
    if args.candidate_method:
        missing = sorted(set(args.candidate_method).difference(methods))
        if missing:
            raise ValueError(f"Missing candidate methods: {missing}")
        methods = {name: methods[name] for name in args.candidate_method}

    train = frame[frame["split"] == "train"]
    pair_support = train.groupby(["assay_src", "assay_dst"]).size().to_dict()
    frame["pair_train_support"] = [
        int(pair_support.get((a, b), 0)) for a, b in zip(frame["assay_src"], frame["assay_dst"])
    ]
    frame["pair_support_bucket"] = frame["pair_train_support"].map(support_bucket)

    val = frame[frame["split"] == "val"].copy()
    choices: dict[str, dict[str, object]] = {}
    for bucket, sub in val.groupby("pair_support_bucket"):
        scores = []
        for method, col in methods.items():
            mae = float(np.mean(np.abs(sub[col].to_numpy(float) - sub["y_dst"].to_numpy(float))))
            scores.append((mae, method, col))
        best_mae, best_method, best_col = min(scores)
        choices[str(bucket)] = {"method": best_method, "column": best_col, "val_micro_mae": best_mae}

    global_scores = []
    for method, col in methods.items():
        mae = float(np.mean(np.abs(val[col].to_numpy(float) - val["y_dst"].to_numpy(float))))
        global_scores.append((mae, method, col))
    global_best = min(global_scores)

    pred = np.empty(len(frame), dtype=float)
    selected_method = np.empty(len(frame), dtype=object)
    for bucket, sub_idx in frame.groupby("pair_support_bucket").groups.items():
        choice = choices.get(
            str(bucket),
            {"method": global_best[1], "column": global_best[2], "val_micro_mae": global_best[0]},
        )
        idx = np.asarray(list(sub_idx), dtype=int)
        pred[idx] = frame.iloc[idx][choice["column"]].to_numpy(float)
        selected_method[idx] = choice["method"]
    frame["pred_support_gated_val_micro"] = pred
    frame["support_gated_selected_method"] = selected_method

    metrics = []
    for split in ("val", "test"):
        metrics.extend(metric_rows(frame, "pred_support_gated_val_micro", "support_gated_val_micro", split))
    metric_df = pd.DataFrame(metrics).sort_values(["split", "scope", "bucket"]).reset_index(drop=True)

    pred_cols = KEY_COLS + [
        "pair_train_support",
        "pair_support_bucket",
        "support_gated_selected_method",
        "pred_support_gated_val_micro",
    ]
    frame[pred_cols].to_parquet(out_dir / "support_gated_predictions.parquet", index=False)
    metric_df.to_csv(out_dir / "support_gated_metrics.csv", index=False)
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "validation_micro_mae_within_pair_support_bucket",
        "candidate_methods": list(methods.keys()),
        "global_best_val": {"method": global_best[1], "val_micro_mae": global_best[0]},
        "choices": choices,
    }
    (out_dir / "support_gated_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
