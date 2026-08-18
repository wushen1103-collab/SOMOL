#!/usr/bin/env python
"""Validation-tolerant guarded stack for the final SOMOL predictor."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLS = [
    "split",
    "compound_target_key",
    "target_chembl_id",
    "target_name",
    "assay_src",
    "assay_dst",
    "y_src",
    "y_dst",
]


STACK_COLS = {
    "micro": "pred_somol_stack_micro",
    "macro_assay_pair": "pred_somol_stack_macro_assay_pair",
    "macro_target": "pred_somol_stack_macro_target",
}


PAIR_COL = "pred_pair_affine_bias_fallback"


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else math.nan


def weight_grid(step: float) -> list[float]:
    n = int(round(1.0 / step))
    return [i / n for i in range(n + 1)]


def objective(frame: pd.DataFrame, pred: np.ndarray, mask: np.ndarray, scope: str) -> float:
    sub = frame.loc[mask, ["target_chembl_id", "assay_src", "assay_dst", "y_dst"]].copy()
    sub["abs_err"] = np.abs(pred[mask] - sub["y_dst"].to_numpy(float))
    if scope == "micro":
        return float(sub["abs_err"].mean())
    if scope == "macro_assay_pair":
        return float(sub.groupby(["assay_src", "assay_dst"])["abs_err"].mean().mean())
    if scope == "macro_target":
        return float(sub.groupby("target_chembl_id")["abs_err"].mean().mean())
    raise ValueError(f"Unknown scope: {scope}")


def metric_rows(frame: pd.DataFrame, pred_col: str, method: str, split: str) -> list[dict[str, object]]:
    sub = frame[frame["split"] == split].copy()
    if sub.empty:
        return []
    err = sub[pred_col].to_numpy(float) - sub["y_dst"].to_numpy(float)
    rows = [
        {
            "method": method,
            "split": split,
            "scope": "micro",
            "bucket": "all",
            "n": int(len(sub)),
            "groups": 1,
            "mae": float(np.mean(np.abs(err))),
            "rmse": rmse(err),
            "bias": float(np.mean(err)),
        }
    ]
    work = sub[["target_chembl_id", "assay_src", "assay_dst"]].copy()
    work["err"] = err
    for scope, cols in {"macro_assay_pair": ["assay_src", "assay_dst"], "macro_target": ["target_chembl_id"]}.items():
        grouped = (
            work.groupby(cols)
            .agg(
                mae=("err", lambda x: float(np.mean(np.abs(x)))),
                rmse=("err", lambda x: rmse(x.to_numpy(float))),
                bias=("err", "mean"),
                n=("err", "size"),
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
    parser.add_argument("--stack-predictions", required=True)
    parser.add_argument("--output-dir", default="results/somol_guarded_stack")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--relative-tolerance", type=float, default=0.005)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_parquet(args.baseline_predictions).reset_index(drop=True)
    frame = pd.read_parquet(args.stack_predictions).reset_index(drop=True)
    if len(baseline) != len(frame):
        raise ValueError(f"Prediction length mismatch: {len(baseline)}, {len(frame)}")
    if PAIR_COL not in baseline.columns:
        raise ValueError(f"Missing baseline prediction column: {PAIR_COL}")
    frame[PAIR_COL] = baseline[PAIR_COL].to_numpy(float)
    for col in ["pair_train_support", "pair_support_bucket", "min_node_degree", "node_degree_bucket", "graph_bucket"]:
        if col in baseline.columns and col not in frame.columns:
            frame[col] = baseline[col].to_numpy()

    weights = weight_grid(args.step)
    val_mask = frame["split"].to_numpy() == "val"
    pair_pred = frame[PAIR_COL].to_numpy(float)
    diagnostics: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "validation-tolerant conservative blend of SOMOL stack and pair-affine bias fallback",
        "pair_column": PAIR_COL,
        "step": args.step,
        "relative_tolerance": args.relative_tolerance,
        "selection_rule": "choose the smallest stack weight whose validation objective is within relative_tolerance of the best candidate",
        "selectors": {},
    }

    method_cols: dict[str, str] = {}
    for scope, stack_col in STACK_COLS.items():
        if stack_col not in frame.columns:
            raise ValueError(f"Missing stack prediction column: {stack_col}")
        stack_pred = frame[stack_col].to_numpy(float)
        candidates = []
        for alpha in weights:
            pred = alpha * stack_pred + (1.0 - alpha) * pair_pred
            candidates.append(
                {
                    "stack_weight": float(alpha),
                    "val_score": objective(frame, pred, val_mask, scope),
                    "prediction": pred,
                }
            )
        best_score = min(c["val_score"] for c in candidates)
        threshold = best_score * (1.0 + args.relative_tolerance)
        feasible = [c for c in candidates if c["val_score"] <= threshold + 1e-12]
        selected = min(feasible, key=lambda c: (c["stack_weight"], c["val_score"]))
        method = f"somol_guarded_stack_{scope}"
        col = f"pred_{method}"
        frame[col] = selected["prediction"]
        method_cols[method] = col
        diagnostics["selectors"][scope] = {
            "stack_column": stack_col,
            "selected_stack_weight": float(selected["stack_weight"]),
            "selected_pair_weight": float(1.0 - selected["stack_weight"]),
            "best_val_score": float(best_score),
            "selected_val_score": float(selected["val_score"]),
            "threshold": float(threshold),
        }

    frame["pred_somol_guarded_stack"] = frame["pred_somol_guarded_stack_micro"].to_numpy()
    method_cols["somol_guarded_stack"] = "pred_somol_guarded_stack"

    rows = []
    for method, col in method_cols.items():
        for split in ["val", "test"]:
            rows.extend(metric_rows(frame, col, method, split))
    metrics = pd.DataFrame(rows).sort_values(["split", "scope", "mae", "method"]).reset_index(drop=True)

    keep_cols = [c for c in KEY_COLS if c in frame.columns]
    extra_cols = ["pair_train_support", "pair_support_bucket", "min_node_degree", "node_degree_bucket", "graph_bucket"]
    pred_cols = keep_cols + [c for c in extra_cols if c in frame.columns] + list(method_cols.values())
    frame[pred_cols].to_parquet(out_dir / "somol_guarded_stack_predictions.parquet", index=False)
    metrics.to_csv(out_dir / "somol_guarded_stack_metrics.csv", index=False)
    (out_dir / "somol_guarded_stack_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
