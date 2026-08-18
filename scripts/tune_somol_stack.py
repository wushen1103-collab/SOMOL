#!/usr/bin/env python
"""Validation-tuned convex stack for the final SOMOL predictor."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from itertools import product
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


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else math.nan


def simplex_grid(step: float) -> list[tuple[float, float, float, float]]:
    n = int(round(1.0 / step))
    weights: list[tuple[float, float, float, float]] = []
    for a, b, c in product(range(n + 1), repeat=3):
        if a + b + c > n:
            continue
        d = n - a - b - c
        weights.append((a / n, b / n, c / n, d / n))
    return weights


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
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--gauge-predictions", required=True)
    parser.add_argument("--graph-predictions", required=True)
    parser.add_argument("--output-dir", default="results/somol_stack")
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = pd.read_parquet(args.calibration_predictions).reset_index(drop=True)
    gauge = pd.read_parquet(args.gauge_predictions).reset_index(drop=True)
    graph = pd.read_parquet(args.graph_predictions).reset_index(drop=True)
    if not (len(cal) == len(gauge) == len(graph)):
        raise ValueError(f"Prediction length mismatch: {len(cal)}, {len(gauge)}, {len(graph)}")
    frame = gauge.copy()
    for col in cal.columns:
        if col.startswith("pred_calibration_"):
            frame[col] = cal[col].to_numpy()
    for col in graph.columns:
        if col.startswith("pred_graph_") or col in {"min_node_degree", "node_degree_bucket", "graph_bucket"}:
            frame[col] = graph[col].to_numpy()

    base_cols = [
        "pred_calibration_grid_selector_macro_assay_pair",
        "pred_gauge_calibration_selector_micro",
        "pred_gauge_calibration_selector_macro_assay_pair",
        "pred_graph_gauge_calibration_selector",
    ]
    missing = [col for col in base_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing base prediction columns: {missing}")
    base_preds = [frame[col].to_numpy(float) for col in base_cols]
    weights = simplex_grid(args.step)
    val_mask = frame["split"].to_numpy() == "val"

    diagnostics: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "convex stack of calibration macro-pair, gauge micro, gauge macro-pair, graph-gauge selector; weights selected on validation only",
        "base_columns": base_cols,
        "step": args.step,
        "n_weight_candidates": len(weights),
        "selectors": {},
    }
    method_cols: dict[str, str] = {}
    for scope in ["micro", "macro_assay_pair", "macro_target"]:
        scored = []
        for weight in weights:
            pred = sum(w * p for w, p in zip(weight, base_preds))
            scored.append((objective(frame, pred, val_mask, scope), weight, pred))
        best_score, best_weight, best_pred = min(scored, key=lambda x: (x[0], x[1]))
        col = f"pred_somol_stack_{scope}"
        method = f"somol_stack_{scope}"
        frame[col] = best_pred
        method_cols[method] = col
        diagnostics["selectors"][scope] = {
            "selected_weights": {name: float(w) for name, w in zip(base_cols, best_weight)},
            f"val_{scope}": float(best_score),
        }
    frame["pred_somol_stack"] = frame["pred_somol_stack_micro"].to_numpy()
    method_cols["somol_stack"] = "pred_somol_stack"

    rows = []
    for method, col in method_cols.items():
        for split in ["val", "test"]:
            rows.extend(metric_rows(frame, col, method, split))
    metrics = pd.DataFrame(rows).sort_values(["split", "scope", "mae", "method"]).reset_index(drop=True)
    keep_cols = [c for c in KEY_COLS if c in frame.columns]
    extra_cols = [
        "pair_train_support",
        "pair_support_bucket",
        "min_node_degree",
        "node_degree_bucket",
        "graph_bucket",
    ]
    pred_cols = keep_cols + [c for c in extra_cols if c in frame.columns] + list(method_cols.values())
    frame[pred_cols].to_parquet(out_dir / "somol_stack_predictions.parquet", index=False)
    metrics.to_csv(out_dir / "somol_stack_metrics.csv", index=False)
    (out_dir / "somol_stack_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
