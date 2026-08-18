#!/usr/bin/env python
"""Validation-tuned blend of SOMOL gauge and graph-gauge predictions."""

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


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else math.nan


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
    parser.add_argument("--gauge-predictions", required=True)
    parser.add_argument("--graph-predictions", required=True)
    parser.add_argument("--output-dir", default="results/somol_blend")
    parser.add_argument("--alpha-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gauge = pd.read_parquet(args.gauge_predictions).reset_index(drop=True)
    graph = pd.read_parquet(args.graph_predictions).reset_index(drop=True)
    if len(gauge) != len(graph):
        raise ValueError(f"Prediction length mismatch: {len(gauge)} != {len(graph)}")
    frame = gauge.copy()
    for col in graph.columns:
        if col.startswith("pred_graph_") or col in {"min_node_degree", "node_degree_bucket", "graph_bucket"}:
            frame[col] = graph[col].to_numpy()

    gauge_col = "pred_gauge_calibration_selector_micro"
    graph_col = "pred_graph_gauge_calibration_selector"
    if gauge_col not in frame.columns or graph_col not in frame.columns:
        raise ValueError(f"Required columns missing: {gauge_col}, {graph_col}")

    alphas = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    val_mask = frame["split"].to_numpy() == "val"
    gauge_pred = frame[gauge_col].to_numpy(float)
    graph_pred = frame[graph_col].to_numpy(float)

    diagnostics: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "pred = alpha * gauge_micro + (1 - alpha) * graph_gauge_selector; alpha selected on validation only",
        "alpha_grid": alphas,
        "selectors": {},
    }
    method_cols: dict[str, str] = {}
    for scope in ["micro", "macro_assay_pair", "macro_target"]:
        scored = []
        for alpha in alphas:
            pred = alpha * gauge_pred + (1.0 - alpha) * graph_pred
            scored.append((objective(frame, pred, val_mask, scope), alpha, pred))
        best_score, best_alpha, best_pred = min(scored, key=lambda x: (x[0], x[1]))
        col = f"pred_somol_gauge_graph_blend_{scope}"
        frame[col] = best_pred
        method = f"somol_gauge_graph_blend_{scope}"
        method_cols[method] = col
        diagnostics["selectors"][scope] = {
            "selected_alpha": float(best_alpha),
            f"val_{scope}": float(best_score),
            "grid": [{"alpha": float(alpha), f"val_{scope}": float(score)} for score, alpha, _ in scored],
        }
    frame["pred_somol_gauge_graph_blend"] = frame["pred_somol_gauge_graph_blend_micro"].to_numpy()
    method_cols["somol_gauge_graph_blend"] = "pred_somol_gauge_graph_blend"

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
    frame[pred_cols].to_parquet(out_dir / "somol_blend_predictions.parquet", index=False)
    metrics.to_csv(out_dir / "somol_blend_metrics.csv", index=False)
    (out_dir / "somol_blend_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
