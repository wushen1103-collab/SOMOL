#!/usr/bin/env python
"""Tune gauge-calibration candidates by pair-support and assay-node-degree buckets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tune_calibration_grid import (
    KEY_COLS,
    build_candidates,
    metric_rows,
    objective_score,
    support_bucket,
)
from tune_gauge_calibration import fit_gauge_candidates


def degree_bucket(n: int) -> str:
    if n <= 0:
        return "00_node_unseen"
    if n < 10:
        return "01_1-9"
    if n < 100:
        return "02_10-99"
    if n < 1000:
        return "03_100-999"
    return "04_1000plus"


def select_by_column(
    frame: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    bucket_col: str,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    val_idx = np.flatnonzero(frame["split"].to_numpy() == "val")
    global_best_score, global_best_name = min(
        (objective_score(frame, val_idx, pred, scope), name) for name, pred in candidates.items()
    )
    selected = np.empty(len(frame), dtype=float)
    selected_name = np.empty(len(frame), dtype=object)
    choices: dict[str, dict[str, object]] = {}
    for bucket, idx in frame.groupby(bucket_col).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        bucket_val_idx = idx_arr[frame.iloc[idx_arr]["split"].to_numpy() == "val"]
        if len(bucket_val_idx) == 0:
            best_score, best_name = global_best_score, global_best_name
        else:
            best_score, best_name = min(
                (objective_score(frame, bucket_val_idx, pred, scope), name) for name, pred in candidates.items()
            )
        selected[idx_arr] = candidates[best_name][idx_arr]
        selected_name[idx_arr] = best_name
        choices[str(bucket)] = {"candidate": best_name, f"val_{scope}": best_score}
    return selected, selected_name, {
        "global_best": {"candidate": global_best_name, f"val_{scope}": global_best_score},
        "bucket_choices": choices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--output-dir", default="results/graph_gauge_calibration")
    parser.add_argument("--min-pairs-grid", default="2,3,5,10,20,30")
    parser.add_argument("--shrink-grid", default="1,2,5,10,20,50,100")
    parser.add_argument("--gauge-lambdas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--gauge-alphas", default="0.1,0.25,0.5,0.75,0.9")
    parser.add_argument("--min-graph-val-count-grid", default="0,100,500,1000")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.baseline_predictions).reset_index(drop=True)
    frame = frame[frame["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    train = frame[frame["split"] == "train"].copy()

    support = train.groupby(["assay_src", "assay_dst"]).size().to_dict()
    frame["pair_train_support"] = [
        int(support.get((a, b), 0)) for a, b in zip(frame["assay_src"], frame["assay_dst"])
    ]
    frame["pair_support_bucket"] = frame["pair_train_support"].map(support_bucket)
    degree = train.groupby("assay_src").size().add(train.groupby("assay_dst").size(), fill_value=0).astype(int).to_dict()
    frame["src_node_degree"] = frame["assay_src"].map(degree).fillna(0).astype(int)
    frame["dst_node_degree"] = frame["assay_dst"].map(degree).fillna(0).astype(int)
    frame["min_node_degree"] = frame[["src_node_degree", "dst_node_degree"]].min(axis=1)
    frame["node_degree_bucket"] = frame["min_node_degree"].map(degree_bucket)
    frame["graph_bucket"] = frame["pair_support_bucket"].astype(str) + "|" + frame["node_degree_bucket"].astype(str)

    min_pairs_grid = [int(x) for x in args.min_pairs_grid.split(",") if x]
    shrink_grid = [float(x) for x in args.shrink_grid.split(",") if x]
    gauge_lambdas = [float(x) for x in args.gauge_lambdas.split(",") if x]
    gauge_alphas = [float(x) for x in args.gauge_alphas.split(",") if x]
    min_graph_val_count_grid = [int(x) for x in args.min_graph_val_count_grid.split(",") if x]
    calibration_candidates, calibration_info = build_candidates(frame, train, min_pairs_grid, shrink_grid)
    gauge_candidates, gauge_info = fit_gauge_candidates(frame, train, gauge_lambdas, gauge_alphas)
    candidates = {**calibration_candidates, **gauge_candidates}

    diagnostics_selectors: dict[str, object] = {}
    method_cols: dict[str, str] = {}
    val_idx = np.flatnonzero(frame["split"].to_numpy() == "val")
    graph_val_counts = frame.loc[val_idx, "graph_bucket"].astype(str).value_counts().to_dict()
    for scope in ["micro", "macro_assay_pair", "macro_target"]:
        pair_selected, pair_selected_name, pair_info = select_by_column(frame, candidates, "pair_support_bucket", scope)
        graph_selected, graph_selected_name, graph_info = select_by_column(frame, candidates, "graph_bucket", scope)
        threshold_records = []
        best_threshold = None
        best_score = float("inf")
        best_selected = None
        best_selected_name = None
        for threshold in min_graph_val_count_grid:
            selected = pair_selected.copy()
            selected_name = pair_selected_name.copy()
            for bucket, idx in frame.groupby("graph_bucket").groups.items():
                if graph_val_counts.get(str(bucket), 0) < threshold:
                    continue
                idx_arr = np.asarray(list(idx), dtype=int)
                selected[idx_arr] = graph_selected[idx_arr]
                selected_name[idx_arr] = graph_selected_name[idx_arr]
            score = objective_score(frame, val_idx, selected, scope)
            threshold_records.append({"min_graph_val_count": int(threshold), f"val_{scope}": float(score)})
            if score < best_score:
                best_threshold = int(threshold)
                best_score = float(score)
                best_selected = selected
                best_selected_name = selected_name
        if best_selected is None or best_selected_name is None:
            raise RuntimeError("No graph threshold candidate was selected.")
        col = f"pred_graph_gauge_calibration_selector_{scope}"
        frame[col] = best_selected
        frame[f"graph_gauge_calibration_selected_candidate_{scope}"] = best_selected_name
        method_cols[f"graph_gauge_calibration_selector_{scope}"] = col
        diagnostics_selectors[scope] = {
            "selected_min_graph_val_count": best_threshold,
            "selected_validation_score": best_score,
            "threshold_grid": threshold_records,
            "pair_support_selector": pair_info,
            "graph_bucket_selector": graph_info,
        }
    frame["pred_graph_gauge_calibration_selector"] = frame[
        "pred_graph_gauge_calibration_selector_macro_target"
    ].to_numpy()
    frame["graph_gauge_calibration_selected_candidate"] = frame[
        "graph_gauge_calibration_selected_candidate_macro_target"
    ].to_numpy()
    method_cols["graph_gauge_calibration_selector"] = "pred_graph_gauge_calibration_selector"

    metrics = []
    for method, col in method_cols.items():
        for split in ("val", "test"):
            metrics.extend(metric_rows(frame, col, method, split))
    metric_df = pd.DataFrame(metrics).sort_values(["split", "scope", "bucket", "mae", "method"]).reset_index(drop=True)
    metric_df.to_csv(out_dir / "graph_gauge_calibration_metrics.csv", index=False)

    pred_cols = KEY_COLS + [
        "pair_train_support",
        "pair_support_bucket",
        "min_node_degree",
        "node_degree_bucket",
        "graph_bucket",
        "graph_gauge_calibration_selected_candidate",
        "pred_graph_gauge_calibration_selector",
        "pred_graph_gauge_calibration_selector_micro",
        "pred_graph_gauge_calibration_selector_macro_assay_pair",
        "pred_graph_gauge_calibration_selector_macro_target",
    ]
    frame[pred_cols].to_parquet(out_dir / "graph_gauge_calibration_predictions.parquet", index=False)
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "validation_graph_bucket_score",
        "selectors": diagnostics_selectors,
        "calibration_candidate_info": calibration_info,
        "gauge_info": gauge_info,
        "n_candidates": int(len(candidates)),
        "min_graph_val_count_grid": min_graph_val_count_grid,
        "graph_bucket_counts_test": frame[frame["split"] == "test"]["graph_bucket"].value_counts().sort_index().astype(int).to_dict(),
    }
    (out_dir / "graph_gauge_calibration_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
