#!/usr/bin/env python
"""Validation-only robust SOMOL stack for measurement-method baselines."""

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


def weight_grid(step: float) -> list[float]:
    n = int(round(1.0 / step))
    return [i / n for i in range(n + 1)]


def micro_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true))) if len(y_true) else math.nan


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


def select_global(frame: pd.DataFrame, candidates: dict[str, np.ndarray]) -> tuple[str, np.ndarray, dict[str, float]]:
    val_mask = frame["split"].to_numpy() == "val"
    y_val = frame.loc[val_mask, "y_dst"].to_numpy(float)
    scores = {name: micro_mae(y_val, pred[val_mask]) for name, pred in candidates.items()}
    selected = min(scores, key=lambda name: (scores[name], name))
    return selected, candidates[selected], scores


def blend_global(frame: pd.DataFrame, somol: np.ndarray, robust: np.ndarray, weights: list[float]) -> tuple[float, np.ndarray, float]:
    val_mask = frame["split"].to_numpy() == "val"
    y_val = frame.loc[val_mask, "y_dst"].to_numpy(float)
    best = (math.inf, 0.0, robust)
    for alpha in weights:
        pred = alpha * somol + (1.0 - alpha) * robust
        score = micro_mae(y_val, pred[val_mask])
        if score < best[0] - 1e-12:
            best = (score, float(alpha), pred)
    return best[1], best[2], best[0]


def select_by_bucket(frame: pd.DataFrame, candidates: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, str]]:
    global_name, _, _ = select_global(frame, candidates)
    selected_pred = np.empty(len(frame), dtype=float)
    choices: dict[str, str] = {}
    buckets = frame["pair_support_bucket"] if "pair_support_bucket" in frame.columns else pd.Series(["all"] * len(frame))
    for bucket, idx in frame.groupby(buckets).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        val_idx = idx_arr[frame.iloc[idx_arr]["split"].to_numpy() == "val"]
        if len(val_idx) == 0:
            best_name = global_name
        else:
            y_val = frame.loc[val_idx, "y_dst"].to_numpy(float)
            best_name = min(candidates, key=lambda name: (micro_mae(y_val, candidates[name][val_idx]), name))
        selected_pred[idx_arr] = candidates[best_name][idx_arr]
        choices[str(bucket)] = best_name
    return selected_pred, choices


def blend_by_bucket(frame: pd.DataFrame, somol: np.ndarray, robust: np.ndarray, weights: list[float]) -> tuple[np.ndarray, dict[str, float]]:
    global_alpha, _, _ = blend_global(frame, somol, robust, weights)
    selected_pred = np.empty(len(frame), dtype=float)
    choices: dict[str, float] = {}
    buckets = frame["pair_support_bucket"] if "pair_support_bucket" in frame.columns else pd.Series(["all"] * len(frame))
    for bucket, idx in frame.groupby(buckets).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        val_idx = idx_arr[frame.iloc[idx_arr]["split"].to_numpy() == "val"]
        if len(val_idx) == 0:
            alpha = global_alpha
        else:
            y_val = frame.loc[val_idx, "y_dst"].to_numpy(float)
            best_score = math.inf
            alpha = 0.0
            for candidate_alpha in weights:
                pred = candidate_alpha * somol[val_idx] + (1.0 - candidate_alpha) * robust[val_idx]
                score = micro_mae(y_val, pred)
                if score < best_score - 1e-12:
                    best_score = score
                    alpha = float(candidate_alpha)
        selected_pred[idx_arr] = alpha * somol[idx_arr] + (1.0 - alpha) * robust[idx_arr]
        choices[str(bucket)] = alpha
    return selected_pred, choices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--guarded-predictions", required=True)
    parser.add_argument("--method-comparison-predictions", required=True)
    parser.add_argument("--output-dir", default="results/somol_robust_stack")
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_parquet(args.baseline_predictions).reset_index(drop=True)
    frame = pd.read_parquet(args.guarded_predictions).reset_index(drop=True)
    method = pd.read_parquet(args.method_comparison_predictions).reset_index(drop=True)
    if not (len(baseline) == len(frame) == len(method)):
        raise ValueError(f"Prediction length mismatch: {len(baseline)}, {len(frame)}, {len(method)}")

    robust_col = "pred_robust_trimmed_bias_fallback"
    somol_col = "pred_somol_guarded_stack_micro"
    if robust_col not in method.columns:
        raise ValueError(f"Missing method-comparison column: {robust_col}")
    if somol_col not in frame.columns:
        raise ValueError(f"Missing guarded-stack column: {somol_col}")

    frame[robust_col] = method[robust_col].to_numpy(float)
    for col in ["pred_deming_bias_fallback", "pred_odr_tls_bias_fallback"]:
        if col in method.columns:
            frame[col] = method[col].to_numpy(float)
    for col in ["pair_train_support", "pair_support_bucket", "min_node_degree", "node_degree_bucket", "graph_bucket"]:
        if col in baseline.columns and col not in frame.columns:
            frame[col] = baseline[col].to_numpy()

    somol = frame[somol_col].to_numpy(float)
    robust = frame[robust_col].to_numpy(float)
    candidates = {"somol_guarded_stack_micro": somol, "robust_trimmed_bias_fallback": robust}
    weights = weight_grid(args.step)

    global_name, global_pred, global_scores = select_global(frame, candidates)
    global_alpha, global_blend_pred, global_blend_score = blend_global(frame, somol, robust, weights)
    bucket_pred, bucket_choices = select_by_bucket(frame, candidates)
    bucket_blend_pred, bucket_blend_alphas = blend_by_bucket(frame, somol, robust, weights)

    method_cols = {
        "somol_robust_selector_global": "pred_somol_robust_selector_global",
        "somol_robust_blend_global": "pred_somol_robust_blend_global",
        "somol_robust_selector_bucket": "pred_somol_robust_selector_bucket",
        "somol_robust_blend_bucket": "pred_somol_robust_blend_bucket",
    }
    frame[method_cols["somol_robust_selector_global"]] = global_pred
    frame[method_cols["somol_robust_blend_global"]] = global_blend_pred
    frame[method_cols["somol_robust_selector_bucket"]] = bucket_pred
    frame[method_cols["somol_robust_blend_bucket"]] = bucket_blend_pred

    rows = []
    for method_name, pred_col in method_cols.items():
        for split in ["val", "test"]:
            rows.extend(metric_rows(frame, pred_col, method_name, split))
    metrics = pd.DataFrame(rows).sort_values(["split", "scope", "mae", "method"]).reset_index(drop=True)

    keep_cols = [c for c in KEY_COLS if c in frame.columns]
    extra_cols = ["pair_train_support", "pair_support_bucket", "min_node_degree", "node_degree_bucket", "graph_bucket"]
    pred_cols = keep_cols + [c for c in extra_cols if c in frame.columns] + [robust_col] + list(method_cols.values())
    frame[pred_cols].to_parquet(out_dir / "somol_robust_stack_predictions.parquet", index=False)
    metrics.to_csv(out_dir / "somol_robust_stack_metrics.csv", index=False)

    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "validation-only robust SOMOL stack combining the existing guarded SOMOL stack with robust trimmed measurement-method calibration",
        "somol_column": somol_col,
        "robust_column": robust_col,
        "step": args.step,
        "global_selector": {"selected": global_name, "val_micro_mae": global_scores},
        "global_blend": {"somol_weight": global_alpha, "robust_weight": 1.0 - global_alpha, "val_micro_mae": global_blend_score},
        "bucket_selector": bucket_choices,
        "bucket_blend_somol_weights": bucket_blend_alphas,
    }
    (out_dir / "somol_robust_stack_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
