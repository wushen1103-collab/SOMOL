#!/usr/bin/env python
"""Tune calibration candidates augmented with a shared assay gauge-bias operator."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from tune_calibration_grid import (
    KEY_COLS,
    build_candidates,
    metric_rows,
    objective_score,
    rmse,
    select_by_bucket,
    support_bucket,
)


def safe_float_name(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def fit_gauge_candidates(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    lambdas: list[float],
    alphas: list[float],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    assays = pd.Index(pd.unique(pd.concat([frame["assay_src"], frame["assay_dst"]], ignore_index=True))).astype(int)
    assay_to_idx = {int(a): i for i, a in enumerate(assays)}
    src = train["assay_src"].map(assay_to_idx).to_numpy()
    dst = train["assay_dst"].map(assay_to_idx).to_numpy()
    n_assays = len(assays)
    n_train = len(train)
    y_delta = (train["y_dst"] - train["y_src"]).to_numpy(float)

    rows = np.arange(n_train)
    design = sparse.coo_matrix(
        (
            np.r_[-np.ones(n_train), np.ones(n_train)],
            (np.r_[rows, rows], np.r_[src, dst]),
        ),
        shape=(n_train, n_assays),
    ).tocsr()
    src_all = frame["assay_src"].map(assay_to_idx).to_numpy()
    dst_all = frame["assay_dst"].map(assay_to_idx).to_numpy()
    y_src = frame["y_src"].to_numpy(float)
    identity_bias = frame["pred_identity_bias"].to_numpy(float)

    candidates: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {
        "n_assays": int(n_assays),
        "n_train_edges": int(n_train),
        "lambdas": lambdas,
        "alphas": alphas,
        "models": {},
    }
    for ridge in lambdas:
        augmented = sparse.vstack([design, sparse.eye(n_assays) * np.sqrt(ridge)], format="csr")
        rhs = np.r_[y_delta, np.zeros(n_assays)]
        result = lsqr(augmented, rhs, atol=1e-8, btol=1e-8, iter_lim=2000)
        bias = result[0]
        bias = bias - bias.mean()
        gauge_pred = y_src + bias[dst_all] - bias[src_all]
        ridge_name = safe_float_name(ridge)
        candidates[f"gauge_lam{ridge_name}"] = gauge_pred
        diagnostics["models"][f"gauge_lam{ridge_name}"] = {
            "ridge": float(ridge),
            "lsqr_istop": int(result[1]),
            "lsqr_iterations": int(result[2]),
            "bias_std": float(np.std(bias)),
            "bias_min": float(np.min(bias)),
            "bias_max": float(np.max(bias)),
        }
        for alpha in alphas:
            alpha_name = safe_float_name(alpha)
            candidates[f"gauge_lam{ridge_name}_blendID_{alpha_name}"] = (
                float(alpha) * gauge_pred + (1.0 - float(alpha)) * identity_bias
            )
    return candidates, diagnostics


def score_candidates(frame: pd.DataFrame, candidates: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for split in ("val", "test"):
        idx = np.flatnonzero(frame["split"].to_numpy() == split)
        y = frame.iloc[idx]["y_dst"].to_numpy(float)
        for name, pred in candidates.items():
            err = pred[idx] - y
            tmp = frame.iloc[idx][["target_chembl_id", "assay_src", "assay_dst"]].copy()
            tmp["abs_err"] = np.abs(err)
            rows.append(
                {
                    "candidate": name,
                    "split": split,
                    "n": int(len(idx)),
                    "micro_mae": float(np.mean(np.abs(err))),
                    "micro_rmse": rmse(err),
                    "macro_assay_pair_mae": float(tmp.groupby(["assay_src", "assay_dst"])["abs_err"].mean().mean()),
                    "macro_target_mae": float(tmp.groupby("target_chembl_id")["abs_err"].mean().mean()),
                    "bias": float(np.mean(err)),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "micro_mae", "candidate"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--output-dir", default="results/gauge_calibration")
    parser.add_argument("--min-pairs-grid", default="2,3,5,10,20,30")
    parser.add_argument("--shrink-grid", default="1,2,5,10,20,50,100")
    parser.add_argument("--gauge-lambdas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--gauge-alphas", default="0.1,0.25,0.5,0.75,0.9")
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

    min_pairs_grid = [int(x) for x in args.min_pairs_grid.split(",") if x]
    shrink_grid = [float(x) for x in args.shrink_grid.split(",") if x]
    gauge_lambdas = [float(x) for x in args.gauge_lambdas.split(",") if x]
    gauge_alphas = [float(x) for x in args.gauge_alphas.split(",") if x]

    calibration_candidates, calibration_info = build_candidates(frame, train, min_pairs_grid, shrink_grid)
    gauge_candidates, gauge_info = fit_gauge_candidates(frame, train, gauge_lambdas, gauge_alphas)
    candidates = {**calibration_candidates, **gauge_candidates}
    candidate_scores = score_candidates(frame, candidates)
    candidate_scores.to_csv(out_dir / "gauge_calibration_candidate_scores.csv", index=False)

    val_idx = np.flatnonzero(frame["split"].to_numpy() == "val")
    selection_scopes = ["micro", "macro_assay_pair", "macro_target"]
    diagnostics_selectors: dict[str, dict[str, object]] = {}
    method_cols: dict[str, str] = {}
    for scope in selection_scopes:
        global_best_score, global_best_name = min(
            (objective_score(frame, val_idx, pred, scope), name) for name, pred in candidates.items()
        )
        selected, selected_name, choices = select_by_bucket(
            frame, candidates, scope, global_best_name, global_best_score
        )
        col = f"pred_gauge_calibration_selector_{scope}"
        frame[col] = selected
        frame[f"gauge_calibration_selected_candidate_{scope}"] = selected_name
        method_cols[f"gauge_calibration_selector_{scope}"] = col
        diagnostics_selectors[scope] = {
            "global_best": {"candidate": global_best_name, f"val_{scope}": global_best_score},
            "bucket_choices": choices,
        }

    frame["pred_gauge_calibration_selector"] = frame["pred_gauge_calibration_selector_micro"].to_numpy()
    frame["gauge_calibration_selected_candidate"] = frame["gauge_calibration_selected_candidate_micro"].to_numpy()
    method_cols["gauge_calibration_selector"] = "pred_gauge_calibration_selector"

    metrics = []
    for method, col in method_cols.items():
        for split in ("val", "test"):
            metrics.extend(metric_rows(frame, col, method, split))
    metric_df = pd.DataFrame(metrics).sort_values(["split", "scope", "bucket", "mae", "method"]).reset_index(drop=True)
    metric_df.to_csv(out_dir / "gauge_calibration_metrics.csv", index=False)

    pred_cols = KEY_COLS + [
        "pair_train_support",
        "pair_support_bucket",
        "gauge_calibration_selected_candidate",
        "pred_gauge_calibration_selector",
        "pred_gauge_calibration_selector_micro",
        "pred_gauge_calibration_selector_macro_assay_pair",
        "pred_gauge_calibration_selector_macro_target",
    ]
    frame[pred_cols].to_parquet(out_dir / "gauge_calibration_predictions.parquet", index=False)
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "validation_bucket_score",
        "selectors": diagnostics_selectors,
        "calibration_candidate_info": calibration_info,
        "gauge_info": gauge_info,
        "n_candidates": int(len(candidates)),
        "support_buckets": frame[frame["split"] == "test"]["pair_support_bucket"].value_counts().sort_index().astype(int).to_dict(),
    }
    (out_dir / "gauge_calibration_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
