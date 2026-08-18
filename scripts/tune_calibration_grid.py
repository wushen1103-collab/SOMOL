#!/usr/bin/env python
"""Tune affine calibration fallbacks and shrinkage on validation support buckets."""

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
    "assay_src",
    "assay_dst",
    "y_src",
    "y_dst",
]


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.nanstd(x) < 1e-8:
        return float(np.nanmean(y)), 0.0
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else math.nan


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


def metric_rows(frame: pd.DataFrame, pred_col: str, method: str, split: str) -> list[dict[str, object]]:
    sub = frame[frame["split"] == split].copy()
    sub["err"] = sub[pred_col] - sub["y_dst"]
    rows = [
        {
            "method": method,
            "split": split,
            "scope": "micro",
            "n": int(len(sub)),
            "groups": 1,
            "mae": float(sub["err"].abs().mean()),
            "rmse": rmse(sub["err"].to_numpy(float)),
            "bias": float(sub["err"].mean()),
        }
    ]
    for scope, group_cols in {"macro_assay_pair": ["assay_src", "assay_dst"], "macro_target": ["target_chembl_id"]}.items():
        grouped = (
            sub.groupby(group_cols)
            .apply(
                lambda g: pd.Series(
                    {
                        "mae": g["err"].abs().mean(),
                        "rmse": rmse(g["err"].to_numpy(float)),
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
                "n": int(grouped["n"].sum()),
                "groups": int(len(grouped)),
                "mae": float(grouped["mae"].mean()),
                "rmse": float(grouped["rmse"].mean()),
                "bias": float(grouped["bias"].mean()),
            }
        )
    for bucket, bucket_df in sub.groupby("pair_support_bucket"):
        err = bucket_df["err"].to_numpy(float)
        rows.append(
            {
                "method": method,
                "split": split,
                "scope": "micro_by_support",
                "n": int(len(bucket_df)),
                "groups": 1,
                "mae": float(np.mean(np.abs(err))),
                "rmse": rmse(err),
                "bias": float(np.mean(err)),
                "bucket": str(bucket),
            }
        )
    for row in rows:
        row.setdefault("bucket", "all")
    return rows


def build_candidates(
    predictions: pd.DataFrame,
    train: pd.DataFrame,
    min_pairs_grid: list[int],
    shrink_grid: list[float],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    global_intercept, global_slope = fit_affine(train["y_src"].to_numpy(float), train["y_dst"].to_numpy(float))
    y_src = predictions["y_src"].to_numpy(float)
    global_pred = global_intercept + global_slope * y_src
    identity = predictions["y_src"].to_numpy(float)
    identity_bias = predictions["pred_identity_bias"].to_numpy(float)

    support = train.groupby(["assay_src", "assay_dst"]).size().to_dict()
    support_arr = np.asarray(
        [int(support.get((a, b), 0)) for a, b in zip(predictions["assay_src"], predictions["assay_dst"])],
        dtype=float,
    )

    pair_models: dict[tuple[int, int], tuple[float, float]] = {}
    for key, group in train.groupby(["assay_src", "assay_dst"]):
        if len(group) >= 2:
            pair_models[(int(key[0]), int(key[1]))] = fit_affine(
                group["y_src"].to_numpy(float), group["y_dst"].to_numpy(float)
            )

    pair_min2_global = np.empty(len(predictions), dtype=float)
    for i, row in enumerate(predictions.itertuples(index=False)):
        model = pair_models.get((int(row.assay_src), int(row.assay_dst)))
        if model is None:
            pair_min2_global[i] = global_pred[i]
        else:
            pair_min2_global[i] = model[0] + model[1] * float(row.y_src)

    candidates: dict[str, np.ndarray] = {
        "identity": identity,
        "identity_bias": identity_bias,
        "global_affine": global_pred,
        "pair_min2_global": pair_min2_global,
    }
    if "pred_pair_affine" in predictions.columns:
        candidates["pair_min5_global_existing"] = predictions["pred_pair_affine"].to_numpy(float)
    if "pred_pair_affine_bias_fallback" in predictions.columns:
        candidates["pair_min5_id_existing"] = predictions["pred_pair_affine_bias_fallback"].to_numpy(float)
    if "pred_pair_affine_shrink_k10" in predictions.columns:
        candidates["shrink_min5_k10_existing"] = predictions["pred_pair_affine_shrink_k10"].to_numpy(float)

    for min_pairs in min_pairs_grid:
        pair_or_global = np.where(support_arr >= min_pairs, pair_min2_global, global_pred)
        candidates[f"pair_min{min_pairs}_global"] = pair_or_global
        candidates[f"pair_min{min_pairs}_id"] = np.where(support_arr >= min_pairs, pair_min2_global, identity_bias)
        zero_global_low_identity = np.where(support_arr == 0, global_pred, np.where(support_arr >= min_pairs, pair_min2_global, identity_bias))
        candidates[f"pair_min{min_pairs}_zeroG_lowID"] = zero_global_low_identity
        for shrink_k in shrink_grid:
            weight = support_arr / (support_arr + shrink_k)
            shrink = weight * pair_min2_global + (1.0 - weight) * identity_bias
            shrink = np.where(support_arr >= min_pairs, shrink, np.where(support_arr == 0, global_pred, identity_bias))
            k_name = str(int(shrink_k)) if float(shrink_k).is_integer() else str(shrink_k).replace(".", "p")
            candidates[f"shrink_min{min_pairs}_k{k_name}_zeroG_lowID"] = shrink

    info = {
        "global_affine": {"intercept": global_intercept, "slope": global_slope},
        "n_pair_models_min2": int(len(pair_models)),
        "min_pairs_grid": min_pairs_grid,
        "shrink_grid": shrink_grid,
    }
    return candidates, info


def score_candidates(frame: pd.DataFrame, candidates: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for split in ("val", "test"):
        mask = frame["split"].to_numpy() == split
        y = frame.loc[mask, "y_dst"].to_numpy(float)
        for name, pred in candidates.items():
            err = pred[mask] - y
            rows.append(
                {
                    "candidate": name,
                    "split": split,
                    "n": int(mask.sum()),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": rmse(err),
                    "bias": float(np.mean(err)),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "mae", "candidate"]).reset_index(drop=True)


def objective_score(frame: pd.DataFrame, idx: np.ndarray, pred: np.ndarray, scope: str) -> float:
    sub = frame.iloc[idx]
    err_abs = np.abs(pred[idx] - sub["y_dst"].to_numpy(float))
    if scope == "micro":
        return float(np.mean(err_abs))
    work = sub[["target_chembl_id", "assay_src", "assay_dst"]].copy()
    work["abs_err"] = err_abs
    if scope == "macro_assay_pair":
        return float(work.groupby(["assay_src", "assay_dst"])["abs_err"].mean().mean())
    if scope == "macro_target":
        return float(work.groupby("target_chembl_id")["abs_err"].mean().mean())
    raise ValueError(f"Unknown selection scope: {scope}")


def select_by_bucket(
    frame: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    scope: str,
    global_best_name: str,
    global_best_score: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, object]]]:
    selected = np.empty(len(frame), dtype=float)
    selected_name = np.empty(len(frame), dtype=object)
    choices: dict[str, dict[str, object]] = {}
    for bucket, idx in frame.groupby("pair_support_bucket").groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        val_idx = idx_arr[frame.iloc[idx_arr]["split"].to_numpy() == "val"]
        if len(val_idx) == 0:
            best_name = global_best_name
            best_score = global_best_score
        else:
            best_score, best_name = min(
                (objective_score(frame, val_idx, pred, scope), name) for name, pred in candidates.items()
            )
        selected[idx_arr] = candidates[best_name][idx_arr]
        selected_name[idx_arr] = best_name
        choices[str(bucket)] = {"candidate": best_name, f"val_{scope}": best_score}
    return selected, selected_name, choices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--output-dir", default="results/calibration_grid")
    parser.add_argument("--min-pairs-grid", default="2,3,5,10,20,30")
    parser.add_argument("--shrink-grid", default="1,2,5,10,20,50,100")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.baseline_predictions).reset_index(drop=True)
    frame = frame[frame["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    train = frame[frame["split"] == "train"].copy()
    min_pairs_grid = [int(x) for x in args.min_pairs_grid.split(",") if x]
    shrink_grid = [float(x) for x in args.shrink_grid.split(",") if x]

    support = train.groupby(["assay_src", "assay_dst"]).size().to_dict()
    frame["pair_train_support"] = [
        int(support.get((a, b), 0)) for a, b in zip(frame["assay_src"], frame["assay_dst"])
    ]
    frame["pair_support_bucket"] = frame["pair_train_support"].map(support_bucket)

    candidates, candidate_info = build_candidates(frame, train, min_pairs_grid, shrink_grid)
    candidate_scores = score_candidates(frame, candidates)
    candidate_scores.to_csv(out_dir / "calibration_grid_candidate_scores.csv", index=False)

    val_idx_all = np.flatnonzero(frame["split"].to_numpy() == "val")
    global_best = min((objective_score(frame, val_idx_all, pred, "micro"), name) for name, pred in candidates.items())

    frame["pred_calibration_grid_global"] = candidates[global_best[1]]
    selection_scopes = ["micro", "macro_assay_pair", "macro_target"]
    selector_choices: dict[str, dict[str, dict[str, object]]] = {}
    for scope in selection_scopes:
        global_scope_best = min((objective_score(frame, val_idx_all, pred, scope), name) for name, pred in candidates.items())
        selected, selected_name, choices = select_by_bucket(frame, candidates, scope, global_scope_best[1], global_scope_best[0])
        suffix = scope.replace("macro_", "macro_")
        frame[f"pred_calibration_grid_selector_{suffix}"] = selected
        frame[f"calibration_grid_selected_candidate_{suffix}"] = selected_name
        selector_choices[scope] = {
            "global_best": {"candidate": global_scope_best[1], f"val_{scope}": global_scope_best[0]},
            "bucket_choices": choices,
        }
    frame["pred_calibration_grid_selector"] = frame["pred_calibration_grid_selector_micro"].to_numpy()
    frame["calibration_grid_selected_candidate"] = frame["calibration_grid_selected_candidate_micro"].to_numpy()

    metrics = []
    method_cols = {
        "calibration_grid_global": "pred_calibration_grid_global",
        "calibration_grid_selector": "pred_calibration_grid_selector",
        "calibration_grid_selector_micro": "pred_calibration_grid_selector_micro",
        "calibration_grid_selector_macro_assay_pair": "pred_calibration_grid_selector_macro_assay_pair",
        "calibration_grid_selector_macro_target": "pred_calibration_grid_selector_macro_target",
    }
    for method, col in method_cols.items():
        for split in ("val", "test"):
            metrics.extend(metric_rows(frame, col, method, split))
    metric_df = pd.DataFrame(metrics).sort_values(["split", "scope", "bucket", "mae", "method"]).reset_index(drop=True)
    metric_df.to_csv(out_dir / "calibration_grid_metrics.csv", index=False)

    pred_cols = KEY_COLS + [
        "pair_train_support",
        "pair_support_bucket",
        "calibration_grid_selected_candidate",
        "pred_calibration_grid_global",
        "pred_calibration_grid_selector",
        "pred_calibration_grid_selector_micro",
        "pred_calibration_grid_selector_macro_assay_pair",
        "pred_calibration_grid_selector_macro_target",
    ]
    frame[pred_cols].to_parquet(out_dir / "calibration_grid_predictions.parquet", index=False)
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "validation_bucket_score",
        "global_best": {"candidate": global_best[1], "val_micro_mae": global_best[0]},
        "selectors": selector_choices,
        "candidate_info": candidate_info,
        "support_buckets": frame[frame["split"] == "test"]["pair_support_bucket"].value_counts().sort_index().astype(int).to_dict(),
    }
    (out_dir / "calibration_grid_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
