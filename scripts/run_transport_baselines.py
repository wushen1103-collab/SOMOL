#!/usr/bin/env python
"""Run direct cross-assay transport calibration baselines."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class Affine:
    intercept: float
    slope: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * x


def fit_affine(x: np.ndarray, y: np.ndarray) -> Affine:
    if len(x) < 2 or np.nanstd(x) < 1e-8:
        return Affine(float(np.nanmean(y)), 0.0)
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return Affine(float(coef[0]), float(coef[1]))


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else math.nan


def metric_rows(frame: pd.DataFrame, pred_col: str, split: str, method: str) -> list[dict[str, object]]:
    sub = frame[frame["split"] == split].copy()
    if sub.empty:
        return []
    sub["err"] = sub[pred_col] - sub["y_dst"]
    rows = [
        {
            "method": method,
            "split": split,
            "scope": "micro",
            "n": int(len(sub)),
            "mae": float(sub["err"].abs().mean()),
            "rmse": rmse(sub["err"].to_numpy()),
            "bias": float(sub["err"].mean()),
            "groups": 1,
        }
    ]
    for scope, group_cols in {
        "macro_assay_pair": ["assay_src", "assay_dst"],
        "macro_target": ["target_chembl_id"],
    }.items():
        grouped = (
            sub.groupby(group_cols)
            .apply(lambda g: pd.Series({"mae": g["err"].abs().mean(), "rmse": rmse(g["err"].to_numpy()), "bias": g["err"].mean(), "n": len(g)}), include_groups=False)
            .reset_index()
        )
        rows.append(
            {
                "method": method,
                "split": split,
                "scope": scope,
                "n": int(grouped["n"].sum()),
                "mae": float(grouped["mae"].mean()),
                "rmse": float(grouped["rmse"].mean()),
                "bias": float(grouped["bias"].mean()),
                "groups": int(len(grouped)),
            }
        )
    return rows


def add_mean_baseline(train: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    global_mean = float(train["y_dst"].mean())
    dst_mean = train.groupby("assay_dst")["y_dst"].mean().to_dict()
    return eval_df["assay_dst"].map(dst_mean).fillna(global_mean).to_numpy(float)


def add_identity_bias_baseline(train: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    global_delta = float((train["y_dst"] - train["y_src"]).mean())
    pair_delta = train.groupby(["assay_src", "assay_dst"]).apply(lambda g: (g["y_dst"] - g["y_src"]).mean(), include_groups=False).to_dict()
    values = []
    for src, dst, y_src in zip(eval_df["assay_src"], eval_df["assay_dst"], eval_df["y_src"]):
        values.append(float(y_src) + float(pair_delta.get((src, dst), global_delta)))
    return np.asarray(values, dtype=float)


def predict_affine(train: pd.DataFrame, eval_df: pd.DataFrame, min_pairs: int, target_conditioned: bool = False) -> tuple[np.ndarray, dict[str, int]]:
    global_model = fit_affine(train["y_src"].to_numpy(float), train["y_dst"].to_numpy(float))
    pair_models: dict[tuple[int, int], Affine] = {}
    for key, group in train.groupby(["assay_src", "assay_dst"]):
        if len(group) >= min_pairs:
            pair_models[key] = fit_affine(group["y_src"].to_numpy(float), group["y_dst"].to_numpy(float))

    target_models: dict[tuple[str, int, int], Affine] = {}
    if target_conditioned:
        for key, group in train.groupby(["target_chembl_id", "assay_src", "assay_dst"]):
            if len(group) >= min_pairs:
                target_models[(str(key[0]), int(key[1]), int(key[2]))] = fit_affine(group["y_src"].to_numpy(float), group["y_dst"].to_numpy(float))

    pred = np.empty(len(eval_df), dtype=float)
    used_target = 0
    used_pair = 0
    used_global = 0
    for i, row in enumerate(eval_df.itertuples(index=False)):
        target_key = (str(row.target_chembl_id), int(row.assay_src), int(row.assay_dst))
        pair_key = (int(row.assay_src), int(row.assay_dst))
        if target_conditioned and target_key in target_models:
            pred[i] = target_models[target_key].predict(np.asarray([row.y_src], dtype=float))[0]
            used_target += 1
        elif pair_key in pair_models:
            pred[i] = pair_models[pair_key].predict(np.asarray([row.y_src], dtype=float))[0]
            used_pair += 1
        else:
            pred[i] = global_model.predict(np.asarray([row.y_src], dtype=float))[0]
            used_global += 1
    return pred, {"target_model": used_target, "pair_model": used_pair, "global_model": used_global, "n_pair_models": len(pair_models), "n_target_models": len(target_models)}


def pair_support(train: pd.DataFrame) -> dict[tuple[int, int], int]:
    return {tuple(map(int, key)): int(value) for key, value in train.groupby(["assay_src", "assay_dst"]).size().to_dict().items()}


def fallback_and_shrink(
    eval_df: pd.DataFrame,
    pair_affine_pred: np.ndarray,
    identity_bias_pred: np.ndarray,
    support: dict[tuple[int, int], int],
    min_pairs: int,
    shrink_k: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    fallback = pair_affine_pred.copy()
    shrink = pair_affine_pred.copy()
    used_pair = 0
    used_fallback = 0
    for i, row in enumerate(eval_df.itertuples(index=False)):
        n = int(support.get((int(row.assay_src), int(row.assay_dst)), 0))
        if n >= min_pairs:
            used_pair += 1
            w = n / (n + shrink_k)
            shrink[i] = w * pair_affine_pred[i] + (1.0 - w) * identity_bias_pred[i]
        else:
            used_fallback += 1
            fallback[i] = identity_bias_pred[i]
            shrink[i] = identity_bias_pred[i]
    usage = {"used_pair": used_pair, "used_identity_bias_fallback": used_fallback, "shrink_k": shrink_k}
    return fallback, shrink, usage


def predict_isotonic(train: pd.DataFrame, eval_df: pd.DataFrame, min_pairs: int, affine_fallback: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    models: dict[tuple[int, int], IsotonicRegression] = {}
    for key, group in train.groupby(["assay_src", "assay_dst"]):
        if len(group) >= min_pairs and group["y_src"].nunique() >= 3:
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(group["y_src"].to_numpy(float), group["y_dst"].to_numpy(float))
            models[key] = model

    pred = affine_fallback.copy()
    used_iso = 0
    for key, idx in eval_df.groupby(["assay_src", "assay_dst"]).groups.items():
        model = models.get(key)
        if model is None:
            continue
        loc = np.fromiter(idx, dtype=int)
        pred[loc] = model.predict(eval_df.iloc[loc]["y_src"].to_numpy(float))
        used_iso += len(loc)
    return pred, {"isotonic_model": used_iso, "fallback": int(len(eval_df) - used_iso), "n_isotonic_models": len(models)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-dir", default="results/transport_baselines")
    parser.add_argument("--min-affine-pairs", type=int, default=5)
    parser.add_argument("--min-isotonic-pairs", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_parquet(args.pairs)
    pairs = pairs[pairs["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    train = pairs[pairs["split"] == "train"].copy()
    if train.empty:
        raise RuntimeError("No train transport pairs; cannot fit baselines.")

    predictions = pairs.copy()
    diagnostics: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": args.pairs,
        "n_pairs": int(len(pairs)),
        "by_split": pairs.groupby("split").size().astype(int).to_dict(),
        "min_affine_pairs": args.min_affine_pairs,
        "min_isotonic_pairs": args.min_isotonic_pairs,
        "model_usage": {},
    }

    predictions["pred_identity"] = predictions["y_src"].to_numpy(float)
    predictions["pred_assay_mean"] = add_mean_baseline(train, predictions)
    predictions["pred_identity_bias"] = add_identity_bias_baseline(train, predictions)
    affine_pred, affine_usage = predict_affine(train, predictions, args.min_affine_pairs, target_conditioned=False)
    predictions["pred_pair_affine"] = affine_pred
    diagnostics["model_usage"]["pair_affine"] = affine_usage
    support = pair_support(train)
    bias_fallback, shrink_k10, fallback_usage = fallback_and_shrink(
        predictions,
        predictions["pred_pair_affine"].to_numpy(float),
        predictions["pred_identity_bias"].to_numpy(float),
        support,
        args.min_affine_pairs,
        shrink_k=10.0,
    )
    predictions["pred_pair_affine_bias_fallback"] = bias_fallback
    predictions["pred_pair_affine_shrink_k10"] = shrink_k10
    diagnostics["model_usage"]["pair_affine_bias_fallback"] = fallback_usage
    target_affine_pred, target_affine_usage = predict_affine(train, predictions, args.min_affine_pairs, target_conditioned=True)
    predictions["pred_target_pair_affine"] = target_affine_pred
    diagnostics["model_usage"]["target_pair_affine"] = target_affine_usage
    iso_pred, iso_usage = predict_isotonic(train, predictions, args.min_isotonic_pairs, predictions["pred_pair_affine"].to_numpy(float))
    predictions["pred_pair_isotonic"] = iso_pred
    diagnostics["model_usage"]["pair_isotonic"] = iso_usage

    metric_list: list[dict[str, object]] = []
    method_cols = {
        "identity": "pred_identity",
        "assay_mean": "pred_assay_mean",
        "identity_bias": "pred_identity_bias",
        "pair_affine": "pred_pair_affine",
        "pair_affine_bias_fallback": "pred_pair_affine_bias_fallback",
        "pair_affine_shrink_k10": "pred_pair_affine_shrink_k10",
        "target_pair_affine": "pred_target_pair_affine",
        "pair_isotonic": "pred_pair_isotonic",
    }
    for method, col in method_cols.items():
        for split in ["val", "test"]:
            metric_list.extend(metric_rows(predictions, col, split, method))

    metrics = pd.DataFrame(metric_list).sort_values(["split", "scope", "mae", "method"]).reset_index(drop=True)
    metrics.to_csv(out_dir / "transport_baseline_metrics.csv", index=False)
    predictions.to_parquet(out_dir / "transport_baseline_predictions.parquet", index=False)
    (out_dir / "transport_baseline_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
