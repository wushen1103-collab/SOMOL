#!/usr/bin/env python3
"""Fit Deming/TLS and trimmed robust assay-edge calibration baselines."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = [
    "split",
    "compound_target_key",
    "target_chembl_id",
    "assay_src",
    "assay_dst",
    "y_src",
    "y_dst",
]


@dataclass(frozen=True)
class Affine:
    intercept: float
    slope: float


def bias_fallback(x: np.ndarray, y: np.ndarray) -> Affine:
    return Affine(float(np.mean(y - x)), 1.0)


def fit_ols(x: np.ndarray, y: np.ndarray) -> Affine:
    if len(x) < 2 or np.std(x) < 1e-8:
        return Affine(float(np.mean(y)), 0.0)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return Affine(float(coefficients[0]), float(coefficients[1]))


def fit_deming(x: np.ndarray, y: np.ndarray) -> Affine:
    fallback = bias_fallback(x, y)
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return fallback
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    sxx = float(np.mean(x_centered**2))
    syy = float(np.mean(y_centered**2))
    sxy = float(np.mean(x_centered * y_centered))
    if abs(sxy) < 1e-12:
        return fallback
    slope = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4.0 * sxy**2)) / (2.0 * sxy)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return Affine(intercept, float(slope))


def fit_tls(x: np.ndarray, y: np.ndarray) -> Affine:
    fallback = bias_fallback(x, y)
    if len(x) < 3 or np.std(x) < 1e-8:
        return fallback
    centered = np.column_stack([x - np.mean(x), y - np.mean(y)])
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    direction = directions[0]
    if abs(direction[0]) < 1e-12:
        return fallback
    slope = float(direction[1] / direction[0])
    intercept = float(np.mean(y) - slope * np.mean(x))
    return Affine(intercept, slope)


def fit_trimmed_ols(x: np.ndarray, y: np.ndarray, trim_fraction: float) -> Affine:
    if len(x) < 3:
        return bias_fallback(x, y)
    if np.std(x) < 1e-8:
        return bias_fallback(x, y)
    initial = fit_ols(x, y)
    residual = np.abs(y - (initial.intercept + initial.slope * x))
    threshold = float(np.quantile(residual, 1.0 - trim_fraction))
    keep = np.flatnonzero(residual <= threshold)
    if len(keep) < 3 or np.std(x[keep]) < 1e-8:
        return initial
    return fit_ols(x[keep], y[keep])


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error**2))) if len(error) else math.nan


def metric_rows(
    frame: pd.DataFrame,
    prediction_column: str,
    method: str,
    split: str,
    endpoint: str,
    seed: int,
) -> list[dict[str, object]]:
    subset = frame[frame["split"].eq(split)].copy()
    if subset.empty:
        return []
    subset["error"] = subset[prediction_column] - subset["y_dst"]
    rows = [{
        "endpoint": endpoint,
        "seed": seed,
        "method": method,
        "split": split,
        "scope": "micro",
        "n": int(len(subset)),
        "groups": 1,
        "mae": float(subset["error"].abs().mean()),
        "rmse": rmse(subset["error"].to_numpy(float)),
        "bias": float(subset["error"].mean()),
    }]
    for scope, columns in {
        "macro_assay_pair": ["assay_src", "assay_dst"],
        "macro_target": ["target_chembl_id"],
    }.items():
        grouped = (
            subset.groupby(columns, sort=False)["error"]
            .agg(
                mae=lambda values: float(np.mean(np.abs(values))),
                rmse=lambda values: rmse(values.to_numpy(float)),
                bias="mean",
                n="size",
            )
            .reset_index()
        )
        rows.append({
            "endpoint": endpoint,
            "seed": seed,
            "method": method,
            "split": split,
            "scope": scope,
            "n": int(grouped["n"].sum()),
            "groups": int(len(grouped)),
            "mae": float(grouped["mae"].mean()),
            "rmse": float(grouped["rmse"].mean()),
            "bias": float(grouped["bias"].mean()),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--trim-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_parquet(args.baseline_predictions).reset_index(drop=True)
    missing = [column for column in KEY_COLUMNS if column not in baseline.columns]
    if missing:
        raise ValueError(f"Missing baseline columns: {missing}")
    train = baseline[baseline["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("No training rows are available for edge calibration.")

    global_delta = float((train["y_dst"] - train["y_src"]).mean())
    models: dict[tuple[int, int], dict[str, Affine]] = {}
    parameter_rows: list[dict[str, float | int]] = []
    for key, group in train.groupby(["assay_src", "assay_dst"], sort=False):
        x = group["y_src"].to_numpy(float)
        y = group["y_dst"].to_numpy(float)
        fallback = bias_fallback(x, y)
        if len(group) < args.min_pairs:
            deming = tls = robust = fallback
        else:
            deming = fit_deming(x, y)
            tls = fit_tls(x, y)
            robust = fit_trimmed_ols(x, y, args.trim_fraction)
        pair_key = (int(key[0]), int(key[1]))
        models[pair_key] = {"deming": deming, "tls": tls, "robust": robust}
        parameter_rows.append({
            "assay_src": pair_key[0],
            "assay_dst": pair_key[1],
            "support": int(len(group)),
            "deming_a": deming.intercept,
            "deming_b": deming.slope,
            "odr_tls_a": tls.intercept,
            "odr_tls_b": tls.slope,
            "robust_trimmed_a": robust.intercept,
            "robust_trimmed_b": robust.slope,
        })

    method_keys = {
        "pred_deming_bias_fallback": "deming",
        "pred_odr_tls_bias_fallback": "tls",
        "pred_robust_trimmed_bias_fallback": "robust",
    }
    predictions = baseline[KEY_COLUMNS].copy()
    for prediction_column, model_key in method_keys.items():
        values = np.empty(len(baseline), dtype=float)
        for index, row in enumerate(baseline.itertuples(index=False)):
            model = models.get((int(row.assay_src), int(row.assay_dst)))
            if model is None:
                values[index] = float(row.y_src) + global_delta
            else:
                affine = model[model_key]
                values[index] = affine.intercept + affine.slope * float(row.y_src)
        predictions[prediction_column] = values

    rows: list[dict[str, object]] = []
    names = {
        "pred_deming_bias_fallback": "deming_bias_fallback",
        "pred_odr_tls_bias_fallback": "odr_tls_bias_fallback",
        "pred_robust_trimmed_bias_fallback": "robust_trimmed_bias_fallback",
    }
    for prediction_column, method in names.items():
        for split in ("val", "test"):
            rows.extend(metric_rows(predictions, prediction_column, method, split, args.endpoint.upper(), args.seed))

    parameters = pd.DataFrame(parameter_rows).sort_values(["assay_src", "assay_dst"]).reset_index(drop=True)
    metrics = pd.DataFrame(rows).sort_values(["split", "scope", "mae", "method"]).reset_index(drop=True)
    predictions.to_parquet(output_dir / "method_comparison_predictions.parquet", index=False)
    parameters.to_csv(output_dir / "method_comparison_edge_params.csv", index=False)
    metrics.to_csv(output_dir / "method_comparison_metrics.csv", index=False)
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint.upper(),
        "seed": args.seed,
        "min_pairs": args.min_pairs,
        "trim_fraction": args.trim_fraction,
        "n_training_edges": len(models),
        "global_mean_delta": global_delta,
    }
    (output_dir / "method_comparison_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
