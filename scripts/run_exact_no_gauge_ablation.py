#!/usr/bin/env python3
"""Refit the complete SOMOL-RBB selector after removing gauge candidates."""

from __future__ import annotations

import argparse
import json
import math
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

BASE_COLUMNS = [
    "pred_pair_affine_bias_fallback",
    "pred_support_gated_val_micro",
    "pred_calibration_grid_selector_micro",
    "pred_calibration_grid_selector_macro_assay_pair",
    "pred_calibration_grid_selector_macro_target",
]


def weight_grid(step: float) -> list[float]:
    count = int(round(1.0 / step))
    return [index / count for index in range(count + 1)]


def simplex_grid(components: int, step: float) -> list[tuple[float, ...]]:
    count = int(round(1.0 / step))
    compositions: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == components - 1:
            compositions.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            visit((*prefix, value), remaining - value)

    visit((), count)
    return [tuple(value / count for value in composition) for composition in compositions]


def objective(frame: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray, scope: str) -> float:
    subset = frame.loc[mask, ["target_chembl_id", "assay_src", "assay_dst", "y_dst"]].copy()
    subset["absolute_error"] = np.abs(prediction[mask] - subset["y_dst"].to_numpy(float))
    if scope == "micro":
        return float(subset["absolute_error"].mean())
    if scope == "macro_assay_pair":
        return float(subset.groupby(["assay_src", "assay_dst"])["absolute_error"].mean().mean())
    if scope == "macro_target":
        return float(subset.groupby("target_chembl_id")["absolute_error"].mean().mean())
    raise ValueError(f"Unknown scope: {scope}")


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


def load_aligned(path: str, expected_length: int | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path).reset_index(drop=True)
    if expected_length is not None and len(frame) != expected_length:
        raise ValueError(f"Prediction length mismatch for {path}: {len(frame)} != {expected_length}")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--support-predictions", required=True)
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--method-comparison-predictions", required=True)
    parser.add_argument("--full-model-predictions", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stack-step", type=float, default=0.1)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--relative-tolerance", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    stack_output = output_root / "no_gauge_stack"
    final_output = output_root / "somol_robust_no_gauge_ablation"
    stack_output.mkdir(parents=True, exist_ok=True)
    final_output.mkdir(parents=True, exist_ok=True)

    baseline = load_aligned(args.baseline_predictions)
    support = load_aligned(args.support_predictions, len(baseline))
    calibration = load_aligned(args.calibration_predictions, len(baseline))
    method_comparison = load_aligned(args.method_comparison_predictions, len(baseline))
    full_model = load_aligned(args.full_model_predictions, len(baseline))

    frame = baseline[[column for column in KEY_COLUMNS if column in baseline.columns]].copy()
    for source in (baseline, support, calibration):
        for column in BASE_COLUMNS:
            if column in source.columns:
                frame[column] = source[column].to_numpy(float)
    missing = [column for column in BASE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing no-gauge candidates: {missing}")

    base_predictions = [frame[column].to_numpy(float) for column in BASE_COLUMNS]
    pair_prediction = frame["pred_pair_affine_bias_fallback"].to_numpy(float)
    validation_mask = frame["split"].to_numpy() == "val"
    stack_weights = simplex_grid(len(BASE_COLUMNS), args.stack_step)
    guard_weights = weight_grid(args.stack_step)
    diagnostics: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_columns": BASE_COLUMNS,
        "guard_relative_tolerance": args.relative_tolerance,
        "selectors": {},
    }

    method_columns: dict[str, str] = {}
    for scope in ("micro", "macro_assay_pair", "macro_target"):
        best_stack_score = math.inf
        best_stack_weight: tuple[float, ...] | None = None
        best_stack_prediction: np.ndarray | None = None
        for weights in stack_weights:
            prediction = sum(weight * candidate for weight, candidate in zip(weights, base_predictions))
            score = objective(frame, prediction, validation_mask, scope)
            if score < best_stack_score - 1e-12:
                best_stack_score = score
                best_stack_weight = weights
                best_stack_prediction = prediction
        assert best_stack_weight is not None and best_stack_prediction is not None

        candidates = []
        for stack_weight in guard_weights:
            prediction = stack_weight * best_stack_prediction + (1.0 - stack_weight) * pair_prediction
            candidates.append((objective(frame, prediction, validation_mask, scope), stack_weight, prediction))
        best_guard_score = min(candidate[0] for candidate in candidates)
        threshold = best_guard_score * (1.0 + args.relative_tolerance)
        selected_score, selected_weight, selected_prediction = min(
            (candidate for candidate in candidates if candidate[0] <= threshold + 1e-12),
            key=lambda candidate: (candidate[1], candidate[0]),
        )
        method = f"somol_no_gauge_stack_{scope}"
        column = f"pred_{method}"
        frame[column] = selected_prediction
        method_columns[method] = column
        diagnostics["selectors"][scope] = {
            "stack_weights": {
                name: float(weight) for name, weight in zip(BASE_COLUMNS, best_stack_weight)
            },
            "stack_val_score": float(best_stack_score),
            "guard_stack_weight": float(selected_weight),
            "selected_val_score": float(selected_score),
        }

    frame["pred_somol_no_gauge_stack"] = frame["pred_somol_no_gauge_stack_micro"]
    method_columns["somol_no_gauge_stack"] = "pred_somol_no_gauge_stack"
    stack_rows: list[dict[str, object]] = []
    for method, column in method_columns.items():
        for split in ("val", "test"):
            stack_rows.extend(metric_rows(frame, column, method, split, args.endpoint.upper(), args.seed))
    stack_metrics = pd.DataFrame(stack_rows)
    stack_columns = [column for column in KEY_COLUMNS if column in frame.columns] + list(method_columns.values())
    frame[stack_columns].to_parquet(stack_output / "no_gauge_stack_predictions.parquet", index=False)
    stack_metrics.to_csv(stack_output / "no_gauge_stack_metrics.csv", index=False)
    (stack_output / "no_gauge_stack_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )

    final_frame = full_model.copy()
    robust_column = "pred_robust_trimmed_bias_fallback"
    final_column = "pred_somol_robust_blend_bucket"
    for required, source in {
        robust_column: method_comparison,
        "pred_somol_no_gauge_stack_micro": frame,
    }.items():
        if required not in source.columns:
            raise ValueError(f"Missing final ablation column: {required}")
        final_frame[required] = source[required].to_numpy(float)
    if final_column not in final_frame.columns or "pair_support_bucket" not in final_frame.columns:
        raise ValueError("Full-model predictions must contain the final prediction and support buckets.")

    no_gauge = final_frame["pred_somol_no_gauge_stack_micro"].to_numpy(float)
    robust = final_frame[robust_column].to_numpy(float)
    selected = np.empty(len(final_frame), dtype=float)
    choices: dict[str, object] = {}
    for bucket, indices in final_frame.groupby("pair_support_bucket", sort=False).groups.items():
        index = np.asarray(list(indices), dtype=int)
        validation_index = index[final_frame.iloc[index]["split"].to_numpy() == "val"]
        if len(validation_index) == 0:
            raise ValueError(f"No validation records in support bucket {bucket}")
        truth = final_frame.loc[validation_index, "y_dst"].to_numpy(float)
        best = (math.inf, 0.0)
        for no_gauge_weight in weight_grid(args.blend_step):
            prediction = (
                no_gauge_weight * no_gauge[validation_index]
                + (1.0 - no_gauge_weight) * robust[validation_index]
            )
            score = float(np.mean(np.abs(prediction - truth)))
            if score < best[0] - 1e-12:
                best = (score, no_gauge_weight)
        selected[index] = best[1] * no_gauge[index] + (1.0 - best[1]) * robust[index]
        choices[str(bucket)] = {
            "no_gauge_stack_weight": float(best[1]),
            "robust_trimmed_weight": float(1.0 - best[1]),
            "val_micro_mae": float(best[0]),
            "n_val": int(len(validation_index)),
        }

    ablation_column = "pred_somol_robust_blend_bucket_exact_no_gauge"
    final_frame[ablation_column] = selected
    final_rows: list[dict[str, object]] = []
    for split in ("val", "test"):
        final_rows.extend(
            metric_rows(
                final_frame,
                ablation_column,
                "somol_robust_blend_bucket_exact_no_gauge",
                split,
                args.endpoint.upper(),
                args.seed,
            )
        )
    final_metrics = pd.DataFrame(final_rows)
    extra_columns = [
        "pair_train_support",
        "pair_support_bucket",
        "min_node_degree",
        "node_degree_bucket",
        "graph_bucket",
    ]
    output_columns = (
        [column for column in KEY_COLUMNS if column in final_frame.columns]
        + [column for column in extra_columns if column in final_frame.columns]
        + [final_column, ablation_column, "pred_somol_no_gauge_stack_micro", robust_column]
    )
    final_frame[output_columns].to_parquet(
        final_output / "somol_robust_no_gauge_predictions.parquet", index=False
    )
    final_metrics.to_csv(final_output / "somol_robust_no_gauge_metrics.csv", index=False)
    final_diagnostics = {
        "endpoint": args.endpoint.upper(),
        "seed": args.seed,
        "definition": (
            "exact no-gauge ablation of the final robust blend: remove every gauge candidate, "
            "refit the stack and guard, then reselect the support-bucket blend"
        ),
        "weight_grid": weight_grid(args.blend_step),
        "bucket_choices": choices,
    }
    (final_output / "somol_robust_no_gauge_diagnostics.json").write_text(
        json.dumps(final_diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(final_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
