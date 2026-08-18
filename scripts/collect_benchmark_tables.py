#!/usr/bin/env python
"""Collect multi-endpoint benchmark tables for SOTA-readiness checks."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_INFO = {
    "identity": ("classic", "identity transport", "ours_rerun_same_split", False, False),
    "assay_mean": ("classic", "destination assay mean", "ours_rerun_same_split", False, False),
    "identity_bias": ("classic", "global/pair bias calibration", "ours_rerun_same_split", False, False),
    "pair_affine": ("classic", "local affine calibration", "ours_rerun_same_split", False, False),
    "pair_affine_bias_fallback": ("classic", "local affine with identity-bias fallback", "ours_rerun_same_split", False, False),
    "pair_affine_shrink_k10": ("classic", "fixed shrinkage calibration", "ours_rerun_same_split", False, False),
    "target_pair_affine": ("classic", "target-conditioned local affine", "ours_rerun_same_split", False, False),
    "pair_isotonic": ("classic", "monotone isotonic calibration", "ours_rerun_same_split", False, False),
    "support_gated_val_micro": ("support_shrinkage", "validation support-bucket selector", "ours_rerun_same_split", False, False),
    "deming_bias_fallback": ("method_comparison", "Deming errors-in-variables method-comparison baseline", "ours_rerun_same_split", False, False),
    "odr_tls_bias_fallback": ("method_comparison", "orthogonal-distance total least-squares method-comparison baseline", "ours_rerun_same_split", False, False),
    "robust_trimmed_bias_fallback": ("method_comparison", "trimmed robust method-comparison baseline", "ours_rerun_same_split", False, False),
    "calibration_grid_global": ("direct_competitor", "validation-selected global affine/shrinkage calibration", "ours_rerun_same_split", False, False),
    "calibration_grid_selector": ("direct_competitor", "support-aware shrinkage selector", "ours_rerun_same_split", False, False),
    "calibration_grid_selector_micro": ("direct_competitor", "support-aware shrinkage selector", "ours_rerun_same_split", False, False),
    "calibration_grid_selector_macro_assay_pair": ("direct_competitor", "support-aware shrinkage selector", "ours_rerun_same_split", False, False),
    "calibration_grid_selector_macro_target": ("direct_competitor", "support-aware shrinkage selector", "ours_rerun_same_split", False, False),
    "gauge_calibration_selector": ("ours", "SOMOL shared assay-gauge selector", "ours_main_same_split", False, False),
    "gauge_calibration_selector_micro": ("ours", "SOMOL shared assay-gauge selector", "ours_main_same_split", False, False),
    "gauge_calibration_selector_macro_assay_pair": ("ours_ablation", "SOMOL gauge selector, macro-pair selected", "ours_ablation_same_split", False, False),
    "gauge_calibration_selector_macro_target": ("ours_ablation", "SOMOL gauge selector, macro-target selected", "ours_ablation_same_split", False, False),
    "somol_gauge_graph_blend": ("ours", "SOMOL validation-tuned gauge/graph blend", "ours_main_same_split", False, False),
    "somol_gauge_graph_blend_micro": ("ours", "SOMOL validation-tuned gauge/graph blend", "ours_main_same_split", False, False),
    "somol_gauge_graph_blend_macro_assay_pair": ("ours_ablation", "SOMOL gauge/graph blend, macro-pair selected", "ours_ablation_same_split", False, False),
    "somol_gauge_graph_blend_macro_target": ("ours_ablation", "SOMOL gauge/graph blend, macro-target selected", "ours_ablation_same_split", False, False),
    "somol_stack": ("ours", "SOMOL validation-tuned calibration/gauge/graph convex stack", "ours_main_same_split", False, False),
    "somol_stack_micro": ("ours", "SOMOL validation-tuned calibration/gauge/graph convex stack", "ours_main_same_split", False, False),
    "somol_stack_macro_assay_pair": ("ours_ablation", "SOMOL convex stack, macro-pair selected", "ours_ablation_same_split", False, False),
    "somol_stack_macro_target": ("ours_ablation", "SOMOL convex stack, macro-target selected", "ours_ablation_same_split", False, False),
    "somol_guarded_stack": ("ours", "SOMOL validation-tolerant guarded stack", "ours_main_same_split", False, False),
    "somol_guarded_stack_micro": ("ours", "SOMOL validation-tolerant guarded stack", "ours_main_same_split", False, False),
    "somol_guarded_stack_macro_assay_pair": ("ours_ablation", "SOMOL guarded stack, macro-pair selected", "ours_ablation_same_split", False, False),
    "somol_guarded_stack_macro_target": ("ours_ablation", "SOMOL guarded stack, macro-target selected", "ours_ablation_same_split", False, False),
    "somol_robust_selector_global": ("ours_ablation", "SOMOL validation selector over guarded stack and robust method-comparison baseline", "ours_ablation_same_split", False, False),
    "somol_robust_blend_global": ("ours_ablation", "SOMOL global convex blend of guarded stack and robust method-comparison baseline", "ours_ablation_same_split", False, False),
    "somol_robust_selector_bucket": ("ours_ablation", "SOMOL support-bucket selector over guarded stack and robust method-comparison baseline", "ours_ablation_same_split", False, False),
    "somol_robust_blend_bucket": ("ours", "SOMOL support-bucket convex blend of guarded stack and robust method-comparison baseline", "ours_main_same_split", False, False),
    "graph_gauge_calibration_selector": ("ours_ablation", "SOMOL graph-degree gauge extension", "ours_ablation_same_split", False, False),
    "graph_gauge_calibration_selector_micro": ("ours_ablation", "SOMOL graph-degree gauge extension", "ours_ablation_same_split", False, False),
    "graph_gauge_calibration_selector_macro_assay_pair": ("ours_ablation", "SOMOL graph-degree gauge extension", "ours_ablation_same_split", False, False),
    "graph_gauge_calibration_selector_macro_target": ("ours_ablation", "SOMOL graph-degree gauge extension", "ours_ablation_same_split", False, False),
    "assay_context_knn_affine_k1": ("modern_assay_context", "assay-context KNN affine, k=1", "ours_rerun_same_split_surrogate", True, False),
    "assay_context_knn_affine_k5": ("modern_assay_context", "assay-context KNN affine, k=5", "ours_rerun_same_split_surrogate", True, False),
    "assay_context_knn_affine_k20": ("modern_assay_context", "assay-context KNN affine, k=20", "ours_rerun_same_split_surrogate", True, False),
    "assay_context_knn_delta_k1": ("modern_assay_context", "assay-context KNN delta, k=1", "ours_rerun_same_split_surrogate", True, False),
    "assay_context_knn_delta_k5": ("modern_assay_context", "assay-context KNN delta, k=5", "ours_rerun_same_split_surrogate", True, False),
    "assay_context_knn_delta_k20": ("modern_assay_context", "assay-context KNN delta, k=20", "ours_rerun_same_split_surrogate", True, False),
    "assay_context_ridge_delta": ("modern_assay_context", "assay-context ridge delta", "ours_rerun_same_split_surrogate", True, False),
    "assay_context_pair_or_knn_k1": ("modern_assay_context", "direct pair else assay-context KNN, k=1", "ours_rerun_same_split_surrogate", True, False),
    "assay_context_pair_or_knn_k5": ("modern_assay_context", "direct pair else assay-context KNN, k=5", "ours_rerun_same_split_surrogate", True, False),
    "assay_context_pair_or_knn_k20": ("modern_assay_context", "direct pair else assay-context KNN, k=20", "ours_rerun_same_split_surrogate", True, False),
    "assay_context_pair_or_ridge": ("modern_assay_context", "direct pair else assay-context ridge", "ours_rerun_same_split_surrogate", True, False),
    "missing_metadata_pair_or_global_delta": ("missing_modality", "direct pair else global delta with assay metadata removed", "ours_rerun_same_split", False, False),
}


RESULT_FILES = {
    "classic": "transport_baselines/transport_baseline_metrics.csv",
    "support_shrinkage": "support_gated_selector/support_gated_metrics.csv",
    "calibration_grid": "calibration_grid/calibration_grid_metrics.csv",
    "gauge": "gauge_calibration/gauge_calibration_metrics.csv",
    "graph_gauge": "graph_gauge_calibration/graph_gauge_calibration_metrics.csv",
    "somol_blend": "somol_blend/somol_blend_metrics.csv",
    "somol_stack": "somol_stack/somol_stack_metrics.csv",
    "somol_guarded_stack": "somol_guarded_stack/somol_guarded_stack_metrics.csv",
    "method_comparison": "method_comparison_baselines/method_comparison_metrics.csv",
    "somol_robust_stack": "somol_robust_stack/somol_robust_stack_metrics.csv",
    "assay_context": "assay_context_surrogates/assay_context_surrogate_metrics.csv",
}


PREDICTION_FILES = [
    "support_gated_selector/support_gated_predictions.parquet",
    "calibration_grid/calibration_grid_predictions.parquet",
    "gauge_calibration/gauge_calibration_predictions.parquet",
    "graph_gauge_calibration/graph_gauge_calibration_predictions.parquet",
    "somol_blend/somol_blend_predictions.parquet",
    "somol_stack/somol_stack_predictions.parquet",
    "somol_guarded_stack/somol_guarded_stack_predictions.parquet",
    "method_comparison_baselines/method_comparison_predictions.parquet",
    "somol_robust_stack/somol_robust_stack_predictions.parquet",
    "assay_context_surrogates/assay_context_surrogate_predictions.parquet",
]


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else math.nan


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def method_info(method: str) -> tuple[str, str, str, bool, bool]:
    return METHOD_INFO.get(method, ("other", "unclassified rerun method", "ours_rerun_same_split", False, False))


def metric_rows(frame: pd.DataFrame, pred_col: str, method: str, endpoint: str, seed: int, split: str, scenario: str) -> list[dict[str, object]]:
    sub = frame[frame["split"] == split].copy()
    if sub.empty:
        return []
    err = sub[pred_col].to_numpy(float) - sub["y_dst"].to_numpy(float)
    rows = [
        {
            "endpoint": endpoint,
            "seed": seed,
            "scenario": scenario,
            "method": method,
            "scope": "micro",
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
            .agg(mae=("err", lambda x: float(np.mean(np.abs(x)))), rmse=("err", lambda x: rmse(x.to_numpy(float))), bias=("err", "mean"), n=("err", "size"))
            .reset_index()
        )
        rows.append(
            {
                "endpoint": endpoint,
                "seed": seed,
                "scenario": scenario,
                "method": method,
                "scope": scope,
                "n": int(grouped["n"].sum()),
                "groups": int(len(grouped)),
                "mae": float(grouped["mae"].mean()),
                "rmse": float(grouped["rmse"].mean()),
                "bias": float(grouped["bias"].mean()),
            }
        )
    return rows


def load_metrics(benchmark_root: Path, endpoints: list[str], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for endpoint in endpoints:
        for seed in seeds:
            run_root = benchmark_root / endpoint.lower() / f"seed_{seed}"
            for family, rel in RESULT_FILES.items():
                path = run_root / rel
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                df["endpoint"] = endpoint.upper()
                df["seed"] = seed
                df["result_family"] = family
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    metrics = pd.concat(rows, ignore_index=True)
    metrics["method_raw"] = metrics["method"]
    metrics["method"] = metrics["method"].str.replace(
        r"^calibration_grid_global_.*$", "calibration_grid_global", regex=True
    )
    if "bucket" not in metrics.columns:
        metrics["bucket"] = "all"
    metrics["bucket"] = metrics["bucket"].fillna("all")
    info = metrics["method"].map(method_info)
    metrics["route_group"] = [x[0] for x in info]
    metrics["route_description"] = [x[1] for x in info]
    metrics["source_type"] = [x[2] for x in info]
    metrics["requires_assay_metadata"] = [x[3] for x in info]
    metrics["requires_compound_features"] = [x[4] for x in info]
    return metrics


def load_prediction_frame(run_root: Path) -> pd.DataFrame | None:
    base_path = run_root / "transport_baselines/transport_baseline_predictions.parquet"
    if not base_path.exists():
        return None
    frame = pd.read_parquet(base_path).reset_index(drop=True)
    for rel in PREDICTION_FILES:
        path = run_root / rel
        if not path.exists():
            continue
        extra = pd.read_parquet(path).reset_index(drop=True)
        if len(extra) != len(frame):
            continue
        for col in extra.columns:
            if col.startswith("pred_") or col in {
                "pair_train_support",
                "pair_support_bucket",
                "min_node_degree",
                "node_degree_bucket",
                "graph_bucket",
            }:
                frame[col] = extra[col].to_numpy()
    train = frame[frame["split"] == "train"].copy()
    if "pair_train_support" not in frame.columns:
        support = train.groupby(["assay_src", "assay_dst"]).size().to_dict()
        frame["pair_train_support"] = [int(support.get((a, b), 0)) for a, b in zip(frame["assay_src"], frame["assay_dst"])]
    if "min_node_degree" not in frame.columns:
        degree = train.groupby("assay_src").size().add(train.groupby("assay_dst").size(), fill_value=0).astype(int).to_dict()
        frame["min_node_degree"] = [
            min(int(degree.get(a, 0)), int(degree.get(b, 0))) for a, b in zip(frame["assay_src"], frame["assay_dst"])
        ]
    target_support = train.groupby("target_chembl_id").size().to_dict()
    frame["target_train_support"] = frame["target_chembl_id"].map(target_support).fillna(0).astype(int)
    global_delta = float((train["y_dst"] - train["y_src"]).mean()) if len(train) else 0.0
    direct = frame["pred_pair_affine_bias_fallback"].to_numpy(float)
    global_pred = frame["y_src"].to_numpy(float) + global_delta
    frame["pred_missing_metadata_pair_or_global_delta"] = np.where(frame["pair_train_support"].to_numpy(int) >= 5, direct, global_pred)
    return frame


def scenario_masks(test: pd.DataFrame) -> dict[str, pd.Series]:
    positive_target = test.loc[test["target_train_support"] > 0, "target_train_support"]
    target_q25 = int(positive_target.quantile(0.25)) if len(positive_target) else 0
    return {
        "all": pd.Series(True, index=test.index),
        "direct_pair_cold_start_zero": test["pair_train_support"].astype(int) == 0,
        "direct_pair_low_support_1_4": test["pair_train_support"].astype(int).between(1, 4),
        "assay_node_ood_unseen": test["min_node_degree"].astype(int) == 0,
        "assay_node_low_degree_1_9": test["min_node_degree"].astype(int).between(1, 9),
        "long_tail_target_q25": test["target_train_support"].astype(int) <= target_q25,
        "pic50_in_2_12": test["y_src"].between(2, 12) & test["y_dst"].between(2, 12),
    }


def collect_special_scenarios(benchmark_root: Path, endpoints: list[str], seeds: list[int]) -> pd.DataFrame:
    methods = [
        "identity_bias",
        "pair_affine_bias_fallback",
        "pair_isotonic",
        "calibration_grid_selector_micro",
        "gauge_calibration_selector_micro",
        "graph_gauge_calibration_selector",
        "somol_gauge_graph_blend_micro",
        "somol_stack_micro",
        "somol_guarded_stack_micro",
        "robust_trimmed_bias_fallback",
        "somol_robust_blend_bucket",
        "assay_context_pair_or_knn_k5",
        "assay_context_pair_or_ridge",
        "missing_metadata_pair_or_global_delta",
    ]
    rows = []
    for endpoint in endpoints:
        for seed in seeds:
            run_root = benchmark_root / endpoint.lower() / f"seed_{seed}"
            frame = load_prediction_frame(run_root)
            if frame is None:
                continue
            test = frame[frame["split"] == "test"].copy()
            masks = scenario_masks(test)
            for scenario, mask in masks.items():
                sub = test[mask].copy()
                if sub.empty:
                    continue
                for method in methods:
                    col = f"pred_{method}"
                    if col not in sub.columns:
                        continue
                    rows.extend(metric_rows(sub, col, method, endpoint.upper(), seed, "test", scenario))
    if not rows:
        return pd.DataFrame()
    special = pd.DataFrame(rows)
    info = special["method"].map(method_info)
    special["route_group"] = [x[0] for x in info]
    special["source_type"] = [x[2] for x in info]
    return special


def aggregate_mean_std(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols)
        .agg(
            n_runs=("mae", "size"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_bias=("bias", "mean"),
            mean_n=("n", "mean"),
            mean_groups=("groups", "mean"),
        )
        .reset_index()
    )
    grouped["std_mae"] = grouped["std_mae"].fillna(0.0)
    grouped["std_rmse"] = grouped["std_rmse"].fillna(0.0)
    return grouped.sort_values(group_cols[:-1] + ["mean_mae"] if group_cols else ["mean_mae"])


def write_method_inventory(out_dir: Path) -> None:
    rows = []
    for method, (group, desc, source, needs_meta, needs_compound) in sorted(METHOD_INFO.items()):
        rows.append(
            {
                "method": method,
                "route_group": group,
                "route_description": desc,
                "source_type": source,
                "numeric_result_source": "rerun under SOMOL same split" if "rerun" in source or "same_split" in source else "not used numerically",
                "requires_assay_metadata": needs_meta,
                "requires_compound_features": needs_compound,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "method_route_inventory.csv", index=False)


def write_sota_readiness(headline: pd.DataFrame, out_dir: Path, primary_method: str, required_runs: int) -> None:
    scoped = headline[(headline["scope"] == "micro") & (headline["endpoint"].notna())].copy()
    scoped = scoped[scoped["n_runs"] >= required_runs].copy()
    rows = []
    for endpoint, group in scoped.groupby("endpoint"):
        ours = group[group["method"] == primary_method]
        if ours.empty:
            continue
        ours_row = ours.sort_values("mean_mae").iloc[0]
        competitors = group[~group["route_group"].isin(["ours", "ours_ablation"])].sort_values("mean_mae")
        best_comp = competitors.iloc[0] if len(competitors) else None
        primary_vs_comp = pd.concat([competitors, ours], ignore_index=True)
        rows.append(
            {
                "endpoint": endpoint,
                "ours_method": primary_method,
                "ours_micro_mae_mean": float(ours_row["mean_mae"]),
                "ours_micro_mae_std": float(ours_row["std_mae"]),
                "best_competitor": None if best_comp is None else best_comp["method"],
                "best_competitor_route": None if best_comp is None else best_comp["route_group"],
                "best_competitor_micro_mae_mean": None if best_comp is None else float(best_comp["mean_mae"]),
                "delta_best_competitor_minus_ours": None if best_comp is None else float(best_comp["mean_mae"] - ours_row["mean_mae"]),
                "rank_among_complete_methods": int((group["mean_mae"] < ours_row["mean_mae"]).sum() + 1),
                "rank_among_complete_non_ours_plus_primary": int((primary_vs_comp["mean_mae"] < ours_row["mean_mae"]).sum() + 1),
                "n_methods": int(group["method"].nunique()),
                "required_runs": int(required_runs),
            }
        )
    readiness = pd.DataFrame(rows)
    readiness.to_csv(out_dir / "sota_readiness_by_endpoint.csv", index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "criterion": f"{primary_method} ranks first among complete non-ours comparators by mean test micro MAE on every endpoint with >=8 rerun methods",
        "n_endpoints": int(len(readiness)),
        "primary_method": primary_method,
        "required_runs": int(required_runs),
        "all_endpoints_rank1_vs_non_ours": bool(len(readiness) > 0 and (readiness["rank_among_complete_non_ours_plus_primary"] == 1).all()),
        "all_endpoints_have_8plus_methods": bool(len(readiness) > 0 and (readiness["n_methods"] >= 8).all()),
        "claim_supported": bool(len(readiness) > 0 and (readiness["rank_among_complete_non_ours_plus_primary"] == 1).all() and (readiness["n_methods"] >= 8).all()),
    }
    (out_dir / "sota_claim_readiness.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--endpoints", default="IC50,KI,KD,EC50")
    parser.add_argument("--seeds", default="13,17,23,29,31")
    parser.add_argument("--primary-method", default="somol_guarded_stack_micro")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    endpoints = [x.upper() for x in parse_csv(args.endpoints)]
    seeds = [int(x) for x in parse_csv(args.seeds)]
    benchmark_root = root / "results/benchmark"
    out_dir = benchmark_root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(benchmark_root, endpoints, seeds)
    if metrics.empty:
        raise RuntimeError("No benchmark metrics found.")
    metrics.to_csv(out_dir / "all_rerun_metrics_long.csv", index=False)

    headline_input = metrics[
        (metrics["split"] == "test")
        & (metrics["bucket"] == "all")
        & (metrics["scope"].isin(["micro", "macro_assay_pair", "macro_target"]))
    ].copy()
    headline = aggregate_mean_std(
        headline_input,
        ["endpoint", "route_group", "source_type", "method", "scope"],
    )
    headline.to_csv(out_dir / "headline_mean_std_by_endpoint.csv", index=False)

    pooled = aggregate_mean_std(
        headline_input,
        ["route_group", "source_type", "method", "scope"],
    )
    pooled.to_csv(out_dir / "headline_mean_std_pooled_endpoints.csv", index=False)

    special = collect_special_scenarios(benchmark_root, endpoints, seeds)
    if not special.empty:
        special.to_csv(out_dir / "special_scenario_metrics_long.csv", index=False)
        special_summary = aggregate_mean_std(
            special,
            ["scenario", "route_group", "source_type", "method", "scope"],
        )
        special_summary.to_csv(out_dir / "special_scenario_mean_std.csv", index=False)

    write_method_inventory(out_dir)
    write_sota_readiness(headline, out_dir, args.primary_method, len(seeds))
    print(f"Wrote benchmark tables to {out_dir}")


if __name__ == "__main__":
    main()
