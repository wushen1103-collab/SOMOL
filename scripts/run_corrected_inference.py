#!/usr/bin/env python3
"""Correct repeated-split inference and stratify block bootstrap by split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ENDPOINTS = ("ic50", "ec50", "ki", "kd")
SEEDS = (13, 17, 23, 29, 31)
BLOCKS = {
    "assay_pair": ["assay_src", "assay_dst"],
    "target": ["target_chembl_id"],
    "assay_dst_node": ["assay_dst"],
}
PRIMARY = "pred_somol_robust_blend_bucket"
COMPARATOR = "pred_robust_trimmed_bias_fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument("--test-train-ratio", type=float, default=0.15 / 0.70)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ac_experiments/tables"))
    return parser.parse_args()


def bootstrap_means(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty(n_bootstrap, dtype=float)
    chunk = 20
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        result[start:stop] = values[indices].mean(axis=1)
    return result


def main() -> None:
    args = parse_args()
    output = args.repo / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    bootstrap_rows = []

    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        split_frames = []
        deltas = []
        for seed in SEEDS:
            path = (
                args.repo / "results" / "benchmark" / endpoint / f"seed_{seed}"
                / "somol_robust_stack" / "somol_robust_stack_predictions.parquet"
            )
            frame = pd.read_parquet(
                path,
                columns=["split", "target_chembl_id", "assay_src", "assay_dst", "y_dst", PRIMARY, COMPARATOR],
            )
            test = frame.loc[frame["split"].eq("test")].copy()
            test["delta"] = (test[COMPARATOR] - test["y_dst"]).abs() - (test[PRIMARY] - test["y_dst"]).abs()
            primary_mae = float((test[PRIMARY] - test["y_dst"]).abs().mean())
            comparator_mae = float((test[COMPARATOR] - test["y_dst"]).abs().mean())
            delta = comparator_mae - primary_mae
            deltas.append(delta)
            split_frames.append(test)
            paired_rows.append({
                "endpoint": endpoint.upper(), "seed": seed,
                "primary_method": "somol_robust_blend_bucket",
                "competitor": "robust_trimmed_bias_fallback",
                "primary_mae": primary_mae, "competitor_mae": comparator_mae,
                "delta_competitor_minus_primary": delta, "n": len(test),
            })

        differences = np.asarray(deltas, dtype=float)
        mean_delta = float(differences.mean())
        standard_deviation = float(differences.std(ddof=1))
        corrected_se = standard_deviation * np.sqrt(1.0 / len(SEEDS) + args.test_train_ratio)
        corrected_t = mean_delta / corrected_se
        corrected_p = float(stats.t.sf(corrected_t, df=len(SEEDS) - 1))
        paired_rows.append({
            "endpoint": endpoint.upper(), "seed": "seed_summary",
            "primary_method": "somol_robust_blend_bucket",
            "competitor": "robust_trimmed_bias_fallback",
            "delta_competitor_minus_primary": mean_delta, "n": len(SEEDS),
            "delta_std": standard_deviation, "corrected_standard_error": corrected_se,
            "corrected_resampled_t": corrected_t, "corrected_p_greater": corrected_p,
            "test_train_ratio": args.test_train_ratio,
        })

        for block_index, (block_name, columns) in enumerate(BLOCKS.items()):
            rng = np.random.default_rng(args.bootstrap_seed + endpoint_index * 100 + block_index)
            split_values = [frame.groupby(columns, sort=False)["delta"].mean().to_numpy() for frame in split_frames]
            within_split = np.vstack([
                bootstrap_means(values, args.n_bootstrap, rng) for values in split_values
            ])
            sampled_splits = rng.integers(0, len(SEEDS), size=(len(SEEDS), args.n_bootstrap))
            replicates = np.take_along_axis(within_split, sampled_splits, axis=0).mean(axis=0)
            point = float(np.mean([values.mean() for values in split_values]))
            low, high = np.quantile(replicates, [0.025, 0.975])
            bootstrap_rows.append({
                "endpoint": endpoint.upper(),
                "comparison": "somol_robust_blend_bucket_vs_robust_trimmed_bias_fallback",
                "block_unit": block_name,
                "delta_mean": point,
                "ci_low": float(low), "ci_high": float(high),
                "p_le_zero": float(np.mean(replicates <= 0)),
                "bootstrap_win_rate": float(np.mean(replicates > 0)),
                "n_blocks": int(sum(len(values) for values in split_values)),
                "n_splits": len(SEEDS),
                "bootstrap_design": "hierarchical_split_then_within_split_block",
            })

    paired = pd.DataFrame(paired_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    paired.to_csv(output / "robust_somol_paired_seed_significance.csv", index=False)
    bootstrap.to_csv(output / "robust_somol_block_bootstrap_significance.csv", index=False)
    print(paired.loc[paired["seed"].eq("seed_summary")].to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
