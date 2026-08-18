#!/usr/bin/env python3
"""Verify compact release tables against the reported benchmark values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"
TOLERANCE = 5e-12


def assert_close(label: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, atol=TOLERANCE, rtol=0.0):
        raise AssertionError(f"{label}: expected {expected:.12g}, found {actual:.12g}")


def main() -> None:
    headline = pd.read_csv(SUMMARY / "headline_mean_std_by_endpoint.csv")
    final = headline[
        headline["method"].eq("somol_robust_blend_bucket")
        & headline["scope"].eq("micro")
    ].set_index("endpoint")
    expected_final = {
        "EC50": 0.16354696429126087,
        "IC50": 0.18747192589121944,
        "KD": 0.22610612093794097,
        "KI": 0.16993791716328244,
    }
    for endpoint, expected in expected_final.items():
        row = final.loc[endpoint]
        assert int(row["n_runs"]) == 5
        assert_close(f"{endpoint} final MAE", float(row["mean_mae"]), expected)

    ablation = pd.read_csv(SUMMARY / "somol_robust_exact_no_gauge_ablation_mean_std.csv")
    no_gauge = ablation[
        ablation["method"].eq("somol_robust_blend_bucket_exact_no_gauge")
        & ablation["scope"].eq("micro")
    ].set_index("endpoint")
    expected_no_gauge = {
        "EC50": 0.16382934631025978,
        "IC50": 0.19026632540674460,
        "KD": 0.23232428979852750,
        "KI": 0.17226530774921445,
    }
    for endpoint, expected in expected_no_gauge.items():
        assert_close(f"{endpoint} no-gauge MAE", float(no_gauge.loc[endpoint, "mean_mae"]), expected)

    cycles = pd.read_csv(SUMMARY / "graph_cycle_consistency_overall.csv")
    expected_cycles = {
        ("EC50", "somol_robust_blend_bucket"): 0.3107932791962855,
        ("IC50", "somol_robust_blend_bucket"): 0.15528084293233368,
        ("KD", "somol_robust_blend_bucket"): 0.2863368986172758,
        ("KI", "somol_robust_blend_bucket"): 0.09268188036992796,
    }
    cycle_index = cycles.set_index(["endpoint", "method"])
    for key, expected in expected_cycles.items():
        assert_close(f"{key[0]} cycle error", float(cycle_index.loc[key, "mean_abs_closure"]), expected)

    inference = pd.read_csv(SUMMARY / "robust_somol_paired_seed_significance.csv")
    ic50 = inference[
        inference["endpoint"].eq("IC50") & inference["seed"].eq("seed_summary")
    ].iloc[0]
    assert_close("IC50 corrected p", float(ic50["corrected_p_greater"]), 0.0002710890414443175)
    print("Verified final MAEs, exact no-gauge ablations, cycle errors, and corrected inference.")


if __name__ == "__main__":
    main()
