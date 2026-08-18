#!/usr/bin/env python
"""Run endpoint x split-seed SOMOL transport benchmarks.

This runner keeps every method on the same frozen endpoint/seed split. It is
intended for paper-level SOTA evidence, not for exploratory one-off runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def split_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        print("\n".join(tail), file=sys.stderr)
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def maybe_clean_path(root: Path, endpoint: str, processed_dir: Path) -> Path:
    lower = endpoint.lower()
    local = processed_dir / f"{lower}_clean.parquet"
    if local.exists():
        return local
    legacy = root / "data/processed" / f"{lower}_clean.parquet"
    if legacy.exists():
        return legacy
    return local


def maybe_duplicate_path(root: Path, endpoint: str, processed_dir: Path) -> Path:
    lower = endpoint.lower()
    local = processed_dir / f"{lower}_duplicate_stats.parquet"
    if local.exists():
        return local
    legacy = root / "data/processed" / f"{lower}_duplicate_stats.parquet"
    if legacy.exists():
        return legacy
    return local


def count_pairs(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--sqlite", default="data/raw/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db")
    parser.add_argument("--endpoints", default="IC50,KI,KD,EC50")
    parser.add_argument("--seeds", default="13,17,23,29,31")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-assay-size", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-assay-context", action="store_true")
    parser.add_argument("--skip-exact-no-gauge", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    py = Path(sys.executable).resolve()
    sqlite_path = (root / args.sqlite).resolve()
    if not sqlite_path.exists():
        raise RuntimeError(f"ChEMBL SQLite not found: {sqlite_path}")

    endpoints = [x.upper() for x in split_csv(args.endpoints)]
    seeds = [int(x) for x in split_csv(args.seeds)]
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    benchmark_root = root / "results/benchmark"
    data_root = root / "data/benchmark"
    log_root = root / "logs/benchmark"
    records: list[dict[str, object]] = []

    for endpoint in endpoints:
        lower = endpoint.lower()
        processed_dir = data_root / lower / "processed"
        endpoint_results = benchmark_root / lower
        table_dir = endpoint_results / "tables"
        gate_dir = endpoint_results / "gate"
        logs = log_root / lower
        processed_dir.mkdir(parents=True, exist_ok=True)
        endpoint_results.mkdir(parents=True, exist_ok=True)

        clean = maybe_clean_path(root, endpoint, processed_dir)
        duplicates = maybe_duplicate_path(root, endpoint, processed_dir)
        if not clean.exists():
            run_command(
                [
                    str(py),
                    str(root / "scripts/extract_endpoint.py"),
                    "--sqlite",
                    str(sqlite_path),
                    "--endpoint",
                    endpoint,
                    "--output-dir",
                    str(processed_dir),
                    "--workers",
                    str(args.workers),
                ],
                logs / "extract.log",
                env,
            )
            clean = processed_dir / f"{lower}_clean.parquet"
            duplicates = processed_dir / f"{lower}_duplicate_stats.parquet"

        assay_metadata = table_dir / "assay_metadata_table.csv"
        if not assay_metadata.exists() or not args.skip_existing:
            run_command(
                [
                    str(py),
                    str(root / "scripts/run_gate.py"),
                    "--clean",
                    str(clean),
                    "--duplicates",
                    str(duplicates),
                    "--endpoint",
                    endpoint,
                    "--output-dir",
                    str(gate_dir),
                    "--table-dir",
                    str(table_dir),
                    "--min-assay-size",
                    str(args.min_assay_size),
                ],
                logs / "gate.log",
                env,
            )

        for seed in seeds:
            run_root = endpoint_results / f"seed_{seed}"
            split_dir = data_root / lower / f"seed_{seed}" / "splits"
            pairs = data_root / lower / f"seed_{seed}" / "transport_pairs.parquet"
            pair_summary = data_root / lower / f"seed_{seed}" / "transport_pairs_summary.json"
            split_dir.mkdir(parents=True, exist_ok=True)
            pairs.parent.mkdir(parents=True, exist_ok=True)
            run_root.mkdir(parents=True, exist_ok=True)

            if not (split_dir / "split_assignments.parquet").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/make_splits.py"),
                        "--clean",
                        str(clean),
                        "--output-dir",
                        str(split_dir),
                        "--seed",
                        str(seed),
                    ],
                    logs / f"seed_{seed}_splits.log",
                    env,
                )
            if not pairs.exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/build_transport_pairs.py"),
                        "--clean",
                        str(clean),
                        "--splits",
                        str(split_dir / "split_assignments.parquet"),
                        "--endpoint",
                        endpoint,
                        "--output",
                        str(pairs),
                        "--summary",
                        str(pair_summary),
                        "--min-assay-size",
                        str(args.min_assay_size),
                    ],
                    logs / f"seed_{seed}_pairs.log",
                    env,
                )

            summary = count_pairs(pair_summary)
            n_test = int(summary.get("by_split", {}).get("test", {}).get("directed_pairs", 0))
            if n_test == 0:
                records.append({"endpoint": endpoint, "seed": seed, "status": "skipped_no_test_pairs", **summary})
                continue

            baseline_dir = run_root / "transport_baselines"
            if not (baseline_dir / "transport_baseline_metrics.csv").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/run_transport_baselines.py"),
                        "--pairs",
                        str(pairs),
                        "--output-dir",
                        str(baseline_dir),
                    ],
                    logs / f"seed_{seed}_baselines.log",
                    env,
                )

            support_dir = run_root / "support_gated_selector"
            if not (support_dir / "support_gated_metrics.csv").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/run_support_gated_selector.py"),
                        "--baseline-predictions",
                        str(baseline_dir / "transport_baseline_predictions.parquet"),
                        "--output-dir",
                        str(support_dir),
                    ],
                    logs / f"seed_{seed}_support_gated.log",
                    env,
                )

            calibration_dir = run_root / "calibration_grid"
            if not (calibration_dir / "calibration_grid_metrics.csv").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/tune_calibration_grid.py"),
                        "--baseline-predictions",
                        str(baseline_dir / "transport_baseline_predictions.parquet"),
                        "--output-dir",
                        str(calibration_dir),
                    ],
                    logs / f"seed_{seed}_calibration_grid.log",
                    env,
                )

            gauge_dir = run_root / "gauge_calibration"
            if not (gauge_dir / "gauge_calibration_metrics.csv").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/tune_gauge_calibration.py"),
                        "--baseline-predictions",
                        str(baseline_dir / "transport_baseline_predictions.parquet"),
                        "--output-dir",
                        str(gauge_dir),
                    ],
                    logs / f"seed_{seed}_gauge.log",
                    env,
                )

            graph_dir = run_root / "graph_gauge_calibration"
            if not (graph_dir / "graph_gauge_calibration_metrics.csv").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/tune_graph_gauge_calibration.py"),
                        "--baseline-predictions",
                        str(baseline_dir / "transport_baseline_predictions.parquet"),
                        "--output-dir",
                        str(graph_dir),
                    ],
                    logs / f"seed_{seed}_graph_gauge.log",
                    env,
                )

            blend_dir = run_root / "somol_blend"
            if not (blend_dir / "somol_blend_metrics.csv").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/tune_somol_blend.py"),
                        "--gauge-predictions",
                        str(gauge_dir / "gauge_calibration_predictions.parquet"),
                        "--graph-predictions",
                        str(graph_dir / "graph_gauge_calibration_predictions.parquet"),
                        "--output-dir",
                        str(blend_dir),
                    ],
                    logs / f"seed_{seed}_somol_blend.log",
                    env,
                )

            stack_dir = run_root / "somol_stack"
            if not (stack_dir / "somol_stack_metrics.csv").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/tune_somol_stack.py"),
                        "--calibration-predictions",
                        str(calibration_dir / "calibration_grid_predictions.parquet"),
                        "--gauge-predictions",
                        str(gauge_dir / "gauge_calibration_predictions.parquet"),
                        "--graph-predictions",
                        str(graph_dir / "graph_gauge_calibration_predictions.parquet"),
                        "--output-dir",
                        str(stack_dir),
                    ],
                    logs / f"seed_{seed}_somol_stack.log",
                    env,
                )

            guarded_dir = run_root / "somol_guarded_stack"
            if not (guarded_dir / "somol_guarded_stack_metrics.csv").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/tune_somol_guarded_stack.py"),
                        "--baseline-predictions",
                        str(baseline_dir / "transport_baseline_predictions.parquet"),
                        "--stack-predictions",
                        str(stack_dir / "somol_stack_predictions.parquet"),
                        "--output-dir",
                        str(guarded_dir),
                    ],
                    logs / f"seed_{seed}_somol_guarded_stack.log",
                    env,
                )

            method_comparison_dir = run_root / "method_comparison_baselines"
            if not (method_comparison_dir / "method_comparison_metrics.csv").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/run_method_comparison_baselines.py"),
                        "--baseline-predictions",
                        str(baseline_dir / "transport_baseline_predictions.parquet"),
                        "--output-dir",
                        str(method_comparison_dir),
                        "--endpoint",
                        endpoint,
                        "--seed",
                        str(seed),
                    ],
                    logs / f"seed_{seed}_method_comparison.log",
                    env,
                )

            robust_dir = run_root / "somol_robust_stack"
            if not (robust_dir / "somol_robust_stack_metrics.csv").exists() or not args.skip_existing:
                run_command(
                    [
                        str(py),
                        str(root / "scripts/tune_somol_robust_stack.py"),
                        "--baseline-predictions",
                        str(baseline_dir / "transport_baseline_predictions.parquet"),
                        "--guarded-predictions",
                        str(guarded_dir / "somol_guarded_stack_predictions.parquet"),
                        "--method-comparison-predictions",
                        str(method_comparison_dir / "method_comparison_predictions.parquet"),
                        "--output-dir",
                        str(robust_dir),
                    ],
                    logs / f"seed_{seed}_somol_robust_stack.log",
                    env,
                )

            if not args.skip_exact_no_gauge:
                ablation_dir = run_root / "somol_robust_no_gauge_ablation"
                if not (ablation_dir / "somol_robust_no_gauge_metrics.csv").exists() or not args.skip_existing:
                    run_command(
                        [
                            str(py),
                            str(root / "scripts/run_exact_no_gauge_ablation.py"),
                            "--baseline-predictions",
                            str(baseline_dir / "transport_baseline_predictions.parquet"),
                            "--support-predictions",
                            str(support_dir / "support_gated_predictions.parquet"),
                            "--calibration-predictions",
                            str(calibration_dir / "calibration_grid_predictions.parquet"),
                            "--method-comparison-predictions",
                            str(method_comparison_dir / "method_comparison_predictions.parquet"),
                            "--full-model-predictions",
                            str(robust_dir / "somol_robust_stack_predictions.parquet"),
                            "--output-root",
                            str(run_root),
                            "--endpoint",
                            endpoint,
                            "--seed",
                            str(seed),
                        ],
                        logs / f"seed_{seed}_exact_no_gauge.log",
                        env,
                    )


            if not args.skip_assay_context:
                context_dir = run_root / "assay_context_surrogates"
                if not (context_dir / "assay_context_surrogate_metrics.csv").exists() or not args.skip_existing:
                    run_command(
                        [
                            str(py),
                            str(root / "scripts/run_assay_context_surrogates.py"),
                            "--baseline-predictions",
                            str(baseline_dir / "transport_baseline_predictions.parquet"),
                            "--assay-metadata",
                            str(assay_metadata),
                            "--output-dir",
                            str(context_dir),
                        ],
                        logs / f"seed_{seed}_assay_context.log",
                        env,
                    )

            records.append(
                {
                    "endpoint": endpoint,
                    "seed": seed,
                    "status": "done",
                    "clean_rows": int(pd.read_parquet(clean, columns=["assay_id"]).shape[0]),
                    "transport_pairs": int(summary.get("n_pairs_directed", 0)),
                    "test_pairs": n_test,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

    manifest = pd.DataFrame(records)
    manifest.to_csv(benchmark_root / "benchmark_run_manifest.csv", index=False)
    print(manifest.to_string(index=False))

    if not args.skip_postprocess:
        endpoint_text = ",".join(endpoints)
        seed_text = ",".join(str(seed) for seed in seeds)
        run_command(
            [
                str(py),
                str(root / "scripts/collect_benchmark_tables.py"),
                "--root",
                str(root),
                "--endpoints",
                endpoint_text,
                "--seeds",
                seed_text,
                "--primary-method",
                "somol_robust_blend_bucket",
            ],
            log_root / "collect_benchmark_tables.log",
            env,
        )
        run_command(
            [str(py), str(root / "scripts/run_corrected_inference.py"), "--repo", str(root)],
            log_root / "corrected_inference.log",
            env,
        )
        run_command(
            [str(py), str(root / "scripts/run_operator_cycle_consistency.py"), "--repo", str(root)],
            log_root / "operator_cycle_consistency.log",
            env,
        )


if __name__ == "__main__":
    main()
