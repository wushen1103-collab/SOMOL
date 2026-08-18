#!/usr/bin/env python3
"""Evaluate cycle consistency by composing the fitted affine transport operators."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ENDPOINTS = ("ec50", "ic50", "kd", "ki")
SEEDS = (13, 17, 23, 29, 31)
METHOD_FILES = {
    "gauge_selector_micro": (
        "gauge_calibration/gauge_calibration_predictions.parquet",
        "pred_gauge_calibration_selector_micro",
    ),
    "somol_robust_blend_bucket": (
        "somol_robust_stack/somol_robust_stack_predictions.parquet",
        "pred_somol_robust_blend_bucket",
    ),
}
KEYS = ["split", "compound_target_key", "assay_src", "assay_dst", "y_src", "y_dst"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--cycle-cap", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ac_experiments/tables"))
    return parser.parse_args()


def canonical_cycle(nodes: tuple[int, ...]) -> tuple[int, ...]:
    rotations = [nodes[i:] + nodes[:i] for i in range(len(nodes))]
    return min(rotations)


def support_bucket(value: int) -> str:
    if value <= 4:
        return "01_1_4"
    if value <= 19:
        return "02_5_19"
    if value <= 99:
        return "03_20_99"
    return "04_100plus"


def degree_bucket(value: int) -> str:
    if value <= 9:
        return "01_1_9"
    if value <= 99:
        return "02_10_99"
    if value <= 999:
        return "03_100_999"
    return "04_1000plus"


def fit_affine(frame: pd.DataFrame, prediction: str) -> tuple[dict, float]:
    work = frame[["assay_src", "assay_dst", "y_src", prediction]].dropna().copy()
    work["x2"] = work["y_src"] ** 2
    work["xy"] = work["y_src"] * work[prediction]
    grouped = work.groupby(["assay_src", "assay_dst"], sort=False)
    stats = grouped.agg(
        n=("y_src", "size"),
        sx=("y_src", "sum"),
        sy=(prediction, "sum"),
        sxx=("x2", "sum"),
        sxy=("xy", "sum"),
    )
    denominator = stats["n"] * stats["sxx"] - stats["sx"] ** 2
    stats = stats.loc[denominator.abs() > 1e-10].copy()
    denominator = denominator.loc[stats.index]
    stats["slope"] = (stats["n"] * stats["sxy"] - stats["sx"] * stats["sy"]) / denominator
    stats["intercept"] = (stats["sy"] - stats["slope"] * stats["sx"]) / stats["n"]

    check = work.merge(stats[["slope", "intercept"]], left_on=["assay_src", "assay_dst"], right_index=True)
    check["residual"] = (check[prediction] - check["intercept"] - check["slope"] * check["y_src"]).abs()
    edge_residual = check.groupby(["assay_src", "assay_dst"])["residual"].max()
    valid = edge_residual.loc[edge_residual <= 1e-7].index
    stats = stats.loc[valid]
    coefficients = {
        (int(src), int(dst)): (float(row.intercept), float(row.slope))
        for (src, dst), row in stats.iterrows()
    }
    return coefficients, float(edge_residual.max()) if len(edge_residual) else np.nan


def load_operators(repo: Path, endpoint: str, seed: int) -> tuple[dict, dict, dict, dict, dict]:
    base = repo / "results" / "benchmark" / endpoint / f"seed_{seed}"
    final_path, final_col = METHOD_FILES["somol_robust_blend_bucket"]
    frame = pd.read_parquet(base / final_path, columns=KEYS + [final_col])

    for method, (relative, prediction) in METHOD_FILES.items():
        if method == "somol_robust_blend_bucket":
            continue
        other = pd.read_parquet(base / relative, columns=KEYS + [prediction])
        if not frame[KEYS].equals(other[KEYS]):
            raise RuntimeError(f"Prediction rows are not aligned for {endpoint} seed {seed}: {method}")
        frame[prediction] = other[prediction].to_numpy()

    train = frame.loc[frame["split"].eq("train")].copy()
    edge_groups = train.groupby(["assay_src", "assay_dst"], sort=False)
    support = edge_groups.size().astype(int).to_dict()
    quantiles = edge_groups["y_src"].quantile([0.25, 0.5, 0.75]).unstack().to_dict("index")

    operators: dict[str, dict] = {
        "empirical_delta": {
            (int(src), int(dst)): (float((block["y_dst"] - block["y_src"]).mean()), 1.0)
            for (src, dst), block in edge_groups
        }
    }
    max_residual = {"empirical_delta": 0.0}
    for method, (_, prediction) in METHOD_FILES.items():
        operators[method], max_residual[method] = fit_affine(frame, prediction)

    common = set(support)
    for coefficients in operators.values():
        common &= set(coefficients)
    operators = {method: {edge: values for edge, values in coefficients.items() if edge in common}
                 for method, coefficients in operators.items()}

    neighbors: dict[int, set[int]] = defaultdict(set)
    undirected: dict[int, set[int]] = defaultdict(set)
    for src, dst in common:
        neighbors[src].add(dst)
        undirected[src].add(dst)
        undirected[dst].add(src)
    degrees = {node: len(adjacent) for node, adjacent in undirected.items()}
    return operators, support, quantiles, neighbors, degrees, max_residual


def sample_cycles(neighbors: dict[int, set[int]], length: int, cap: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    nodes = np.array(sorted(neighbors), dtype=np.int64)
    predecessors: dict[int, set[int]] = defaultdict(set)
    for src, adjacent in neighbors.items():
        for dst in adjacent:
            predecessors[dst].add(src)
    cycles: set[tuple[int, ...]] = set()
    attempts = max(cap * 100, 100_000)
    for _ in range(attempts):
        start = int(rng.choice(nodes))
        second_candidates = list(neighbors[start] - {start})
        if not second_candidates:
            continue
        second = int(rng.choice(second_candidates))
        if length == 3:
            closing = list((neighbors[second] & predecessors[start]) - {start, second})
            if not closing:
                continue
            path = (start, second, int(rng.choice(closing)))
        else:
            third_candidates = list(neighbors[second] - {start, second})
            if not third_candidates:
                continue
            third = int(rng.choice(third_candidates))
            closing = list((neighbors[third] & predecessors[start]) - {start, second, third})
            if not closing:
                continue
            path = (start, second, third, int(rng.choice(closing)))
        cycles.add(canonical_cycle(path))
        if len(cycles) >= cap:
            break
    return sorted(cycles)


def compose_cycle(cycle: tuple[int, ...], coefficients: dict, quantiles: dict) -> tuple[float, float]:
    signed = []
    n = len(cycle)
    for rotation in range(n):
        ordered = cycle[rotation:] + cycle[:rotation]
        first_edge = (ordered[0], ordered[1])
        y_values = np.array(list(quantiles[first_edge].values()), dtype=float)
        for y0 in y_values:
            y = float(y0)
            for index, src in enumerate(ordered):
                dst = ordered[(index + 1) % n]
                intercept, slope = coefficients[(src, dst)]
                y = intercept + slope * y
            signed.append(y - y0)
    residual = np.asarray(signed, dtype=float)
    return float(np.mean(np.abs(residual))), float(np.mean(residual))


def main() -> None:
    args = parse_args()
    output = args.repo / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    long_rows = []
    diagnostics = []

    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        for seed in SEEDS:
            operators, support, quantiles, neighbors, degrees, residuals = load_operators(args.repo, endpoint, seed)
            for length in (3, 4):
                rng = np.random.default_rng(20260818 + endpoint_index * 1000 + seed * 10 + length)
                cycles = sample_cycles(neighbors, length, args.cycle_cap, rng)
                diagnostics.append({
                    "endpoint": endpoint.upper(),
                    "seed": seed,
                    "cycle_len": length,
                    "n_cycles": len(cycles),
                    "n_common_directed_edges": sum(len(values) for values in neighbors.values()),
                    "max_gauge_affine_fit_residual": residuals["gauge_selector_micro"],
                    "max_somol_affine_fit_residual": residuals["somol_robust_blend_bucket"],
                })
                print(endpoint.upper(), seed, f"length={length}", f"cycles={len(cycles)}", flush=True)
                for cycle_index, cycle in enumerate(cycles):
                    edges = [(cycle[i], cycle[(i + 1) % length]) for i in range(length)]
                    min_support = min(support[edge] for edge in edges)
                    min_degree = min(degrees[node] for node in cycle)
                    for method, coefficients in operators.items():
                        absolute, signed = compose_cycle(cycle, coefficients, quantiles)
                        long_rows.append({
                            "endpoint": endpoint.upper(),
                            "seed": seed,
                            "cycle_len": length,
                            "cycle_id": f"{seed}_{length}_{cycle_index}",
                            "min_edge_support": min_support,
                            "min_edge_support_bucket": support_bucket(min_support),
                            "min_node_degree": min_degree,
                            "min_node_degree_bucket": degree_bucket(min_degree),
                            "method": method,
                            "cycle_abs_closure": absolute,
                            "cycle_signed_closure": signed,
                        })

    long = pd.DataFrame(long_rows)
    summary = (
        long.groupby(
            ["endpoint", "cycle_len", "method", "min_edge_support_bucket", "min_node_degree_bucket"],
            as_index=False,
        )
        .agg(
            n=("cycle_abs_closure", "size"),
            mean_abs_closure=("cycle_abs_closure", "mean"),
            std_abs_closure=("cycle_abs_closure", "std"),
            median_abs_closure=("cycle_abs_closure", "median"),
            mean_signed_closure=("cycle_signed_closure", "mean"),
        )
    )
    overall = (
        long.groupby(["endpoint", "method"], as_index=False)
        .agg(
            n=("cycle_abs_closure", "size"),
            mean_abs_closure=("cycle_abs_closure", "mean"),
            median_abs_closure=("cycle_abs_closure", "median"),
            mean_signed_closure=("cycle_signed_closure", "mean"),
        )
    )
    long.to_csv(output / "graph_cycle_consistency_long.csv", index=False)
    summary.to_csv(output / "graph_cycle_consistency_summary.csv", index=False)
    overall.to_csv(output / "graph_cycle_consistency_overall.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "graph_cycle_consistency_diagnostics.csv", index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
