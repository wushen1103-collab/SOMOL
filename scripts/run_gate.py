#!/usr/bin/env python
"""Run SOMOL Phase 0 Go/No-Go gates and produce paper-ready tables."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


STRUCTURED_METADATA_FIELDS = [
    "assay_type",
    "bao_format",
    "assay_organism",
    "assay_tissue",
    "assay_cell_type",
    "assay_subcellular_fraction",
    "relationship_type",
    "confidence_score",
    "src_id",
    "pub_year",
    "doc_type",
    "journal",
]


def nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "") & (series.astype(str).str.lower() != "nan")


def bool_gate(value: bool) -> str:
    return "PASS" if value else "FAIL"


def load_endpoint(clean_path: Path, duplicate_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    df = pd.read_parquet(clean_path)
    duplicates = pd.read_parquet(duplicate_path) if duplicate_path and duplicate_path.exists() else None
    return df, duplicates


def assay_metadata_table(df: pd.DataFrame, y_col: str, min_assay_size: int) -> pd.DataFrame:
    agg = {
        y_col: ["size", "mean", "std"],
        "description": "first",
    }
    for field in STRUCTURED_METADATA_FIELDS:
        if field in df.columns:
            agg[field] = "first"
    assay = df.groupby("assay_id").agg(agg)
    assay.columns = ["_".join(c).rstrip("_") for c in assay.columns.to_flat_index()]
    assay = assay.rename(columns={f"{y_col}_size": "n_measurements", f"{y_col}_mean": "assay_mean", f"{y_col}_std": "assay_scale"})
    assay = assay.reset_index()
    assay["assay_scale"] = assay["assay_scale"].fillna(0.0)
    assay["has_description"] = nonempty(assay["description_first"])
    structured_cols = [f"{c}_first" for c in STRUCTURED_METADATA_FIELDS if f"{c}_first" in assay.columns]
    assay["n_structured_fields"] = 0
    for col in structured_cols:
        assay["n_structured_fields"] += nonempty(assay[col]).astype(int)
    assay["metadata_complete"] = assay["has_description"] & (assay["n_structured_fields"] >= 2)
    return assay[assay["n_measurements"] >= min_assay_size].copy()


def compute_overlap(
    df: pd.DataFrame,
    y_col: str,
    edge_min_anchors: int,
    max_cross_diffs_per_target: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[float]]]:
    overlap_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    cross_diffs: dict[str, list[float]] = {}

    for target, target_df in df.groupby("target_chembl_id", sort=False):
        edge_counts: Counter[tuple[int, int]] = Counter()
        target_diffs: list[float] = []
        target_name = str(target_df["target_name"].dropna().iloc[0]) if target_df["target_name"].notna().any() else ""

        for _, sub in target_df.groupby("compound_target_key", sort=False):
            assay_vals = sub[["assay_id", y_col]].drop_duplicates("assay_id").sort_values("assay_id")
            if len(assay_vals) < 2:
                continue
            assays = assay_vals["assay_id"].astype(int).tolist()
            values = assay_vals[y_col].astype(float).tolist()
            for i, j in combinations(range(len(assays)), 2):
                a, b = assays[i], assays[j]
                edge_counts[(a, b)] += 1
                if len(target_diffs) < max_cross_diffs_per_target:
                    target_diffs.append(abs(values[i] - values[j]))

        graph = nx.Graph()
        for (a, b), n in edge_counts.items():
            if n >= edge_min_anchors:
                graph.add_edge(a, b, n_anchors=n)
                overlap_rows.append(
                    {
                        "target_chembl_id": target,
                        "target_name": target_name,
                        "assay_a": a,
                        "assay_b": b,
                        "n_anchors": n,
                    }
                )

        components = [len(c) for c in nx.connected_components(graph)] if graph.number_of_nodes() else []
        target_rows.append(
            {
                "target_chembl_id": target,
                "target_name": target_name,
                "n_measurements": len(target_df),
                "n_assays": int(target_df["assay_id"].nunique()),
                "n_compounds": int(target_df["canonical_smiles_somol"].nunique()),
                "n_transport_edges": graph.number_of_edges(),
                "n_transport_nodes": graph.number_of_nodes(),
                "n_connected_components": len(components),
                "max_component_size": max(components) if components else 0,
                "n_components_ge4": int(sum(1 for c in components if c >= 4)),
                "sampled_cross_diff_median": float(np.median(target_diffs)) if target_diffs else math.nan,
                "sampled_cross_diff_n": len(target_diffs),
            }
        )
        cross_diffs[str(target)] = target_diffs

    return pd.DataFrame(overlap_rows), pd.DataFrame(target_rows), cross_diffs


def assay_effect_gate(
    target_stats: pd.DataFrame,
    duplicates: pd.DataFrame | None,
    cross_diffs: dict[str, list[float]],
    min_targets: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if duplicates is None or duplicates.empty or "range_p" not in duplicates.columns:
        global_repeat = np.array([0.0])
        repeat_by_target: dict[str, np.ndarray] = {}
    else:
        dup = duplicates[duplicates["n_repeats"] > 1].copy()
        global_repeat = dup["range_p"].dropna().astype(float).to_numpy()
        if len(global_repeat) == 0:
            global_repeat = np.array([0.0])
        repeat_by_target = {
            str(k): g["range_p"].dropna().astype(float).to_numpy()
            for k, g in dup.groupby("target_chembl_id")
            if len(g) >= 5
        }

    rows: list[dict[str, object]] = []
    for target, diffs in cross_diffs.items():
        cross = np.asarray(diffs, dtype=float)
        cross = cross[np.isfinite(cross)]
        if len(cross) < 30:
            continue
        repeat = repeat_by_target.get(target, global_repeat)
        repeat = repeat[np.isfinite(repeat)]
        if len(repeat) == 0:
            repeat = np.array([0.0])
        p_value = math.nan
        if len(cross) >= 5 and len(repeat) >= 5:
            try:
                p_value = float(mannwhitneyu(cross, repeat, alternative="greater").pvalue)
            except ValueError:
                p_value = math.nan
        rows.append(
            {
                "target_chembl_id": target,
                "cross_diff_n": len(cross),
                "repeat_noise_n": len(repeat),
                "cross_diff_median": float(np.median(cross)),
                "repeat_noise_median": float(np.median(repeat)),
                "median_gap": float(np.median(cross) - np.median(repeat)),
                "mannwhitney_p": p_value,
                "assay_effect_replicated": bool(np.median(cross) > np.median(repeat) + 0.05 and (math.isnan(p_value) or p_value < 0.05)),
            }
        )
    effect = pd.DataFrame(rows)
    summary = {
        "targets_with_cross_assay_effect": int(effect["assay_effect_replicated"].sum()) if len(effect) else 0,
        "required_targets": min_targets,
        "global_repeat_noise_median": float(np.median(global_repeat)) if len(global_repeat) else math.nan,
    }
    summary["pass"] = summary["targets_with_cross_assay_effect"] >= min_targets
    return effect, summary


def metadata_predictability(assay: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, dict[str, object]]:
    if len(assay) < 10:
        result = pd.DataFrame()
        return result, {"pass": False, "reason": "fewer than 10 assays"}

    feature_cols = [f"{c}_first" for c in STRUCTURED_METADATA_FIELDS if f"{c}_first" in assay.columns]
    work = assay.copy()
    work["description_first"] = work["description_first"].fillna("").astype(str)
    for col in feature_cols:
        work[col] = work[col].fillna("__MISSING__").astype(str)
    work["log_n_measurements"] = np.log1p(work["n_measurements"].astype(float))
    numeric_cols = ["log_n_measurements"]

    def make_pipeline() -> Pipeline:
        return Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        [
                            ("text", TfidfVectorizer(max_features=512, min_df=2, ngram_range=(1, 2)), "description_first"),
                            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), feature_cols),
                            ("num", StandardScaler(), numeric_cols),
                        ],
                        remainder="drop",
                    ),
                ),
                ("model", Ridge(alpha=10.0)),
            ]
        )

    cv = KFold(n_splits=min(5, len(work)), shuffle=True, random_state=random_state)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_state)
    shuffled = work.copy()
    shuffled["description_first"] = rng.permutation(shuffled["description_first"].to_numpy())

    for target_col in ["assay_mean", "assay_scale"]:
        y = work[target_col].astype(float).to_numpy()
        if np.nanstd(y) <= 1e-8:
            rows.append({"target": target_col, "cv_r2": math.nan, "randomized_text_cv_r2": math.nan, "delta_vs_random": math.nan})
            continue
        pred = cross_val_predict(make_pipeline(), work, y, cv=cv, n_jobs=1)
        rand_pred = cross_val_predict(make_pipeline(), shuffled, y, cv=cv, n_jobs=1)
        r2 = float(r2_score(y, pred))
        rand_r2 = float(r2_score(y, rand_pred))
        rows.append({"target": target_col, "cv_r2": r2, "randomized_text_cv_r2": rand_r2, "delta_vs_random": r2 - rand_r2})

    result = pd.DataFrame(rows)
    valid = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["cv_r2", "randomized_text_cv_r2"])
    passed = bool(len(valid) and ((valid["cv_r2"] > 0.0) & (valid["delta_vs_random"] > 0.0)).any())
    summary = {
        "pass": passed,
        "best_cv_r2": float(valid["cv_r2"].max()) if len(valid) else math.nan,
        "best_delta_vs_random": float(valid["delta_vs_random"].max()) if len(valid) else math.nan,
    }
    return result, summary


def write_dataset_card(
    path: Path,
    endpoint: str,
    df: pd.DataFrame,
    main_df: pd.DataFrame,
    gates: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    lines = [
        f"# SOMOL Phase 0 Dataset Card: {endpoint}",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Cleaned Data",
        "",
        f"- Measurements: {len(df):,}",
        f"- Assays: {df['assay_id'].nunique():,}",
        f"- Targets: {df['target_chembl_id'].nunique():,}",
        f"- Compounds: {df['canonical_smiles_somol'].nunique():,}",
        f"- Main assays (n >= {summary['min_assay_size']}): {main_df['assay_id'].nunique():,}",
        f"- Main measurements: {len(main_df):,}",
        "",
        "## Gate Summary",
        "",
        "| Gate | Status | Value | Threshold |",
        "|---|---:|---:|---:|",
    ]
    for gate in gates:
        lines.append(f"| {gate['gate']} | {gate['status']} | {gate['value']} | {gate['threshold']} |")
    lines.extend(
        [
            "",
            "## Route Decision",
            "",
            str(summary["route_decision"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--duplicates")
    parser.add_argument("--endpoint", default="IC50")
    parser.add_argument("--output-dir", default="results/gate")
    parser.add_argument("--table-dir", default="results/tables")
    parser.add_argument("--min-assay-size", type=int, default=20)
    parser.add_argument("--transport-edge-min", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-cross-diffs-per-target", type=int, default=200_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    table_dir = Path(args.table_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    endpoint = args.endpoint.upper()
    y_col = f"p{endpoint}"
    df, duplicates = load_endpoint(Path(args.clean), Path(args.duplicates) if args.duplicates else None)
    df = df.dropna(subset=[y_col, "assay_id", "target_chembl_id", "canonical_smiles_somol"]).copy()
    main_assays = df.groupby("assay_id").size()
    main_assay_ids = main_assays[main_assays >= args.min_assay_size].index
    main_df = df[df["assay_id"].isin(main_assay_ids)].copy()

    assay_meta = assay_metadata_table(df, y_col, args.min_assay_size)
    metadata_complete_fraction = float(assay_meta["metadata_complete"].mean()) if len(assay_meta) else 0.0

    overlap, target_components, cross_diffs = compute_overlap(
        main_df,
        y_col,
        args.transport_edge_min,
        args.max_cross_diffs_per_target,
    )
    if overlap.empty:
        overlap = pd.DataFrame(columns=["target_chembl_id", "target_name", "assay_a", "assay_b", "n_anchors"])
    if target_components.empty:
        target_components = pd.DataFrame()

    effect_table, effect_summary = assay_effect_gate(target_components, duplicates, cross_diffs, min_targets=10)
    predict_table, predict_summary = metadata_predictability(assay_meta, args.random_state)

    global_edges = overlap.groupby(["assay_a", "assay_b"], as_index=False)["n_anchors"].sum() if len(overlap) else pd.DataFrame(columns=["assay_a", "assay_b", "n_anchors"])
    transport_assay_pairs = int(len(global_edges[global_edges["n_anchors"] >= args.transport_edge_min])) if len(global_edges) else 0
    transport_paired_observations = int(overlap["n_anchors"].sum()) if len(overlap) else 0
    transport_targets = int(overlap["target_chembl_id"].nunique()) if len(overlap) else 0
    components_ge4 = int(target_components["n_components_ge4"].sum()) if len(target_components) and "n_components_ge4" in target_components else 0
    max_component = int(target_components["max_component_size"].max()) if len(target_components) and "max_component_size" in target_components else 0

    gates = [
        {
            "gate": "G0-1 scale",
            "status": bool_gate(len(main_df) >= 100_000 and main_df["assay_id"].nunique() >= 300 and main_df["target_chembl_id"].nunique() >= 30),
            "value": f"{len(main_df):,} rows / {main_df['assay_id'].nunique():,} assays / {main_df['target_chembl_id'].nunique():,} targets",
            "threshold": ">=100,000 rows / >=300 assays / >=30 targets",
        },
        {
            "gate": "G0-2 metadata",
            "status": bool_gate(metadata_complete_fraction >= 0.80),
            "value": f"{metadata_complete_fraction:.3f}",
            "threshold": ">=0.800 complete main assays",
        },
        {
            "gate": "G0-3 transport anchors",
            "status": bool_gate(transport_assay_pairs >= 50 and transport_paired_observations >= 5_000 and transport_targets >= 10),
            "value": f"{transport_assay_pairs:,} assay-pairs / {transport_paired_observations:,} anchors / {transport_targets:,} targets",
            "threshold": ">=50 assay-pairs / >=5,000 anchors / >=10 targets",
        },
        {
            "gate": "G0-4 graph connectivity",
            "status": bool_gate(components_ge4 >= 10),
            "value": f"{components_ge4:,} components>=4 assays; max={max_component}",
            "threshold": ">=10 target-level components with >=4 assays",
        },
        {
            "gate": "G0-5 assay effect",
            "status": bool_gate(bool(effect_summary["pass"])),
            "value": f"{effect_summary['targets_with_cross_assay_effect']:,} targets",
            "threshold": ">=10 targets",
        },
        {
            "gate": "G0-6 metadata predictability",
            "status": bool_gate(bool(predict_summary["pass"])),
            "value": f"best R2={predict_summary.get('best_cv_r2', math.nan):.3f}, delta={predict_summary.get('best_delta_vs_random', math.nan):.3f}",
            "threshold": "CV R2 > 0 and > randomized-text control",
        },
    ]

    hard_transport_pass = gates[2]["status"] == "PASS" and gates[3]["status"] == "PASS" and gates[4]["status"] == "PASS"
    all_pass = all(g["status"] == "PASS" for g in gates)
    if all_pass:
        route_decision = "GO: proceed with SOMOL direct transport and unseen-assay experiments."
    elif not hard_transport_pass:
        route_decision = "NO-GO for direct transport headline: downgrade to metadata-conditioned few-shot measurement adaptation unless later endpoint replication rescues anchors."
    elif gates[5]["status"] != "PASS":
        route_decision = "PARTIAL: anchors exist, but zero-shot metadata operator is weak; prioritize few-shot adaptation and metadata ablations."
    else:
        route_decision = "PARTIAL: proceed cautiously and inspect failing gates before expensive neural training."

    summary = {
        "endpoint": endpoint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "min_assay_size": args.min_assay_size,
        "transport_edge_min": args.transport_edge_min,
        "rows_clean": int(len(df)),
        "rows_main": int(len(main_df)),
        "assays_clean": int(df["assay_id"].nunique()),
        "assays_main": int(main_df["assay_id"].nunique()),
        "targets_clean": int(df["target_chembl_id"].nunique()),
        "targets_main": int(main_df["target_chembl_id"].nunique()),
        "compounds_clean": int(df["canonical_smiles_somol"].nunique()),
        "metadata_complete_fraction": metadata_complete_fraction,
        "transport_assay_pairs": transport_assay_pairs,
        "transport_paired_observations": transport_paired_observations,
        "transport_targets": transport_targets,
        "components_ge4": components_ge4,
        "max_component_size": max_component,
        "assay_effect": effect_summary,
        "metadata_predictability": predict_summary,
        "gates": gates,
        "route_decision": route_decision,
    }

    assay_meta.to_csv(table_dir / "assay_metadata_table.csv", index=False)
    overlap.to_csv(table_dir / "assay_overlap_stats.csv", index=False)
    global_edges.to_csv(table_dir / "assay_overlap_global_edges.csv", index=False)
    target_components.to_csv(table_dir / "target_component_stats.csv", index=False)
    effect_table.to_csv(table_dir / "assay_effect_stats.csv", index=False)
    predict_table.to_csv(table_dir / "metadata_predictability.csv", index=False)
    pd.DataFrame(gates).to_csv(table_dir / "gate_summary.csv", index=False)

    (output_dir / "gate_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_dataset_card(output_dir / "dataset_card.md", endpoint, df, main_df, gates, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
