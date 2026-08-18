#!/usr/bin/env python
"""Collect CSV tables for SOMOL transport experiments."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {
    "pair_affine_bias_fallback": "pred_pair_affine_bias_fallback",
    "calibration_grid_selector_micro": "pred_calibration_grid_selector_micro",
    "gauge_calibration_selector_micro": "pred_gauge_calibration_selector_micro",
    "gauge_calibration_selector_macro_assay_pair": "pred_gauge_calibration_selector_macro_assay_pair",
    "gauge_calibration_selector_macro_target": "pred_gauge_calibration_selector_macro_target",
    "graph_gauge_calibration_selector": "pred_graph_gauge_calibration_selector",
    "graph_gauge_calibration_selector_macro_target": "pred_graph_gauge_calibration_selector_macro_target",
}


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


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


def degree_bucket(n: int) -> str:
    if n <= 0:
        return "00_node_unseen"
    if n < 10:
        return "01_1-9"
    if n < 100:
        return "02_10-99"
    if n < 1000:
        return "03_100-999"
    return "04_1000plus"


def load_predictions(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "results/transport_baselines/transport_baseline_predictions.parquet").reset_index(drop=True)
    for path in [
        root / "results/calibration_grid/calibration_grid_predictions.parquet",
        root / "results/gauge_calibration/gauge_calibration_predictions.parquet",
        root / "results/graph_gauge_calibration/graph_gauge_calibration_predictions.parquet",
        root / "results/assay_context_surrogates/assay_context_surrogate_predictions.parquet",
    ]:
        extra = pd.read_parquet(path).reset_index(drop=True)
        if len(extra) != len(frame):
            raise ValueError(f"Prediction length mismatch for {path}")
        for col in [c for c in extra.columns if c.startswith("pred_")]:
            frame[col] = extra[col].to_numpy()
    return frame


def load_all_metrics(root: Path) -> pd.DataFrame:
    files = {
        "baseline": "results/transport_baselines/transport_baseline_metrics.csv",
        "operator": "results/operator_affine/operator_affine_metrics.csv",
        "latent": "results/latent_affine_id/latent_affine_metrics.csv",
        "support": "results/support_gated_selector/support_gated_metrics.csv",
        "residual": "results/residual_xgb/residual_xgb_metrics.csv",
        "grid": "results/calibration_grid/calibration_grid_metrics.csv",
        "gauge": "results/gauge_calibration/gauge_calibration_metrics.csv",
        "graph_gauge": "results/graph_gauge_calibration/graph_gauge_calibration_metrics.csv",
        "assay_context": "results/assay_context_surrogates/assay_context_surrogate_metrics.csv",
    }
    rows = []
    for family, rel in files.items():
        path = root / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["family"] = family
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def write_overall_rankings(metrics: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for scope in ["micro", "macro_assay_pair", "macro_target"]:
        sub = metrics[(metrics["split"] == "test") & (metrics["scope"] == scope)].sort_values("mae").copy()
        sub["rank"] = np.arange(1, len(sub) + 1)
        rows.append(sub[["rank", "family", "method", "scope", "n", "groups", "mae", "rmse", "bias"]].head(20))
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "test_overall_top20_by_scope.csv", index=False)


def write_ablation(metrics: pd.DataFrame, root: Path, out_dir: Path) -> None:
    keep = [
        ("baseline", "identity_bias", "identity plus train pair/global bias"),
        ("baseline", "pair_affine_bias_fallback", "local pair affine with identity-bias fallback"),
        ("support", "support_gated_val_micro", "validation support-bucket selector, no grid"),
        ("grid", "calibration_grid_selector_micro", "support-aware tuned shrinkage, no gauge"),
        ("gauge", "gauge_calibration_selector_micro", "gauge plus support selector, micro-selected"),
        ("gauge", "gauge_calibration_selector_macro_assay_pair", "gauge plus support selector, macro-pair-selected"),
        ("gauge", "gauge_calibration_selector_macro_target", "gauge plus support selector, macro-target-selected"),
        ("graph_gauge", "graph_gauge_calibration_selector", "gauge plus pair-support and assay-degree selector"),
        ("graph_gauge", "graph_gauge_calibration_selector_macro_target", "graph-gauge selector, macro-target-selected"),
        ("assay_context", "assay_context_pair_or_knn_k5", "assay metadata KNN fallback surrogate"),
        ("assay_context", "assay_context_pair_or_ridge", "assay metadata ridge fallback surrogate"),
    ]
    rows = []
    for family, method, description in keep:
        sub = metrics[
            (metrics["family"] == family)
            & (metrics["method"] == method)
            & (metrics["split"] == "test")
            & (~metrics["scope"].astype(str).str.startswith("micro_by"))
        ]
        rec = {"family": family, "method": method, "description": description}
        for _, row in sub.iterrows():
            rec[f"{row.scope}_mae"] = row.mae
            rec[f"{row.scope}_rmse"] = row.rmse
            rec[f"{row.scope}_bias"] = row.bias
        rows.append(rec)
    cand = pd.read_csv(root / "results/gauge_calibration/gauge_calibration_candidate_scores.csv")
    gauge_val = cand[(cand["split"] == "val") & (cand["candidate"].str.startswith("gauge_"))].sort_values("micro_mae").iloc[0]
    gauge_test = cand[(cand["split"] == "test") & (cand["candidate"] == gauge_val.candidate)].iloc[0]
    rows.append(
        {
            "family": "gauge_candidate",
            "method": gauge_val.candidate,
            "description": "best standalone gauge candidate by validation micro, no support selector",
            "micro_mae": gauge_test.micro_mae,
            "micro_rmse": gauge_test.micro_rmse,
            "micro_bias": gauge_test.bias,
            "macro_assay_pair_mae": gauge_test.macro_assay_pair_mae,
            "macro_target_mae": gauge_test.macro_target_mae,
        }
    )
    pd.DataFrame(rows).sort_values("micro_mae").to_csv(out_dir / "gauge_ablation_table.csv", index=False)


def write_support_table(root: Path, out_dir: Path) -> None:
    metrics = pd.read_csv(root / "results/transport_compare/support_stratified_metrics.csv")
    sub = metrics[(metrics["grouping"] == "assay_pair_train_support") & (metrics["scope"] == "micro")]
    methods = [
        "identity_bias",
        "pair_affine_bias_fallback",
        "calibration_grid_selector_micro",
        "gauge_calibration_selector_micro",
        "gauge_calibration_selector_macro_assay_pair",
        "gauge_calibration_selector_macro_target",
        "graph_gauge_calibration_selector",
        "graph_gauge_calibration_selector_macro_target",
    ]
    rows = []
    for bucket in sorted(sub["bucket"].unique()):
        rec = {"support_bucket": bucket, "n": int(sub[sub["bucket"] == bucket]["n"].iloc[0])}
        for method in methods:
            row = sub[(sub["bucket"] == bucket) & (sub["method"] == method)]
            if not row.empty:
                rec[f"{method}_mae"] = float(row["mae"].iloc[0])
        rows.append(rec)
    out = pd.DataFrame(rows)
    baseline = "pair_affine_bias_fallback_mae"
    for method in methods:
        col = f"{method}_mae"
        if col in out.columns:
            out[f"{method}_delta_vs_pair_affine_bias_fallback"] = out[baseline] - out[col]
    out.to_csv(out_dir / "support_bucket_main_methods.csv", index=False)


def write_pic50_robustness(frame: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for split in ["val", "test"]:
        split_mask = frame["split"] == split
        range_mask = split_mask & frame["y_src"].between(2, 12) & frame["y_dst"].between(2, 12)
        for label, mask in [("all", split_mask), ("src_dst_in_2_12", range_mask)]:
            sub = frame[mask].copy()
            for method, col in METHODS.items():
                err = sub[col] - sub["y_dst"]
                tmp = sub[["target_chembl_id", "assay_src", "assay_dst"]].copy()
                tmp["abs_err"] = err.abs().to_numpy()
                rows.append(
                    {
                        "split": split,
                        "eval_subset": label,
                        "method": method,
                        "n": int(len(sub)),
                        "excluded_from_split": int(split_mask.sum() - len(sub)),
                        "micro_mae": float(err.abs().mean()),
                        "rmse": rmse(err.to_numpy(float)),
                        "macro_assay_pair_mae": float(tmp.groupby(["assay_src", "assay_dst"])["abs_err"].mean().mean()),
                        "macro_target_mae": float(tmp.groupby("target_chembl_id")["abs_err"].mean().mean()),
                        "bias": float(err.mean()),
                    }
                )
    pd.DataFrame(rows).sort_values(["split", "eval_subset", "micro_mae"]).to_csv(out_dir / "pic50_range_robustness.csv", index=False)


def write_bootstrap(frame: pd.DataFrame, out_dir: Path, seed: int, n_boot: int) -> None:
    rng = np.random.default_rng(seed)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    baseline = "pred_pair_affine_bias_fallback"
    comparisons = {
        "gauge_micro": "pred_gauge_calibration_selector_micro",
        "gauge_macro_pair": "pred_gauge_calibration_selector_macro_assay_pair",
        "gauge_macro_target": "pred_gauge_calibration_selector_macro_target",
        "graph_gauge": "pred_graph_gauge_calibration_selector",
        "graph_gauge_macro_target": "pred_graph_gauge_calibration_selector_macro_target",
        "calibration_grid_micro": "pred_calibration_grid_selector_micro",
    }

    def boot(values: np.ndarray) -> tuple[float, float, float, float]:
        n = len(values)
        draws = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), float((draws <= 0).mean())

    rows = []
    for name, col in comparisons.items():
        delta = (test[baseline] - test["y_dst"]).abs().to_numpy() - (test[col] - test["y_dst"]).abs().to_numpy()
        mean, lo, hi, p = boot(delta)
        rows.append({"comparison": name, "scope": "micro_rows", "n_units": len(delta), "delta_mae_baseline_minus_method": mean, "ci95_lo": lo, "ci95_hi": hi, "p_improvement_le_0": p})
        for scope, group_cols in [("macro_assay_pair", ["assay_src", "assay_dst"]), ("macro_target", ["target_chembl_id"])]:
            tmp = test[group_cols].copy()
            tmp["delta"] = delta
            values = tmp.groupby(group_cols)["delta"].mean().to_numpy()
            mean, lo, hi, p = boot(values)
            rows.append({"comparison": name, "scope": scope, "n_units": len(values), "delta_mae_baseline_minus_method": mean, "ci95_lo": lo, "ci95_hi": hi, "p_improvement_le_0": p})
    pd.DataFrame(rows).to_csv(out_dir / "bootstrap_delta_vs_pair_affine_bias_fallback.csv", index=False)


def write_target_delta(frame: pd.DataFrame, out_dir: Path) -> None:
    test = frame[frame["split"] == "test"].copy()
    for name, col in METHODS.items():
        test[f"abs_{name}"] = (test[col] - test["y_dst"]).abs()
    rows = []
    for target, group in test.groupby("target_chembl_id"):
        rec = {
            "target_chembl_id": target,
            "target_name": group["target_name"].iloc[0] if "target_name" in group.columns else "",
            "n": int(len(group)),
            "n_assay_pairs": int(group.groupby(["assay_src", "assay_dst"]).ngroups),
        }
        for name in METHODS:
            rec[f"{name}_mae"] = float(group[f"abs_{name}"].mean())
            rec[f"delta_baseline_minus_{name}"] = float(group["abs_pair_affine_bias_fallback"].mean() - group[f"abs_{name}"].mean())
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("delta_baseline_minus_gauge_calibration_selector_micro", ascending=False)
    out.to_csv(out_dir / "target_level_gauge_delta.csv", index=False)
    summary = {
        "n_targets": int(len(out)),
        "gauge_micro_improves_targets": int((out["delta_baseline_minus_gauge_calibration_selector_micro"] > 0).sum()),
        "gauge_micro_worsens_targets": int((out["delta_baseline_minus_gauge_calibration_selector_micro"] < 0).sum()),
        "median_delta": float(out["delta_baseline_minus_gauge_calibration_selector_micro"].median()),
        "mean_delta": float(out["delta_baseline_minus_gauge_calibration_selector_micro"].mean()),
    }
    pd.Series(summary).to_json(out_dir / "target_level_gauge_delta_summary.json", indent=2)


def load_target_families(db_path: Path, target_ids: pd.Series) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame({"target_chembl_id": pd.unique(target_ids), "target_family": "unclassified"})
    con = sqlite3.connect(db_path)
    hierarchy = pd.read_sql_query(
        """
        SELECT
            protein_class_id,
            parent_id,
            pref_name,
            protein_class_desc,
            class_level
        FROM protein_classification
        """,
        con,
    )
    classes = pd.read_sql_query(
        """
        SELECT
            td.chembl_id AS target_chembl_id,
            pc.protein_class_id,
            pc.parent_id,
            pc.class_level,
            pc.pref_name,
            pc.protein_class_desc
        FROM target_dictionary td
        JOIN target_components tc ON td.tid = tc.tid
        JOIN component_class cc ON tc.component_id = cc.component_id
        JOIN protein_classification pc ON cc.protein_class_id = pc.protein_class_id
        """,
        con,
    )
    con.close()
    target_set = set(target_ids.astype(str))
    classes = classes[classes["target_chembl_id"].astype(str).isin(target_set)].copy()
    if classes.empty:
        return pd.DataFrame({"target_chembl_id": sorted(target_set), "target_family": "unclassified"})
    hierarchy = hierarchy.set_index("protein_class_id")

    def top_level_name(class_id: int) -> str:
        current = class_id
        seen: set[int] = set()
        while current in hierarchy.index and current not in seen:
            seen.add(current)
            row = hierarchy.loc[current]
            if int(row["class_level"]) == 1:
                return str(row["pref_name"])
            parent = row["parent_id"]
            if pd.isna(parent):
                break
            current = int(parent)
        return "unclassified"

    classes["target_family"] = classes["protein_class_id"].astype(int).map(top_level_name)
    rows = []
    for target, group in classes.groupby("target_chembl_id"):
        family = group["target_family"].value_counts().index[0]
        deepest = group.sort_values("class_level", ascending=False).iloc[0]
        rows.append(
            {
                "target_chembl_id": target,
                "target_family": family,
                "deepest_class_level": int(deepest["class_level"]),
                "deepest_class_name": deepest["pref_name"],
                "class_paths": "; ".join(sorted(pd.unique(group["protein_class_desc"].astype(str)))),
            }
        )
    family = pd.DataFrame(rows)
    missing = sorted(target_set.difference(set(family["target_chembl_id"].astype(str))))
    if missing:
        family = pd.concat(
            [
                family,
                pd.DataFrame(
                    {
                        "target_chembl_id": missing,
                        "target_family": "unclassified",
                        "deepest_class_level": 0,
                        "deepest_class_name": "unclassified",
                        "class_paths": "unclassified",
                    }
                ),
            ],
            ignore_index=True,
        )
    return family


def write_target_family_table(frame: pd.DataFrame, out_dir: Path, db_path: Path) -> None:
    test = frame[frame["split"] == "test"].copy()
    family = load_target_families(db_path, test["target_chembl_id"])
    test = test.merge(family, on="target_chembl_id", how="left")
    test["target_family"] = test["target_family"].fillna("unclassified")
    rows = []
    for family_name, group in test.groupby("target_family"):
        rec = {
            "target_family": family_name,
            "n": int(len(group)),
            "n_targets": int(group["target_chembl_id"].nunique()),
            "n_assay_pairs": int(group.groupby(["assay_src", "assay_dst"]).ngroups),
        }
        abs_errors = {
            method: np.abs((group[col] - group["y_dst"]).to_numpy(float))
            for method, col in METHODS.items()
        }
        baseline_mae = float(abs_errors["pair_affine_bias_fallback"].mean())
        for method, col in METHODS.items():
            err = (group[col] - group["y_dst"]).to_numpy(float)
            rec[f"{method}_mae"] = float(abs_errors[method].mean())
            rec[f"{method}_rmse"] = rmse(err)
            rec[f"delta_baseline_minus_{method}"] = baseline_mae - rec[f"{method}_mae"]
        rows.append(rec)
    pd.DataFrame(rows).sort_values("n", ascending=False).to_csv(out_dir / "target_family_gauge_delta.csv", index=False)


def write_degree_stratified(frame: pd.DataFrame, out_dir: Path) -> None:
    train = frame[frame["split"] == "train"]
    degree = train.groupby("assay_src").size().add(train.groupby("assay_dst").size(), fill_value=0).astype(int).to_dict()
    pair_support = train.groupby(["assay_src", "assay_dst"]).size().to_dict()
    test = frame[frame["split"] == "test"].copy()
    test["src_degree"] = test["assay_src"].map(degree).fillna(0).astype(int)
    test["dst_degree"] = test["assay_dst"].map(degree).fillna(0).astype(int)
    test["min_node_degree"] = test[["src_degree", "dst_degree"]].min(axis=1)
    test["node_degree_bucket"] = test["min_node_degree"].map(degree_bucket)
    test["pair_support_bucket"] = [support_bucket(pair_support.get((a, b), 0)) for a, b in zip(test["assay_src"], test["assay_dst"])]
    methods = {
        "baseline": "pred_pair_affine_bias_fallback",
        "gauge_micro": "pred_gauge_calibration_selector_micro",
        "gauge_macro_target": "pred_gauge_calibration_selector_macro_target",
    }
    rows = []
    grouped_sets = [
        ("node_degree_bucket", test.groupby("node_degree_bucket")),
        ("pair_x_node", test.groupby(["pair_support_bucket", "node_degree_bucket"])),
    ]
    for grouping, groups in grouped_sets:
        for key, group in groups:
            rec = {"grouping": grouping, "bucket": str(key), "n": int(len(group)), "min_node_degree_median": float(group["min_node_degree"].median())}
            for name, col in methods.items():
                rec[f"{name}_mae"] = float((group[col] - group["y_dst"]).abs().mean())
            rec["delta_baseline_minus_gauge_micro"] = rec["baseline_mae"] - rec["gauge_micro_mae"]
            rows.append(rec)
    pd.DataFrame(rows).sort_values(["grouping", "bucket"]).to_csv(out_dir / "assay_graph_degree_stratified.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--chembl-db",
        default="data/raw/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db",
    )
    args = parser.parse_args()

    root = Path(args.root)
    tables_dir = root / "results/tables"
    compare_dir = root / "results/transport_compare"
    tables_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    frame = load_predictions(root)
    metrics = load_all_metrics(root)
    write_overall_rankings(metrics, compare_dir)
    write_ablation(metrics, root, tables_dir)
    write_support_table(root, tables_dir)
    write_pic50_robustness(frame, tables_dir)
    write_bootstrap(frame, compare_dir, args.seed, args.n_boot)
    write_target_delta(frame, tables_dir)
    write_target_family_table(frame, tables_dir, root / args.chembl_db)
    write_degree_stratified(frame, tables_dir)
    print(f"Wrote tables to {tables_dir} and {compare_dir}")


if __name__ == "__main__":
    main()
