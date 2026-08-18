#!/usr/bin/env python
"""Metadata-conditioned direct transport diagnostic baseline.

This is intentionally lighter than SOMOL: it uses y_src plus assay metadata
text/structured fields to predict y_dst. It does not use label-derived assay
statistics so that it remains a leakage-safe diagnostic for zero/few-shot
operator metadata signal.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


META_BASE = [
    "description_first",
    "assay_type_first",
    "bao_format_first",
    "assay_organism_first",
    "assay_tissue_first",
    "assay_cell_type_first",
    "assay_subcellular_fraction_first",
    "relationship_type_first",
    "confidence_score_first",
    "src_id_first",
    "pub_year_first",
    "doc_type_first",
    "journal_first",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(frame: pd.DataFrame, pred_col: str, method: str, split: str) -> list[dict[str, object]]:
    sub = frame[frame["split"] == split].copy()
    sub["err"] = sub[pred_col] - sub["y_dst"]
    rows = [
        {
            "method": method,
            "split": split,
            "scope": "micro",
            "n": int(len(sub)),
            "mae": float(sub["err"].abs().mean()),
            "rmse": rmse(sub["y_dst"].to_numpy(float), sub[pred_col].to_numpy(float)),
            "bias": float(sub["err"].mean()),
            "groups": 1,
        }
    ]
    for scope, cols in {"macro_assay_pair": ["assay_src", "assay_dst"], "macro_target": ["target_chembl_id"]}.items():
        grouped = (
            sub.groupby(cols)
            .apply(
                lambda g: pd.Series(
                    {
                        "mae": g["err"].abs().mean(),
                        "rmse": rmse(g["y_dst"].to_numpy(float), g[pred_col].to_numpy(float)),
                        "bias": g["err"].mean(),
                        "n": len(g),
                    }
                ),
                include_groups=False,
            )
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


def prepare_features(pairs: pd.DataFrame, assay_meta: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["assay_id"] + [c for c in META_BASE if c in assay_meta.columns]
    meta = assay_meta[meta_cols].drop_duplicates("assay_id").copy()
    for col in meta.columns:
        if col != "assay_id":
            meta[col] = meta[col].fillna("__MISSING__").astype(str)

    df = pairs.merge(meta.add_prefix("src_"), left_on="assay_src", right_on="src_assay_id", how="left")
    df = df.merge(meta.add_prefix("dst_"), left_on="assay_dst", right_on="dst_assay_id", how="left")
    for prefix in ("src", "dst"):
        desc = f"{prefix}_description_first"
        if desc not in df.columns:
            df[desc] = ""
        df[desc] = df[desc].fillna("").astype(str)
    df["pair_text"] = "SRC " + df["src_description_first"] + " DST " + df["dst_description_first"]
    df["y_src_sq"] = df["y_src"].astype(float) ** 2
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--assay-metadata", required=True)
    parser.add_argument("--output-dir", default="results/metadata_transport")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-text-features", type=int, default=2048)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_parquet(args.pairs)
    assay_meta = pd.read_csv(args.assay_metadata)
    df = prepare_features(pairs, assay_meta)

    cat_cols = []
    for base in META_BASE:
        if base == "description_first":
            continue
        for prefix in ("src", "dst"):
            col = f"{prefix}_{base}"
            if col in df.columns:
                cat_cols.append(col)
    numeric_cols = ["y_src", "y_src_sq"]

    feature_block = ColumnTransformer(
        [
            ("text", TfidfVectorizer(max_features=args.max_text_features, min_df=3, ngram_range=(1, 2)), "pair_text"),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    train = df["split"] == "train"
    model = Pipeline([("features", feature_block), ("ridge", Ridge(alpha=args.alpha))])
    model.fit(df.loc[train], df.loc[train, "y_dst"].to_numpy(float))
    df["pred_metadata_ridge"] = model.predict(df)
    delta_model = Pipeline([("features", clone(feature_block)), ("ridge", Ridge(alpha=args.alpha))])
    delta_model.fit(df.loc[train], (df.loc[train, "y_dst"] - df.loc[train, "y_src"]).to_numpy(float))
    df["pred_metadata_delta_ridge"] = df["y_src"].to_numpy(float) + delta_model.predict(df)

    metric_rows: list[dict[str, object]] = []
    for method, col in {
        "metadata_ridge": "pred_metadata_ridge",
        "metadata_delta_ridge": "pred_metadata_delta_ridge",
    }.items():
        for split in ("val", "test"):
            metric_rows.extend(metrics(df, col, method, split))
    metric_df = pd.DataFrame(metric_rows).sort_values(["split", "scope", "mae", "method"]).reset_index(drop=True)
    pred_cols = [
        "split",
        "compound_target_key",
        "target_chembl_id",
        "assay_src",
        "assay_dst",
        "y_src",
        "y_dst",
        "pred_metadata_ridge",
        "pred_metadata_delta_ridge",
    ]
    df[pred_cols].to_parquet(out_dir / "metadata_transport_predictions.parquet", index=False)
    metric_df.to_csv(out_dir / "metadata_transport_metrics.csv", index=False)
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": args.pairs,
        "assay_metadata": args.assay_metadata,
        "alpha": args.alpha,
        "max_text_features": args.max_text_features,
        "n_rows": int(len(df)),
        "by_split": df.groupby("split").size().astype(int).to_dict(),
        "cat_cols": cat_cols,
        "numeric_cols": numeric_cols,
    }
    (out_dir / "metadata_transport_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
