#!/usr/bin/env python
"""Assay-context surrogate baselines for external-method positioning.

These are not reimplementations of ActFound/ChemPFN/AssayMatch. They are
same-protocol surrogates that test the relevant ingredient reviewers will ask
about: can assay metadata/context alone choose or predict a useful cross-assay
calibration when direct pair support is sparse?
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from train_operator_affine import build_assay_features


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.nanstd(x) < 1e-8:
        return float(np.nanmean(y)), 0.0
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else math.nan


def metric_rows(frame: pd.DataFrame, pred_col: str, method: str, split: str) -> list[dict[str, object]]:
    sub = frame[frame["split"] == split].copy()
    sub["err"] = sub[pred_col] - sub["y_dst"]
    rows = [
        {
            "method": method,
            "split": split,
            "scope": "micro",
            "n": int(len(sub)),
            "groups": 1,
            "mae": float(sub["err"].abs().mean()),
            "rmse": rmse(sub["err"].to_numpy(float)),
            "bias": float(sub["err"].mean()),
        }
    ]
    for scope, group_cols in {"macro_assay_pair": ["assay_src", "assay_dst"], "macro_target": ["target_chembl_id"]}.items():
        grouped = (
            sub.groupby(group_cols)
            .apply(
                lambda g: pd.Series(
                    {
                        "mae": g["err"].abs().mean(),
                        "rmse": rmse(g["err"].to_numpy(float)),
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
                "groups": int(len(grouped)),
                "mae": float(grouped["mae"].mean()),
                "rmse": float(grouped["rmse"].mean()),
                "bias": float(grouped["bias"].mean()),
            }
        )
    return rows


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


def pair_signature(features: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    src = features[src_idx]
    dst = features[dst_idx]
    return np.concatenate([src, dst, dst - src, np.abs(dst - src)], axis=1).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--assay-metadata", required=True)
    parser.add_argument("--output-dir", default="results/assay_context_surrogates")
    parser.add_argument("--min-pair-support", type=int, default=2)
    parser.add_argument("--svd-dim", type=int, default=128)
    parser.add_argument("--max-text-features", type=int, default=2048)
    parser.add_argument("--knn-list", default="1,5,20")
    parser.add_argument("--ridge-alpha-list", default="0.1,1,10,100")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.baseline_predictions).reset_index(drop=True)
    frame = frame[frame["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    assay_meta = pd.read_csv(args.assay_metadata)
    assay_features, assay_to_idx, feature_info = build_assay_features(
        frame, assay_meta, args.svd_dim, args.max_text_features, args.seed
    )
    frame["src_idx"] = frame["assay_src"].map(assay_to_idx).astype("int64")
    frame["dst_idx"] = frame["assay_dst"].map(assay_to_idx).astype("int64")
    train = frame[frame["split"] == "train"].copy()

    support = train.groupby(["assay_src", "assay_dst"]).size().to_dict()
    frame["pair_train_support"] = [
        int(support.get((a, b), 0)) for a, b in zip(frame["assay_src"], frame["assay_dst"])
    ]
    frame["pair_support_bucket"] = frame["pair_train_support"].map(support_bucket)

    model_rows = []
    for key, group in train.groupby(["assay_src", "assay_dst"]):
        if len(group) < args.min_pair_support:
            continue
        intercept, slope = fit_affine(group["y_src"].to_numpy(float), group["y_dst"].to_numpy(float))
        model_rows.append(
            {
                "assay_src": int(key[0]),
                "assay_dst": int(key[1]),
                "support": int(len(group)),
                "intercept": intercept,
                "slope": slope,
                "delta": float((group["y_dst"] - group["y_src"]).mean()),
                "src_idx": int(group["src_idx"].iloc[0]),
                "dst_idx": int(group["dst_idx"].iloc[0]),
            }
        )
    pair_models = pd.DataFrame(model_rows).reset_index(drop=True)
    if pair_models.empty:
        raise RuntimeError("No train pair models available for assay-context surrogate.")

    model_sig = pair_signature(
        assay_features, pair_models["src_idx"].to_numpy(int), pair_models["dst_idx"].to_numpy(int)
    )
    unique_pairs = frame[["assay_src", "assay_dst", "src_idx", "dst_idx"]].drop_duplicates().reset_index(drop=True)
    unique_sig = pair_signature(
        assay_features, unique_pairs["src_idx"].to_numpy(int), unique_pairs["dst_idx"].to_numpy(int)
    )

    knn_values = [int(x) for x in args.knn_list.split(",") if x]
    max_k = max(knn_values) + 5
    nn = NearestNeighbors(n_neighbors=min(max_k, len(pair_models)), metric="cosine", algorithm="brute")
    nn.fit(model_sig)
    distances, indices = nn.kneighbors(unique_sig)

    pair_key_to_unique = {
        (int(row.assay_src), int(row.assay_dst)): i for i, row in unique_pairs.iterrows()
    }
    unique_key_to_model_idx = {
        (int(row.assay_src), int(row.assay_dst)): i for i, row in pair_models.iterrows()
    }
    frame_unique_idx = np.asarray(
        [pair_key_to_unique[(int(a), int(b))] for a, b in zip(frame["assay_src"], frame["assay_dst"])], dtype=int
    )

    for k in knn_values:
        intercept = np.empty(len(unique_pairs), dtype=float)
        slope = np.empty(len(unique_pairs), dtype=float)
        delta = np.empty(len(unique_pairs), dtype=float)
        for row_i, row in unique_pairs.iterrows():
            self_model_idx = unique_key_to_model_idx.get((int(row.assay_src), int(row.assay_dst)))
            picked = []
            weights = []
            for dist, idx in zip(distances[row_i], indices[row_i]):
                if self_model_idx is not None and int(idx) == self_model_idx:
                    continue
                picked.append(int(idx))
                weights.append(1.0 / (float(dist) + 1e-3))
                if len(picked) >= k:
                    break
            if not picked:
                picked = [int(indices[row_i][0])]
                weights = [1.0]
            weights_np = np.asarray(weights, dtype=float)
            weights_np = weights_np / weights_np.sum()
            sub = pair_models.iloc[picked]
            intercept[row_i] = float(np.dot(weights_np, sub["intercept"].to_numpy(float)))
            slope[row_i] = float(np.dot(weights_np, sub["slope"].to_numpy(float)))
            delta[row_i] = float(np.dot(weights_np, sub["delta"].to_numpy(float)))
        frame[f"pred_assay_context_knn_affine_k{k}"] = intercept[frame_unique_idx] + slope[frame_unique_idx] * frame[
            "y_src"
        ].to_numpy(float)
        frame[f"pred_assay_context_knn_delta_k{k}"] = frame["y_src"].to_numpy(float) + delta[frame_unique_idx]
        # Natural adapted baseline: trust direct calibration when enough direct support exists, otherwise use context.
        direct = frame["pred_pair_affine_bias_fallback"].to_numpy(float)
        fallback = frame[f"pred_assay_context_knn_affine_k{k}"].to_numpy(float)
        frame[f"pred_assay_context_pair_or_knn_k{k}"] = np.where(
            frame["pair_train_support"].to_numpy(int) >= 5, direct, fallback
        )

    pair_model_targets = pair_models["delta"].to_numpy(float)
    ridge_alphas = [float(x) for x in args.ridge_alpha_list.split(",") if x]
    val_mask = frame["split"].to_numpy() == "val"
    best_ridge = None
    best_val = float("inf")
    ridge_records = []
    for alpha in ridge_alphas:
        model = Ridge(alpha=alpha, random_state=args.seed)
        model.fit(model_sig, pair_model_targets, sample_weight=np.sqrt(pair_models["support"].to_numpy(float)))
        unique_delta = model.predict(unique_sig)
        pred = frame["y_src"].to_numpy(float) + unique_delta[frame_unique_idx]
        mae = float(np.mean(np.abs(pred[val_mask] - frame.loc[val_mask, "y_dst"].to_numpy(float))))
        ridge_records.append({"alpha": alpha, "val_micro_mae": mae})
        if mae < best_val:
            best_val = mae
            best_ridge = (alpha, pred)
    if best_ridge is not None:
        frame["pred_assay_context_ridge_delta"] = best_ridge[1]
        frame["pred_assay_context_pair_or_ridge"] = np.where(
            frame["pair_train_support"].to_numpy(int) >= 5,
            frame["pred_pair_affine_bias_fallback"].to_numpy(float),
            best_ridge[1],
        )

    metrics = []
    method_cols = {c.replace("pred_", ""): c for c in frame.columns if c.startswith("pred_assay_context_")}
    for method, col in method_cols.items():
        for split in ("val", "test"):
            metrics.extend(metric_rows(frame, col, method, split))
    metric_df = pd.DataFrame(metrics).sort_values(["split", "scope", "mae", "method"]).reset_index(drop=True)
    keep_cols = [
        "split",
        "compound_target_key",
        "target_chembl_id",
        "target_name",
        "assay_src",
        "assay_dst",
        "y_src",
        "y_dst",
        "pair_train_support",
        "pair_support_bucket",
    ] + list(method_cols.values())
    frame[keep_cols].to_parquet(out_dir / "assay_context_surrogate_predictions.parquet", index=False)
    metric_df.to_csv(out_dir / "assay_context_surrogate_metrics.csv", index=False)
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_info": feature_info,
        "n_pair_models": int(len(pair_models)),
        "n_unique_eval_pairs": int(len(unique_pairs)),
        "knn_values": knn_values,
        "ridge_records": ridge_records,
        "best_ridge_alpha": None if best_ridge is None else float(best_ridge[0]),
        "interpretation": "Same-protocol assay-context surrogate baseline; not a faithful reimplementation of external models.",
    }
    (out_dir / "assay_context_surrogate_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
