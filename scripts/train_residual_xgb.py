#!/usr/bin/env python
"""Train a sparse-feature XGBoost residual corrector for transport predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_extraction import FeatureHasher


KEY_COLS = [
    "split",
    "compound_target_key",
    "target_chembl_id",
    "assay_src",
    "assay_dst",
    "y_src",
    "y_dst",
]


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else math.nan


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


def load_predictions(baseline_path: str, prediction_dirs: list[str]) -> pd.DataFrame:
    base = pd.read_parquet(baseline_path).reset_index(drop=True)
    for pred_dir in [Path(p) for p in prediction_dirs]:
        if not pred_dir.exists():
            continue
        paths = sorted(pred_dir.glob("*_predictions.parquet"))
        if not paths:
            paths = sorted(pred_dir.glob("*predictions.parquet"))
        for path in paths:
            frame = pd.read_parquet(path).reset_index(drop=True)
            if len(frame) != len(base):
                raise ValueError(f"Prediction length mismatch for {path}: {len(frame)} != {len(base)}")
            for col in [c for c in frame.columns if c.startswith("pred_")]:
                base[col] = frame[col].to_numpy()
            for col in ["pair_train_support", "pair_support_bucket", "support_gated_selected_method"]:
                if col in frame.columns:
                    base[col] = frame[col].to_numpy()
    return base


def fingerprint_bits(smiles: str, fp_bits: int, radius: int) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_bits)
    return list(generator.GetFingerprint(mol).GetOnBits())


def build_fingerprint_matrix(smiles: pd.Series, fp_bits: int, radius: int, n_jobs: int) -> tuple[sp.csr_matrix, dict[str, int]]:
    unique_smiles = pd.Index(pd.unique(smiles.astype(str)))
    smiles_to_idx = {s: i for i, s in enumerate(unique_smiles)}
    if n_jobs == 1:
        bit_lists = [fingerprint_bits(s, fp_bits, radius) for s in unique_smiles]
    else:
        bit_lists = Parallel(n_jobs=n_jobs, backend="loky", batch_size=256)(
            delayed(fingerprint_bits)(s, fp_bits, radius) for s in unique_smiles
        )

    row_idx: list[int] = []
    col_idx: list[int] = []
    for row, bits in enumerate(bit_lists):
        row_idx.extend([row] * len(bits))
        col_idx.extend(bits)
    data = np.ones(len(row_idx), dtype=np.float32)
    unique_fp = sp.csr_matrix((data, (row_idx, col_idx)), shape=(len(unique_smiles), fp_bits), dtype=np.float32)
    row_codes = smiles.astype(str).map(smiles_to_idx).to_numpy()
    return unique_fp[row_codes], {"n_unique_smiles": int(len(unique_smiles)), "n_invalid_smiles": int(sum(len(b) == 0 for b in bit_lists))}


def build_features(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[sp.csr_matrix, dict[str, object]]:
    work = frame.copy()
    if "pair_train_support" not in work.columns:
        train = work[work["split"] == "train"]
        support = train.groupby(["assay_src", "assay_dst"]).size().to_dict()
        work["pair_train_support"] = [
            int(support.get((a, b), 0)) for a, b in zip(work["assay_src"], work["assay_dst"])
        ]
    if "pair_support_bucket" not in work.columns:
        work["pair_support_bucket"] = work["pair_train_support"].map(support_bucket)

    pred_cols = sorted(c for c in work.columns if c.startswith("pred_"))
    dense = pd.DataFrame(index=work.index)
    dense["y_src"] = work["y_src"].astype(float)
    dense["pair_train_support"] = work["pair_train_support"].astype(float)
    dense["log1p_pair_train_support"] = np.log1p(dense["pair_train_support"])
    for col in pred_cols:
        dense[col] = work[col].astype(float)
    if {"pred_identity_bias", "pred_identity"}.issubset(work.columns):
        dense["delta_identity_bias"] = work["pred_identity_bias"].astype(float) - work["pred_identity"].astype(float)
    if {"pred_pair_affine", "pred_identity_bias"}.issubset(work.columns):
        dense["delta_pair_affine_identity_bias"] = work["pred_pair_affine"].astype(float) - work["pred_identity_bias"].astype(float)
    if {"pred_pair_affine_shrink_k10", "pred_pair_affine"}.issubset(work.columns):
        dense["delta_shrink_pair_affine"] = work["pred_pair_affine_shrink_k10"].astype(float) - work["pred_pair_affine"].astype(float)
    dense = dense.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_dense = sp.csr_matrix(dense.to_numpy(np.float32))

    cat_rows = [
        [
            f"target={target}",
            f"src={src}",
            f"dst={dst}",
            f"pair={src}>{dst}",
            f"bucket={bucket}",
        ]
        for target, src, dst, bucket in zip(
            work["target_chembl_id"].astype(str),
            work["assay_src"].astype(str),
            work["assay_dst"].astype(str),
            work["pair_support_bucket"].astype(str),
        )
    ]
    x_hash = FeatureHasher(n_features=args.hash_features, input_type="string", alternate_sign=False).transform(cat_rows)
    smiles = work["compound_target_key"].astype(str).str.split("||", n=1, regex=False).str[0]
    x_fp, fp_info = build_fingerprint_matrix(smiles, args.fp_bits, args.fp_radius, args.fingerprint_jobs)
    features = sp.hstack([x_dense, x_hash, x_fp], format="csr", dtype=np.float32)
    info = {
        "dense_columns": dense.columns.tolist(),
        "n_dense_features": int(x_dense.shape[1]),
        "n_hash_features": int(args.hash_features),
        "n_fp_bits": int(args.fp_bits),
        "n_rows": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "nnz": int(features.nnz),
        **fp_info,
    }
    return features, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--prediction-dir", action="append", default=[])
    parser.add_argument("--output-dir", default="results/residual_xgb")
    parser.add_argument("--base-method", default="support_gated_val_micro")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-round", type=int, default=800)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--lambda-l2", type=float, default=5.0)
    parser.add_argument("--alpha-l1", type=float, default=0.1)
    parser.add_argument("--nthread", type=int, default=96)
    parser.add_argument("--hash-features", type=int, default=32768)
    parser.add_argument("--fp-bits", type=int, default=1024)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fingerprint-jobs", type=int, default=48)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.nthread))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_predictions(args.baseline_predictions, args.prediction_dir)
    base_col = f"pred_{args.base_method}"
    if base_col not in frame.columns:
        available = sorted(c.replace("pred_", "") for c in frame.columns if c.startswith("pred_"))
        raise ValueError(f"Base prediction {base_col} not found. Available: {available}")
    features, feature_info = build_features(frame, args)

    split = frame["split"].astype(str).to_numpy()
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    y = frame["y_dst"].to_numpy(np.float32)
    base_pred = frame[base_col].to_numpy(np.float32)
    residual = y - base_pred

    dtrain = xgb.DMatrix(features[train_idx], label=residual[train_idx])
    dval = xgb.DMatrix(features[val_idx], label=residual[val_idx])
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.lambda_l2,
        "alpha": args.alpha_l1,
        "max_bin": 256,
        "seed": args.seed,
        "nthread": args.nthread,
    }
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.patience,
        evals_result=evals_result,
        verbose_eval=25,
    )
    best_iteration = int(getattr(booster, "best_iteration", args.num_round - 1))

    dall = xgb.DMatrix(features)
    pred_residual = booster.predict(dall, iteration_range=(0, best_iteration + 1))
    raw = base_pred + pred_residual
    train_y = y[train_idx]
    clipped = np.clip(raw, float(np.quantile(train_y, 0.001)), float(np.quantile(train_y, 0.999)))

    y_val = y[val_idx]
    base_val = base_pred[val_idx]
    resid_val = pred_residual[val_idx]
    best_alpha = 0.0
    best_mae = float(np.mean(np.abs(base_val - y_val)))
    for alpha in np.linspace(0.0, 1.0, 41):
        pred = base_val + alpha * resid_val
        mae = float(np.mean(np.abs(pred - y_val)))
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    blend = base_pred + best_alpha * pred_residual

    frame["pred_residual_xgb_raw"] = raw
    frame["pred_residual_xgb_clipped"] = clipped
    frame["pred_residual_xgb_blend"] = blend

    metric_list: list[dict[str, object]] = []
    for method, col in {
        "residual_xgb_raw": "pred_residual_xgb_raw",
        "residual_xgb_clipped": "pred_residual_xgb_clipped",
        "residual_xgb_blend": "pred_residual_xgb_blend",
        f"base_{args.base_method}": base_col,
    }.items():
        for split_name in ("val", "test"):
            metric_list.extend(metric_rows(frame, col, method, split_name))
    metrics = pd.DataFrame(metric_list).sort_values(["split", "scope", "mae", "method"]).reset_index(drop=True)

    pred_cols = KEY_COLS + [
        base_col,
        "pred_residual_xgb_raw",
        "pred_residual_xgb_clipped",
        "pred_residual_xgb_blend",
    ]
    if "pair_train_support" in frame.columns:
        pred_cols.extend(["pair_train_support", "pair_support_bucket"])
    frame[pred_cols].to_parquet(out_dir / "residual_xgb_predictions.parquet", index=False)
    metrics.to_csv(out_dir / "residual_xgb_metrics.csv", index=False)
    booster.save_model(out_dir / "residual_xgb_model.json")
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_method": args.base_method,
        "best_iteration": best_iteration,
        "best_score": float(getattr(booster, "best_score", math.nan)),
        "best_alpha": best_alpha,
        "best_val_blend_micro_mae": best_mae,
        "splits": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "xgboost_params": params,
        "feature_info": feature_info,
    }
    (out_dir / "residual_xgb_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(evals_result["train"]).rename(columns={"mae": "train_mae"}).join(
        pd.DataFrame(evals_result["val"]).rename(columns={"mae": "val_mae"})
    ).to_csv(out_dir / "residual_xgb_history.csv", index_label="round")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
