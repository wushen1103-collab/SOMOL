#!/usr/bin/env python
"""Train a metadata-conditioned affine measurement-operator prototype.

This is a deliberately small SOMOL sanity model:

    g_a(z) = softplus(s_a) * z + b_a
    T_{a->b}(y) = g_b(g_a^{-1}(y))

The operator parameters are generated from assay metadata. Optionally, a small
trainable per-assay residual is added for seen-assay calibration; zero-shot
assays can still fall back to metadata-only parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_rows(frame: pd.DataFrame, pred_col: str, method: str, split: str) -> list[dict[str, object]]:
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


def build_assay_features(
    pairs: pd.DataFrame,
    assay_meta: pd.DataFrame,
    svd_dim: int,
    max_text_features: int,
    seed: int,
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    assay_ids = pd.Index(pd.unique(pd.concat([pairs["assay_src"], pairs["assay_dst"]], ignore_index=True))).astype(int)
    meta_cols = ["assay_id"] + [c for c in META_BASE if c in assay_meta.columns]
    meta = assay_meta[meta_cols].drop_duplicates("assay_id").copy()
    meta = pd.DataFrame({"assay_id": assay_ids}).merge(meta, on="assay_id", how="left")
    for col in meta.columns:
        if col != "assay_id":
            meta[col] = meta[col].fillna("__MISSING__").astype(str)

    cat_cols = [c for c in META_BASE if c in meta.columns and c != "description_first"]
    if "description_first" not in meta.columns:
        meta["description_first"] = ""

    train_assays = set(pd.unique(pd.concat([pairs.loc[pairs["split"] == "train", "assay_src"], pairs.loc[pairs["split"] == "train", "assay_dst"]])))
    train_mask = meta["assay_id"].isin(train_assays).to_numpy()

    pre = ColumnTransformer(
        [
            ("text", TfidfVectorizer(max_features=max_text_features, min_df=2, ngram_range=(1, 2)), "description_first"),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols),
        ],
        remainder="drop",
    )
    x_train = pre.fit_transform(meta.loc[train_mask])
    x_all = pre.transform(meta)
    n_components = min(svd_dim, max(2, min(x_train.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    dense_train = svd.fit_transform(x_train)
    dense_all = svd.transform(x_all)
    scaler = StandardScaler()
    scaler.fit(dense_train)
    dense_all = scaler.transform(dense_all).astype("float32")
    if sparse.issparse(dense_all):
        dense_all = dense_all.toarray()

    id_to_idx = {int(a): int(i) for i, a in enumerate(meta["assay_id"].tolist())}
    info = {
        "n_assays": int(len(meta)),
        "n_train_assays": int(len(train_assays)),
        "raw_feature_dim": int(x_all.shape[1]),
        "svd_dim": int(dense_all.shape[1]),
        "cat_cols": cat_cols,
        "max_text_features": max_text_features,
    }
    return dense_all, id_to_idx, info


class AffineOperator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, n_assays: int, use_assay_residual: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.use_assay_residual = use_assay_residual
        self.residual = nn.Embedding(n_assays, 2) if use_assay_residual else None
        if self.residual is not None:
            nn.init.zeros_(self.residual.weight)

    def params_for(self, features: torch.Tensor, assay_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(features)
        if self.residual is not None:
            raw = raw + self.residual(assay_idx)
        log_scale = torch.clamp(raw[:, 0], -3.0, 3.0)
        bias = raw[:, 1]
        scale = torch.nn.functional.softplus(log_scale) / math.log(2.0)
        return scale, bias

    def forward(self, assay_features: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor, y_src: torch.Tensor) -> torch.Tensor:
        src_feat = assay_features[src_idx]
        dst_feat = assay_features[dst_idx]
        s_src, b_src = self.params_for(src_feat, src_idx)
        s_dst, b_dst = self.params_for(dst_feat, dst_idx)
        z = (y_src - b_src) / (s_src + 1e-6)
        return s_dst * z + b_dst


@torch.no_grad()
def predict(model: AffineOperator, assay_features: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor, y_src: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(y_src), batch_size):
        end = min(start + batch_size, len(y_src))
        pred = model(assay_features, src_idx[start:end], dst_idx[start:end], y_src[start:end])
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--assay-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="somol_affine_meta")
    parser.add_argument("--use-assay-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--svd-dim", type=int, default=128)
    parser.add_argument("--max-text-features", type=int, default=2048)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_parquet(args.pairs).reset_index(drop=True)
    assay_meta = pd.read_csv(args.assay_metadata)
    features_np, id_to_idx, feature_info = build_assay_features(pairs, assay_meta, args.svd_dim, args.max_text_features, args.seed)

    pairs["src_idx"] = pairs["assay_src"].map(id_to_idx).astype("int64")
    pairs["dst_idx"] = pairs["assay_dst"].map(id_to_idx).astype("int64")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assay_features = torch.tensor(features_np, dtype=torch.float32, device=device)
    src_idx = torch.tensor(pairs["src_idx"].to_numpy(), dtype=torch.long, device=device)
    dst_idx = torch.tensor(pairs["dst_idx"].to_numpy(), dtype=torch.long, device=device)
    y_src = torch.tensor(pairs["y_src"].to_numpy("float32"), dtype=torch.float32, device=device)
    y_dst = torch.tensor(pairs["y_dst"].to_numpy("float32"), dtype=torch.float32, device=device)

    train_idx = np.flatnonzero(pairs["split"].to_numpy() == "train")
    val_idx = np.flatnonzero(pairs["split"].to_numpy() == "val")
    train_ds = TensorDataset(
        torch.tensor(train_idx, dtype=torch.long),
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = AffineOperator(features_np.shape[1], args.hidden_dim, features_np.shape[0], args.use_assay_residual).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    history: list[dict[str, float]] = []
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (batch_idx_cpu,) in loader:
            idx = batch_idx_cpu.to(device)
            pred = model(assay_features, src_idx[idx], dst_idx[idx], y_src[idx])
            loss = loss_fn(pred, y_dst[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(assay_features, src_idx[val_idx_t], dst_idx[val_idx_t], y_src[val_idx_t])
            val_mae = torch.mean(torch.abs(val_pred - y_dst[val_idx_t])).item()
        train_loss = float(np.mean(losses)) if losses else math.nan
        history.append({"epoch": epoch, "train_smooth_l1": train_loss, "val_micro_mae": val_mae})
        print(json.dumps(history[-1]), flush=True)
        if val_mae < best_val - 1e-5:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    pred = predict(model, assay_features, src_idx, dst_idx, y_src, args.batch_size)
    pairs[f"pred_{args.method_name}"] = pred
    metrics: list[dict[str, object]] = []
    for split in ("val", "test"):
        metrics.extend(metric_rows(pairs, f"pred_{args.method_name}", args.method_name, split))
    metric_df = pd.DataFrame(metrics).sort_values(["split", "scope", "mae"]).reset_index(drop=True)

    pred_cols = ["split", "compound_target_key", "target_chembl_id", "assay_src", "assay_dst", "y_src", "y_dst", f"pred_{args.method_name}"]
    pairs[pred_cols].to_parquet(out_dir / f"{args.method_name}_predictions.parquet", index=False)
    metric_df.to_csv(out_dir / f"{args.method_name}_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / f"{args.method_name}_history.csv", index=False)
    torch.save({"state_dict": best_state, "args": vars(args), "feature_info": feature_info}, out_dir / f"{args.method_name}.pt")
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_name": args.method_name,
        "best_epoch": int(best_epoch),
        "best_val_micro_mae": float(best_val),
        "device": str(device),
        "use_assay_residual": bool(args.use_assay_residual),
        "feature_info": feature_info,
    }
    (out_dir / f"{args.method_name}_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
