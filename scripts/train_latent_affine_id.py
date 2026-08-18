#!/usr/bin/env python
"""Train an observation-assisted latent affine operator prototype.

This is not the final SOMOL model. It tests whether adding a learned latent
compound-target signal helps direct transport beyond pairwise calibration.

Latent:
    z_dp = MLP([compound_embedding, target_embedding])

Operator:
    y = softplus(s_a) * z_dp + b_a

Direct transport:
    T(y_src, a, b) = g_b(g_a^{-1}(y_src))

Observation-assisted transport:
    alpha * T(y_src, a, b) + (1-alpha) * g_b(z_dp), alpha tuned on val.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def train_vocab(values: pd.Series, train_mask: np.ndarray) -> dict[str, int]:
    unique = pd.Index(pd.unique(values[train_mask].astype(str)))
    return {str(v): i + 1 for i, v in enumerate(unique)}


def map_vocab(values: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    return values.astype(str).map(vocab).fillna(0).astype("int64").to_numpy()


class LatentAffine(nn.Module):
    def __init__(self, n_compounds: int, n_targets: int, n_assays: int, emb_dim: int, target_dim: int, hidden_dim: int):
        super().__init__()
        self.compound = nn.Embedding(n_compounds + 1, emb_dim, padding_idx=0)
        self.target = nn.Embedding(n_targets + 1, target_dim, padding_idx=0)
        self.latent = nn.Sequential(
            nn.Linear(emb_dim + target_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.assay = nn.Embedding(n_assays + 1, 2, padding_idx=0)
        nn.init.normal_(self.compound.weight, std=0.02)
        nn.init.normal_(self.target.weight, std=0.02)
        nn.init.zeros_(self.assay.weight)
        with torch.no_grad():
            self.compound.weight[0].zero_()
            self.target.weight[0].zero_()
            self.assay.weight[0].zero_()

    def z(self, compound_idx: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.compound(compound_idx), self.target(target_idx)], dim=-1)
        return self.latent(h).squeeze(-1)

    def operator(self, assay_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.assay(assay_idx)
        log_scale = torch.clamp(raw[:, 0], -4.0, 4.0)
        scale = torch.nn.functional.softplus(log_scale) / math.log(2.0)
        bias = raw[:, 1]
        return scale, bias

    def observe(self, compound_idx: torch.Tensor, target_idx: torch.Tensor, assay_idx: torch.Tensor) -> torch.Tensor:
        z = self.z(compound_idx, target_idx)
        scale, bias = self.operator(assay_idx)
        return scale * z + bias

    def transport(self, y_src: torch.Tensor, assay_src: torch.Tensor, assay_dst: torch.Tensor) -> torch.Tensor:
        s_src, b_src = self.operator(assay_src)
        s_dst, b_dst = self.operator(assay_dst)
        z = (y_src - b_src) / (s_src + 1e-6)
        return s_dst * z + b_dst


@torch.no_grad()
def predict_pairs(
    model: LatentAffine,
    arrays: dict[str, torch.Tensor],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = len(arrays["pair_y_src"])
    pred_t = []
    pred_obs = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pred_t.append(
            model.transport(
                arrays["pair_y_src"][start:end],
                arrays["pair_assay_src"][start:end],
                arrays["pair_assay_dst"][start:end],
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred_obs.append(
            model.observe(
                arrays["pair_compound"][start:end],
                arrays["pair_target"][start:end],
                arrays["pair_assay_dst"][start:end],
            )
            .detach()
            .cpu()
            .numpy()
        )
    return np.concatenate(pred_t), np.concatenate(pred_obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-dir", default="results/latent_affine_id")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--target-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = pd.read_parquet(
        args.clean,
        columns=["canonical_smiles_somol", "target_chembl_id", "assay_id", "compound_target_key", "pIC50"],
    ).reset_index(drop=True)
    clean["row_id"] = np.arange(len(clean), dtype=np.int64)
    splits = pd.read_parquet(args.splits, columns=["row_id", "S6_transport_pair"])
    obs = clean.merge(splits, on="row_id", how="inner")
    obs_train_mask = obs["S6_transport_pair"].to_numpy() == "train"
    obs_val_mask = obs["S6_transport_pair"].to_numpy() == "val"

    compound_vocab = train_vocab(obs["canonical_smiles_somol"], obs_train_mask)
    target_vocab = train_vocab(obs["target_chembl_id"], obs_train_mask)
    assay_vocab = train_vocab(obs["assay_id"].astype(str), obs_train_mask)
    obs["compound_idx"] = map_vocab(obs["canonical_smiles_somol"], compound_vocab)
    obs["target_idx"] = map_vocab(obs["target_chembl_id"], target_vocab)
    obs["assay_idx"] = map_vocab(obs["assay_id"].astype(str), assay_vocab)

    pairs = pd.read_parquet(args.pairs).reset_index(drop=True)
    # Recover compound identity from the compound-target key: SMILES || target_chembl_id.
    pairs["canonical_smiles_somol"] = pairs["compound_target_key"].astype(str).str.split("||", regex=False).str[0]
    pairs["compound_idx"] = map_vocab(pairs["canonical_smiles_somol"], compound_vocab)
    pairs["target_idx"] = map_vocab(pairs["target_chembl_id"], target_vocab)
    pairs["assay_src_idx"] = map_vocab(pairs["assay_src"].astype(str), assay_vocab)
    pairs["assay_dst_idx"] = map_vocab(pairs["assay_dst"].astype(str), assay_vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LatentAffine(
        n_compounds=len(compound_vocab),
        n_targets=len(target_vocab),
        n_assays=len(assay_vocab),
        emb_dim=args.emb_dim,
        target_dim=args.target_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    obs_arrays = {
        "compound": torch.tensor(obs["compound_idx"].to_numpy(), dtype=torch.long, device=device),
        "target": torch.tensor(obs["target_idx"].to_numpy(), dtype=torch.long, device=device),
        "assay": torch.tensor(obs["assay_idx"].to_numpy(), dtype=torch.long, device=device),
        "y": torch.tensor(obs["pIC50"].to_numpy("float32"), dtype=torch.float32, device=device),
    }
    pair_arrays = {
        "pair_compound": torch.tensor(pairs["compound_idx"].to_numpy(), dtype=torch.long, device=device),
        "pair_target": torch.tensor(pairs["target_idx"].to_numpy(), dtype=torch.long, device=device),
        "pair_assay_src": torch.tensor(pairs["assay_src_idx"].to_numpy(), dtype=torch.long, device=device),
        "pair_assay_dst": torch.tensor(pairs["assay_dst_idx"].to_numpy(), dtype=torch.long, device=device),
        "pair_y_src": torch.tensor(pairs["y_src"].to_numpy("float32"), dtype=torch.float32, device=device),
    }

    train_idx = np.flatnonzero(obs_train_mask)
    val_idx = np.flatnonzero(obs_val_mask)
    train_loader = DataLoader(TensorDataset(torch.tensor(train_idx, dtype=torch.long)), batch_size=args.batch_size, shuffle=True)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (idx_cpu,) in train_loader:
            idx = idx_cpu.to(device)
            pred = model.observe(obs_arrays["compound"][idx], obs_arrays["target"][idx], obs_arrays["assay"][idx])
            loss = loss_fn(pred, obs_arrays["y"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model.observe(
                obs_arrays["compound"][val_idx_t],
                obs_arrays["target"][val_idx_t],
                obs_arrays["assay"][val_idx_t],
            )
            val_mae = torch.mean(torch.abs(val_pred - obs_arrays["y"][val_idx_t])).item()
        record = {"epoch": epoch, "train_smooth_l1": float(np.mean(losses)), "val_obs_micro_mae": val_mae}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_mae < best_val - 1e-5:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    pred_transport, pred_obs = predict_pairs(model, pair_arrays, args.batch_size)
    pairs["pred_latent_affine_transport"] = pred_transport
    pairs["pred_latent_affine_obs"] = pred_obs

    val_mask = pairs["split"].to_numpy() == "val"
    y_val = pairs.loc[val_mask, "y_dst"].to_numpy(float)
    best_alpha = 1.0
    best_blend = float("inf")
    for alpha in np.linspace(0.0, 1.0, 21):
        pred = alpha * pairs.loc[val_mask, "pred_latent_affine_transport"].to_numpy(float) + (1 - alpha) * pairs.loc[val_mask, "pred_latent_affine_obs"].to_numpy(float)
        mae = float(np.mean(np.abs(pred - y_val)))
        if mae < best_blend:
            best_blend = mae
            best_alpha = float(alpha)
    pairs["pred_latent_affine_blend"] = best_alpha * pairs["pred_latent_affine_transport"] + (1 - best_alpha) * pairs["pred_latent_affine_obs"]

    metrics = []
    for method, col in {
        "latent_affine_transport": "pred_latent_affine_transport",
        "latent_affine_obs": "pred_latent_affine_obs",
        "latent_affine_blend": "pred_latent_affine_blend",
    }.items():
        for split in ("val", "test"):
            metrics.extend(metric_rows(pairs, col, method, split))
    metric_df = pd.DataFrame(metrics).sort_values(["split", "scope", "mae", "method"]).reset_index(drop=True)

    pred_cols = [
        "split",
        "compound_target_key",
        "target_chembl_id",
        "assay_src",
        "assay_dst",
        "y_src",
        "y_dst",
        "pred_latent_affine_transport",
        "pred_latent_affine_obs",
        "pred_latent_affine_blend",
    ]
    pairs[pred_cols].to_parquet(out_dir / "latent_affine_predictions.parquet", index=False)
    metric_df.to_csv(out_dir / "latent_affine_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "latent_affine_history.csv", index=False)
    torch.save({"state_dict": best_state, "args": vars(args)}, out_dir / "latent_affine.pt")
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_obs_micro_mae": float(best_val),
        "best_blend_alpha_transport": float(best_alpha),
        "best_val_blend_pair_micro_mae": float(best_blend),
        "n_compounds_train_vocab": int(len(compound_vocab)),
        "n_targets_train_vocab": int(len(target_vocab)),
        "n_assays_train_vocab": int(len(assay_vocab)),
    }
    (out_dir / "latent_affine_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
