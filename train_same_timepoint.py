"""
train_same_timepoint.py
─────────────────────────
ONE model, trained jointly across ALL timepoints' real same-cell
RNA/protein pairs (no OT -- at each single timepoint, RNA and protein
are already correctly paired per real cell). No ensembling, no
per-timepoint separate models: one set of weights, sees every
timepoint's real pairs during training.

This is the direct, fair baseline comparison against scTranslator's
own same-cell fine-tuning task, but done properly: one model across
the full time range (not one model per timepoint, not capped at 100
cells), with results broken out per timepoint via t_src/t_tgt columns
in the output CSVs so you can inspect performance across time.

Usage:
    python train_same_timepoint.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy hvg --n_genes 100 \
        --model film_post \
        --epochs 100 --outdir results/same_timepoint_joint
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from models import build_model


def normalize(x, eps=1e-8):
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        return np.full_like(x, 1e-8, dtype=np.float32)
    return (1e-8 + (x - xmin) / (xmax - xmin + eps)).astype(np.float32)


def concordance_correlation_coefficient(y_true, y_pred):
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return 0.0 if denom == 0 else float((2 * covariance) / denom)


class SameCellDataset(Dataset):
    """Real, directly-paired RNA/protein for the SAME cells at ONE timepoint.
    t_src == t_tgt == that timepoint (no cross-time transition here)."""
    def __init__(self, rna, prot, timepoint):
        self.rna = rna.astype(np.float32)
        self.prot = prot.astype(np.float32)
        self.timepoint = timepoint
        self.t_info_val = np.array([np.log1p(timepoint), 0.0], dtype=np.float32)

    def __len__(self):
        return len(self.rna)

    def __getitem__(self, idx):
        x = normalize(self.rna[idx])
        y = normalize(self.prot[idx])
        return torch.tensor(x), torch.tensor(y), torch.tensor(self.t_info_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--condition", default="aIg")
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], required=True)
    parser.add_argument("--n_genes", required=True)
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--variance_penalty", type=float, default=0.0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    rna_adata = ad.read_h5ad(args.rna)
    prot_adata = ad.read_h5ad(args.protein)
    mask = rna_adata.obs["condition"] == args.condition
    rna_adata = rna_adata[mask].copy()
    prot_adata = prot_adata[mask].copy()

    rna_full = rna_adata.layers["counts"]
    prot_full = prot_adata.layers["counts"]
    rna_full = (rna_full.toarray() if hasattr(rna_full, "toarray") else rna_full).astype(np.float32)
    prot_full = (prot_full.toarray() if hasattr(prot_full, "toarray") else prot_full).astype(np.float32)
    protein_names = list(prot_adata.var_names)
    obs_time = rna_adata.obs["time_min"].values.astype(float)

    gene_idx = select_genes(rna_adata, args.gene_strategy, n_genes_arg, groupby="time_min")
    print(f"Selected {len(gene_idx)} genes via {args.gene_strategy}.")
    rna_selected = rna_full[:, gene_idx]

    timepoints = sorted(np.unique(obs_time).tolist())
    print(f"Timepoints found: {timepoints} -- building same-cell pairs for EACH, "
          f"pooling into ONE joint training set.")

    rng = np.random.default_rng(args.seed)
    train_ds_by_t, val_ds_by_t, test_ds_by_t = {}, {}, {}
    for t in timepoints:
        cell_idx = np.where(np.isclose(obs_time, t, atol=1e-3))[0]
        rng.shuffle(cell_idx)
        n_train = int(len(cell_idx) * args.train_ratio)
        n_val = int(len(cell_idx) * args.val_ratio)
        train_idx = cell_idx[:n_train]
        val_idx = cell_idx[n_train:n_train + n_val]
        test_idx = cell_idx[n_train + n_val:]
        print(f"  t={int(t)}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        train_ds_by_t[t] = SameCellDataset(rna_selected[train_idx], prot_full[train_idx], t)
        val_ds_by_t[t] = SameCellDataset(rna_selected[val_idx], prot_full[val_idx], t)
        test_ds_by_t[t] = SameCellDataset(rna_selected[test_idx], prot_full[test_idx], t)

    # --- ONE joint training set, pooled across all timepoints ---
    full_train = ConcatDataset(list(train_ds_by_t.values()))
    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True)

    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    ckpt_path = os.path.join(args.outdir, "same_timepoint_joint_best.pt")
    best_val_loss = float("inf")

    def val_loss_over_all_timepoints():
        losses = []
        for t, va in val_ds_by_t.items():
            if len(va) == 0:
                continue
            loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for x, y, t_info in loader:
                    x, y, t_info = x.to(device), y.to(device), t_info.to(device)
                    vl += criterion(model(x, t_info), y).item()
            losses.append(vl / max(1, len(loader)))
        return float(np.mean(losses)) if losses else float("inf")

    print("\n=== JOINT TRAINING across all timepoints (one model, same-cell pairs) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, t_info in train_loader:
            x, y, t_info = x.to(device), y.to(device), t_info.to(device)
            optimizer.zero_grad()
            y_hat = model(x, t_info)
            loss = criterion(y_hat, y)
            if args.variance_penalty > 0:
                var_true, var_pred = y.var(dim=0), y_hat.var(dim=0)
                var_gap = torch.clamp(var_true - var_pred, min=0).mean()
                loss = loss + args.variance_penalty * var_gap
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        val_loss = val_loss_over_all_timepoints()
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss}, ckpt_path)
        if epoch % 10 == 0 or epoch == 1 or improved:
            flag = " *improved, checkpoint saved*" if improved else ""
            print(f"  Epoch {epoch:>4}: train_loss={train_loss:.4f} val_loss(avg over timepoints)={val_loss:.4f}{flag}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- Evaluate the ONE model, separately per timepoint, t_src=t_tgt=that timepoint ---
    print("\n=== FINAL: single joint model evaluated per timepoint ===")
    all_pred_rows, all_truth_rows = [], []
    summary_rows = []
    for t, test_ds in test_ds_by_t.items():
        if len(test_ds) == 0:
            continue
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        y_hat_all, y_all = [], []
        with torch.no_grad():
            for x, y, t_info in loader:
                x, t_info = x.to(device), t_info.to(device)
                y_hat_all.append(model(x, t_info).cpu().numpy())
                y_all.append(y.numpy())
        y_hat_all = np.concatenate(y_hat_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        pred_df = pd.DataFrame(y_hat_all, columns=protein_names)
        pred_df.insert(0, "t_src", t)
        pred_df.insert(1, "t_tgt", t)
        truth_df = pd.DataFrame(y_all, columns=protein_names)
        truth_df.insert(0, "t_src", t)
        truth_df.insert(1, "t_tgt", t)
        all_pred_rows.append(pred_df)
        all_truth_rows.append(truth_df)

        mse = float(np.mean((y_hat_all - y_all) ** 2))
        per_r, per_ccc = [], []
        for i in range(y_all.shape[1]):
            if np.std(y_all[:, i]) > 1e-8 and np.std(y_hat_all[:, i]) > 1e-8:
                r, _ = pearsonr(y_all[:, i], y_hat_all[:, i])
                per_r.append(r)
                per_ccc.append(concordance_correlation_coefficient(y_all[:, i], y_hat_all[:, i]))
        mean_r = float(np.nanmean(per_r)) if per_r else float("nan")
        mean_ccc = float(np.mean(per_ccc)) if per_ccc else float("nan")
        print(f"    t={int(t)}: mse={mse:.4f} mean_r={mean_r:.4f} ccc={mean_ccc:.4f} (n={len(test_ds)})")
        summary_rows.append({"t_src": t, "t_tgt": t, "n_test": len(test_ds), "mse": mse,
                              "mean_pearson_r": mean_r, "mean_per_protein_ccc": mean_ccc})

    # --- Single combined CSVs across ALL timepoints, t_src/t_tgt distinguish rows ---
    pd.concat(all_pred_rows, ignore_index=True).to_csv(
        os.path.join(args.outdir, "y_pred.csv"), index=False)
    pd.concat(all_truth_rows, ignore_index=True).to_csv(
        os.path.join(args.outdir, "y_truth.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.outdir, "same_timepoint_summary.csv"), index=False)

    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({"config": vars(args), "timepoints": timepoints,
                    "n_genes_actual": len(gene_idx), "best_val_loss": best_val_loss}, f, indent=2)

    print(f"\nSaved combined y_pred.csv / y_truth.csv (t_src/t_tgt columns distinguish timepoints)")
    print(f"Saved per-timepoint summary to same_timepoint_summary.csv")
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()