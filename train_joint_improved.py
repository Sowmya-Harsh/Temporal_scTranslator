"""
train_joint_improved.py
───────────────────────────
Same as train_joint.py (one model, all pairs, no ensembling), but with
training_improvements.py wired in as usable CLI flags:

  --weight_decay FLOAT       AdamW weight decay (0 = off, plain Adam)
  --early_stop_patience INT  stop if val loss doesn't improve for N
                              epochs (0 = off, always run full --epochs)
  --lr_scheduler              enable ReduceLROnPlateau (halves LR when
                              val loss plateaus for 5 epochs)
  --correlation_loss_weight   weight on (1 - per-protein Pearson r)
                              auxiliary loss term, added to MSE (0=off).
                              Targets R specifically.
  --ccc_loss_weight           NEW: weight on (1 - per-protein CCC)
                              auxiliary loss term, added to MSE (0=off).
                              Targets CCC specifically -- also penalizes
                              scale/location mismatch that correlation
                              loss alone does not. Can be combined with
                              --correlation_loss_weight (both added
                              together) or used alone.

Everything else (gene selection, OT matching, model architectures,
curriculum, variance_penalty, shuffle controls) is identical to
train_joint.py -- only the training loop itself changed.

Usage (test R-loss, CCC-loss, and both, on the small FiLM-pre model
first, before porting whichever wins to the pretrained-encoder /
full-encdec scripts):

    # R-loss only (original thesis config, Table 5.6)
    python train_joint_improved.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 --ot_type soft --ot_reg 0.01 \
        --model film_pre --correlation_loss_weight 0.3 \
        --epochs 100 --outdir results/joint_corrloss_only

    # CCC-loss only
    python train_joint_improved.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 --ot_type soft --ot_reg 0.01 \
        --model film_pre --ccc_loss_weight 0.3 \
        --epochs 100 --outdir results/joint_cccloss_only

    # Both together
    python train_joint_improved.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 --ot_type soft --ot_reg 0.01 \
        --model film_pre --correlation_loss_weight 0.3 --ccc_loss_weight 0.3 \
        --epochs 100 --outdir results/joint_both_losses
"""

import argparse
import os
import sys
import json
import csv

import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from models import build_model
from training_improvements import EarlyStopper, build_optimizer, build_lr_scheduler, correlation_loss, ccc_loss

# Reuse everything else directly from train_joint.py -- no duplication
from train_joint import (
    CURRICULUM, PairDataset, split_cells_per_timepoint, get_ot_targets,
    build_pair_dataset, evaluate_per_pair, concordance_correlation_coefficient,
)


def run_epoch_improved(model, loader, device, optimizer=None, criterion=None,
                        variance_penalty=0.0, corr_loss_weight=0.0, ccc_loss_weight=0.0,
                        grad_clip=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for x, y, t_info in loader:
            x, y, t_info = x.to(device), y.to(device), t_info.to(device)
            if is_train:
                optimizer.zero_grad()
            y_hat = model(x, t_info)
            loss = criterion(y_hat, y)

            if is_train and variance_penalty > 0:
                var_true, var_pred = y.var(dim=0), y_hat.var(dim=0)
                var_gap = torch.clamp(var_true - var_pred, min=0).mean()
                loss = loss + variance_penalty * var_gap

            if is_train and corr_loss_weight > 0:
                loss = loss + corr_loss_weight * correlation_loss(y_hat, y)

            if is_train and ccc_loss_weight > 0:
                loss = loss + ccc_loss_weight * ccc_loss(y_hat, y)

            if is_train:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--condition", default="aIg")
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], required=True)
    parser.add_argument("--n_genes", required=True)
    parser.add_argument("--ot_type", choices=["soft", "hard", "hard_assignment"], required=True)
    parser.add_argument("--ot_reg", type=float, default=0.1)
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post", "gated", "gated_post"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--variance_penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)

    # --- training_improvements flags ---
    parser.add_argument("--weight_decay", type=float, default=0.0,
                         help="AdamW weight decay (0=off, uses plain Adam)")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                         help="stop if val loss doesn't improve for N epochs (0=off)")
    parser.add_argument("--lr_scheduler", action="store_true",
                         help="enable ReduceLROnPlateau (halves LR on val plateau)")
    parser.add_argument("--correlation_loss_weight", type=float, default=0.0,
                         help="weight on (1 - per-protein Pearson r) auxiliary loss (0=off). Targets R.")
    parser.add_argument("--ccc_loss_weight", type=float, default=0.0,
                         help="weight on (1 - per-protein CCC) auxiliary loss (0=off). Targets CCC "
                              "specifically, also penalizes scale/location mismatch. Can combine "
                              "with --correlation_loss_weight.")
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Config: gene_strategy={args.gene_strategy} n_genes={n_genes_arg} "
          f"ot_type={args.ot_type} ot_reg={args.ot_reg} model={args.model} "
          f"variance_penalty={args.variance_penalty} | "
          f"weight_decay={args.weight_decay} early_stop_patience={args.early_stop_patience} "
          f"lr_scheduler={args.lr_scheduler} correlation_loss_weight={args.correlation_loss_weight} "
          f"ccc_loss_weight={args.ccc_loss_weight}")

    # --- Load data (identical to train_joint.py) ---
    rna_adata = ad.read_h5ad(args.rna)
    prot_adata = ad.read_h5ad(args.protein)
    mask = rna_adata.obs["condition"] == args.condition
    rna_adata = rna_adata[mask].copy()
    prot_adata = prot_adata[mask].copy()

    if "counts" not in rna_adata.layers or "counts" not in prot_adata.layers:
        raise RuntimeError("Expected '.layers[\"counts\"]' in both h5ad files.")
    rna_full = rna_adata.layers["counts"]
    prot_full = prot_adata.layers["counts"]
    rna_full = (rna_full.toarray() if hasattr(rna_full, "toarray") else rna_full).astype(np.float32)
    prot_full = (prot_full.toarray() if hasattr(prot_full, "toarray") else prot_full).astype(np.float32)
    protein_names = list(prot_adata.var_names)
    obs_time = rna_adata.obs["time_min"].values.astype(float)

    gene_idx = select_genes(rna_adata, args.gene_strategy, n_genes_arg, groupby="time_min")
    print(f"Selected {len(gene_idx)} genes via {args.gene_strategy}.")
    rna_selected = rna_full[:, gene_idx]

    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)
    available_pairs = [(s, t) for s, t in CURRICULUM if s in train_cells and t in train_cells]
    print(f"Available timepoint pairs: {available_pairs}")

    print("\nBuilding datasets for all pairs...")
    train_ds_by_pair, val_ds_by_pair, test_ds_by_pair = {}, {}, {}
    for (t_src, t_tgt) in available_pairs:
        tr, va, te = build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                         t_src, t_tgt, args.ot_type, args.ot_reg)
        if tr is not None:
            train_ds_by_pair[(t_src, t_tgt)] = tr
            val_ds_by_pair[(t_src, t_tgt)] = va
            test_ds_by_pair[(t_src, t_tgt)] = te

    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)

    # --- Optimizer: AdamW w/ weight_decay if requested, else plain Adam ---
    if args.weight_decay > 0:
        optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        print(f"Using AdamW, weight_decay={args.weight_decay}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print("Using plain Adam (weight_decay=0)")

    scheduler = build_lr_scheduler(optimizer, patience=5, factor=0.5) if args.lr_scheduler else None
    if scheduler:
        print("LR scheduler enabled: ReduceLROnPlateau (patience=5, factor=0.5)")

    stopper = EarlyStopper(patience=args.early_stop_patience) if args.early_stop_patience > 0 else None
    if stopper:
        print(f"Early stopping enabled: patience={args.early_stop_patience}")

    criterion = nn.MSELoss()
    ckpt_path = os.path.join(args.outdir, "joint_model_best.pt")
    best_val_loss = float("inf")

    def val_loss_over_all_pairs():
        losses = []
        for pair, va in val_ds_by_pair.items():
            if va is None or len(va) == 0:
                continue
            loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)
            losses.append(run_epoch_improved(model, loader, device, optimizer=None, criterion=criterion))
        return float(np.mean(losses)) if losses else float("inf")

    full_train = ConcatDataset(list(train_ds_by_pair.values()))
    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True)

    print("\n=== JOINT TRAINING (improved: weight_decay/early_stop/lr_scheduler/corr_loss/ccc_loss) ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch_improved(model, train_loader, device, optimizer=optimizer,
                                         criterion=criterion, variance_penalty=args.variance_penalty,
                                         corr_loss_weight=args.correlation_loss_weight,
                                         ccc_loss_weight=args.ccc_loss_weight, grad_clip=1.0)
        val_loss = val_loss_over_all_pairs()

        if scheduler:
            scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss}, ckpt_path)

        if epoch % 10 == 0 or epoch == 1 or improved:
            flag = " *improved, checkpoint saved*" if improved else ""
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>4}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={cur_lr:.6f}{flag}")

        if stopper is not None and stopper.step(val_loss):
            print(f"  Early stopping triggered at epoch {epoch} (no improvement for "
                  f"{args.early_stop_patience} epochs).")
            break

    print(f"\nLoading best checkpoint (val_loss={best_val_loss:.4f}) for final evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n=== FINAL: single joint model evaluated per timepoint pair ===")
    per_pair_results = evaluate_per_pair(model, test_ds_by_pair, device, protein_names, outdir=args.outdir)

    summary_path = os.path.join(args.outdir, "joint_results_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_src", "t_tgt", "n_test", "mse", "mean_pearson_r", "ccc"])
        for (t_src, t_tgt), res in per_pair_results.items():
            writer.writerow([t_src, t_tgt, res["n_test"], res["mse"], res["mean_pearson_r"], res["ccc"]])

    with open(os.path.join(args.outdir, "joint_config.json"), "w") as f:
        json.dump({"config": vars(args), "n_genes_actual": len(gene_idx),
                    "best_val_loss": best_val_loss, "checkpoint_path": ckpt_path,
                    "available_pairs": available_pairs}, f, indent=2)

    print(f"\nSaved per-pair results to {summary_path}")
    print(f"Saved single joint model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()