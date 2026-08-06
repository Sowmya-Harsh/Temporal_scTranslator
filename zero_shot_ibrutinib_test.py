"""
zero_shot_ibrutinib_test.py
───────────────────────────────
Tests the trained model (from-scratch, deg/1000/film_pre/correlation_loss,
R=0.319 on aIg) in a TRUE zero-shot setting: the aIg+Ibru condition was
held out entirely from training from the very start of this project
(decided early on: only 2 timepoints available -- 6, 180 min -- not
enough for a temporal trajectory, reserved purely as a final
perturbation/generalization test).

Uses the SAME 1000 genes the model was actually trained on (selected
from aIg data) -- matched by gene name, not recomputed on the Ibru
condition, since recomputing DEG on Ibru would select different genes
and silently break the checkpoint's expected input.

Only one timepoint pair is available in aIg+Ibru: 6->180 min.

Usage:
    python zero_shot_ibrutinib_test.py \
        --rna_aig Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --rna_ibru Final_Dataset/qurie_RNA_clean_aIg_Ibru.h5ad \
        --protein_ibru Final_Dataset/qurie_Protein_clean_aIg_Ibru.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --ot_type soft --ot_reg 0.01 \
        --model film_pre \
        --checkpoint run_results/correlationlosswt_test_deg1000_filmpre/joint_model_best.pt \
        --outdir results/zero_shot_ibrutinib
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from models import build_model
from train_joint import get_ot_targets, concordance_correlation_coefficient, normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_aig", required=True,
                         help="aIg RNA h5ad -- used ONLY to reconstruct which genes "
                              "were selected during training, not for inference.")
    parser.add_argument("--rna_ibru", required=True,
                         help="aIg+Ibru RNA h5ad -- the actual zero-shot test data.")
    parser.add_argument("--protein_ibru", required=True)
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], required=True)
    parser.add_argument("--n_genes", required=True)
    parser.add_argument("--ot_type", choices=["soft", "hard", "hard_assignment"], required=True)
    parser.add_argument("--ot_reg", type=float, default=0.1)
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Config: ZERO-SHOT test on aIg+Ibru. gene_strategy={args.gene_strategy} "
          f"n_genes={n_genes_arg} ot_type={args.ot_type} ot_reg={args.ot_reg} model={args.model}")

    print("\nReconstructing training gene selection from aIg data...")
    rna_aig = ad.read_h5ad(args.rna_aig)
    mask_aig = rna_aig.obs["condition"] == "aIg"
    rna_aig = rna_aig[mask_aig].copy()
    gene_idx_aig = select_genes(rna_aig, args.gene_strategy, n_genes_arg, groupby="time_min")
    trained_gene_names = np.array(rna_aig.var_names)[gene_idx_aig]
    print(f"Model was trained on {len(trained_gene_names)} genes (selected from aIg).")

    print("\nLoading aIg+Ibru (zero-shot test) data...")
    rna_ibru = ad.read_h5ad(args.rna_ibru)
    prot_ibru = ad.read_h5ad(args.protein_ibru)
    mask_ibru = rna_ibru.obs["condition"] == "aIg+Ibru"
    if mask_ibru.sum() == 0:
        for candidate in ["aIg+Ibru", "aIg_Ibru", "aIg Ibru"]:
            mask_ibru = rna_ibru.obs["condition"] == candidate
            if mask_ibru.sum() > 0:
                print(f"  Matched condition label: '{candidate}'")
                break
    rna_ibru = rna_ibru[mask_ibru].copy()
    prot_ibru = prot_ibru[mask_ibru].copy()
    print(f"aIg+Ibru cells: {rna_ibru.n_obs}")

    ibru_gene_names = np.array(rna_ibru.var_names)
    name_to_idx = {name: i for i, name in enumerate(ibru_gene_names)}
    missing = [g for g in trained_gene_names if g not in name_to_idx]
    if missing:
        print(f"WARNING: {len(missing)} of {len(trained_gene_names)} trained genes "
              f"not found in aIg+Ibru data (first few: {missing[:5]}). These will be "
              f"excluded -- model may perform worse due to missing input features.")
    matched_gene_idx = [name_to_idx[g] for g in trained_gene_names if g in name_to_idx]
    print(f"Matched {len(matched_gene_idx)}/{len(trained_gene_names)} trained genes in Ibru data.")

    rna_full = rna_ibru.layers["counts"]
    prot_full = prot_ibru.layers["counts"]
    rna_full = (rna_full.toarray() if hasattr(rna_full, "toarray") else rna_full).astype(np.float32)
    prot_full = (prot_full.toarray() if hasattr(prot_full, "toarray") else prot_full).astype(np.float32)
    protein_names = list(prot_ibru.var_names)
    obs_time = rna_ibru.obs["time_min"].values.astype(float)

    rna_selected = rna_full[:, matched_gene_idx]
    n_genes_actual = rna_selected.shape[1]
    if n_genes_actual != len(trained_gene_names):
        print(f"NOTE: model expects {len(trained_gene_names)} input genes but only "
              f"{n_genes_actual} were found. If the model architecture has a fixed "
              f"input size, this WILL crash -- gene sets must match exactly for a "
              f"from-scratch (non-transformer) model.")

    timepoints = sorted(np.unique(obs_time).tolist())
    print(f"Available timepoints in aIg+Ibru: {timepoints}")
    if len(timepoints) < 2:
        print("ERROR: need at least 2 timepoints. Aborting.")
        return
    t_src, t_tgt = timepoints[0], timepoints[-1]
    print(f"Testing zero-shot transition: t{int(t_src)} -> t{int(t_tgt)}")

    src_cells = np.where(np.isclose(obs_time, t_src, atol=1e-3))[0]
    tgt_cells = np.where(np.isclose(obs_time, t_tgt, atol=1e-3))[0]
    print(f"Source cells (t={t_src}): {len(src_cells)}  Target cells (t={t_tgt}): {len(tgt_cells)}")

    print("\nComputing OT targets for zero-shot test set...")
    targets = get_ot_targets(rna_selected[src_cells], rna_selected[tgt_cells],
                              prot_full[tgt_cells], args.ot_type, args.ot_reg)

    model = build_model(args.model, n_genes=n_genes_actual, n_proteins=len(protein_names)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        print(f"\nERROR loading checkpoint -- likely a gene-count mismatch:")
        print(e)
        return
    model.eval()
    print(f"Loaded checkpoint (epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')})")

    x = np.array([normalize(rna_selected[i]) for i in src_cells], dtype=np.float32)
    delta_t = t_tgt - t_src
    t_info = np.tile(np.array([np.log1p(t_src), np.log1p(delta_t)], dtype=np.float32), (len(src_cells), 1))

    with torch.no_grad():
        x_t = torch.tensor(x).to(device)
        t_info_t = torch.tensor(t_info).to(device)
        y_hat = model(x_t, t_info_t).cpu().numpy()

    mse = float(np.mean((y_hat - targets) ** 2))
    per_r, per_ccc = [], []
    for i in range(targets.shape[1]):
        if np.std(targets[:, i]) > 1e-8 and np.std(y_hat[:, i]) > 1e-8:
            r, _ = pearsonr(targets[:, i], y_hat[:, i])
            per_r.append(r)
            per_ccc.append(concordance_correlation_coefficient(targets[:, i], y_hat[:, i]))
    mean_r = float(np.nanmean(per_r)) if per_r else float("nan")
    ccc = float(np.mean(per_ccc)) if per_ccc else float("nan")

    print(f"\n=== ZERO-SHOT RESULT: t{int(t_src)}->t{int(t_tgt)} (aIg+Ibru, never seen in training) ===")
    print(f"mse={mse:.4f}  mean_pearson_r={mean_r:.4f}  ccc={ccc:.4f}  n_cells={len(src_cells)}")

    pred_df = pd.DataFrame(y_hat, columns=protein_names)
    pred_df.insert(0, "t_src", t_src)
    pred_df.insert(1, "t_tgt", t_tgt)
    pred_df.to_csv(os.path.join(args.outdir, "zeroshot_ibru_y_pred.csv"), index=False)

    truth_df = pd.DataFrame(targets, columns=protein_names)
    truth_df.insert(0, "t_src", t_src)
    truth_df.insert(1, "t_tgt", t_tgt)
    truth_df.to_csv(os.path.join(args.outdir, "zeroshot_ibru_y_truth.csv"), index=False)

    with open(os.path.join(args.outdir, "zeroshot_ibru_result.json"), "w") as f:
        json.dump({"config": vars(args), "t_src": t_src, "t_tgt": t_tgt,
                    "n_cells": len(src_cells), "mse": mse, "mean_pearson_r": mean_r, "ccc": ccc,
                    "n_genes_matched": n_genes_actual, "n_genes_trained": len(trained_gene_names)},
                   f, indent=2)
    print(f"\nSaved to {args.outdir}/")


if __name__ == "__main__":
    main()