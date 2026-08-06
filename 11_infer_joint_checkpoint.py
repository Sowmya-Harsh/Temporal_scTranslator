"""
infer_joint_checkpoint.py
────────────────────────────
Loads an already-trained train_joint.py checkpoint and re-runs
evaluation only (no training) to produce the per-pair y_pred/y_truth
CSVs that an older run (before CSV export was added) didn't save.

Requires passing the EXACT SAME config used for the original training
run (--gene_strategy, --n_genes, --ot_type, --ot_reg, --model, --seed)
so that gene selection, OT targets, and the train/val/test split are
reconstructed identically -- deterministic given the same seed and
config, so this recovers the true original test sets, not new ones.

Usage:
    python infer_joint_checkpoint.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy hvg --n_genes 100 \
        --ot_type soft --ot_reg 0.01 \
        --model film_post \
        --checkpoint run_results/joint/joint_model_best.pt \
        --outdir run_results/joint_csv_recovery
"""

import argparse
import os
import sys

import numpy as np
import anndata as ad
import torch

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from models import build_model
from train_joint import (
    CURRICULUM, split_cells_per_timepoint, build_pair_dataset, evaluate_per_pair,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--condition", default="aIg")
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], required=True)
    parser.add_argument("--n_genes", required=True)
    parser.add_argument("--ot_type", choices=["soft", "hard", "hard_assignment"], required=True)
    parser.add_argument("--ot_reg", type=float, default=0.1)
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1105,
                         help="MUST match the seed used in the original training run "
                              "(train_joint.py's default is 1105 -- only change this if "
                              "you explicitly passed a different --seed originally)")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reconstructing test sets for: gene_strategy={args.gene_strategy} "
          f"n_genes={n_genes_arg} ot_type={args.ot_type} ot_reg={args.ot_reg} "
          f"seed={args.seed} -- MUST match original training run exactly.")

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

    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)
    available_pairs = [(s, t) for s, t in CURRICULUM if s in train_cells and t in train_cells]
    print(f"Available pairs: {available_pairs}")

    print("\nRebuilding test datasets (OT recomputed identically for the same seed/config)...")
    test_ds_by_pair = {}
    for (t_src, t_tgt) in available_pairs:
        _, _, te = build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                       t_src, t_tgt, args.ot_type, args.ot_reg)
        if te is not None:
            test_ds_by_pair[(t_src, t_tgt)] = te

    # --- Load model architecture + checkpoint weights ---
    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint} (epoch={ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?')})")

    print("\n=== Re-running evaluation only (no training), saving CSVs ===")
    results = evaluate_per_pair(model, test_ds_by_pair, device, protein_names, outdir=args.outdir)

    print(f"\nDone. CSVs saved to {args.outdir}/ -- one joint_y_pred_t{{src}}to{{tgt}}.csv "
          f"and joint_y_truth_t{{src}}to{{tgt}}.csv per pair.")
    print("\nSanity check: compare the mean_r/ccc printed above against the original job's "
          "console log -- they should match closely (small differences possible only from "
          "OT recomputation if your POT library version changed).")


if __name__ == "__main__":
    main()