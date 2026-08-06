"""
14_kfold_export_predictions.py

Purpose: The k-fold run only saved model checkpoints (fold_1/model_best.pt
... fold_5/model_best.pt) and a summary CSV -- no per-cell predictions.
This reconstructs each fold's exact test set (same seed, same gene
selection, same OT config used originally) and runs inference with that
fold's saved checkpoint, exporting joint_y_pred_t{src}to{tgt}.csv /
joint_y_truth_t{src}to{tgt}.csv per fold, in the SAME format as the main
pipeline, so 11_per_protein_bar_charts.py and 13_prediction_vs_truth_scatter.py
work unchanged on any fold's output.

Reuses split_cells_per_timepoint / build_pair_dataset / select_genes /
build_model from your existing train_joint.py / gene_selection.py /
models.py -- same approach as posthoc_shuffle_tests.py, extended to
actually save predictions rather than just compute shuffle diagnostics.

IMPORTANT: seeds below (1105, 2105, 3105, 4105, 5105) are inferred from
your k-fold log output ("FOLD 1/5 (seed=1105)" ... "FOLD 5/5 (seed=5105)").
If your actual k-fold script used a different seed-per-fold convention,
adjust FOLD_SEEDS below to match exactly, or the reconstructed test split
will not match what the checkpoint was actually evaluated on.

Usage:
    python 14_kfold_export_predictions.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --ot_type soft --ot_reg 0.01 \
        --model film_pre \
        --kfold_dir run_results/kfold_final_deg1000_filmpre \
        --fold 1
    # repeat with --fold 2, 3, 4, 5 as needed
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from models import build_model
from train_joint import CURRICULUM, split_cells_per_timepoint, build_pair_dataset

FOLD_SEEDS = {1: 1105, 2: 2105, 3: 3105, 4: 4105, 5: 5105}


def run_inference(model, test_ds_by_pair, device):
    model.eval()
    pred_by_pair, true_by_pair = {}, {}
    for pair, ds in test_ds_by_pair.items():
        if ds is None or len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        y_hat_all, y_all = [], []
        with torch.no_grad():
            for x, y, t_info in loader:
                x, t_info = x.to(device), t_info.to(device)
                y_hat = model(x, t_info).cpu().numpy()
                y_hat_all.append(y_hat)
                y_all.append(y.numpy())
        pred_by_pair[pair] = np.concatenate(y_hat_all, axis=0)
        true_by_pair[pair] = np.concatenate(y_all, axis=0)
    return pred_by_pair, true_by_pair


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
    parser.add_argument("--kfold_dir", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = FOLD_SEEDS[args.fold]
    torch.manual_seed(seed)
    np.random.seed(seed)

    fold_dir = os.path.join(args.kfold_dir, f"fold_{args.fold}")
    checkpoint_path = os.path.join(fold_dir, "model_best.pt")
    print(f"Fold {args.fold}: seed={seed}, checkpoint={checkpoint_path}")

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
    rna_selected = rna_full[:, gene_idx]

    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=seed)
    available_pairs = [(s, t) for s, t in CURRICULUM if s in train_cells and t in train_cells]

    test_ds_by_pair = {}
    for (t_src, t_tgt) in available_pairs:
        _, _, te = build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                       t_src, t_tgt, args.ot_type, args.ot_reg)
        if te is not None:
            test_ds_by_pair[(t_src, t_tgt)] = te

    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')})")

    pred_by_pair, true_by_pair = run_inference(model, test_ds_by_pair, device)

    for (t_src, t_tgt), pred in pred_by_pair.items():
        truth = true_by_pair[(t_src, t_tgt)]
        pred_df = pd.DataFrame(pred, columns=protein_names)
        pred_df.insert(0, "t_tgt", t_tgt)
        pred_df.insert(0, "t_src", t_src)
        truth_df = pd.DataFrame(truth, columns=protein_names)
        truth_df.insert(0, "t_tgt", t_tgt)
        truth_df.insert(0, "t_src", t_src)

        pred_path = os.path.join(fold_dir, f"joint_y_pred_t{t_src}to{t_tgt}.csv")
        truth_path = os.path.join(fold_dir, f"joint_y_truth_t{t_src}to{t_tgt}.csv")
        pred_df.to_csv(pred_path, index=False)
        truth_df.to_csv(truth_path, index=False)
        print(f"  Saved {pred_path} and {truth_path} ({len(pred)} cells)")

    print(f"\nDone. Fold {args.fold} predictions exported to {fold_dir}/")
    print(f"Now usable directly with, e.g.:")
    print(f"  python 11_per_protein_bar_charts.py --run_dir {fold_dir} --pool_all --top_n 15 --out_prefix kfold_fold{args.fold}")


if __name__ == "__main__":
    main()
