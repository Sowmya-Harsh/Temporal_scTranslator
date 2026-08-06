"""
kfold_cv.py
──────────────
K-fold cross-validation for the joint (all-pairs) training pipeline.
Your current train/val/test split is a SINGLE fixed split (seed=1105);
test sets are only 70-97 cells per pair, so reported mean_r/ccc numbers
could be somewhat lucky/unlucky just from which cells landed in test.

This script repeats the full train_joint.py pipeline K times, each
with a DIFFERENT random split (different seed each fold), and reports
mean +/- std of mean_pearson_r and ccc per pair across folds -- gives
you real error bars for your thesis figures/tables, not just a single
point estimate.

NOTE: this is K times more expensive than a single train_joint.py run.
Given your runs take ~2 minutes each, K=5 folds is ~10 minutes total --
still cheap, worth doing for your final chosen config specifically
(not for every config in a big sweep).

Usage:
    python kfold_cv.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy hvg --n_genes 100 \
        --ot_type soft --ot_reg 0.01 \
        --model film_post \
        --k_folds 5 --epochs 100 \
        --outdir results/kfold_cv
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(__file__))
# Reuses train_joint.py's main() logic by importing its pieces directly,
# looping the whole pipeline K times with different seeds.
import train_joint as tj
from training_improvements import correlation_loss


def run_one_fold(args, fold_seed, fold_outdir):
    """Runs the equivalent of train_joint.py's main() for one fold,
    returns the per-pair results dict (does not write files itself --
    caller aggregates and saves)."""
    os.makedirs(fold_outdir, exist_ok=True)
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import anndata as ad
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

    from gene_selection import select_genes
    gene_idx = select_genes(rna_adata, args.gene_strategy, n_genes_arg, groupby="time_min")
    rna_selected = rna_full[:, gene_idx]

    # KEY: different seed -> different train/val/test split this fold
    train_cells, val_cells, test_cells = tj.split_cells_per_timepoint(obs_time, seed=fold_seed)
    available_pairs = [(s, t) for s, t in tj.CURRICULUM if s in train_cells and t in train_cells]

    from torch.utils.data import DataLoader, ConcatDataset
    train_ds_by_pair, val_ds_by_pair, test_ds_by_pair = {}, {}, {}
    for (t_src, t_tgt) in available_pairs:
        tr, va, te = tj.build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                            t_src, t_tgt, args.ot_type, args.ot_reg)
        if tr is not None:
            train_ds_by_pair[(t_src, t_tgt)] = tr
            val_ds_by_pair[(t_src, t_tgt)] = va
            test_ds_by_pair[(t_src, t_tgt)] = te

    from models import build_model
    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)

    full_train = ConcatDataset(list(train_ds_by_pair.values()))
    full_val = ConcatDataset([v for v in val_ds_by_pair.values() if v is not None])
    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(full_val, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    ckpt_path = os.path.join(fold_outdir, "model_best.pt")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for x, y, t_info in train_loader:
            x, y, t_info = x.to(device), y.to(device), t_info.to(device)
            optimizer.zero_grad()
            y_hat = model(x, t_info)
            loss = criterion(y_hat, y)
            if args.correlation_loss_weight > 0:
                loss = loss + args.correlation_loss_weight * correlation_loss(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        val_loss = tj.run_epoch(model, val_loader, device, optimizer=None, criterion=criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    results = tj.evaluate_per_pair(model, test_ds_by_pair, device, protein_names, outdir=None)
    return results


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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--correlation_loss_weight", type=float, default=0.0,
                         help="Set to 0.3 to match the final selected model's training objective.")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Running {args.k_folds}-fold CV: gene_strategy={args.gene_strategy} "
          f"n_genes={args.n_genes} ot_type={args.ot_type} model={args.model}")

    all_fold_results = []  # list of {pair: {mse, mean_pearson_r, ccc, n_test}}
    for fold in range(args.k_folds):
        fold_seed = args.base_seed + fold * 1000  # distinct seed per fold
        print(f"\n{'='*60}\nFOLD {fold+1}/{args.k_folds} (seed={fold_seed})\n{'='*60}")
        fold_outdir = os.path.join(args.outdir, f"fold_{fold+1}")
        results = run_one_fold(args, fold_seed, fold_outdir)
        all_fold_results.append(results)

    # --- Aggregate: mean +/- std per pair across folds ---
    all_pairs = set()
    for res in all_fold_results:
        all_pairs.update(res.keys())

    rows = []
    for pair in sorted(all_pairs):
        rs = [res[pair]["mean_pearson_r"] for res in all_fold_results if pair in res]
        cccs = [res[pair]["ccc"] for res in all_fold_results if pair in res]
        mses = [res[pair]["mse"] for res in all_fold_results if pair in res]
        rows.append({
            "t_src": pair[0], "t_tgt": pair[1], "n_folds": len(rs),
            "mean_pearson_r_avg": np.mean(rs), "mean_pearson_r_std": np.std(rs),
            "ccc_avg": np.mean(cccs), "ccc_std": np.std(cccs),
            "mse_avg": np.mean(mses), "mse_std": np.std(mses),
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(args.outdir, "kfold_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\n{'='*60}\nK-FOLD SUMMARY (mean +/- std across {args.k_folds} folds)\n{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")

    with open(os.path.join(args.outdir, "kfold_config.json"), "w") as f:
        json.dump({"config": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
