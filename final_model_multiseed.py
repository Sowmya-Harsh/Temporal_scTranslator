"""
final_model_multiseed.py
──────────────────────────
Multi-seed rerun of the FINAL model config only (deg-1000, film_pre,
correlation-loss 0.3, per the Table 5.6 decision) -- NOT the full
36-config ablation grid. This isolates TRAINING/INIT stochasticity
(different torch/np seed for model weight init and minibatch order)
while holding the DATA PARTITION FIXED (same single split every seed,
seed=1105 by default) -- a different question from true_kfold_cv.py,
which holds init fixed and varies the data partition instead.

Together, true_kfold_cv.py + this script separate the two sources of
variance in your reported R/CCC: how much comes from which cells
happened to land in test (kfold), vs how much comes from training
randomness alone (this script).

Usage:
    python final_model_multiseed.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --ot_type soft --ot_reg 0.01 \
        --model film_pre --correlation_loss_weight 0.3 \
        --seeds 1105,2210,3315,4420,5525 \
        --epochs 100 --outdir results/final_model_multiseed
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(os.path.dirname(__file__))
import train_joint as tj
from gene_selection import select_genes
from models import build_model
from training_improvements import correlation_loss, ccc_loss


def run_one_seed(args, rna_adata, rna_full, prot_full, protein_names, obs_time,
                  train_cells, val_cells, test_cells, gene_idx, rna_selected,
                  available_pairs, seed, seed_outdir, device):
    """DATA PARTITION IS FIXED (passed in, built once outside this function)
    -- only the model init/training seed varies here."""
    os.makedirs(seed_outdir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds_by_pair, val_ds_by_pair, test_ds_by_pair = {}, {}, {}
    for (t_src, t_tgt) in available_pairs:
        tr, va, te = tj.build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                            t_src, t_tgt, args.ot_type, args.ot_reg, seed=seed)
        if tr is not None:
            train_ds_by_pair[(t_src, t_tgt)] = tr
            val_ds_by_pair[(t_src, t_tgt)] = va
            test_ds_by_pair[(t_src, t_tgt)] = te

    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    ckpt_path = os.path.join(seed_outdir, "model_best.pt")
    best_val_loss = float("inf")

    full_train = ConcatDataset(list(train_ds_by_pair.values()))
    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True)

    def val_loss_over_all_pairs():
        losses = []
        for pair, va in val_ds_by_pair.items():
            if va is None or len(va) == 0:
                continue
            loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)
            losses.append(tj.run_epoch(model, loader, device, optimizer=None, criterion=criterion))
        return float(np.mean(losses)) if losses else float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, t_info in train_loader:
            x, y, t_info = x.to(device), y.to(device), t_info.to(device)
            optimizer.zero_grad()
            y_hat = model(x, t_info)
            loss = criterion(y_hat, y)
            if args.correlation_loss_weight > 0:
                loss = loss + args.correlation_loss_weight * correlation_loss(y_hat, y)
            if args.ccc_loss_weight > 0:
                loss = loss + args.ccc_loss_weight * ccc_loss(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss = val_loss_over_all_pairs()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ckpt_path)
        if epoch % 20 == 0 or epoch == 1:
            print(f"    seed={seed} epoch {epoch:>4}: val_loss={val_loss:.4f} (best={best_val_loss:.4f})")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_results = tj.evaluate_per_pair(model, val_ds_by_pair, device, protein_names, outdir=None)
    test_results = tj.evaluate_per_pair(model, test_ds_by_pair, device, protein_names, outdir=None)
    return val_results, test_results


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
    parser.add_argument("--correlation_loss_weight", type=float, default=0.0)
    parser.add_argument("--ccc_loss_weight", type=float, default=0.0)
    parser.add_argument("--split_seed", type=int, default=1105,
                         help="FIXED data-partition seed, same for every model seed below "
                              "(isolates training stochasticity from partition variance).")
    parser.add_argument("--seeds", type=str, default="1105,2210,3315,4420,5525",
                         help="Comma-separated list of model init/training seeds.")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)

    print(f"Final model multi-seed: gene_strategy={args.gene_strategy} n_genes={n_genes_arg} "
          f"model={args.model} correlation_loss_weight={args.correlation_loss_weight} "
          f"ccc_loss_weight={args.ccc_loss_weight} seeds={seeds} (fixed split_seed={args.split_seed})")

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

    # --- FIXED partition, built ONCE, reused for every seed below ---
    train_cells, val_cells, test_cells = tj.split_cells_per_timepoint(obs_time, seed=args.split_seed)
    train_idx_all = np.concatenate([idx for idx in train_cells.values() if len(idx) > 0])
    rna_adata_train = rna_adata[train_idx_all].copy()
    gene_idx = select_genes(rna_adata_train, args.gene_strategy, n_genes_arg, groupby="time_min")
    rna_selected = rna_full[:, gene_idx]
    print(f"Selected {len(gene_idx)} genes (fixed across all seeds, train-split cells only: "
          f"{len(train_idx_all)}/{rna_adata.n_obs})")

    available_pairs = [(s, t) for s, t in tj.CURRICULUM if s in train_cells and t in train_cells]

    all_val_results, all_test_results = [], []
    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        seed_outdir = os.path.join(args.outdir, f"seed_{seed}")
        val_res, test_res = run_one_seed(
            args, rna_adata, rna_full, prot_full, protein_names, obs_time,
            train_cells, val_cells, test_cells, gene_idx, rna_selected,
            available_pairs, seed, seed_outdir, device)
        all_val_results.append(val_res)
        all_test_results.append(test_res)

    def aggregate(all_results, split_name):
        all_pairs = set()
        for res in all_results:
            all_pairs.update(res.keys())
        rows = []
        for pair in sorted(all_pairs):
            rs = [res[pair]["mean_pearson_r"] for res in all_results if pair in res]
            cccs = [res[pair]["ccc"] for res in all_results if pair in res]
            rows.append({
                "split": split_name, "t_src": pair[0], "t_tgt": pair[1], "n_seeds": len(rs),
                "mean_pearson_r_avg": np.nanmean(rs), "mean_pearson_r_std": np.nanstd(rs),
                "ccc_avg": np.nanmean(cccs), "ccc_std": np.nanstd(cccs),
            })
        return rows

    rows = aggregate(all_val_results, "val") + aggregate(all_test_results, "test")
    df = pd.DataFrame(rows)
    out_path = os.path.join(args.outdir, "multiseed_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\n{'='*60}\nMULTI-SEED SUMMARY (mean +/- std across {len(seeds)} seeds, fixed data split)\n{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")

    for split_name, results_list in [("val", all_val_results), ("test", all_test_results)]:
        r_all = [r["mean_pearson_r"] for res in results_list for r in res.values()]
        ccc_all = [r["ccc"] for res in results_list for r in res.values()]
        print(f"Pooled {split_name}: mean_R={np.nanmean(r_all):.4f} +/- {np.nanstd(r_all):.4f} | "
              f"mean_CCC={np.nanmean(ccc_all):.4f} +/- {np.nanstd(ccc_all):.4f}")

    with open(os.path.join(args.outdir, "multiseed_config.json"), "w") as f:
        json.dump({"config": vars(args), "seeds": seeds, "n_genes_actual": len(gene_idx)}, f, indent=2)


if __name__ == "__main__":
    main()
