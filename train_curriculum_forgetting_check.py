"""
train_curriculum_forgetting_check.py
──────────────────────────────────────
Runs curriculum training (0->2, then +2->6, then +6->60, ...) and, after
EVERY stage, evaluates the SAME model on ALL pairs introduced so far
(not just the newest one) -- this is what actually lets you see
forgetting: if stage 3 (adds 6->60) makes performance on stage 1's
pair (0->2) get WORSE, that's catastrophic forgetting. If it stays
stable or improves, the curriculum is working as intended.

Output: one growing CSV, `forgetting_check.csv`, with columns
[stage, stage_pairs_so_far, evaluated_pair_t_src, evaluated_pair_t_tgt,
mean_pearson_r, ccc, mse] -- one row per (stage, previously-seen pair)
combination. Plot t_src/t_tgt vs stage in a spreadsheet/notebook to see
the forgetting curve directly.

Usage:
    python train_curriculum_forgetting_check.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy hvg --n_genes 100 \
        --ot_type soft --ot_reg 0.01 \
        --model film_post \
        --epochs_per_stage 20 --outdir results/curriculum_forgetting
"""

import argparse
import os
import sys
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from scipy.stats import pearsonr
import anndata as ad

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from models import build_model
from train_joint import (
    CURRICULUM, split_cells_per_timepoint, build_pair_dataset,
    concordance_correlation_coefficient,
)


def evaluate_pair(model, ds, device):
    if ds is None or len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    y_hat_all, y_all = [], []
    model.eval()
    with torch.no_grad():
        for x, y, t_info in loader:
            x, t_info = x.to(device), t_info.to(device)
            y_hat_all.append(model(x, t_info).cpu().numpy())
            y_all.append(y.numpy())
    y_hat_all = np.concatenate(y_hat_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    mse = float(np.mean((y_hat_all - y_all) ** 2))
    per_r, per_ccc = [], []
    for i in range(y_all.shape[1]):
        if np.std(y_all[:, i]) > 1e-8 and np.std(y_hat_all[:, i]) > 1e-8:
            r, _ = pearsonr(y_all[:, i], y_hat_all[:, i])
            per_r.append(r)
            per_ccc.append(concordance_correlation_coefficient(y_all[:, i], y_hat_all[:, i]))
    mean_r = float(np.nanmean(per_r)) if per_r else float("nan")
    ccc = float(np.mean(per_ccc)) if per_ccc else float("nan")
    return {"mse": mse, "mean_pearson_r": mean_r, "ccc": ccc, "n": len(ds)}


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
    parser.add_argument("--epochs_per_stage", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    rna_selected = rna_full[:, gene_idx]

    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)
    available_pairs = [(s, t) for s, t in CURRICULUM if s in train_cells and t in train_cells]
    print(f"Curriculum order: {available_pairs}")

    print("\nBuilding datasets for all pairs up front...")
    train_ds_by_pair, test_ds_by_pair = {}, {}
    for (t_src, t_tgt) in available_pairs:
        tr, va, te = build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                         t_src, t_tgt, args.ot_type, args.ot_reg)
        if tr is not None:
            train_ds_by_pair[(t_src, t_tgt)] = tr
            test_ds_by_pair[(t_src, t_tgt)] = te

    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    csv_path = os.path.join(args.outdir, "forgetting_check.csv")
    rows = []

    active_pairs = []
    global_epoch = 0
    for stage_i, pair in enumerate(available_pairs, 1):
        active_pairs.append(pair)
        print(f"\n--- Stage {stage_i}/{len(available_pairs)}: training on {active_pairs} ---")
        stage_train = ConcatDataset([train_ds_by_pair[p] for p in active_pairs])
        train_loader = DataLoader(stage_train, batch_size=args.batch_size, shuffle=True)

        for e in range(1, args.epochs_per_stage + 1):
            global_epoch += 1
            model.train()
            train_loss = 0.0
            for x, y, t_info in train_loader:
                x, y, t_info = x.to(device), y.to(device), t_info.to(device)
                optimizer.zero_grad()
                y_hat = model(x, t_info)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(1, len(train_loader))
            if e % 10 == 0 or e == 1:
                print(f"    Stage {stage_i} epoch {e:>3} (global {global_epoch}): train_loss={train_loss:.4f}")

        # --- KEY STEP: after this stage, evaluate on EVERY pair seen so far,
        #     not just the newest one -- this is what reveals forgetting ---
        print(f"  Evaluating on all {len(active_pairs)} pairs seen so far (stage {stage_i}):")
        for (es, et) in active_pairs:
            res = evaluate_pair(model, test_ds_by_pair.get((es, et)), device)
            if res is None:
                continue
            print(f"    [after stage {stage_i}] t{int(es)}->t{int(et)}: "
                  f"mean_r={res['mean_pearson_r']:.4f} ccc={res['ccc']:.4f} mse={res['mse']:.4f}")
            rows.append({
                "stage": stage_i, "global_epoch": global_epoch,
                "stage_pairs_so_far": str(active_pairs),
                "evaluated_t_src": es, "evaluated_t_tgt": et,
                "mean_pearson_r": res["mean_pearson_r"], "ccc": res["ccc"],
                "mse": res["mse"], "n_test": res["n"],
                "is_newest_pair": (es, et) == pair,
            })

        # write incrementally so partial progress isn't lost if the job is killed
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved forgetting-check results to {csv_path}")
    print("\nINTERPRETATION: for each evaluated_t_src/t_tgt pair, look at how mean_pearson_r")
    print("changes across increasing 'stage' values (after the pair was first introduced).")
    print("Stable or improving r across later stages = no forgetting, curriculum working.")
    print("Declining r in later stages (after the pair stops being 'is_newest_pair') =")
    print("catastrophic forgetting as new, later pairs get added.")


if __name__ == "__main__":
    main()