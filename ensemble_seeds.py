"""
ensemble_seeds.py
────────────────────
Trains the SAME config N times with different random seeds, then
averages the resulting predictions (a simple, standard ensemble). Two
benefits:
  1. Often a modest but real accuracy bump vs any single run.
  2. Tells you how much run-to-run variance exists in your results --
     if predictions vary a lot across seeds, single-seed numbers
     shouldn't be over-trusted.

Cheap given your runs take ~2 minutes each -- 5 seeds is ~10 minutes.

Usage:
    python ensemble_seeds.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy hvg --n_genes 100 \
        --ot_type soft --ot_reg 0.01 \
        --model film_post \
        --n_seeds 5 --epochs 100 \
        --outdir results/ensemble
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(__file__))
import train_joint as tj


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
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE: uses the SAME train/val/test split (data_seed fixed) across all
    # ensemble members -- only the MODEL's random init/training seed varies.
    # This isolates "does averaging multiple trained models help" from
    # "does a different data split help" (that's kfold_cv.py's job instead).
    data_seed = args.base_seed

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

    train_cells, val_cells, test_cells = tj.split_cells_per_timepoint(obs_time, seed=data_seed)
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

    full_train = ConcatDataset(list(train_ds_by_pair.values()))
    full_val = ConcatDataset([v for v in val_ds_by_pair.values() if v is not None])
    from models import build_model

    # --- Train N models, different init seed each, collect test predictions ---
    all_seed_preds = {pair: [] for pair in test_ds_by_pair}
    truth_by_pair = {}

    for seed_i in range(args.n_seeds):
        model_seed = args.base_seed + seed_i * 777
        print(f"\n{'='*60}\nENSEMBLE MEMBER {seed_i+1}/{args.n_seeds} (model_seed={model_seed})\n{'='*60}")
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)

        model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()
        train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(full_val, batch_size=args.batch_size, shuffle=False)
        ckpt_path = os.path.join(args.outdir, f"seed_{seed_i}_best.pt")
        best_val_loss = float("inf")

        for epoch in range(1, args.epochs + 1):
            train_loss = tj.run_epoch(model, train_loader, device, optimizer=optimizer,
                                       criterion=criterion, grad_clip=1.0)
            val_loss = tj.run_epoch(model, val_loader, device, optimizer=None, criterion=criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        for pair, ds in test_ds_by_pair.items():
            if ds is None or len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            y_hat_all, y_all = [], []
            with torch.no_grad():
                for x, y, t_info in loader:
                    x, t_info = x.to(device), t_info.to(device)
                    y_hat_all.append(model(x, t_info).cpu().numpy())
                    y_all.append(y.numpy())
            all_seed_preds[pair].append(np.concatenate(y_hat_all, axis=0))
            if pair not in truth_by_pair:
                truth_by_pair[pair] = np.concatenate(y_all, axis=0)

    # --- Average predictions across seeds, compute ensemble metrics ---
    print(f"\n{'='*60}\nENSEMBLE RESULT (averaged across {args.n_seeds} seeds)\n{'='*60}")
    summary_rows = []
    for pair, preds_list in all_seed_preds.items():
        if not preds_list:
            continue
        ensemble_pred = np.mean(preds_list, axis=0)
        truth = truth_by_pair[pair]

        per_r, per_ccc = [], []
        for i in range(truth.shape[1]):
            if np.std(truth[:, i]) > 1e-8 and np.std(ensemble_pred[:, i]) > 1e-8:
                r, _ = pearsonr(truth[:, i], ensemble_pred[:, i])
                per_r.append(r)
                per_ccc.append(tj.concordance_correlation_coefficient(truth[:, i], ensemble_pred[:, i]))
        mean_r = float(np.nanmean(per_r)) if per_r else float("nan")
        ccc = float(np.mean(per_ccc)) if per_ccc else float("nan")
        mse = float(np.mean((ensemble_pred - truth) ** 2))
        t_src, t_tgt = pair
        print(f"    t{int(t_src)}->t{int(t_tgt)}: ensemble mse={mse:.4f} mean_r={mean_r:.4f} ccc={ccc:.4f}")
        summary_rows.append({"t_src": t_src, "t_tgt": t_tgt, "mse": mse, "mean_pearson_r": mean_r, "ccc": ccc})

        pred_df = pd.DataFrame(ensemble_pred, columns=protein_names)
        pred_df.insert(0, "t_src", t_src); pred_df.insert(1, "t_tgt", t_tgt)
        pred_df.to_csv(os.path.join(args.outdir, f"ensemble_y_pred_t{int(t_src)}to{int(t_tgt)}.csv"), index=False)

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.outdir, "ensemble_summary.csv"), index=False)
    with open(os.path.join(args.outdir, "ensemble_config.json"), "w") as f:
        json.dump({"config": vars(args)}, f, indent=2)
    print(f"\nSaved to {args.outdir}/. Compare against your single-seed train_joint.py "
          f"result for the same config to see if ensembling actually helped.")


if __name__ == "__main__":
    main()
