"""
23_zeroshot_ibrutinib.py

Purpose: Zero-shot generalization test on the held-out aIg+Ibru
(ibrutinib) condition, matching Section methods-zeroshot exactly:
trained model applied directly to the 6->180 transition, no fine-
tuning, same gene panel matched by name, OT pools ALL available
aIg+Ibru cells (no train/val/test split needed since none of this
data is used for training), full diagnostic protocol applied.

Generalized to accept --model {mlp,film_pre,film_post,gated,gated_post}
so it works for the gated checkpoint too, not just FiLM-pre.

Usage:
    python 23_zeroshot_ibrutinib.py \
        --rna_aIg Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein_aIg Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --rna_ibru Final_Dataset/qurie_RNA_clean_aIg_Ibru.h5ad \
        --protein_ibru Final_Dataset/qurie_Protein_clean_aIg_Ibru.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --ot_type soft --ot_reg 0.01 \
        --model gated \
        --checkpoint run_results/gated_corr0.3_ccc0.3_test_deg1000/joint_model_best.pt \
        --seed 1105 \
        --outdir results/zeroshot_gated_ibrutinib
"""
import argparse
from html import parser
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from models import build_model
from train_joint import PairDataset, get_ot_targets, concordance_correlation_coefficient


def per_protein_r_ccc(y_true, y_pred):
    rs, cccs = [], []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_pred[:, j]
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            r, _ = pearsonr(yt, yp)
            if not np.isnan(r):
                rs.append(r)
                cccs.append(concordance_correlation_coefficient(yt, yp))
    return (float(np.mean(rs)) if rs else float("nan"),
            float(np.mean(cccs)) if cccs else float("nan"))


def run_inference(model, ds, device, shuffle_rna=False, shuffle_time=False, seed=42):
    rng = np.random.default_rng(seed)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    y_hat_all, y_all = [], []
    model.eval()
    with torch.no_grad():
        for x, y, t_info in loader:
            if shuffle_rna:
                perm = torch.randperm(x.shape[0])
                x = x[perm]
            if shuffle_time:
                fake_t_src = torch.tensor(rng.uniform(0, 180, size=x.shape[0]), dtype=torch.float32)
                fake_delta = torch.tensor(rng.uniform(0, 180, size=x.shape[0]), dtype=torch.float32)
                t_info = torch.stack([torch.log1p(fake_t_src), torch.log1p(fake_delta)], dim=1)
            x, t_info = x.to(device), t_info.to(device)
            y_hat = model(x, t_info).cpu().numpy()
            y_hat_all.append(y_hat)
            y_all.append(y.numpy())
    return np.concatenate(y_hat_all), np.concatenate(y_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_aIg", required=True, help="Used only to compute the SAME gene panel training used")
    parser.add_argument("--protein_aIg", required=True)
    parser.add_argument("--rna_ibru", required=True)
    parser.add_argument("--protein_ibru", required=True)
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], required=True)
    parser.add_argument("--n_genes", required=True)
    parser.add_argument("--ot_type", choices=["soft", "hard", "hard_assignment"], required=True)
    parser.add_argument("--ot_reg", type=float, default=0.01)
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post", "gated", "gated_post"], required=True)
    parser.add_argument("--time_embed", choices=["raw", "sinusoidal"], default="raw",
                     help="MUST match the time_embed used to train the checkpoint being loaded.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Load aIg to recompute the SAME gene panel used for training (must match by name)
    print("Loading aIg data to recompute matching gene panel...")
    rna_aIg = ad.read_h5ad(args.rna_aIg)
    gene_idx = select_genes(rna_aIg, args.gene_strategy, n_genes_arg, groupby="time_min")
    gene_names = rna_aIg.var_names[gene_idx].tolist()
    print(f"Selected {len(gene_names)} genes (matched by name against aIg+Ibru below).")

    print("Loading aIg+Ibru (zero-shot) data...")
    rna_ibru = ad.read_h5ad(args.rna_ibru)
    prot_ibru = ad.read_h5ad(args.protein_ibru)

    # Match gene panel by NAME against the ibru RNA var_names (not by index --
    # panels may differ in ordering/coverage between files)
    common_genes = [g for g in gene_names if g in rna_ibru.var_names]
    print(f"{len(common_genes)}/{len(gene_names)} training genes found in aIg+Ibru data.")
    rna_full = rna_ibru[:, common_genes].layers["counts"]
    rna_selected = (rna_full.toarray() if hasattr(rna_full, "toarray") else rna_full).astype(np.float32)

    prot_full = prot_ibru.layers["counts"]
    prot_full = (prot_full.toarray() if hasattr(prot_full, "toarray") else prot_full).astype(np.float32)
    protein_names = list(prot_ibru.var_names)

    times = rna_ibru.obs["time_min"].values.astype(float)
    t_src, t_tgt = 6.0, 180.0
    src_idx = np.where(np.isclose(times, t_src))[0]
    tgt_idx = np.where(np.isclose(times, t_tgt))[0]
    print(f"Zero-shot transition {t_src}->{t_tgt}: {len(src_idx)} src cells, {len(tgt_idx)} tgt cells")

    print("Computing OT targets (pooling ALL available cells, no train/val/test split)...")
    targets = get_ot_targets(rna_selected[src_idx], rna_selected[tgt_idx], prot_full[tgt_idx],
                              args.ot_type, args.ot_reg)

    ds = PairDataset(rna_selected[src_idx], targets, t_src, t_tgt)

    model = build_model(args.model, n_genes=len(common_genes), n_proteins=len(protein_names),
                     time_embed=args.time_embed).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (epoch={ckpt.get('epoch', '?')})")

    print("\n=== REAL (zero-shot) ===")
    y_hat, y_true = run_inference(model, ds, device)
    real_r, real_ccc = per_protein_r_ccc(y_true, y_hat)
    print(f"Zero-shot R={real_r:.4f}  CCC={real_ccc:.4f}  (n={len(src_idx)})")

    pred_df = pd.DataFrame(y_hat, columns=protein_names)
    pred_df.insert(0, "t_tgt", t_tgt)
    pred_df.insert(0, "t_src", t_src)
    pred_df.to_csv(os.path.join(args.outdir, f"zeroshot_y_pred_t{int(t_src)}to{int(t_tgt)}.csv"), index=False)
    truth_df = pd.DataFrame(y_true, columns=protein_names)
    truth_df.insert(0, "t_tgt", t_tgt)
    truth_df.insert(0, "t_src", t_src)
    truth_df.to_csv(os.path.join(args.outdir, f"zeroshot_y_truth_t{int(t_src)}to{int(t_tgt)}.csv"), index=False)

    print("\n=== DIAGNOSTIC 1: PREDICTION SHUFFLE ===")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y_hat))
    r_pred, _ = per_protein_r_ccc(y_true, y_hat[perm])
    print(f"Real R={real_r:.4f} | Shuffled R={r_pred:.4f} | Diff={real_r - r_pred:.4f}")

    print("\n=== DIAGNOSTIC 2: TIME SHUFFLE ===")
    y_hat_tshuf, _ = run_inference(model, ds, device, shuffle_time=True, seed=args.seed)
    r_time, _ = per_protein_r_ccc(y_true, y_hat_tshuf)
    print(f"Real R={real_r:.4f} | Time-shuffled R={r_time:.4f} | Diff={real_r - r_time:.4f}")

    print("\n=== DIAGNOSTIC 3: RNA SHUFFLE ===")
    y_hat_rshuf, _ = run_inference(model, ds, device, shuffle_rna=True, seed=args.seed)
    r_rna, _ = per_protein_r_ccc(y_true, y_hat_rshuf)
    print(f"Real R={real_r:.4f} | RNA-shuffled R={r_rna:.4f} | Diff={real_r - r_rna:.4f}")

    print(f"\nSaved predictions/truth to {args.outdir}/")
    print(f"\nSUMMARY: model={args.model}, zero-shot R={real_r:.4f}, CCC={real_ccc:.4f}")
    print(f"  pred-shuffle diff={real_r - r_pred:.4f} | time-shuffle diff={real_r - r_time:.4f} | rna-shuffle diff={real_r - r_rna:.4f}")


if __name__ == "__main__":
    main()
