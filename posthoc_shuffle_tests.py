"""
posthoc_shuffle_tests.py
────────────────────────────
Loads an ALREADY-TRAINED train_joint.py checkpoint and runs three
sanity checks purely at inference time -- no retraining needed, much
faster than the --shuffle_rna/--shuffle_time retrain-based controls.

1. PREDICTION-SHUFFLE: permute the model's own predictions across
   cells (within each timepoint pair), re-measure correlation against
   truth. Sanity floor check.

2. TIME-SHUFFLE (inference-time): feed the trained model random
   t_info instead of the real values. If the model genuinely uses
   time conditioning, R should drop noticeably.

3. RNA-SHUFFLE (inference-time): feed the trained model permuted RNA
   input, keep t_info real. If the model genuinely uses RNA content,
   R should drop noticeably.

Requires the EXACT SAME config used for the original training run
(gene_strategy, n_genes, ot_type, ot_reg, model, seed) to reconstruct
the same test sets -- same requirement as infer_joint_checkpoint.py.

Usage:
    python posthoc_shuffle_tests.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --ot_type soft --ot_reg 0.01 \
        --model film_pre \
        --checkpoint run_results/full_sweep/deg_1000_film_pre/joint_model_best.pt \
        --outdir results/posthoc_shuffle_deg1000_filmpre
"""

import argparse
import os
import sys
import json

import numpy as np
import anndata as ad
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from models import build_model
from train_joint import CURRICULUM, split_cells_per_timepoint, build_pair_dataset


def within_pair_r(pred, true, min_cells=5):
    all_rs = []
    for pair in pred:
        p, t = pred[pair], true[pair]
        if len(p) < min_cells:
            continue
        for j in range(p.shape[1]):
            if p[:, j].std() > 1e-6 and t[:, j].std() > 1e-6:
                r, _ = pearsonr(p[:, j], t[:, j])
                all_rs.append(r)
    return float(np.mean(all_rs)) if all_rs else 0.0


def run_inference(model, test_ds_by_pair, device, shuffle_time=False,
                   shuffle_rna=False, seed=42):
    rng = np.random.default_rng(seed)
    model.eval()
    pred_by_pair, true_by_pair = {}, {}

    for pair, ds in test_ds_by_pair.items():
        if ds is None or len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        y_hat_all, y_all = [], []
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
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post", "gated", "gated_post"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1105,
                         help="MUST match the original training run's seed")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reconstructing test sets: gene_strategy={args.gene_strategy} n_genes={n_genes_arg} "
          f"ot_type={args.ot_type} ot_reg={args.ot_reg} seed={args.seed}")

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

    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)
    train_idx_all = np.concatenate([idx for idx in train_cells.values() if len(idx) > 0])
    rna_adata_train = rna_adata[train_idx_all].copy()
    gene_idx = select_genes(rna_adata_train, args.gene_strategy, n_genes_arg, groupby="time_min")
    print(f"Selected {len(gene_idx)} genes (train-split cells only: {len(train_idx_all)}/{rna_adata.n_obs}).")
    rna_selected = rna_full[:, gene_idx]

    available_pairs = [(s, t) for s, t in CURRICULUM if s in train_cells and t in train_cells]

    test_ds_by_pair = {}
    for (t_src, t_tgt) in available_pairs:
        _, _, te = build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                       t_src, t_tgt, args.ot_type, args.ot_reg)
        if te is not None:
            test_ds_by_pair[(t_src, t_tgt)] = te

    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')})")

    results = {}

    print("\n=== REAL (baseline) ===")
    pred_real, true_np = run_inference(model, test_ds_by_pair, device)
    r_real = within_pair_r(pred_real, true_np)
    print(f"Within-pair R (real): {r_real:.4f}")
    results["real"] = r_real

    print("\n=== TEST 1: PREDICTION SHUFFLE ===")
    rng = np.random.default_rng(42)
    pred_shuf = {}
    for pair, p in pred_real.items():
        perm = rng.permutation(len(p))
        pred_shuf[pair] = p[perm]
    r_pred_shuf = within_pair_r(pred_shuf, true_np)
    print(f"Within-pair R (predictions shuffled): {r_pred_shuf:.4f}")
    print(f"Difference from real: {r_real - r_pred_shuf:.4f}")
    results["prediction_shuffle"] = r_pred_shuf
    print("VERDICT:", "GENUINE (real >> shuffled)" if r_real > r_pred_shuf + 0.05 else "WARNING: real not clearly above shuffled")

    print("\n=== TEST 2: TIME SHUFFLE (inference-time, model unchanged) ===")
    pred_tshuf, _ = run_inference(model, test_ds_by_pair, device, shuffle_time=True, seed=42)
    r_time_shuf = within_pair_r(pred_tshuf, true_np)
    print(f"Within-pair R (real time):   {r_real:.4f}")
    print(f"Within-pair R (random time): {r_time_shuf:.4f}")
    print(f"Difference: {r_real - r_time_shuf:.4f}")
    results["time_shuffle"] = r_time_shuf
    if r_real > r_time_shuf + 0.05:
        print("VERDICT: model GENUINELY uses temporal information")
    else:
        print("VERDICT: WARNING -- performance similar with random time, "
              "time conditioning may not be contributing meaningfully")

    print("\n=== TEST 3: RNA SHUFFLE (inference-time, model unchanged) ===")
    pred_rshuf, _ = run_inference(model, test_ds_by_pair, device, shuffle_rna=True, seed=42)
    r_rna_shuf = within_pair_r(pred_rshuf, true_np)
    print(f"Within-pair R (real RNA):     {r_real:.4f}")
    print(f"Within-pair R (shuffled RNA): {r_rna_shuf:.4f}")
    print(f"Difference: {r_real - r_rna_shuf:.4f}")
    results["rna_shuffle"] = r_rna_shuf
    if r_real > r_rna_shuf + 0.05:
        print("VERDICT: model GENUINELY uses RNA signal")
    else:
        print("VERDICT: WARNING -- performance similar with shuffled RNA, "
              "model may not be using RNA content meaningfully")

    with open(os.path.join(args.outdir, "posthoc_shuffle_results.json"), "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved to {args.outdir}/posthoc_shuffle_results.json")


if __name__ == "__main__":
    main()