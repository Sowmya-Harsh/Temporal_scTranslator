"""
19_permutation_test_pvalues.py

Purpose: Upgrade the existing single-shuffle diagnostics (posthoc_shuffle_
tests.py) into a proper permutation test with empirical p-values. Instead
of comparing real R against ONE shuffled R, this repeats each shuffle
n_permutations times (default 1000), builds an empirical null distribution
of R under random shuffling, and reports:

  p-value = (# permuted trials with R >= real R) / n_permutations

A small p-value (e.g. < 0.05, or < 0.001 at 1000 permutations) means the
real, unshuffled R is not something you'd expect to see by chance under
the null hypothesis that shuffling doesn't matter -- i.e. genuine,
statistically significant signal, not just "R dropped, so it's probably
fine."

Reuses the exact same test-set reconstruction, model loading, and
inference logic as posthoc_shuffle_tests.py (same seed/config contract),
just repeats the three shuffle tests many times instead of once.

Usage:
    python 19_permutation_test_pvalues.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --ot_type soft --ot_reg 0.01 \
        --model film_pre \
        --checkpoint run_results/correlationlosswt_test_deg1000_filmpre/joint_model_best.pt \
        --seed 1105 \
        --n_permutations 1000 \
        --outdir results/permutation_test_filmpre
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


def permutation_test_prediction_shuffle(pred_real, true_np, real_r, n_permutations, seed):
    """Repeatedly shuffle the model's own predictions across cells (within
    each pair) and recompute R, n_permutations times."""
    rng = np.random.default_rng(seed)
    null_rs = []
    for _ in range(n_permutations):
        pred_shuf = {}
        for pair, p in pred_real.items():
            perm = rng.permutation(len(p))
            pred_shuf[pair] = p[perm]
        null_rs.append(within_pair_r(pred_shuf, true_np))
    null_rs = np.array(null_rs)
    p_value = float(np.mean(null_rs >= real_r))
    return null_rs, p_value


def permutation_test_inference_shuffle(model, test_ds_by_pair, device, true_np, real_r,
                                        n_permutations, shuffle_time, shuffle_rna, base_seed):
    """Repeatedly re-run inference with time-shuffle or RNA-shuffle applied,
    each with a different random seed, n_permutations times. More
    expensive than the prediction-shuffle version since it re-runs the
    model forward pass each time, not just a reordering."""
    null_rs = []
    for i in range(n_permutations):
        pred_shuf, _ = run_inference(model, test_ds_by_pair, device,
                                      shuffle_time=shuffle_time, shuffle_rna=shuffle_rna,
                                      seed=base_seed + i)
        null_rs.append(within_pair_r(pred_shuf, true_np))
    null_rs = np.array(null_rs)
    p_value = float(np.mean(null_rs >= real_r))
    return null_rs, p_value


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
    parser.add_argument("--n_permutations", type=int, default=1000,
                         help="Number of repeated shuffles for the permutation test. "
                              "1000 gives p-value resolution down to 0.001. Note: "
                              "time-shuffle and RNA-shuffle permutations each require a "
                              "full model forward pass, so runtime scales with this and "
                              "with dataset/model size -- start with e.g. 200 if 1000 is "
                              "too slow, and note the reduced p-value resolution.")
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

    gene_idx = select_genes(rna_adata, args.gene_strategy, n_genes_arg, groupby="time_min")
    rna_selected = rna_full[:, gene_idx]

    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)
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

    print("\n=== REAL (baseline) ===")
    pred_real, true_np = run_inference(model, test_ds_by_pair, device)
    real_r = within_pair_r(pred_real, true_np)
    print(f"Real within-pair R: {real_r:.4f}")

    results = {"real_r": real_r, "n_permutations": args.n_permutations, "seed": args.seed}

    print(f"\n=== PERMUTATION TEST 1: PREDICTION SHUFFLE ({args.n_permutations} permutations) ===")
    null_rs_pred, p_pred = permutation_test_prediction_shuffle(
        pred_real, true_np, real_r, args.n_permutations, seed=args.seed)
    print(f"Null distribution: mean={null_rs_pred.mean():.4f}, std={null_rs_pred.std():.4f}, "
          f"95th pct={np.percentile(null_rs_pred, 95):.4f}")
    print(f"Real R ({real_r:.4f}) vs null distribution -> p-value = {p_pred:.4f}"
          f"{' (< 1/n_permutations, report as p < %.4f)' % (1/args.n_permutations) if p_pred == 0 else ''}")
    results["prediction_shuffle"] = {
        "p_value": p_pred, "null_mean": float(null_rs_pred.mean()), "null_std": float(null_rs_pred.std()),
        "null_95th_percentile": float(np.percentile(null_rs_pred, 95)),
    }

    print(f"\n=== PERMUTATION TEST 2: TIME SHUFFLE ({args.n_permutations} permutations, slower) ===")
    null_rs_time, p_time = permutation_test_inference_shuffle(
        model, test_ds_by_pair, device, true_np, real_r,
        args.n_permutations, shuffle_time=True, shuffle_rna=False, base_seed=args.seed * 1000)
    print(f"Null distribution: mean={null_rs_time.mean():.4f}, std={null_rs_time.std():.4f}, "
          f"95th pct={np.percentile(null_rs_time, 95):.4f}")
    print(f"Real R ({real_r:.4f}) vs null distribution -> p-value = {p_time:.4f}")
    results["time_shuffle"] = {
        "p_value": p_time, "null_mean": float(null_rs_time.mean()), "null_std": float(null_rs_time.std()),
        "null_95th_percentile": float(np.percentile(null_rs_time, 95)),
    }

    print(f"\n=== PERMUTATION TEST 3: RNA SHUFFLE ({args.n_permutations} permutations, slower) ===")
    null_rs_rna, p_rna = permutation_test_inference_shuffle(
        model, test_ds_by_pair, device, true_np, real_r,
        args.n_permutations, shuffle_time=False, shuffle_rna=True, base_seed=args.seed * 2000)
    print(f"Null distribution: mean={null_rs_rna.mean():.4f}, std={null_rs_rna.std():.4f}, "
          f"95th pct={np.percentile(null_rs_rna, 95):.4f}")
    print(f"Real R ({real_r:.4f}) vs null distribution -> p-value = {p_rna:.4f}")
    results["rna_shuffle"] = {
        "p_value": p_rna, "null_mean": float(null_rs_rna.mean()), "null_std": float(null_rs_rna.std()),
        "null_95th_percentile": float(np.percentile(null_rs_rna, 95)),
    }

    np.save(os.path.join(args.outdir, "null_dist_prediction_shuffle.npy"), null_rs_pred)
    np.save(os.path.join(args.outdir, "null_dist_time_shuffle.npy"), null_rs_time)
    np.save(os.path.join(args.outdir, "null_dist_rna_shuffle.npy"), null_rs_rna)
    with open(os.path.join(args.outdir, "permutation_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results and null distributions to {args.outdir}/")
    print("\n" + "=" * 60)
    print("SUMMARY (for thesis reporting)")
    print("=" * 60)
    print(f"Real R = {real_r:.4f}")
    for name, key in [("Prediction-shuffle", "prediction_shuffle"),
                       ("Time-shuffle", "time_shuffle"),
                       ("RNA-shuffle", "rna_shuffle")]:
        p = results[key]["p_value"]
        p_str = f"p < {1/args.n_permutations:.4f}" if p == 0 else f"p = {p:.4f}"
        print(f"  {name}: {p_str} (n={args.n_permutations} permutations)")


if __name__ == "__main__":
    main()
