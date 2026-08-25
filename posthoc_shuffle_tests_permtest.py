"""
posthoc_shuffle_tests.py  (permutation-test + sanity-check version)
────────────────────────────────────────────────────────────────────
Loads an ALREADY-TRAINED train_joint_improved.py checkpoint and runs
three diagnostics as a proper permutation test (n_permutations repeated
shuffles, empirical p-values), purely at inference time.

--- WHY THIS VERSION EXISTS ---
An earlier attempt at this (19_permutation_test_pvalues.py) reconstructed
the test sets with gene selection computed BEFORE the train/val/test
split, on the full population, whereas train_joint_improved.py (which
actually produced the checkpoints) selects genes AFTER splitting, using
train-split cells only. That mismatch alone was fixed, but the "real R"
baseline it then computed (0.084) still did not match the checkpoint's
own reported test performance (~0.32, from train_joint_improved.py's own
evaluate_per_pair output / joint_results_summary.csv).

This version starts from posthoc_shuffle_tests.py, which ALREADY uses
the correct post-split gene selection order (matching
train_joint_improved.py exactly), and adds a mandatory sanity check:
before running any shuffle test, it calls evaluate_per_pair() directly
(the exact function train_joint_improved.py uses to produce its reported
numbers) on the reconstructed test sets, and prints its result alongside
this script's own within_pair_r() computation on the same predictions.

  - If the two match: the data reconstruction (split, gene selection, OT
    matching) is correct, and any remaining discrepancy is isolated to
    scoring/aggregation logic in within_pair_r() -- worth inspecting or
    replacing with evaluate_per_pair's own internals directly.
  - If the two do NOT match: something in the reconstructed test sets
    themselves (OT matching, cell split, or gene panel) still differs
    from what the checkpoint was trained/evaluated on, independent of
    scoring, and needs further investigation before the permutation
    p-values can be trusted.

Diagnostics:
1. PREDICTION-SHUFFLE: permute the model's own predictions across cells
   (within each timepoint pair). Sanity floor check, unaffected by any
   of the issues above.
2. TIME-SHUFFLE: draw each cell's (t_src, delta) conditioning input from
   the POOL of real, observed transitions across ALL TEN pairs (not from
   within its own pair -- t_info is constant within a single pair by
   construction, so a within-pair permutation is a no-op). Every value
   supplied is genuinely observed; nothing synthetic or out-of-range.
3. RNA-SHUFFLE: permute RNA input across cells within each pair's full
   test set (RNA genuinely varies within a pair, so this is meaningful).

Both shuffles operate on each pair's FULL test set, not per-DataLoader-
batch, to avoid understating null variance.

Requires the EXACT SAME config used for the original training run
(gene_strategy, n_genes, ot_type, ot_reg, model, seed) to reconstruct
the same test sets.

Usage:
    python posthoc_shuffle_tests.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --ot_type soft --ot_reg 0.01 \
        --model film_pre \
        --checkpoint .../final_model_corrloss_only_leakfix/joint_model_best.pt \
        --seed 1105 \
        --n_permutations 1000 \
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
from train_joint import CURRICULUM, split_cells_per_timepoint, build_pair_dataset, evaluate_per_pair


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
    """
    Run inference on each pair's full test set, optionally shuffling RNA
    (within-pair) or time-conditioning (POOLED ACROSS PAIRS, since t_info
    is constant within any single pair -- see module docstring point 2).
    """
    rng = np.random.default_rng(seed)
    model.eval()
    pred_by_pair, true_by_pair = {}, {}

    loaded = {}
    all_t_info = []
    for pair, ds in test_ds_by_pair.items():
        if ds is None or len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        x_all, y_all, t_info_all = next(iter(loader))
        loaded[pair] = (x_all, y_all, t_info_all)
        all_t_info.append(t_info_all)

    if shuffle_time:
        pooled_t_info = torch.cat(all_t_info, dim=0)
        n_pool = pooled_t_info.shape[0]

    for pair, (x_all, y_all, t_info_all) in loaded.items():
        if shuffle_rna:
            perm = torch.from_numpy(rng.permutation(x_all.shape[0]))
            x_all = x_all[perm]

        if shuffle_time:
            draw_idx = rng.integers(0, n_pool, size=x_all.shape[0])
            t_info_all = pooled_t_info[draw_idx]

        x_all, t_info_all = x_all.to(device), t_info_all.to(device)
        with torch.no_grad():
            y_hat = model(x_all, t_info_all).cpu().numpy()

        pred_by_pair[pair] = y_hat
        true_by_pair[pair] = y_all.numpy()

    return pred_by_pair, true_by_pair


def permutation_test_prediction_shuffle(pred_real, true_np, real_r, n_permutations, seed):
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
    parser.add_argument("--n_permutations", type=int, default=1000)
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

    # Split FIRST, then select genes on train-split cells only -- matches
    # train_joint_improved.py exactly (this ordering was already correct
    # in the original posthoc_shuffle_tests.py).
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

    # ------------------------------------------------------------------
    # MANDATORY SANITY CHECK: compare evaluate_per_pair (the function
    # train_joint_improved.py actually used to produce reported numbers)
    # against this script's own within_pair_r() on freshly-reconstructed
    # test sets, BEFORE trusting any permutation p-values below.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SANITY CHECK: evaluate_per_pair() vs. within_pair_r()")
    print("=" * 60)
    official_results = evaluate_per_pair(model, test_ds_by_pair, device, protein_names, outdir=None)
    official_rs = [r["mean_pearson_r"] for r in official_results.values() if not np.isnan(r["mean_pearson_r"])]
    official_mean_r = float(np.mean(official_rs)) if official_rs else float("nan")
    print(f"evaluate_per_pair() pooled mean R: {official_mean_r:.4f}   <-- should match "
          f"train_joint_improved.py's own reported test R for this checkpoint")
    for (t_src, t_tgt), res in official_results.items():
        print(f"    t{t_src}->t{t_tgt}: mean_r={res['mean_pearson_r']:.4f} (n={res['n_test']})")

    pred_real, true_np = run_inference(model, test_ds_by_pair, device)
    own_real_r = within_pair_r(pred_real, true_np)
    print(f"\nThis script's own within_pair_r() on the same test sets: {own_real_r:.4f}")

    gap = abs(official_mean_r - own_real_r)
    if gap < 0.01:
        print(f"MATCH (gap={gap:.4f}): data reconstruction and scoring are consistent. "
              f"Proceeding with permutation tests using within_pair_r().")
    else:
        print(f"*** MISMATCH (gap={gap:.4f}): evaluate_per_pair() and within_pair_r() disagree "
              f"substantially on the SAME reconstructed test sets and SAME model. This means the "
              f"discrepancy is in scoring/aggregation logic, not in data reconstruction (split/gene "
              f"selection/OT matching all match already). Inspect within_pair_r() against "
              f"evaluate_per_pair()'s internals (e.g. per-protein exclusion rules, normalisation, "
              f"or aggregation order) before trusting the permutation p-values below. ***")
    print("=" * 60 + "\n")

    real_r = own_real_r
    results = {
        "real_r": real_r,
        "official_evaluate_per_pair_r": official_mean_r,
        "sanity_check_gap": gap,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
    }

    print(f"\n=== PERMUTATION TEST 1: PREDICTION SHUFFLE ({args.n_permutations} permutations) ===")
    null_rs_pred, p_pred = permutation_test_prediction_shuffle(
        pred_real, true_np, real_r, args.n_permutations, seed=args.seed)
    print(f"Null distribution: mean={null_rs_pred.mean():.4f}, std={null_rs_pred.std():.4f}, "
          f"95th pct={np.percentile(null_rs_pred, 95):.4f}")
    print(f"Real R ({real_r:.4f}) vs null distribution -> p-value = {p_pred:.4f}")
    results["prediction_shuffle"] = {
        "p_value": p_pred, "null_mean": float(null_rs_pred.mean()), "null_std": float(null_rs_pred.std()),
        "null_95th_percentile": float(np.percentile(null_rs_pred, 95)),
    }

    print(f"\n=== PERMUTATION TEST 2: TIME SHUFFLE ({args.n_permutations} permutations, slower) ===")
    print("(Drawing REAL observed (t_src, delta) pairs from the POOL across all timepoint "
          "pairs -- within-pair permutation is a no-op since t_info is constant within a "
          "single pair. No synthetic or out-of-range values are used.)")
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
    print("(Shuffling RNA across cells within each pair's full test set, not per-batch.)")
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
    print(f"evaluate_per_pair() R (official) = {official_mean_r:.4f}")
    print(f"within_pair_r() R (this script)  = {real_r:.4f}   (gap={gap:.4f})")
    for name, key in [("Prediction-shuffle", "prediction_shuffle"),
                       ("Time-shuffle", "time_shuffle"),
                       ("RNA-shuffle", "rna_shuffle")]:
        p = results[key]["p_value"]
        p_str = f"p < {1/args.n_permutations:.4f}" if p == 0 else f"p = {p:.4f}"
        print(f"  {name}: {p_str} (n={args.n_permutations} permutations)")


if __name__ == "__main__":
    main()