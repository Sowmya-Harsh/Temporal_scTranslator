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

--- CHANGES IN THIS VERSION (v3) ---
Three successive bugs were found and fixed in run_inference() across
earlier revisions of this script:

1. (v1 bug) Time-shuffle originally drew t_src and delta INDEPENDENTLY
   from Uniform(0, 180) each. This could produce (t_src, delta)
   combinations that never occur in the real data, and t_tgt values
   exceeding 180 minutes, outside the dataset's range -- confounding a
   genuine time-blindness test with plain extrapolation failure.

2. (v2 bug, introduced while fixing #1) Time-shuffle was changed to
   permute t_info WITHIN each pair's own test set. This is a no-op:
   within a single timepoint pair, every cell shares the SAME (t_src,
   delta) value by construction (e.g. every cell in the "0->2" test set
   has t_src=0, delta=2 -- that is what defines the pair). Permuting a
   list of identical values leaves it unchanged, which produced a null
   distribution with zero variance, exactly matching real R, and a
   p-value of 1.0 -- not evidence that time doesn't matter, but evidence
   the shuffle wasn't actually changing anything.

3. (this version, v3) Time-shuffle now draws each cell's (t_src, delta)
   from the POOL of real, observed transitions across ALL TEN pairs, not
   from within its own pair. Every drawn value is still a genuinely
   observed transition (never synthetic, never out-of-range), but a
   cell's RNA/target can now be paired with a *different* pair's real
   time-conditioning value, producing genuine, meaningful variation.

   Also, both time-shuffle and RNA-shuffle now operate on each pair's
   FULL test set (55-97 cells), not within 32-cell DataLoader batches,
   which had understated null variance in earlier runs.

Prediction-shuffle (shuffling the model's own output predictions across
cells within a pair) was never affected by any of the above and is
unchanged throughout.

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
    """
    Run inference on each pair's full test set, optionally shuffling RNA
    input (within-pair) or time-conditioning input (POOLED ACROSS PAIRS).

    IMPORTANT: within a single timepoint pair's test set, every cell has
    the SAME (t_src, delta) value by construction (e.g. every cell in the
    "0->2" test set has t_src=0, delta=2). Permuting t_info WITHIN one
    pair is therefore a no-op -- it just reorders a list of identical
    values. To genuinely test whether the model uses time-conditioning,
    the shuffle must draw from the pool of real (t_src, delta) values
    observed ACROSS ALL PAIRS, so a cell's RNA/target can be paired with
    a different, but still genuinely observed, transition's time-
    conditioning value. RNA-shuffle does not have this issue, since RNA
    genuinely varies across cells within a pair, so it stays within-pair.

    Both shuffles operate on each pair's full test set (not per-batch),
    to avoid the earlier micro-batch-scoped shuffling artefact.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    pred_by_pair, true_by_pair = {}, {}

    # First pass: load every pair's full test set once, and build the pool
    # of real (t_src, delta) values observed across ALL pairs, for the
    # time-shuffle draw below.
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
        # Pool of real, observed (t_src, delta) conditioning vectors
        # across all ten pairs. Every row here is a genuinely observed
        # transition; no synthetic or out-of-range values are introduced.
        pooled_t_info = torch.cat(all_t_info, dim=0)
        n_pool = pooled_t_info.shape[0]

    for pair, (x_all, y_all, t_info_all) in loaded.items():
        if shuffle_rna:
            # RNA genuinely varies across cells within a pair, so a
            # within-pair permutation is meaningful here.
            perm = torch.from_numpy(rng.permutation(x_all.shape[0]))
            x_all = x_all[perm]

        if shuffle_time:
            # Draw, with replacement, from the pool of real (t_src, delta)
            # values observed across ALL pairs -- one draw per cell in
            # this pair's test set. Every drawn value is a genuinely
            # observed transition (possibly from a different pair), so no
            # synthetic or out-of-range time value is ever supplied, while
            # still producing genuine, non-trivial variation (unlike a
            # within-pair permutation, which is a no-op since t_info is
            # constant within a single pair).
            draw_idx = rng.integers(0, n_pool, size=x_all.shape[0])
            t_info_all = pooled_t_info[draw_idx]

        x_all, t_info_all = x_all.to(device), t_info_all.to(device)
        with torch.no_grad():
            y_hat = model(x_all, t_info_all).cpu().numpy()

        pred_by_pair[pair] = y_hat
        true_by_pair[pair] = y_all.numpy()

    return pred_by_pair, true_by_pair


def permutation_test_prediction_shuffle(pred_real, true_np, real_r, n_permutations, seed):
    """Repeatedly shuffle the model's own predictions across cells (within
    each pair) and recompute R, n_permutations times. Unaffected by the
    fixes above: this only reorders the model's own real, already-computed
    outputs, never draws new synthetic values or re-runs inference."""
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
          f"ot_type={args.ot_type} ot_reg={args.ot_reg} seed={args.seed} "
          f"(gene selection computed post-split, train-split cells only, matching "
          f"train_joint_improved.py's leak-fixed procedure)")

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

    # --- Split FIRST, then select genes using ONLY the train-split cells ---
    # Must match train_joint_improved.py's procedure exactly: gene
    # selection is computed on the train-split subset only, AFTER
    # splitting, not on the full population before splitting. Using the
    # wrong order here reconstructs a different gene panel than the one
    # the checkpoint was actually trained on, silently feeding the model
    # RNA features in the wrong identity/order and producing a spuriously
    # low R that reflects a data-reconstruction mismatch, not genuine
    # model performance.
    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)

    train_idx_all = np.concatenate([idx for idx in train_cells.values() if len(idx) > 0])
    rna_adata_train = rna_adata[train_idx_all].copy()
    gene_idx = select_genes(rna_adata_train, args.gene_strategy, n_genes_arg, groupby="time_min")
    print(f"Selected {len(gene_idx)} genes via {args.gene_strategy} "
          f"(train-split cells only: {len(train_idx_all)}/{rna_adata.n_obs}).")
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
    print(f"Real R = {real_r:.4f}")
    for name, key in [("Prediction-shuffle", "prediction_shuffle"),
                       ("Time-shuffle", "time_shuffle"),
                       ("RNA-shuffle", "rna_shuffle")]:
        p = results[key]["p_value"]
        p_str = f"p < {1/args.n_permutations:.4f}" if p == 0 else f"p = {p:.4f}"
        print(f"  {name}: {p_str} (n={args.n_permutations} permutations)")


if __name__ == "__main__":
    main()