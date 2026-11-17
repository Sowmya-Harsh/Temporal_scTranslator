"""
ot_coverage_comparison.py
───────────────────────────
Answers two open items together, since they're the same experiment:

  1. The original soft-vs-hard-vs-Hungarian OT matching table used
     HVG-100 instead of DEG, with no confirmed reason -- everywhere
     else in the thesis argues DEG beats HVG. This reruns the
     comparison with DEG (gene selection computed on train-split
     cells only, per the leakage fix already applied elsewhere).

  2. Hard matching's coverage (fraction of the target population ever
     selected by ANY source cell) was reported as 8-16%. Soft
     matching's equivalent number was never computed. This computes
     it directly, same style, side by side.

Coverage definitions (population-level, NOT the per-source IPR/
"effective blend width" from 09_sweep_ot_reg.py / 20_myeloid_ot_blending_check.py
-- this is a different question: what FRACTION of the target
population ever gets used, not how many targets each source blends):

  - HARD (exact, argmax): matches ot_matching.py's existing print --
    fraction of target cells matched >=1x across all source cells.
  - SOFT (Sinkhorn): a target cell counts as "meaningfully used" if
    ANY source cell assigns it weight above `--coverage_threshold`
    times the uniform baseline (1/n_tgt). Default threshold=2.0 (i.e.
    at least double the "no information" baseline weight) -- this is
    a judgment call, reported explicitly so it's auditable; the script
    also reports the raw distribution so you can see how sensitive
    the number is to threshold choice.

IMPORTANT -- matches the actual EVALUATION methodology, not training:
source cells are the TEST split of t_src (small, held-out), matched
against the TRAIN split of t_tgt (the full target pool used to build
OT targets everywhere else in this pipeline). This mirrors what
happens at evaluation time and is what the thesis's original "8-16%"
hard-coverage figure measured -- train-vs-train matching gives a much
easier, more balanced problem and is NOT comparable to that figure.

Runs across all 10 CURRICULUM timepoint pairs, using each pair's
TRAIN-split cells only (source and target), consistent with the rest
of the leakage-fixed pipeline.

Usage:
    python ot_coverage_comparison.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 100 \
        --ot_reg 0.01 --coverage_threshold 2.0 \
        --outdir results/ot_coverage_comparison
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics.pairwise import cosine_distances

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from train_joint import CURRICULUM, split_cells_per_timepoint

try:
    import ot as pot
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    print("ERROR: POT not installed. pip install pot --break-system-packages")
    sys.exit(1)


def _select_top_var_genes(rna_src, rna_tgt, max_genes):
    gene_var = rna_src.var(axis=0) + rna_tgt.var(axis=0)
    return np.argsort(gene_var)[-max_genes:]


def soft_coverage(rna_src, rna_tgt, reg, threshold_mult, max_genes=2000):
    """Returns (coverage_pct, n_used, n_untouched, n_total, T_norm) for
    Sinkhorn soft matching. n_used: how many target cells receive weight
    >= threshold_mult/n_tgt from AT LEAST ONE source cell. n_untouched:
    how many target cells never reach that threshold from ANY source
    cell -- i.e. cells the model effectively never learns from, at this
    threshold."""
    n_src, n_tgt = len(rna_src), len(rna_tgt)
    top_genes = _select_top_var_genes(rna_src, rna_tgt, max_genes)
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)
    a = np.ones(n_src, dtype=np.float64) / n_src
    b = np.ones(n_tgt, dtype=np.float64) / n_tgt
    T = pot.sinkhorn(a, b, C, reg=reg)
    T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-8)  # row-normalized, per source cell

    uniform_weight = 1.0 / n_tgt
    threshold = threshold_mult * uniform_weight
    # A target column is "used" if ANY row (source cell) assigns it >= threshold
    used_mask = (T_norm >= threshold).any(axis=0)
    n_used = int(used_mask.sum())
    n_untouched = n_tgt - n_used
    coverage_pct = 100.0 * n_used / n_tgt
    return coverage_pct, n_used, n_untouched, n_tgt, T_norm


def hard_coverage(rna_src, rna_tgt, max_genes=2000):
    """Returns (coverage_pct, n_used, n_untouched, n_total) for exact
    hard OT (argmax assignment), matching ot_matching.py's
    ot_hard_targets_exact print exactly. n_used: target cells selected
    by at least one source cell's argmax. n_untouched: target cells
    NEVER selected by any source cell under this assignment."""
    n_src, n_tgt = len(rna_src), len(rna_tgt)
    top_genes = _select_top_var_genes(rna_src, rna_tgt, max_genes)
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)
    a = np.ones(n_src, dtype=np.float64) / n_src
    b = np.ones(n_tgt, dtype=np.float64) / n_tgt
    T = pot.emd(a, b, C)
    match_idx = T.argmax(axis=1)
    match_counts = np.bincount(match_idx, minlength=n_tgt)
    n_used = int((match_counts > 0).sum())
    n_untouched = n_tgt - n_used
    coverage_pct = 100.0 * n_used / n_tgt
    return coverage_pct, n_used, n_untouched, n_tgt


def hungarian_coverage(rna_src, rna_tgt, max_genes=2000):
    """Returns (coverage_pct, n_used, n_untouched, n_total) for
    hard-assignment (Hungarian, bijective) matching -- strict
    one-to-one, every USED target cell is used at most once:
    'the cost matrix is passed to the Hungarian algorithm, producing a
    strict one-to-one assignment in which every target cell is used at
    most once'.

    scipy's linear_sum_assignment handles rectangular cost matrices by
    matching min(n_src, n_tgt) pairs. So coverage here is mechanical,
    not learned:
      - n_tgt <= n_src: every target gets matched exactly once -> 100%,
        n_untouched=0.
      - n_tgt >  n_src: only n_src of the n_tgt targets are ever used,
        the rest are structurally unreachable under a bijection ->
        n_untouched = n_tgt - n_src, regardless of RNA similarity.
    This is reported for completeness alongside soft/hard-exact, not
    because it's a meaningful matching-QUALITY number -- it's a direct
    consequence of population sizes under a bijective constraint.
    """
    from scipy.optimize import linear_sum_assignment
    n_src, n_tgt = len(rna_src), len(rna_tgt)
    top_genes = _select_top_var_genes(rna_src, rna_tgt, max_genes)
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(C)
    n_used = len(set(col_ind.tolist()))
    n_untouched = n_tgt - n_used
    coverage_pct = 100.0 * n_used / n_tgt
    return coverage_pct, n_used, n_untouched, n_tgt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--condition", default="aIg")
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], default="deg",
                         help="Default 'deg', not 'hvg' -- this rerun exists specifically "
                              "to replace the original HVG-100 comparison.")
    parser.add_argument("--n_genes", default="100")
    parser.add_argument("--ot_reg", type=float, default=0.01)
    parser.add_argument("--coverage_threshold", type=float, default=2.0,
                         help="Soft-match target counts as 'covered' if any source cell "
                              "gives it weight >= threshold * uniform baseline (1/n_tgt).")
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)

    rna_adata = ad.read_h5ad(args.rna)
    prot_adata = ad.read_h5ad(args.protein)
    mask = rna_adata.obs["condition"] == args.condition
    rna_adata = rna_adata[mask].copy()
    rna_full = rna_adata.layers["counts"]
    rna_full = (rna_full.toarray() if hasattr(rna_full, "toarray") else rna_full).astype(np.float32)
    obs_time = rna_adata.obs["time_min"].values.astype(float)

    # --- Split first, gene selection on train cells only (leakage fix) ---
    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)
    train_idx_all = np.concatenate([idx for idx in train_cells.values() if len(idx) > 0])
    rna_adata_train = rna_adata[train_idx_all].copy()
    gene_idx = select_genes(rna_adata_train, args.gene_strategy, n_genes_arg, groupby="time_min")
    print(f"Selected {len(gene_idx)} genes via {args.gene_strategy} "
          f"(train-split cells only: {len(train_idx_all)}/{rna_adata.n_obs})")
    rna_selected = rna_full[:, gene_idx]

    available_pairs = [(s, t) for s, t in CURRICULUM if s in train_cells and t in train_cells]

    rows = []
    for (t_src, t_tgt) in available_pairs:
        # Match the actual evaluation methodology: a SMALL held-out source
        # population (test split) matched against the FULL training target
        # pool -- not train-vs-train, which is a much more balanced, easier
        # matching problem and gives a misleadingly high hard-coverage number.
        # This is what the original "8-16%" hard-coverage figure measured.
        src_idx = test_cells[t_src]
        tgt_idx = train_cells[t_tgt]
        if len(src_idx) < 2 or len(tgt_idx) < 2:
            print(f"  SKIP {t_src}->{t_tgt}: too few cells (src={len(src_idx)}, tgt={len(tgt_idx)}).")
            continue

        soft_cov, soft_used, soft_untouched, n_tgt_total, T_norm = soft_coverage(
            rna_selected[src_idx], rna_selected[tgt_idx], args.ot_reg, args.coverage_threshold)
        hard_cov, hard_used, hard_untouched, _ = hard_coverage(rna_selected[src_idx], rna_selected[tgt_idx])
        hun_cov, hun_used, hun_untouched, _ = hungarian_coverage(rna_selected[src_idx], rna_selected[tgt_idx])

        print(f"  {t_src}->{t_tgt}: n_src(test)={len(src_idx)} n_tgt(train pool)={n_tgt_total} | "
              f"soft: {soft_used}/{n_tgt_total} used, {soft_untouched} untouched ({soft_cov:.1f}%) | "
              f"hard: {hard_used}/{n_tgt_total} used, {hard_untouched} untouched ({hard_cov:.1f}%) | "
              f"hungarian: {hun_used}/{n_tgt_total} used, {hun_untouched} untouched ({hun_cov:.1f}%)")

        rows.append({
            "t_src": t_src, "t_tgt": t_tgt, "n_src_test": len(src_idx), "n_tgt_train_pool": n_tgt_total,
            "soft_used": soft_used, "soft_untouched": soft_untouched, "soft_coverage_pct": soft_cov,
            "hard_used": hard_used, "hard_untouched": hard_untouched, "hard_coverage_pct": hard_cov,
            "hungarian_used": hun_used, "hungarian_untouched": hun_untouched, "hungarian_coverage_pct": hun_cov,
            "coverage_threshold_mult": args.coverage_threshold,
            "gene_strategy": args.gene_strategy, "n_genes": len(gene_idx),
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(args.outdir, "ot_coverage_comparison.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}\nOVERALL MEANS (gene_strategy={args.gene_strategy}, n_genes={len(gene_idx)})\n{'='*60}")
    print(f"Soft: {df['soft_used'].sum()}/{df['n_tgt_train_pool'].sum()} target-cell-slots used across all pairs "
          f"({df['soft_untouched'].sum()} untouched) | mean coverage {df['soft_coverage_pct'].mean():.1f}% "
          f"(range {df['soft_coverage_pct'].min():.1f}-{df['soft_coverage_pct'].max():.1f}%)")
    print(f"Hard: {df['hard_used'].sum()}/{df['n_tgt_train_pool'].sum()} target-cell-slots used across all pairs "
          f"({df['hard_untouched'].sum()} untouched) | mean coverage {df['hard_coverage_pct'].mean():.1f}% "
          f"(range {df['hard_coverage_pct'].min():.1f}-{df['hard_coverage_pct'].max():.1f}%)")
    print(f"Hungarian: {df['hungarian_used'].sum()}/{df['n_tgt_train_pool'].sum()} target-cell-slots used "
          f"({df['hungarian_untouched'].sum()} untouched) | mean coverage {df['hungarian_coverage_pct'].mean():.1f}% "
          f"-- mechanical consequence of n_src/n_tgt sizes, not matching quality")
    print(f"\nSaved to {out_path}")

    with open(os.path.join(args.outdir, "ot_coverage_config.json"), "w") as f:
        json.dump({"config": vars(args), "n_genes_actual": len(gene_idx)}, f, indent=2)


if __name__ == "__main__":
    main()
