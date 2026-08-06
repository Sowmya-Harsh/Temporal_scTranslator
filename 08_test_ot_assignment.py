"""
08_test_ot_assignment.py

Purpose: Run all three OT matching strategies (soft, hard exact,
hard bijective assignment) on a chosen t_src -> t_tgt pair + gene
selection, and report exactly how cells get matched in each case:
  - how many target cells are used at all
  - how many times each target is reused
  - which (if any) source cells are left unmatched
  - cost matrix spread, as a diagnostic for whether the match is
    likely meaningful or close to noise (relevant for long delta_t)

Usage:
    python 08_test_ot_assignment.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --t_src 0 --t_tgt 180 \
        --gene_strategy hvg --n_genes 200
"""

import argparse
import sys
import os
import numpy as np
import anndata as ad
from sklearn.metrics.pairwise import cosine_distances
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))
from gene_selection import select_genes

try:
    import ot
except ImportError:
    print("ERROR: POT (Python Optimal Transport) not installed. pip install pot --break-system-packages")
    sys.exit(1)


def split_cells_per_timepoint(obs_time, train_ratio=0.8, val_ratio=0.1, seed=1105):
    rng = np.random.default_rng(seed)
    timepoints = sorted(np.unique(obs_time).tolist())
    train_cells = {}
    for t in timepoints:
        cells = np.where(np.isclose(obs_time, t, atol=1e-3))[0]
        rng.shuffle(cells)
        n_train = int(len(cells) * train_ratio)
        train_cells[t] = cells[:n_train]
    return train_cells


def report_cost_spread(C, label):
    print(f"\n  Cost matrix spread ({label}):")
    print(f"    min={C.min():.4f}  max={C.max():.4f}  mean={C.mean():.4f}  std={C.std():.4f}")
    rel_spread = C.std() / (C.mean() + 1e-8)
    print(f"    relative spread (std/mean) = {rel_spread:.4f}")
    if rel_spread < 0.05:
        print("    -> WARNING: very low relative spread. Costs are nearly uniform, "
              "meaning source cells are roughly equidistant from ALL target cells. "
              "OT matches here are likely close to noise / arbitrary, not "
              "biologically meaningful similarity. Treat results at this Δt "
              "cautiously, expect them to look similar to a random baseline.")
    elif rel_spread < 0.15:
        print("    -> CAUTION: moderate spread. Some signal, but matches may be "
              "weakly determined. Compare against the mean-target baseline.")
    else:
        print("    -> OK: meaningful spread, matches likely reflect real similarity structure.")


def run_soft(rna_src, rna_tgt, top_genes, reg=0.1):
    print("\n" + "=" * 70)
    print("SOFT OT (Sinkhorn, averaging)")
    print("=" * 70)
    n_src, n_tgt = len(rna_src), len(rna_tgt)
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)
    report_cost_spread(C, "soft")

    a = np.ones(n_src) / n_src
    b = np.ones(n_tgt) / n_tgt
    T = ot.sinkhorn(a, b, C, reg=reg)
    T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-8)

    # "effective number of targets used" per source cell = 1 / sum(weight^2)
    # (inverse participation ratio -- 1 means fully concentrated on one
    # target, n_tgt means fully spread evenly across all targets)
    eff_n_targets = 1.0 / (T_norm ** 2).sum(axis=1)
    print(f"\n  Every source cell gets a target: YES (weighted average, always defined).")
    print(f"  Effective # targets blended per source cell (1=hard-like, {n_tgt}=fully uniform):")
    print(f"    mean={eff_n_targets.mean():.2f}  min={eff_n_targets.min():.2f}  max={eff_n_targets.max():.2f}")
    if eff_n_targets.mean() > n_tgt * 0.5:
        print(f"    -> Soft OT is blending across roughly HALF or more of all target "
              f"cells on average. This is close to just using the target mean. "
              f"Likely indicator of low match confidence (see cost spread above).")


def run_hard_exact(rna_src, rna_tgt, top_genes):
    print("\n" + "=" * 70)
    print("HARD OT (exact EMD + argmax, no averaging)")
    print("=" * 70)
    n_src, n_tgt = len(rna_src), len(rna_tgt)
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)
    report_cost_spread(C, "hard exact")

    a = np.ones(n_src) / n_src
    b = np.ones(n_tgt) / n_tgt
    T = ot.emd(a, b, C)
    match_idx = T.argmax(axis=1)
    match_counts = np.bincount(match_idx, minlength=n_tgt)

    print(f"\n  Every source cell gets a target: YES (deterministic argmax, always defined).")
    print(f"  Target cells used at least once: {(match_counts > 0).sum()} / {n_tgt} "
          f"({100*(match_counts > 0).mean():.1f}%)")
    print(f"  Target cells NEVER used: {(match_counts == 0).sum()} / {n_tgt} "
          f"({100*(match_counts == 0).mean():.1f}%)")
    print(f"  Most-reused target cell: matched {match_counts.max()}x")
    if match_counts.max() > n_src * 0.3:
        print(f"    -> WARNING: one target cell absorbs >{30}% of all source cells' "
              f"matches. Likely sign of a weak/ambiguous cost landscape (a few "
              f"target cells look 'closest' to almost everything).")


def run_hard_assignment(rna_src, rna_tgt, top_genes):
    print("\n" + "=" * 70)
    print("HARD ASSIGNMENT (Hungarian, bijective 1-to-1)")
    print("=" * 70)
    n_src, n_tgt = len(rna_src), len(rna_tgt)
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)
    report_cost_spread(C, "hard assignment")

    row_idx, col_idx = linear_sum_assignment(C)
    n_matched = len(row_idx)
    unmatched_src = n_src - n_matched
    unmatched_tgt = n_tgt - n_matched

    print(f"\n  Cells matched: {n_matched} (1-to-1, each target used EXACTLY once)")
    if n_src == n_tgt:
        print(f"  Balanced case: all {n_src} source cells matched, all {n_tgt} "
              f"target cells used exactly once. Nothing left unmatched.")
    else:
        print(f"  UNBALANCED case (n_src={n_src} != n_tgt={n_tgt}):")
        print(f"    Source cells LEFT UNMATCHED: {unmatched_src} "
              f"({'none -- all matched, some targets unused' if unmatched_src <= 0 else str(unmatched_src)+' excluded from training entirely'})")
        print(f"    Target cells never used: {unmatched_tgt if unmatched_tgt > 0 else 0}")
        print(f"    -> These excluded cells are DROPPED, not averaged or approximated. "
              f"If n_src >> n_tgt or vice versa, this can throw away a lot of data. "
              f"Check whether this pair's split sizes make hard_assignment practical.")


def main(rna_path, protein_path, t_src, t_tgt, gene_strategy, n_genes, condition="aIg"):
    rna_adata = ad.read_h5ad(rna_path)
    prot_adata = ad.read_h5ad(protein_path)
    mask = rna_adata.obs["condition"] == condition
    rna_adata = rna_adata[mask].copy()
    prot_adata = prot_adata[mask].copy()

    rna_counts = rna_adata.layers["counts"]
    rna_counts = (rna_counts.toarray() if hasattr(rna_counts, "toarray") else rna_counts).astype(np.float32)
    obs_time = rna_adata.obs["time_min"].values.astype(float)

    n_genes_arg = n_genes if n_genes == "all" else int(n_genes)
    gene_idx = select_genes(rna_adata, gene_strategy, n_genes_arg, groupby="time_min")
    rna_selected = rna_counts[:, gene_idx]

    train_cells = split_cells_per_timepoint(obs_time)
    if t_src not in train_cells or t_tgt not in train_cells:
        print(f"ERROR: t_src={t_src} or t_tgt={t_tgt} not in data. Available: {sorted(train_cells.keys())}")
        return

    src_idx = train_cells[t_src]
    tgt_idx = train_cells[t_tgt]

    print(f"Config: t_src={t_src} t_tgt={t_tgt} (Δt={t_tgt-t_src}) "
          f"gene_strategy={gene_strategy} n_genes={n_genes_arg} (actual={len(gene_idx)})")
    print(f"Source (train) cells at t={t_src}: {len(src_idx)}")
    print(f"Target (train) cells at t={t_tgt}: {len(tgt_idx)}")

    rna_src = rna_selected[src_idx]
    rna_tgt = rna_selected[tgt_idx]
    top_genes = np.arange(rna_src.shape[1])  # already selected upstream

    run_soft(rna_src, rna_tgt, top_genes)
    run_hard_exact(rna_src, rna_tgt, top_genes)
    run_hard_assignment(rna_src, rna_tgt, top_genes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--t_src", type=float, required=True)
    parser.add_argument("--t_tgt", type=float, required=True)
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], default="hvg")
    parser.add_argument("--n_genes", default="200")
    parser.add_argument("--condition", default="aIg")
    args = parser.parse_args()
    main(args.rna, args.protein, args.t_src, args.t_tgt, args.gene_strategy, args.n_genes, args.condition)