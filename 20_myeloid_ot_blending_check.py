"""
20_myeloid_ot_blending_check.py

Purpose: The myeloid ground truth targets look mean-collapsed (near-
identical values across cells). This checks whether that's caused by
OT over-blending -- same failure mode already diagnosed for QuRIE-seq
in Section 4.3.1 (eps=0.1 blended ~half the population; eps=0.01/0.005
fixed it). Myeloid's target populations are much larger than QuRIE-
seq's (up to 1,263 cells vs 896), so the SAME ot_reg=0.01 that works
fine on QuRIE-seq could be over-blending on myeloid's bigger populations.

Computes the inverse participation ratio (1 / sum(pi_ij^2)), the
effective number of target cells blended per source cell, at several
ot_reg values, so you can see directly whether 0.01 is too high here
and what value actually fixes it.

Usage:
    python 20_myeloid_ot_blending_check.py \
        --rna_path myeloid_geo_data/myeloid_RNA_standardized.h5ad \
        --prot_path myeloid_geo_data/myeloid_Protein_standardized.h5ad \
        --gene_panel_size 1000
"""
import argparse

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.metrics.pairwise import cosine_distances

try:
    import ot
except ImportError:
    raise SystemExit("POT (pip install pot) required for this script.")


def select_deg_genes(rna_adata, time_col, panel_size):
    a = rna_adata.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.obs["_time_group"] = a.obs[time_col].astype(str)
    sc.tl.rank_genes_groups(a, groupby="_time_group", method="wilcoxon")
    best_rank = {}
    for g in a.obs["_time_group"].unique():
        names = a.uns["rank_genes_groups"]["names"][g]
        for rank, gene in enumerate(names):
            if gene not in best_rank or rank < best_rank[gene]:
                best_rank[gene] = rank
    ranked = sorted(best_rank.keys(), key=lambda g: best_rank[g])
    return ranked[:panel_size]


def blending_width(rna_src, rna_tgt, reg, max_genes=2000):
    n_src, n_tgt = len(rna_src), len(rna_tgt)
    gene_var = rna_src.var(axis=0) + rna_tgt.var(axis=0)
    top_genes = np.argsort(gene_var)[-max_genes:]
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)
    a = np.ones(n_src) / n_src
    b = np.ones(n_tgt) / n_tgt
    T = ot.sinkhorn(a, b, C, reg=reg)
    T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-8)
    ipr_per_source = 1.0 / (T_norm ** 2).sum(axis=1)  # effective # targets blended
    return ipr_per_source


def main(rna_path, prot_path, gene_panel_size):
    print(f"Loading {rna_path}...")
    rna = ad.read_h5ad(rna_path)
    print(f"Loading {prot_path}...")
    prot = ad.read_h5ad(prot_path)

    print(f"Selecting top {gene_panel_size} DEG genes...")
    deg_genes = select_deg_genes(rna, "time_min", gene_panel_size)
    deg_genes = [g for g in deg_genes if g in rna.var_names]
    rna_X_full = rna[:, deg_genes].X
    rna_X = rna_X_full.toarray() if hasattr(rna_X_full, "toarray") else np.asarray(rna_X_full)

    times = sorted(rna.obs["time_min"].unique().tolist())
    print(f"Timepoints: {times}")

    time_masks = {t: np.where(np.isclose(rna.obs["time_min"].values.astype(float), t))[0] for t in times}

    print("\n" + "=" * 70)
    print("OT blending width (effective # target cells per source cell)")
    print("=" * 70)

    for i, t_src in enumerate(times):
        for t_tgt in times[i + 1:]:
            src_idx = time_masks[t_src]
            tgt_idx = time_masks[t_tgt]
            print(f"\n{t_src} -> {t_tgt}: {len(src_idx)} src, {len(tgt_idx)} tgt")
            for reg in [0.1, 0.05, 0.01, 0.005, 0.001]:
                try:
                    ipr = blending_width(rna_X[src_idx], rna_X[tgt_idx], reg)
                    pct_of_pop = 100 * ipr.mean() / len(tgt_idx)
                    print(f"  ot_reg={reg:<7}: mean effective targets blended = {ipr.mean():.1f} "
                          f"({pct_of_pop:.1f}% of the {len(tgt_idx)}-cell target population)")
                except Exception as e:
                    print(f"  ot_reg={reg}: FAILED ({e})")

    print("\n" + "=" * 70)
    print("INTERPRETATION: if a given ot_reg blends >20-30% of the target")
    print("population per source cell, that's consistent with near-mean-")
    print("collapse (same finding as Section 4.3.1's eps=0.1 QuRIE-seq case).")
    print("Look for the SMALLEST ot_reg that keeps blending width low AND")
    print("stays numerically stable (no failures) -- that's your fix.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", type=str, required=True)
    parser.add_argument("--prot_path", type=str, required=True)
    parser.add_argument("--gene_panel_size", type=int, default=1000)
    args = parser.parse_args()
    main(args.rna_path, args.prot_path, args.gene_panel_size)
