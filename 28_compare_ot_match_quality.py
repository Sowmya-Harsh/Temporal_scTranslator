"""
28_compare_ot_match_quality.py

Purpose: Test the hypothesis that myeloid's much larger cell populations
produce genuinely cleaner OT matches (a legitimate reason for easier
learning) versus a more concerning possibility: that source and target
cells are so numerous/similar that OT is matching near-duplicate cells,
making the "forecasting" task trivially easy in a way that doesn't
reflect real biological difficulty. Neither of these is "leakage" in
the technical sense, but the second would still mean the R=0.82 result
doesn't demonstrate what it appears to demonstrate.

Usage:
    python 28_compare_ot_match_quality.py \
        --quries_rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --myeloid_rna myeloid_geo_data/myeloid_RNA_standardized.h5ad \
        --gene_panel_size 1000
"""
import argparse

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.metrics.pairwise import cosine_distances


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
    return sorted(best_rank.keys(), key=lambda g: best_rank[g])[:panel_size]


def best_match_similarity(rna_src, rna_tgt, max_genes=2000):
    gene_var = rna_src.var(axis=0) + rna_tgt.var(axis=0)
    top_genes = np.argsort(gene_var)[-max_genes:]
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes])
    best_sim_per_source = 1 - C.min(axis=1)
    return best_sim_per_source


def report(name, rna, time_col, t_src, t_tgt, gene_panel_size):
    print(f"\n{'='*60}\n{name}: {t_src} -> {t_tgt}\n{'='*60}")
    deg_genes = select_deg_genes(rna, time_col, gene_panel_size)
    deg_genes = [g for g in deg_genes if g in rna.var_names]
    X = rna[:, deg_genes].X
    X = (X.toarray() if hasattr(X, "toarray") else X).astype(np.float32)

    times = rna.obs[time_col].values.astype(float)
    src_idx = np.where(np.isclose(times, t_src))[0]
    tgt_idx = np.where(np.isclose(times, t_tgt))[0]
    print(f"  {len(src_idx)} source cells, {len(tgt_idx)} target cells")

    sims = best_match_similarity(X[src_idx], X[tgt_idx])
    print(f"  Best-match cosine similarity: mean={sims.mean():.4f}, median={np.median(sims):.4f}, "
          f"95th pct={np.percentile(sims, 95):.4f}")
    n_near_dup = (sims > 0.99).sum()
    print(f"  Source cells with a near-duplicate target (similarity > 0.99): "
          f"{n_near_dup}/{len(src_idx)} ({100*n_near_dup/len(src_idx):.1f}%)")
    return sims


def main(quries_rna_path, myeloid_rna_path, gene_panel_size):
    print("Loading QuRIE-seq...")
    quries = ad.read_h5ad(quries_rna_path)
    sims_q = report("QuRIE-seq", quries, "time_min", 0.0, 60.0, gene_panel_size)

    print("\nLoading myeloid...")
    myeloid = ad.read_h5ad(myeloid_rna_path)
    sims_m = report("Myeloid", myeloid, "time_min", 0.0, 10080.0, gene_panel_size)

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"QuRIE-seq mean best-match similarity: {sims_q.mean():.4f}")
    print(f"Myeloid   mean best-match similarity: {sims_m.mean():.4f}")
    if sims_m.mean() > sims_q.mean() + 0.1:
        print("\nMyeloid's matches are substantially closer/more similar than QuRIE-seq's.")
        print("This is CONSISTENT with (not proof of) the 'larger population -> cleaner")
        print("matches -> easier legitimate task' explanation, but also check the near-")
        print("duplicate percentage above -- if that is high, matches may be TOO close,")
        print("effectively making this closer to same-cell reconstruction than genuine")
        print("forecasting, needing a more careful interpretation of the R=0.82 result.")
    else:
        print("\nNo dramatic difference in match quality -- the R gap is NOT well explained")
        print("by OT match quality alone; look elsewhere (e.g. protein panel noise")
        print("characteristics, or re-check for leakage more carefully).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quries_rna", type=str, required=True)
    parser.add_argument("--myeloid_rna", type=str, required=True)
    parser.add_argument("--gene_panel_size", type=int, default=1000)
    args = parser.parse_args()
    main(args.quries_rna, args.myeloid_rna, args.gene_panel_size)
