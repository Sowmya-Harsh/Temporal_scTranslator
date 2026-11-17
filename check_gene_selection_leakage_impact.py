"""
check_gene_selection_leakage_impact.py

Checks whether DEG/HVG gene selection actually changes, in practice,
when computed on train-split cells only, versus the full population
(all cells, including validation and test).

This does NOT fix the leakage. It answers a narrower, practical
question first: does the leakage actually change which genes get
selected, meaningfully, or is the gene list nearly identical either
way? If nearly identical, that's real, honest evidence the leakage
had negligible practical impact, worth citing directly in a
Limitations note, without needing to rerun the entire pipeline.

Usage:
    python check_gene_selection_leakage_impact.py \
        --rna_path Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --n_genes 1000 \
        --strategy deg \
        --seed 1105
"""
import argparse

import anndata as ad
import numpy as np
import scanpy as sc

from gene_selection import get_top_deg, get_top_hvg


def split_cells_per_timepoint(rna_adata, seed, train_frac=0.8, val_frac=0.1):
    """Same split logic as train_joint.py: 80/10/10, independently per
    timepoint. Returns a boolean mask over all cells, True for train."""
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(rna_adata.n_obs, dtype=bool)
    times = sorted(rna_adata.obs["time_min"].unique().tolist())
    for t in times:
        idx = np.where(np.isclose(rna_adata.obs["time_min"].values.astype(float), t))[0]
        rng.shuffle(idx)
        n_train = int(train_frac * len(idx))
        train_mask[idx[:n_train]] = True
    return train_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna_path", required=True)
    ap.add_argument("--n_genes", type=int, default=1000)
    ap.add_argument("--strategy", choices=["deg", "hvg"], default="deg")
    ap.add_argument("--seed", type=int, default=1105)
    args = ap.parse_args()

    rna = ad.read_h5ad(args.rna_path)
    print(f"Loaded {rna.n_obs} cells, {rna.n_vars} genes")

    # --- Selection 1: current (leaky) approach, full population ---
    if args.strategy == "deg":
        idx_full = get_top_deg(rna, args.n_genes, groupby="time_min")
    else:
        idx_full = get_top_hvg(rna, args.n_genes)
    genes_full = set(rna.var_names[idx_full])
    print(f"\n[Full population] Selected {len(genes_full)} genes")

    # --- Selection 2: honest approach, train-split cells only ---
    train_mask = split_cells_per_timepoint(rna, seed=args.seed)
    rna_train = rna[train_mask].copy()
    print(f"[Train split] {rna_train.n_obs} of {rna.n_obs} cells "
          f"({100*train_mask.mean():.1f}%)")

    if args.strategy == "deg":
        idx_train = get_top_deg(rna_train, args.n_genes, groupby="time_min")
    else:
        idx_train = get_top_hvg(rna_train, args.n_genes)
    genes_train = set(rna_train.var_names[idx_train])
    print(f"[Train-only] Selected {len(genes_train)} genes")

    # --- Compare ---
    overlap = genes_full & genes_train
    only_full = genes_full - genes_train
    only_train = genes_train - genes_full

    print(f"\n=== Comparison ===")
    print(f"Overlap: {len(overlap)} / {args.n_genes} "
          f"({100*len(overlap)/args.n_genes:.1f}%)")
    print(f"Genes only in full-population selection: {len(only_full)}")
    print(f"Genes only in train-only selection: {len(only_train)}")

    if len(only_full) > 0:
        print(f"\nExample genes dropped when restricting to train-only "
              f"(up to 10 shown): {sorted(only_full)[:10]}")
    if len(only_train) > 0:
        print(f"\nExample genes newly added when restricting to train-only "
              f"(up to 10 shown): {sorted(only_train)[:10]}")

    print(f"\n=== Verdict ===")
    pct = 100 * len(overlap) / args.n_genes
    if pct >= 95:
        print(f"Overlap is {pct:.1f}%, very high. The leakage very likely "
              f"had negligible practical impact on the selected gene panel. "
              f"Worth citing this check directly, but a full pipeline rerun "
              f"may not be strictly necessary for this specific fix.")
    elif pct >= 85:
        print(f"Overlap is {pct:.1f}%, high but not negligible. Worth a "
              f"closer look at which genes differ, and whether any of "
              f"them are biologically important (e.g. re-run the BCR "
              f"pathway gene check on the differing genes specifically).")
    else:
        print(f"Overlap is only {pct:.1f}%, meaningfully different. This "
              f"suggests the leakage genuinely could have influenced "
              f"results, and a full rerun with train-only selection is "
              f"worth prioritizing.")


if __name__ == "__main__":
    main()
