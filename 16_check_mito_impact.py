"""
16_check_mito_impact.py

Purpose: Quick, read-only check -- does fixing the mitochondrial-gene
matching pattern (13 genes, correct 'MT-' prefix) actually change which
cells get excluded at the 20% cutoff, compared to what was actually used
(effectively 1 gene, MT-CO2)? This does NOT retrain or rerun anything
else -- purely answers "is this worth fixing the pipeline for, or is the
practical difference negligible."

Usage:
    python 16_check_mito_impact.py --rna_path Final_Dataset/qurie_RNA.h5ad
"""
import argparse

import anndata as ad
import numpy as np

CANONICAL_MT_GENES = [
    "MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6",
    "MT-CO3", "MT-ND3", "MT-ND4L", "MT-ND4", "MT-ND5", "MT-ND6", "MT-CYB",
]


def main(rna_path, cutoff, min_genes):
    print(f"Loading {rna_path}...")
    a = ad.read_h5ad(rna_path)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    var_names = a.var_names.astype(str)

    total_counts = X.sum(axis=1)

    # Original (buggy): only MT-CO2
    mt_co2_idx = np.where(var_names == "MT-CO2")[0]
    pct_mt_original = 100 * X[:, mt_co2_idx].sum(axis=1) / (total_counts + 1e-8)

    # Corrected: all 13 canonical genes
    mt_all_idx = [i for i, g in enumerate(var_names) if g in CANONICAL_MT_GENES]
    print(f"Found {len(mt_all_idx)}/13 canonical MT genes for corrected calculation.")
    pct_mt_corrected = 100 * X[:, mt_all_idx].sum(axis=1) / (total_counts + 1e-8)

    n_genes_per_cell = (X > 0).sum(axis=1)

    # Apply the SAME two-part cleaning rule as originally (Section 3.6):
    # remove if < min_genes detected OR mito% > cutoff
    keep_original = (n_genes_per_cell >= min_genes) & (pct_mt_original <= cutoff)
    keep_corrected = (n_genes_per_cell >= min_genes) & (pct_mt_corrected <= cutoff)

    n_total = len(a)
    n_removed_original = n_total - keep_original.sum()
    n_removed_corrected = n_total - keep_corrected.sum()

    # Cells that differ between the two filtering decisions
    n_differ = (keep_original != keep_corrected).sum()
    newly_removed = keep_original & ~keep_corrected  # kept before, now removed
    newly_kept = ~keep_original & keep_corrected      # removed before, now kept

    print(f"\n{'='*60}")
    print(f"Total cells: {n_total}")
    print(f"\nOriginal (MT-CO2 only) mito%: mean={pct_mt_original.mean():.3f}, "
          f"median={np.median(pct_mt_original):.3f}, max={pct_mt_original.max():.3f}")
    print(f"Corrected (13 genes) mito%:   mean={pct_mt_corrected.mean():.3f}, "
          f"median={np.median(pct_mt_corrected):.3f}, max={pct_mt_corrected.max():.3f}")
    print(f"\nCells removed under ORIGINAL rule:  {n_removed_original} ({100*n_removed_original/n_total:.2f}%)")
    print(f"Cells removed under CORRECTED rule: {n_removed_corrected} ({100*n_removed_corrected/n_total:.2f}%)")
    print(f"\nCells whose keep/remove DECISION actually changes: {n_differ} ({100*n_differ/n_total:.3f}%)")
    print(f"  Newly removed (kept before, removed now): {newly_removed.sum()}")
    print(f"  Newly kept (removed before, kept now):    {newly_kept.sum()}")
    print(f"{'='*60}")

    if n_differ / n_total < 0.01:
        print("\nVERDICT: Practical impact is negligible (<1% of cells change "
              "status). Safe to just fix the QC FIGURE/TEXT with the corrected "
              "numbers and note the correction -- no need to rerun training.")
    else:
        print(f"\nVERDICT: {100*n_differ/n_total:.2f}% of cells change status -- "
              "this is material enough that you should discuss whether to "
              "redo cleaning + downstream training with the corrected filter.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", type=str, required=True)
    parser.add_argument("--cutoff", type=float, default=20.0)
    parser.add_argument("--min_genes", type=int, default=200)
    args = parser.parse_args()
    main(args.rna_path, args.cutoff, args.min_genes)
