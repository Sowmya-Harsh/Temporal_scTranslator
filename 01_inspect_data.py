"""
01_inspect_h5ad.py

Purpose: Full diagnostic report of an .h5ad file BEFORE any cleaning.
Never clean blind — this tells you what you're actually dealing with:
structure, sparsity, missingness, QC distributions, duplicate/empty
entries, and whether obs/var metadata is usable for filtering.

Usage:
    python 01_inspect_h5ad.py /path/to/file.h5ad

Requires: scanpy, anndata, numpy, pandas
    pip install scanpy anndata numpy pandas --break-system-packages
"""

import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def inspect(path):
    section(f"LOADING: {path}")
    adata = sc.read_h5ad(path)
    print(adata)

    section("BASIC SHAPE")
    print(f"n_obs (cells/samples): {adata.n_obs}")
    print(f"n_vars (genes/features): {adata.n_vars}")

    section("OBS (cell/sample metadata) — columns and dtypes")
    print(adata.obs.dtypes)
    print("\nFirst 5 rows:")
    print(adata.obs.head())

    section("OBS — missingness per column")
    print(adata.obs.isna().sum())

    section("OBS — check for time / timepoint columns")
    time_like = [c for c in adata.obs.columns
                 if any(k in c.lower() for k in ["time", "hour", "min", "day", "stage", "timepoint"])]
    print(f"Candidate time columns: {time_like}")
    for c in time_like:
        print(f"\n  {c} value counts:")
        print(adata.obs[c].value_counts(dropna=False))

    section("OBS — check for batch/condition/replicate columns")
    batch_like = [c for c in adata.obs.columns
                  if any(k in c.lower() for k in ["batch", "sample", "condition", "replicate", "donor", "patient"])]
    print(f"Candidate batch columns: {batch_like}")
    for c in batch_like:
        print(f"\n  {c} value counts:")
        print(adata.obs[c].value_counts(dropna=False))

    section("VAR (gene/feature metadata)")
    print(adata.var.dtypes)
    print(adata.var.head())

    section("VAR — duplicate feature names")
    dup_count = adata.var_names.duplicated().sum()
    print(f"Duplicated var_names: {dup_count}")

    section("OBS — duplicate obs names (cell barcodes)")
    dup_obs = adata.obs_names.duplicated().sum()
    print(f"Duplicated obs_names: {dup_obs}")

    section("X MATRIX — type, sparsity, value range")
    X = adata.X
    print(f"Type: {type(X)}")
    print(f"dtype: {X.dtype}")
    if hasattr(X, "toarray"):
        # sparse — sample a subset to avoid loading huge dense array
        nnz = X.nnz
        total = X.shape[0] * X.shape[1]
        print(f"Sparse matrix. Non-zero entries: {nnz} / {total} ({100*nnz/total:.2f}% dense)")
        sample = X[:min(1000, X.shape[0])].toarray()
    else:
        sample = np.asarray(X[:min(1000, X.shape[0])])
        zero_frac = (sample == 0).mean()
        print(f"Dense matrix. Zero fraction (first 1000 rows): {100*zero_frac:.2f}%")

    print(f"\nValue range (first 1000 rows sample): min={sample.min():.4f}, max={sample.max():.4f}, mean={sample.mean():.4f}")
    print(f"Any NaN in sample: {np.isnan(sample).any()}")
    print(f"Any negative values in sample: {(sample < 0).any()}")
    # Heuristic: is this raw counts or already normalized/log?
    is_integer_like = np.allclose(sample, np.round(sample))
    print(f"Values look like raw integer counts: {is_integer_like}")

    section("LAYERS (alternative matrices, e.g. raw counts vs normalized)")
    print(list(adata.layers.keys()) if adata.layers else "No layers found.")

    section("OBSM (embeddings, e.g. PCA/UMAP)")
    print(list(adata.obsm.keys()) if adata.obsm else "No obsm found.")

    section("UNS (unstructured metadata, e.g. processing history)")
    print(list(adata.uns.keys()) if adata.uns else "No uns found.")

    section("QC METRICS — per-cell basic stats (computed fresh)")
    try:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"] if adata.var["mt"].any() else [],
                                    percent_top=None, log1p=False, inplace=True)
        print(adata.obs[["n_genes_by_counts", "total_counts"]].describe())
        if adata.var["mt"].any():
            print("\nMitochondrial %:")
            print(adata.obs["pct_counts_mt"].describe())
        else:
            print("\nNo MT- genes detected (may not be human, or var_names aren't gene symbols).")
    except Exception as e:
        print(f"Could not compute QC metrics automatically: {e}")

    section("EMPTY CELLS / GENES CHECK")
    if hasattr(X, "toarray"):
        counts_per_cell = np.asarray(X.sum(axis=1)).flatten()
        counts_per_gene = np.asarray(X.sum(axis=0)).flatten()
    else:
        counts_per_cell = np.asarray(X).sum(axis=1)
        counts_per_gene = np.asarray(X).sum(axis=0)
    print(f"Cells with zero total counts: {(counts_per_cell == 0).sum()}")
    print(f"Genes with zero total counts across all cells: {(counts_per_gene == 0).sum()}")

    section("SUMMARY / NEXT STEPS")
    print("""
Report the following back so we can design the cleaning pipeline precisely:
  1. Does obs contain a usable time/timepoint column? What are its values?
  2. Are values raw counts (integers) or already normalized/log-transformed?
  3. Sparsity % and whether it matches expected scRNA-seq sparsity (~90-95% zeros typical)
  4. Any duplicate obs_names or var_names?
  5. % mitochondrial reads distribution (high tail = dying cells to filter)
  6. Are there layers already storing raw vs processed data?
  7. Any columns indicating paired RNA/protein/phospho sample IDs for matching across modalities?
""")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 01_inspect_h5ad.py /path/to/file.h5ad")
        sys.exit(1)
    inspect(sys.argv[1])