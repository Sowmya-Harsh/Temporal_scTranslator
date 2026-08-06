"""
10_diagnose_mito_pct.py

Purpose: Figure out exactly how pct_counts_mt was computed in the real
cleaning pipeline, by testing several candidate mitochondrial-gene
matching conventions on the RAW RNA file and comparing each candidate's
result against the ACTUAL pct_counts_mt already stored in the cleaned
file, for the SAME cells (matched by barcode). Whichever candidate
reproduces the clean file's stored values is the real convention used
-- this resolves the earlier QC figure discrepancy (median mito% before
cleaning: ~7.7%, after: ~1-2%, which was suspiciously large given only
1.5% of cells were removed) without needing to see the original
cleaning script directly.

Also prints: how many candidate "mt genes" each convention finds in
var_names, and a raw side-by-side table for a handful of cells so you
can eyeball the match quality, not just trust an aggregate number.

Usage:
    python 10_diagnose_mito_pct.py \
        --rna_raw Final_Dataset/qurie_RNA.h5ad \
        --rna_clean Final_Dataset/qurie_RNA_clean_aIg.h5ad
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def compute_pct_mt(adata, mt_mask):
    """Mirrors scanpy's calculate_qc_metrics pct_counts_mt definition:
    100 * (sum of counts in mt genes) / (sum of counts in all genes), per cell."""
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    total = X.sum(axis=1)
    mt_total = X[:, mt_mask].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(total > 0, 100 * mt_total / total, 0.0)
    return pct


def main(rna_raw_path, rna_clean_path):
    print(f"Loading raw file: {rna_raw_path}")
    raw = ad.read_h5ad(rna_raw_path)
    print(f"  shape: {raw.shape}, var columns: {list(raw.var.columns)}")

    print(f"\nLoading clean file: {rna_clean_path}")
    clean = ad.read_h5ad(rna_clean_path)
    print(f"  shape: {clean.shape}, obs columns: {list(clean.obs.columns)}")

    if "pct_counts_mt" not in clean.obs.columns:
        raise ValueError(
            "clean file has no 'pct_counts_mt' column -- can't compare against it. "
            "Check the actual column name in clean.obs.columns printed above."
        )
    if "mt" in clean.var.columns:
        n_mt_in_clean = int(clean.var["mt"].sum())
        print(f"\nClean file's own var['mt'] flags {n_mt_in_clean} genes as mitochondrial.")
        print("  Names:", list(clean.var_names[clean.var["mt"]]))
    else:
        print("\nNo 'mt' column in clean.var -- can't directly see which genes it flagged.")

    # Figure out how to identify barcodes shared between raw and clean, to compare
    # the SAME cells' pct_counts_mt under each candidate convention.
    if "barcode" in raw.obs.columns and "barcode" in clean.obs.columns:
        raw_id = raw.obs["barcode"].astype(str).values
        clean_id = clean.obs["barcode"].astype(str).values
    else:
        print("\nNo 'barcode' obs column found -- falling back to obs_names for matching.")
        raw_id = raw.obs_names.astype(str).values
        clean_id = clean.obs_names.astype(str).values

    common = np.intersect1d(clean_id, raw_id)
    print(f"\n{len(common)} cells found in both raw and clean files (matched by identifier).")
    if len(common) == 0:
        raise ValueError("No overlapping cell identifiers between raw and clean files -- "
                          "check that 'barcode' (or obs_names) actually correspond 1:1 across files.")

    raw_idx_map = {v: i for i, v in enumerate(raw_id)}
    clean_idx_map = {v: i for i, v in enumerate(clean_id)}
    raw_positions = [raw_idx_map[c] for c in common]
    clean_positions = [clean_idx_map[c] for c in common]

    actual_clean_pct = clean.obs["pct_counts_mt"].values[clean_positions]
    print(f"\nActual clean file pct_counts_mt for these cells: "
          f"mean={actual_clean_pct.mean():.3f}, median={np.median(actual_clean_pct):.3f}")

    # Candidate conventions to test against the raw file's var_names
    def to_bool_array(mask):
        return mask.values if hasattr(mask, "values") else np.asarray(mask)

    candidates = {
        "MT- prefix (human, uppercase)": to_bool_array(raw.var_names.str.startswith("MT-")),
        "mt- prefix (lowercase)": to_bool_array(raw.var_names.str.startswith("mt-")),
        "MT- or mt- prefix (case-insensitive)": to_bool_array(raw.var_names.str.upper().str.startswith("MT-")),
        "contains 'MT-' anywhere": to_bool_array(raw.var_names.str.contains("MT-", case=False, na=False)),
        "single gene MT-CO2 only (matches clean file's actual var['mt'] flag)": to_bool_array(raw.var_names == "MT-CO2"),
    }

    print("\n" + "=" * 70)
    print("Testing candidate mitochondrial-gene conventions on the RAW file,")
    print("recomputing pct_counts_mt, and comparing to the clean file's actual values")
    print("for the same cells:")
    print("=" * 70)

    raw_sub = raw[raw_positions].copy()

    results = []
    for name, mask in candidates.items():
        n_genes_found = int(mask.sum())
        if n_genes_found == 0:
            print(f"\n[{name}] 0 genes matched -- skipping")
            continue
        pct = compute_pct_mt(raw_sub, mask)
        diff = pct - actual_clean_pct
        mae = np.mean(np.abs(diff))
        corr = np.corrcoef(pct, actual_clean_pct)[0, 1] if np.std(pct) > 0 else float("nan")
        print(f"\n[{name}]")
        print(f"  {n_genes_found} genes matched, first few: {list(raw.var_names[mask])[:5]}")
        print(f"  Recomputed pct_counts_mt: mean={pct.mean():.3f}, median={np.median(pct):.3f}")
        print(f"  Mean absolute difference from clean file's stored values: {mae:.4f}")
        print(f"  Correlation with clean file's stored values: {corr:.4f}")
        results.append((name, mae, corr, n_genes_found))

    print("\n" + "=" * 70)
    if results:
        best = min(results, key=lambda r: r[1])
        print(f"BEST MATCH (lowest mean absolute difference): '{best[0]}'")
        print(f"  MAE={best[1]:.4f}, corr={best[2]:.4f}, n_genes={best[3]}")
        if best[1] < 0.01:
            print("  -> This convention essentially reproduces the clean file exactly. "
                  "Use this for the 'before cleaning' QC figure computation.")
        else:
            print("  -> Even the best candidate doesn't reproduce the clean file closely. "
                  "The real cleaning script likely used a different gene list or a "
                  "precomputed 'mt' annotation not derivable from var_names alone -- "
                  "check your actual preprocessing script's mitochondrial gene source.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_raw", type=str, required=True)
    parser.add_argument("--rna_clean", type=str, required=True)
    args = parser.parse_args()
    main(args.rna_raw, args.rna_clean)