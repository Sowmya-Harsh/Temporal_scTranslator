"""
04_check_dropped_channels.py

Purpose: Verify whether the 22 protein channels flagged as "zero across
all cells" are genuinely zero (real assay/staining dropout) or an
artifact of file loading / indexing. Checks raw file directly, per
condition, and reports basic stats + a spot-check of raw values.

Usage:
    python 04_check_dropped_channels.py /path/to/qurie_Protein.h5ad
"""

import sys
import numpy as np
import scanpy as sc

DROPPED_CHANNELS = [
    "Histone H3", "p-SYK", "p-BTK", "Cyclin E", "c-MYC", "p-RB",
    "p-c-JUN", "active Caspase-3", "p-AKT", "ERK 1/2", "AKT", "SYK",
    "BTK", "p-ERK 1/2", "p-Histon H3", "Cyclin A", "p-Histon H2A.X",
    "Cyclin B1", "p-SRC", "p-IKK a/b", "p-PLC-y2 (Y759)", "BCL6",
]

def main(path):
    print(f"Loading {path} ...")
    adata = sc.read_h5ad(path)
    print(f"Shape: {adata.shape}")
    print(f"var_names dtype/index type: {type(adata.var_names)}")

    missing = [g for g in DROPPED_CHANNELS if g not in adata.var_names]
    if missing:
        print(f"\nWARNING: these channels are not even present in var_names: {missing}")

    print("\n" + "=" * 80)
    print(f"{'Channel':<20}{'sum':>12}{'max':>10}{'nonzero_cells':>16}{'nonzero_%':>12}")
    print("=" * 80)

    X = adata.X
    is_sparse = hasattr(X, "toarray")

    for gene in DROPPED_CHANNELS:
        if gene not in adata.var_names:
            continue
        idx = adata.var_names.get_loc(gene)
        col = X[:, idx]
        col = col.toarray().flatten() if is_sparse else np.asarray(col).flatten()
        nz = np.count_nonzero(col)
        print(f"{gene:<20}{col.sum():>12.2f}{col.max():>10.2f}{nz:>16d}{100*nz/len(col):>11.2f}%")

    print("\n" + "=" * 80)
    print("PER-CONDITION BREAKDOWN (in case dropout is condition-specific)")
    print("=" * 80)
    if "condition" in adata.obs.columns:
        for cond, sub in adata.obs.groupby("condition", observed=True):
            print(f"\n--- condition = {cond} ({len(sub)} cells) ---")
            sub_adata = adata[sub.index]
            for gene in DROPPED_CHANNELS[:5]:  # sample first 5 to keep output short
                if gene not in adata.var_names:
                    continue
                idx = sub_adata.var_names.get_loc(gene)
                col = sub_adata.X[:, idx]
                col = col.toarray().flatten() if is_sparse else np.asarray(col).flatten()
                print(f"  {gene:<20} sum={col.sum():.2f} nonzero_cells={np.count_nonzero(col)}")
    else:
        print("No 'condition' column found in obs.")

    print("\n" + "=" * 80)
    print("RAW FILE SPOT-CHECK (first 10 raw values for first dropped channel)")
    print("=" * 80)
    first_gene = next((g for g in DROPPED_CHANNELS if g in adata.var_names), None)
    if first_gene:
        idx = adata.var_names.get_loc(first_gene)
        col = X[:, idx]
        col = col.toarray().flatten() if is_sparse else np.asarray(col).flatten()
        print(f"{first_gene} — first 10 raw values: {col[:10]}")
        print(f"{first_gene} — dtype: {col.dtype}")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
- If sum/max/nonzero_cells are all exactly 0 for a channel across the WHOLE
  dataset (not just after your earlier filtering) -> genuine zero signal.
  Likely explanation: antibody/staining failure for that channel in this
  experimental run, or the channel was included in panel design but not
  actually acquired in this particular dataset. Legitimate to exclude and
  document in methods.

- If sum/max/nonzero_cells are NON-zero here but the earlier cleaning
  script reported them as all-zero -> something is inconsistent between
  runs (different file loaded, var_names mismatch/reordering, or a bug
  in the earlier zero-filtering step). Re-run 03_clean_qurie_data.py's
  zero-check logic manually to compare against this script's numbers.

- If nonzero_cells > 0 but very low (e.g. <1% of cells) -> not a hard
  zero, but very sparse/noisy channel. Worth a judgment call: keep as
  informative-but-sparse, or exclude for stability. Report the % here
  and we'll decide together.
""")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 04_check_dropped_channels.py /path/to/qurie_Protein.h5ad")
        sys.exit(1)
    main(sys.argv[1])