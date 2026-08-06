"""
15_check_mito_genes.py

Purpose: Directly inspect the raw RNA file's var_names for anything that
looks mitochondrial, under several possible naming conventions, to find
out definitively whether the canonical mitochondrial genes (MT-CO1,
MT-ND1-6, MT-CYB, MT-ATP6/8, etc.) are:
  (a) present but named differently than expected (fixable -- just widen
      the matching pattern), or
  (b) genuinely absent from this gene panel (not fixable by renaming --
      would need to note this as a real panel-coverage limitation).

Usage:
    python 15_check_mito_genes.py --rna_path Final_Dataset/qurie_RNA.h5ad
"""
import argparse

import anndata as ad
import pandas as pd

# Canonical human mitochondrial protein-coding genes (13 total)
CANONICAL_MT_GENES = [
    "MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6",
    "MT-CO3", "MT-ND3", "MT-ND4L", "MT-ND4", "MT-ND5", "MT-ND6", "MT-CYB",
]


def main(rna_path):
    print(f"Loading {rna_path} (backed mode, var_names only)...")
    a = ad.read_h5ad(rna_path, backed="r")
    var_names = a.var_names.astype(str)
    print(f"Total genes in panel: {len(var_names)}")

    print("\n--- Exact match against canonical 13 MT genes ---")
    for g in CANONICAL_MT_GENES:
        found = g in var_names
        print(f"  {g}: {'FOUND' if found else 'missing'}")

    print("\n--- Broad search: anything containing 'MT' (case-insensitive) near start ---")
    candidates = var_names[var_names.str.upper().str.contains("^MT[-_]?", regex=True, na=False)]
    print(f"  {len(candidates)} candidates found:")
    for c in sorted(candidates.tolist()):
        print(f"    {c}")

    print("\n--- Broad search: contains 'MT' anywhere (may include false positives like MTOR, MTHFR) ---")
    broad = var_names[var_names.str.upper().str.contains("MT", na=False)]
    print(f"  {len(broad)} genes contain 'MT' anywhere (showing first 40):")
    for c in sorted(broad.tolist())[:40]:
        print(f"    {c}")

    print("\n--- Check var columns for any pre-existing 'mt' annotation ---")
    print(f"  var columns available: {list(a.var.columns)}")
    if "mt" in a.var.columns:
        n_flagged = int(a.var["mt"].sum())
        print(f"  'mt' column already exists, flags {n_flagged} genes: "
              f"{list(var_names[a.var['mt']])}")

    print("\n" + "=" * 60)
    found_canonical = [g for g in CANONICAL_MT_GENES if g in var_names]
    print(f"SUMMARY: {len(found_canonical)}/13 canonical MT genes found by exact name match.")
    if len(found_canonical) < 13:
        print("Missing genes may be present under different naming (check the")
        print("broad search results above), or may genuinely be absent/filtered")
        print("out earlier in preprocessing (e.g. during the 25,463->18,603 gene")
        print("panel reduction in Section 3.5 -- check if they were zero-count)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", type=str, required=True)
    args = parser.parse_args()
    main(args.rna_path)
