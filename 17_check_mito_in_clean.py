"""
17_check_mito_in_clean.py

Purpose: The raw file has all 13 canonical mitochondrial genes (confirmed
via 15_check_mito_genes.py). This checks whether all 13 survive into the
CLEANED file's gene panel (after the 25,463->18,603 gene reduction from
Section 3.5 -- zero-count filtering + identifier mapping), which is a
DIFFERENT question from the cell-level QC bug already found. If some
genes were dropped here, that's a separate, earlier-stage reason the
mito% calculation could be incomplete even after fixing the matching
pattern.

Usage:
    python 17_check_mito_in_clean.py \
        --rna_raw Final_Dataset/qurie_RNA.h5ad \
        --rna_clean Final_Dataset/qurie_RNA_clean_aIg.h5ad
"""
import argparse

import anndata as ad

CANONICAL_MT_GENES = [
    "MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6",
    "MT-CO3", "MT-ND3", "MT-ND4L", "MT-ND4", "MT-ND5", "MT-ND6", "MT-CYB",
]


def main(rna_raw, rna_clean):
    print(f"Loading raw: {rna_raw}")
    raw = ad.read_h5ad(rna_raw, backed="r")
    raw_names = set(raw.var_names.astype(str))

    print(f"Loading clean: {rna_clean}")
    clean = ad.read_h5ad(rna_clean, backed="r")
    clean_names = set(clean.var_names.astype(str))

    print(f"\nRaw panel size:   {len(raw_names)}")
    print(f"Clean panel size: {len(clean_names)}")

    print(f"\n{'Gene':<12} {'In raw?':<10} {'In clean?':<10}")
    print("-" * 32)
    n_in_raw = n_in_clean = 0
    for g in CANONICAL_MT_GENES:
        in_raw = g in raw_names
        in_clean = g in clean_names
        n_in_raw += in_raw
        n_in_clean += in_clean
        print(f"{g:<12} {'yes' if in_raw else 'NO':<10} {'yes' if in_clean else 'NO':<10}")

    print(f"\n{n_in_raw}/13 present in raw file")
    print(f"{n_in_clean}/13 present in cleaned file")

    if n_in_clean < n_in_raw:
        dropped = [g for g in CANONICAL_MT_GENES if g in raw_names and g not in clean_names]
        print(f"\n{len(dropped)} mito gene(s) were dropped during the raw->clean gene panel "
              f"reduction (Section 3.5's zero-count filtering / identifier mapping): {dropped}")
        print("This means even a corrected mito% calculation run on the CLEANED file's own")
        print("panel would still be incomplete -- the fix must recompute mito% from the RAW")
        print("file (which has all 13), not the already-gene-filtered clean file.")
    else:
        print("\nAll genes found in raw are also present in the cleaned file's panel.")
        print("Confirms: mito% should be recomputed from the RAW file, and doing so will")
        print("have access to all 13 genes with no gene-panel-reduction complication.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_raw", type=str, required=True)
    parser.add_argument("--rna_clean", type=str, required=True)
    args = parser.parse_args()
    main(args.rna_raw, args.rna_clean)
