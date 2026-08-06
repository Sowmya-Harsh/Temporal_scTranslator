"""
02_list_protein_vars.py

Purpose: Extract the full 80-feature protein panel var_names, and give a
first-pass automatic split into phospho markers (prefix 'p-') vs total
protein / other markers. Paste the full output back.

Usage:
    python 02_list_protein_vars.py /path/to/qurie_Protein.h5ad
"""

import sys
import scanpy as sc

def main(path):
    adata = sc.read_h5ad(path)

    print("=" * 70)
    print(f"Total features: {adata.n_vars}")
    print("=" * 70)

    all_vars = adata.var_names.tolist()
    for i, v in enumerate(all_vars):
        print(f"{i+1:3d}. {v}")

    print("\n" + "=" * 70)
    print("AUTO-SPLIT: phospho (prefix 'p-') vs other")
    print("=" * 70)

    phospho = [v for v in all_vars if v.lower().startswith("p-")]
    other = [v for v in all_vars if not v.lower().startswith("p-")]

    print(f"\nPhospho markers ({len(phospho)}):")
    for v in phospho:
        print(f"  {v}")

    print(f"\nOther / total-protein markers ({len(other)}):")
    for v in other:
        print(f"  {v}")

    print("\n" + "=" * 70)
    print("NOTE: Review the 'other' list manually — some panels use")
    print("different prefixes (e.g. 'pY-', 'Phospho-', or no prefix at")
    print("all for phospho markers). Paste both lists back so we can")
    print("confirm the split is correct before building the cleaning")
    print("and modeling pipeline.")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 02_list_protein_vars.py /path/to/qurie_Protein.h5ad")
        sys.exit(1)
    main(sys.argv[1])