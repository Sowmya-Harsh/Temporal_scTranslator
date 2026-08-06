"""
06_verify_myid_integrity.py

Purpose: Confirm 'my_Id' survived the clean+map pipeline correctly:
  - column exists in both RNA and Protein var
  - no NaN/missing my_Id values
  - values are within valid embedding table range (0, 85500)
  - dtype is convertible to int (scTranslator casts to .astype(int))
  - no unexpected duplicate my_Id within a single modality's feature set
    (duplicates would mean two different genes/proteins sharing one
    positional embedding, silently conflating them)

Usage:
    python 06_verify_myid_integrity.py \
        --rna /path/to/qurie_RNA_clean_aIg.h5ad \
        --protein /path/to/qurie_Protein_clean_aIg.h5ad
"""

import argparse
import numpy as np
import scanpy as sc

def check(adata, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    if "my_Id" not in adata.var.columns:
        print("FAIL: 'my_Id' column missing entirely.")
        return
    myid = adata.var["my_Id"]
    n_missing = myid.isna().sum()
    print(f"Total features: {len(myid)}")
    print(f"Missing my_Id: {n_missing}")

    try:
        as_int = myid.astype(int)
        print("dtype conversion to int: OK")
    except Exception as e:
        print(f"FAIL: could not convert my_Id to int: {e}")
        return

    print(f"my_Id range: min={as_int.min()}, max={as_int.max()}")
    if as_int.max() >= 85500 or as_int.min() < 0:
        print("FAIL: my_Id out of valid embedding table range [0, 85499]!")
    else:
        print("Range check: OK (within [0, 85499])")

    n_dupes = as_int.duplicated().sum()
    if n_dupes > 0:
        print(f"WARNING: {n_dupes} duplicate my_Id values found within this modality:")
        dup_ids = as_int[as_int.duplicated(keep=False)]
        dup_table = adata.var.loc[dup_ids.index, ["my_Id"]] if "my_Id" in adata.var else None
        for did in sorted(set(dup_ids)):
            features = adata.var_names[as_int == did].tolist()
            print(f"  my_Id={did} shared by: {features}")
    else:
        print("Duplicate check: OK (all my_Id unique within this file)")

def main(rna_path, protein_path):
    rna = sc.read_h5ad(rna_path)
    prot = sc.read_h5ad(protein_path)
    check(rna, f"RNA: {rna_path}")
    check(prot, f"PROTEIN: {protein_path}")

    # Cross-modality check: RNA and protein should NOT share my_Id values
    # except in the intentional case of a protein marker mapped to its own
    # coding gene's RNA entry (which is fine and expected -- e.g. AKT
    # protein and AKT1 RNA legitimately differ in ID since AKT wasn't
    # mapped this round, but flag any surprises).
    if "my_Id" in rna.var.columns and "my_Id" in prot.var.columns:
        rna_ids = set(rna.var["my_Id"].astype(int))
        prot_ids = set(prot.var["my_Id"].astype(int))
        overlap = rna_ids & prot_ids
        print(f"\n{'='*60}\nCROSS-MODALITY OVERLAP CHECK\n{'='*60}")
        print(f"RNA my_Id count: {len(rna_ids)}, Protein my_Id count: {len(prot_ids)}")
        print(f"Overlapping IDs (same gene appears as both RNA feature and protein feature): {len(overlap)}")
        if overlap:
            for oid in sorted(overlap):
                rna_feat = rna.var_names[rna.var["my_Id"].astype(int) == oid].tolist()
                prot_feat = prot.var_names[prot.var["my_Id"].astype(int) == oid].tolist()
                print(f"  my_Id={oid}: RNA={rna_feat}  Protein={prot_feat}")
            print("\nNote: overlap here is EXPECTED and fine — it means, e.g., "
                  "the AKT1 gene shows up as an RNA feature and its protein "
                  "product also maps to the same real gene ID. This is a "
                  "correct, intentional shared identity, not a bug. Only "
                  "worry if unrelated genes/proteins share an ID.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    args = parser.parse_args()
    main(args.rna, args.protein)