"""
05_map_protein_to_myID.py

Purpose: Assign scTranslator 'my_Id' values to your 58 cleaned protein
panel features.

Strategy:
  - Total-protein / non-phospho markers (36 features): map to their
    underlying gene's real scTranslator my_Id via the existing
    hugo_symbol -> Entrez -> my_Id dictionaries (reuses pretrained
    gene identity embedding).
  - Phospho markers (22 features): NOT mapped to the parent gene's ID
    (would collide/conflate phospho vs total signal in the decoder's
    positional embedding). Instead assigned NEW ids in the unused
    range [75523, 85499] of the embedding table (verified unused by
    inspecting code/model/ID_dic/EntrezID_to_myID.pkl: real my_Id
    values only span 1-75522, out of a table sized 85500). No model
    surgery needed — these are legitimate, already-existing embedding
    rows, just never assigned to a real gene.

A mapping table (var_name -> my_Id, mapping_type) is printed and saved
alongside the output so it can go directly into your thesis methods /
supplementary as a documented, reproducible design choice.

Usage:
    python 05_map_protein_to_myID.py \
        --protein_h5ad /path/to/qurie_Protein_clean_aIg.h5ad \
        --outdir /path/to/output_dir
    (run once per condition file: aIg and aIg+Ibru)
"""

import argparse
import os
import pickle
import scanpy as sc
import pandas as pd

# Names of protein markers with a known/likely direct gene-symbol match.
# Verify each mapping succeeds — printed warnings flag anything that fails.
GENE_MAP_CANDIDATES = {
    "Histone H3": "H3-3A",       # canonical histone H3.3 gene; verify against panel intent
    "CD70": "CD70",
    "IgD": "IGHD",
    "CD86": "CD86",
    "CD20": "MS4A1",
    "CD79a": "CD79A",
    "CD5": "CD5",
    "S6": "RPS6",
    "ERK 1/2": "MAPK1",          # ERK2; MAPK3 (ERK1) is the paralog — flagged, single gene chosen
    "p38": "MAPK14",
    "AKT": "AKT1",
    "SYK": "SYK",
    "BTK": "BTK",
    "p65": "RELA",
    "CD19": "CD19",
    "IgM": "IGHM",
    "Cyclin A": "CCNA2",
    "Cyclin B1": "CCNB1",
    "Ki-67": "MKI67",
    "JNK": "MAPK8",
    "SHP-1": "PTPN6",
    "GAPDH": "GAPDH",
    "BLNK": "BLNK",
    "CD53": "CD53",
    "CD38": "CD38",
    "CD45": "PTPRC",
    "T-bet": "TBX21",
    "CD27": "CD27",
    "IL10": "IL10",
    "CCL3/4": "CCL3",            # ambiguous compound marker — picked CCL3, flagged
    "CD80": "CD80",
    "KLF6": "KLF6",
    "AID": "AICDA",
    "IgG": "IGHG1",               # ambiguous isotype — flagged
    "CD24": "CD24",
    "IgA": "IGHA1",
    "IgE": "IGHE",
    "CD23": "FCER2",
    "BLIMP1": "PRDM1",
    "IRF8": "IRF8",
    "IRF4": "IRF4",
    "BAFF-R": "TNFRSF13C",
    "XBP1": "XBP1",
}

# Phospho markers get NEW reserved IDs, not gene mapping.
# (This list should match whatever survived your zero-filtering step —
# update if your final feature list differs.)
PHOSPHO_RESERVED_START = 75523  # first unused index in the 85500-slot table

def hugo_symbol_to_myID(hugo_symbol, hgs_to_EntrezID_dict, EntrezID_to_myID_dict):
    entrez = hgs_to_EntrezID_dict.get(hugo_symbol)
    if entrez is None:
        return None
    return EntrezID_to_myID_dict.get(entrez)

def main(protein_h5ad, outdir, dic_path):
    adata = sc.read_h5ad(protein_h5ad)
    print(f"Loaded {protein_h5ad}: {adata.n_vars} protein features")

    with open(os.path.join(dic_path, "hgs_to_EntrezID.pkl"), "rb") as f:
        hgs_to_EntrezID_dict = pickle.load(f)
    with open(os.path.join(dic_path, "EntrezID_to_myID.pkl"), "rb") as f:
        EntrezID_to_myID_dict = pickle.load(f)

    records = []
    next_reserved_id = PHOSPHO_RESERVED_START

    my_ids = []
    for feature in adata.var_names:
        is_phospho = feature.lower().startswith("p-")
        if not is_phospho and feature in GENE_MAP_CANDIDATES:
            gene_symbol = GENE_MAP_CANDIDATES[feature]
            myid = hugo_symbol_to_myID(gene_symbol, hgs_to_EntrezID_dict, EntrezID_to_myID_dict)
            if myid is None:
                print(f"WARNING: no mapping found for '{feature}' (gene '{gene_symbol}') "
                      f"-> falling back to reserved ID")
                myid = str(next_reserved_id)
                next_reserved_id += 1
                mapping_type = "reserved_fallback_mapping_failed"
            else:
                mapping_type = f"gene_mapped({gene_symbol})"
        else:
            # phospho marker, or non-phospho marker not in our candidate map
            myid = str(next_reserved_id)
            next_reserved_id += 1
            mapping_type = "reserved_new_id"
            if not is_phospho:
                print(f"NOTE: '{feature}' is not phospho but has no gene-map candidate "
                      f"-> using reserved ID instead. Add it to GENE_MAP_CANDIDATES if "
                      f"a real gene mapping should be used.")

        my_ids.append(myid)
        records.append({"feature": feature, "my_Id": myid, "mapping_type": mapping_type})

    adata.var["my_Id"] = my_ids

    mapping_df = pd.DataFrame(records)
    print("\n" + "=" * 70)
    print("MAPPING TABLE")
    print("=" * 70)
    print(mapping_df.to_string(index=False))

    if next_reserved_id > 85499:
        print(f"\nWARNING: reserved IDs exceeded table bound (85499). "
              f"Last assigned: {next_reserved_id - 1}. This should not happen "
              f"with only ~44 non-gene-mapped features, but double-check if it does.")

    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(protein_h5ad))[0]
    out_h5ad = os.path.join(outdir, f"{base}_mapped.h5ad")
    out_csv = os.path.join(outdir, f"{base}_myId_mapping_table.csv")

    adata.write_h5ad(out_h5ad)
    mapping_df.to_csv(out_csv, index=False)

    print(f"\nSaved mapped h5ad -> {out_h5ad}")
    print(f"Saved mapping table (for thesis methods/supplementary) -> {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_h5ad", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dic_path", default="code/model/ID_dic/",
                         help="path to scTranslator's ID_dic folder (run from repo root, "
                              "or pass absolute path)")
    args = parser.parse_args()
    main(args.protein_h5ad, args.outdir, args.dic_path)