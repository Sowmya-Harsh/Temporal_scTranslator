"""
07_subset_single_timepoint.py

Purpose: Extract a single timepoint from the cleaned+mapped aIg RNA and
Protein files, for a baseline sanity check of scTranslator's pretrained
RNA->protein mapping (no time dimension, no fine-tuning) on same-
timepoint data.

Usage:
    python 07_subset_single_timepoint.py \
        --rna /path/to/qurie_RNA_clean_aIg.h5ad \
        --protein /path/to/qurie_Protein_clean_aIg.h5ad \
        --timepoint 180 \
        --outdir /path/to/output_dir
"""

import argparse
import scanpy as sc

def main(rna_path, protein_path, timepoint, outdir):
    rna = sc.read_h5ad(rna_path)
    prot = sc.read_h5ad(protein_path)

    assert rna.obs_names.equals(prot.obs_names), "RNA/Protein obs mismatch before subsetting!"

    mask = rna.obs["time_min"] == timepoint
    n = mask.sum()
    print(f"Timepoint {timepoint} min: {n} cells found (out of {rna.n_obs} total)")
    if n == 0:
        print(f"ERROR: no cells found at time_min={timepoint}. "
              f"Available timepoints: {sorted(rna.obs['time_min'].unique().tolist())}")
        return

    rna_sub = rna[mask].copy()
    prot_sub = prot[mask].copy()
    assert rna_sub.obs_names.equals(prot_sub.obs_names), "Mismatch after subsetting!"

    rna_out = f"{outdir}/qurie_RNA_clean_aIg_t{timepoint}.h5ad"
    prot_out = f"{outdir}/qurie_Protein_clean_aIg_t{timepoint}.h5ad"
    rna_sub.write_h5ad(rna_out)
    prot_sub.write_h5ad(prot_out)

    print(f"Saved RNA subset -> {rna_out}")
    print(f"Saved Protein subset -> {prot_out}")
    print(f"\nCell count: {n}")
    print(f"RNA genes: {rna_sub.n_vars}, Protein features: {prot_sub.n_vars}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--timepoint", type=int, required=True,
                         help="value from obs['time_min'] to subset, e.g. 180")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    main(args.rna, args.protein, args.timepoint, args.outdir)