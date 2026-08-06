"""
03_clean_qurie_data.py

Purpose: Clean and align paired RNA + Protein QuRIE-seq h5ad files.
QC is computed and applied SEPARATELY per condition (aIg vs aIg+Ibru)
to avoid one condition's outlier distribution skewing thresholds for
the other, then both conditions are recombined into single cleaned
RNA and Protein h5ad outputs with 'condition' retained as an obs column.

Usage:
    python 03_clean_qurie_data.py \
        --rna /path/to/qurie_RNA.h5ad \
        --protein /path/to/qurie_Protein.h5ad \
        --outdir /path/to/output_dir

Requires: scanpy, anndata, numpy, pandas
"""

import argparse
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

PHOSPHO_MARKERS = [
    "p-p65", "p-SYK", "p-JNK", "p-p38", "p-PLC-y2", "p-BTK", "p-SHP-2",
    "p-SHP-1", "p-RB", "p-c-JUN", "p-AKT", "p-S6", "p-ERK 1/2",
    "p-Histon H3", "p-Histon H2A.X", "p-SRC", "p-TOR", "p-BLNK",
    "p-PKC-b1", "p-IRAK4", "p-CD79a", "p-CDK1", "p-CDK4", "p-AMPK-a1/2",
    "p-AMPK-b1", "p-IKK a/b", "p-JAK1", "p-PLC-y2 (Y759)", "p-CDK6",
    "p-STAT1", "p-STAT3", "p-STAT5", "p-STAT6",
]

def log(msg):
    print(f"[clean] {msg}")

def qc_filter_rna_per_condition(adata, min_genes=200, max_pct_mt=20.0):
    """Compute QC + filter RNA cells, separately per condition."""
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    kept_chunks = []
    for cond, sub in adata.obs.groupby("condition", observed=True):
        chunk = adata[sub.index].copy()
        sc.pp.calculate_qc_metrics(
            chunk, qc_vars=["mt"] if chunk.var["mt"].any() else [],
            percent_top=None, log1p=False, inplace=True
        )
        n_before = chunk.n_obs
        gene_mask = chunk.obs["n_genes_by_counts"] >= min_genes
        if "pct_counts_mt" in chunk.obs:
            mt_mask = chunk.obs["pct_counts_mt"] <= max_pct_mt
        else:
            mt_mask = pd.Series(True, index=chunk.obs.index)
        keep = gene_mask & mt_mask
        chunk = chunk[keep].copy()
        log(f"RNA condition='{cond}': {n_before} -> {chunk.n_obs} cells "
            f"(min_genes>={min_genes}, mt<={max_pct_mt}%)")
        kept_chunks.append(chunk)
    merged = ad.concat(kept_chunks, join="outer", merge="same")
    merged.obs_names_make_unique()
    return merged

def qc_filter_protein_per_condition(adata, min_features=None):
    """Filter protein cells per condition. Panel is small (80 features),
    so we filter mainly on total signal, not gene-count-style thresholds."""
    kept_chunks = []
    for cond, sub in adata.obs.groupby("condition", observed=True):
        chunk = adata[sub.index].copy()
        total_counts = np.asarray(chunk.X).sum(axis=1)
        n_before = chunk.n_obs
        # Flag cells with near-zero total signal across the whole panel
        # (likely failed/empty droplets) — threshold set generously since
        # panel is targeted, not whole-proteome.
        thresh = np.percentile(total_counts, 1)  # bottom 1% as floor
        keep = total_counts > max(thresh, 0)
        chunk = chunk[keep].copy()
        log(f"Protein condition='{cond}': {n_before} -> {chunk.n_obs} cells "
            f"(removed bottom 1% by total signal, floor={thresh:.1f})")
        kept_chunks.append(chunk)
    merged = ad.concat(kept_chunks, join="outer", merge="same")
    merged.obs_names_make_unique()
    return merged

def main(rna_path, protein_path, outdir):
    log("Loading RNA and Protein h5ad files...")
    rna = sc.read_h5ad(rna_path)
    prot = sc.read_h5ad(protein_path)

    # --- 1. Verify pairing is exact, not just count-matched ---
    log("Verifying RNA and Protein obs_names match exactly (same cells)...")
    if not rna.obs_names.equals(prot.obs_names):
        common = rna.obs_names.intersection(prot.obs_names)
        log(f"WARNING: obs_names do NOT match exactly. "
            f"RNA={rna.n_obs}, Protein={prot.n_obs}, common={len(common)}.")
        log("Subsetting both to common obs_names before proceeding.")
        rna = rna[common].copy()
        prot = prot[common].copy()
    else:
        log("Confirmed: RNA and Protein are exactly paired 1:1.")

    # --- 2. Remove genes/features with zero signal across all cells ---
    log("Removing RNA genes with zero counts across all cells...")
    gene_counts = np.asarray(rna.X.sum(axis=0)).flatten() if hasattr(rna.X, "sum") else rna.X.sum(axis=0)
    nonzero_genes = gene_counts > 0
    log(f"RNA genes: {rna.n_vars} -> {nonzero_genes.sum()}")
    rna = rna[:, nonzero_genes].copy()

    log("Removing Protein features with zero counts across all cells...")
    prot_counts = np.asarray(prot.X).sum(axis=0)
    nonzero_prot = prot_counts > 0
    log(f"Protein features: {prot.n_vars} -> {nonzero_prot.sum()}")
    if (~nonzero_prot).sum() > 0:
        dropped = prot.var_names[~nonzero_prot].tolist()
        log(f"Dropped zero-signal protein features: {dropped}")
    prot = prot[:, nonzero_prot].copy()

    # --- 3. QC filter cells, per condition, per modality ---
    log("QC filtering RNA cells (per condition)...")
    rna = qc_filter_rna_per_condition(rna)

    log("QC filtering Protein cells (per condition)...")
    prot = qc_filter_protein_per_condition(prot)

    # --- 4. Re-sync pairing after independent filtering ---
    common_after_qc = rna.obs_names.intersection(prot.obs_names)
    log(f"After independent QC: RNA={rna.n_obs}, Protein={prot.n_obs}, "
        f"common (final paired set)={len(common_after_qc)}")
    rna = rna[common_after_qc].copy()
    prot = prot[common_after_qc].copy()

    # --- 5. Normalize ---
    log("Storing raw counts in .layers['counts'] before normalizing...")
    rna.layers["counts"] = rna.X.copy()
    prot.layers["counts"] = prot.X.copy()

    log("Normalizing RNA: total-count normalize + log1p...")
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)

    log("Normalizing Protein: arcsinh transform (cofactor=5, standard for "
        "mass-cytometry / targeted-panel protein data)...")
    cofactor = 5.0
    prot.X = np.arcsinh(np.asarray(prot.X) / cofactor)

    # --- 6. Tag phospho vs total-protein features in var ---
    prot.var["is_phospho"] = prot.var_names.isin(PHOSPHO_MARKERS)
    log(f"Tagged {prot.var['is_phospho'].sum()} phospho features, "
        f"{(~prot.var['is_phospho']).sum()} total-protein/other features.")

    # --- 7. Final sanity checks ---
    assert rna.obs_names.equals(prot.obs_names), "Final RNA/Protein obs mismatch!"
    assert not np.isnan(np.asarray(rna.X)).any() if not hasattr(rna.X, "toarray") else not np.isnan(rna.X.toarray()).any(), "NaNs in RNA X!"
    assert not np.isnan(np.asarray(prot.X)).any(), "NaNs in Protein X!"
    log("Sanity checks passed: obs aligned, no NaNs in either matrix.")

    # --- 8. Save — split by condition into separate h5ad files ---
    conditions = rna.obs["condition"].unique().tolist()
    log(f"Saving separate h5ad files per condition: {conditions}")

    def safe_tag(cond):
        # filesystem-safe suffix, e.g. 'aIg+Ibru' -> 'aIg_Ibru'
        return str(cond).replace("+", "_").replace(" ", "_")

    saved_paths = []
    for cond in conditions:
        rna_c = rna[rna.obs["condition"] == cond].copy()
        prot_c = prot[prot.obs["condition"] == cond].copy()
        assert rna_c.obs_names.equals(prot_c.obs_names), \
            f"Obs mismatch for condition {cond}!"

        tag = safe_tag(cond)
        rna_out = f"{outdir}/qurie_RNA_clean_{tag}.h5ad"
        prot_out = f"{outdir}/qurie_Protein_clean_{tag}.h5ad"
        rna_c.write_h5ad(rna_out)
        prot_c.write_h5ad(prot_out)
        saved_paths.append((cond, rna_out, prot_out))
        log(f"Condition '{cond}': saved RNA -> {rna_out}")
        log(f"Condition '{cond}': saved Protein -> {prot_out}")

    # --- 9. Final report ---
    log("=" * 60)
    log("FINAL REPORT")
    log(f"Total final paired cells (both conditions): {rna.n_obs}")
    log(f"RNA genes retained: {rna.n_vars}")
    log(f"Protein features retained: {prot.n_vars} "
        f"({prot.var['is_phospho'].sum()} phospho, "
        f"{(~prot.var['is_phospho']).sum()} other)")
    for cond, rna_out, prot_out in saved_paths:
        sub = rna.obs[rna.obs["condition"] == cond]
        log(f"--- Condition: {cond} ---")
        log(f"  Cells: {len(sub)}")
        log(f"  Timepoint distribution:")
        print(sub["time_min"].value_counts())
        log(f"  Files: {rna_out} | {prot_out}")
    log("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    main(args.rna, args.protein, args.outdir)