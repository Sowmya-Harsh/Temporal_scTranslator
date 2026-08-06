"""
08_make_synthetic_test_data.py

Purpose: Generate a small, biologically-toy but structurally-realistic
synthetic time-resolved RNA + protein dataset, DIFFERENT in shape from
QuRIE-seq (different gene count, different protein count, different
timepoint set), purely to sanity-check that 06_generalized_finetune.py
runs end-to-end without crashing before spending time on a real external
dataset (Kaggle HSPC or otherwise).

This is NOT a claim of biological validity or a real generalization
result — it only proves the pipeline's plumbing (gene selection, OT
matching across an arbitrary timepoint set, adapter-based fine-tuning,
diagnostics) works on a dataset whose shape the original checkpoint was
never trained on. Report it in the thesis as exactly that: a functional
test, not an external validation result.

Design: genes are given weak-to-moderate ground-truth relationships to
protein channels via a random linear map, plus per-gene time-dependent
drift, so the diagnostic protocol (Section 4.8) has something real to
detect (i.e. shuffling really should hurt performance here, unlike pure
noise where nothing would be detectable either way).

Usage:
    python 08_make_synthetic_test_data.py \
        --n_cells_per_timepoint 300 \
        --n_genes 500 \
        --n_proteins 40 \
        --timepoints 0 5 15 45 \
        --out_rna synthetic_test_RNA.h5ad \
        --out_prot synthetic_test_Protein.h5ad
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd


def main(n_cells_per_timepoint, n_genes, n_proteins, timepoints, out_rna, out_prot, seed):
    rng = np.random.default_rng(seed)

    gene_names = [f"SYN_GENE_{i}" for i in range(n_genes)]
    protein_names = [f"SYN_PROT_{i}" for i in range(n_proteins)]

    # Ground-truth linear map from genes -> proteins (sparse-ish, moderate weight)
    W = rng.normal(0, 0.3, size=(n_genes, n_proteins))
    mask = rng.random((n_genes, n_proteins)) < 0.1  # only 10% of gene-protein pairs are "real"
    W = W * mask

    # Per-protein time-dependent drift (so temporal conditioning has something real to learn)
    time_drift = rng.normal(0, 0.05, size=(len(timepoints), n_proteins))

    all_rna_rows, all_prot_rows, all_times = [], [], []

    for t_idx, t in enumerate(timepoints):
        base_expr = rng.gamma(2.0, 1.0, size=n_genes)  # baseline gene expression profile
        for _ in range(n_cells_per_timepoint):
            noise = rng.normal(0, 0.3, size=n_genes)
            rna_cell = np.clip(base_expr + noise + 0.02 * t_idx, 0, None)

            protein_signal = rna_cell @ W  # real gene->protein relationship
            protein_cell = protein_signal + time_drift[t_idx] + rng.normal(0, 0.1, size=n_proteins)
            protein_cell = np.clip(protein_cell, 0, None)

            all_rna_rows.append(rna_cell)
            all_prot_rows.append(protein_cell)
            all_times.append(t)

    rna_X = np.array(all_rna_rows, dtype=np.float32)
    prot_X = np.array(all_prot_rows, dtype=np.float32)
    times = np.array(all_times, dtype=float)

    cell_ids = [f"cell_{i}" for i in range(len(times))]

    rna_adata = ad.AnnData(
        X=rna_X,
        obs=pd.DataFrame({"time_min": times}, index=cell_ids),
        var=pd.DataFrame(index=gene_names),
    )
    prot_adata = ad.AnnData(
        X=prot_X,
        obs=pd.DataFrame({"time_min": times}, index=cell_ids),
        var=pd.DataFrame(index=protein_names),
    )

    rna_adata.write_h5ad(out_rna)
    prot_adata.write_h5ad(out_prot)

    print(f"Saved {out_rna} ({rna_adata.shape})")
    print(f"Saved {out_prot} ({prot_adata.shape})")
    print(f"Timepoints: {timepoints}")
    print(f"\nSanity check command:")
    print(f"  python 06_generalized_finetune.py --rna_path {out_rna} --prot_path {out_prot} "
          f"--time_col time_min --checkpoint <your_checkpoint.pt> "
          f"--gene_panel_size 100 --epochs 20 --out_dir synthetic_test_results")
    print(f"\nExpected outcome if the pipeline is working correctly: prediction-shuffle and")
    print(f"RNA-shuffle diagnostics should show a clear R drop (real signal was built in above),")
    print(f"since ~10% of gene-protein pairs carry genuine linear signal by construction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cells_per_timepoint", type=int, default=300)
    parser.add_argument("--n_genes", type=int, default=500)
    parser.add_argument("--n_proteins", type=int, default=40)
    parser.add_argument("--timepoints", type=float, nargs="+", default=[0, 5, 15, 45])
    parser.add_argument("--out_rna", type=str, default="synthetic_test_RNA.h5ad")
    parser.add_argument("--out_prot", type=str, default="synthetic_test_Protein.h5ad")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.n_cells_per_timepoint, args.n_genes, args.n_proteins, args.timepoints,
         args.out_rna, args.out_prot, args.seed)