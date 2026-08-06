"""
09_sweep_ot_reg.py

Purpose: Sweep Sinkhorn regularization (`reg`) for soft OT on a chosen
t_src -> t_tgt pair, to find the point where the transport plan stops
over-blending (near-uniform average) and starts producing meaningfully
concentrated, hard-like matches -- without going so low that Sinkhorn
becomes numerically unstable.

For each reg value, reports:
  - effective # targets blended per source cell (lower = more concentrated)
  - whether Sinkhorn converged without numerical warnings/errors
  - comparison against hard OT's cost-matrix-implied "ideal" concentration

Usage:
    python 09_sweep_ot_reg.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --t_src 0 --t_tgt 180 \
        --gene_strategy hvg --n_genes 200 \
        --reg_values 0.001,0.005,0.01,0.05,0.1,0.5,1.0
"""

import argparse
import sys
import os
import warnings
import numpy as np
import anndata as ad
from sklearn.metrics.pairwise import cosine_distances

sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))
from gene_selection import select_genes

try:
    import ot
except ImportError:
    print("ERROR: POT not installed. pip install pot --break-system-packages")
    sys.exit(1)


def split_cells_per_timepoint(obs_time, train_ratio=0.8, seed=1105):
    rng = np.random.default_rng(seed)
    timepoints = sorted(np.unique(obs_time).tolist())
    train_cells = {}
    for t in timepoints:
        cells = np.where(np.isclose(obs_time, t, atol=1e-3))[0]
        rng.shuffle(cells)
        n_train = int(len(cells) * train_ratio)
        train_cells[t] = cells[:n_train]
    return train_cells


def main(rna_path, protein_path, t_src, t_tgt, gene_strategy, n_genes, reg_values, condition="aIg"):
    rna_adata = ad.read_h5ad(rna_path)
    mask = rna_adata.obs["condition"] == condition
    rna_adata = rna_adata[mask].copy()

    rna_counts = rna_adata.layers["counts"]
    rna_counts = (rna_counts.toarray() if hasattr(rna_counts, "toarray") else rna_counts).astype(np.float32)
    obs_time = rna_adata.obs["time_min"].values.astype(float)

    n_genes_arg = n_genes if n_genes == "all" else int(n_genes)
    gene_idx = select_genes(rna_adata, gene_strategy, n_genes_arg, groupby="time_min")
    rna_selected = rna_counts[:, gene_idx]

    train_cells = split_cells_per_timepoint(obs_time)
    src_idx = train_cells[t_src]
    tgt_idx = train_cells[t_tgt]
    rna_src = rna_selected[src_idx]
    rna_tgt = rna_selected[tgt_idx]

    n_src, n_tgt = len(rna_src), len(rna_tgt)
    C = cosine_distances(rna_src, rna_tgt).astype(np.float64)

    print(f"Config: t_src={t_src} t_tgt={t_tgt} n_src={n_src} n_tgt={n_tgt} "
          f"gene_strategy={gene_strategy} n_genes={len(gene_idx)}")
    print(f"Cost matrix: min={C.min():.4f} max={C.max():.4f} mean={C.mean():.4f} std={C.std():.4f}")

    # Hard OT reference point: what concentration looks like with NO
    # entropy smoothing at all (the ceiling of how "sharp" a match can be)
    a = np.ones(n_src) / n_src
    b = np.ones(n_tgt) / n_tgt
    T_hard = ot.emd(a, b, C)
    T_hard_norm = T_hard / (T_hard.sum(axis=1, keepdims=True) + 1e-8)
    eff_hard = 1.0 / (T_hard_norm ** 2).sum(axis=1)
    print(f"\nHARD OT reference: effective targets blended = {eff_hard.mean():.2f} "
          f"(this is the sharpest possible -- soft OT should approach this as "
          f"reg -> 0, and approach {n_tgt} as reg -> large)\n")

    print(f"{'reg':>10} | {'eff_targets_mean':>18} | {'eff_targets_min':>16} | {'converged':>10} | note")
    print("-" * 90)

    for reg in reg_values:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                T = ot.sinkhorn(a, b, C, reg=reg, numItermax=1000)
                converged = not any("did not converge" in str(warn.message).lower() for warn in w)
                has_nan = np.isnan(T).any()
            except Exception as e:
                print(f"{reg:>10.4f} | FAILED: {e}")
                continue

        if has_nan:
            print(f"{reg:>10.4f} | {'NaN in transport plan':>18} | {'':>16} | {'NO':>10} | "
                  f"reg too low, numerically unstable -- do not use")
            continue

        T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-8)
        eff_n = 1.0 / (T_norm ** 2).sum(axis=1)

        note = ""
        if eff_n.mean() > n_tgt * 0.5:
            note = "still heavily blended (>50% of targets)"
        elif eff_n.mean() < eff_hard.mean() * 3:
            note = "close to hard-OT sharpness -- good candidate"
        else:
            note = "moderate blending"

        conv_str = "yes" if converged else "NO (warn)"
        print(f"{reg:>10.4f} | {eff_n.mean():>18.2f} | {eff_n.min():>16.2f} | {conv_str:>10} | {note}")

    print("\nGuidance: pick the SMALLEST reg that (a) converges cleanly (no NaN, "
          "no non-convergence warning) and (b) gets eff_targets_mean reasonably "
          "close to the hard-OT reference above. Going lower than that buys "
          "sharper matches at the cost of numerical stability, with no real benefit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--t_src", type=float, required=True)
    parser.add_argument("--t_tgt", type=float, required=True)
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], default="hvg")
    parser.add_argument("--n_genes", default="200")
    parser.add_argument("--condition", default="aIg")
    parser.add_argument("--reg_values", default="0.001,0.005,0.01,0.05,0.1,0.5,1.0",
                         help="comma-separated list of reg values to test")
    args = parser.parse_args()
    reg_list = [float(x) for x in args.reg_values.split(",")]
    main(args.rna, args.protein, args.t_src, args.t_tgt, args.gene_strategy,
         args.n_genes, reg_list, args.condition)