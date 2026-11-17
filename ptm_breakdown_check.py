"""
ptm_breakdown_check.py
─────────────────────────
Closes TODO #6: recomputes the phospho-specific vs total-protein R/CCC
breakdown directly from the final model's already-saved per-pair
predictions, using the `is_phospho` tag already present in the
cleaned protein AnnData's .var (set in 03_clean_qurie_data.py). No
retraining needed.

Usage:
    python ptm_breakdown_check.py \
        --dir new_results/final_model_corrloss_only_leakfix \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import pearsonr


def concordance_correlation_coefficient(y_true, y_pred):
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return 0.0 if denom == 0 else float((2 * covariance) / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--protein", required=True, help="Path to cleaned protein h5ad with is_phospho tag")
    args = parser.parse_args()

    prot_adata = ad.read_h5ad(args.protein)
    if "is_phospho" not in prot_adata.var.columns:
        raise ValueError("is_phospho column not found -- check 03_clean_qurie_data.py was used to produce this file")
    phospho_map = dict(zip(prot_adata.var_names, prot_adata.var["is_phospho"]))
    print(f"Loaded phospho tags: {sum(phospho_map.values())} phospho, "
          f"{len(phospho_map) - sum(phospho_map.values())} total-protein/other")

    pred_files = sorted(glob.glob(os.path.join(args.dir, "joint_y_pred_t*.csv")))
    all_r_by_protein = {}
    all_ccc_by_protein = {}

    for pred_path in pred_files:
        truth_path = pred_path.replace("y_pred", "y_truth")
        if not os.path.exists(truth_path):
            continue
        pred_df = pd.read_csv(pred_path)
        truth_df = pd.read_csv(truth_path)
        meta_cols = ["t_src", "t_tgt"]
        protein_names = [c for c in pred_df.columns if c not in meta_cols]

        for name in protein_names:
            if name not in phospho_map:
                continue
            yt, yp = truth_df[name].values, pred_df[name].values
            if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
                continue
            r, _ = pearsonr(yt, yp)
            ccc = concordance_correlation_coefficient(yt, yp)
            all_r_by_protein.setdefault(name, []).append(r)
            all_ccc_by_protein.setdefault(name, []).append(ccc)

    phospho_r, phospho_ccc, total_r, total_ccc = [], [], [], []
    for name in all_r_by_protein:
        mean_r = float(np.mean(all_r_by_protein[name]))
        mean_ccc = float(np.mean(all_ccc_by_protein[name]))
        if phospho_map[name]:
            phospho_r.append(mean_r)
            phospho_ccc.append(mean_ccc)
        else:
            total_r.append(mean_r)
            total_ccc.append(mean_ccc)

    print(f"\n{'='*50}")
    print(f"PTM (phospho) channels: n={len(phospho_r)}")
    print(f"  Mean R:   {np.mean(phospho_r):.4f}")
    print(f"  Mean CCC: {np.mean(phospho_ccc):.4f}")
    print(f"\nTotal-protein/other channels: n={len(total_r)}")
    print(f"  Mean R:   {np.mean(total_r):.4f}")
    print(f"  Mean CCC: {np.mean(total_ccc):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
