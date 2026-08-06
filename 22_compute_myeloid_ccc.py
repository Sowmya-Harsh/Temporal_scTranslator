"""
22_compute_myeloid_ccc.py

Purpose: Compute per-protein R and CCC (thesis convention: per-protein,
then averaged, never pooled) for the myeloid external validation
results, from the saved joint_y_pred / joint_y_truth CSVs. This has
been missing from the myeloid results so far -- only R was reported.

Usage:
    python 22_compute_myeloid_ccc.py \
        --pred_csv myeloid_finetune_results/joint_y_pred_t0to10080.csv \
        --truth_csv myeloid_finetune_results/joint_y_truth_t0to10080.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def per_protein_r_ccc(y_true, y_pred, protein_cols):
    rows = []
    for p in protein_cols:
        yt, yp = y_true[p].values, y_pred[p].values
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            rows.append({"protein": p, "R": np.nan, "CCC": np.nan})
            continue
        r, _ = pearsonr(yt, yp)
        mean_t, mean_p = yt.mean(), yp.mean()
        var_t, var_p = yt.var(), yp.var()
        cov = np.mean((yt - mean_t) * (yp - mean_p))
        ccc = (2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12)
        rows.append({"protein": p, "R": r, "CCC": ccc})
    return pd.DataFrame(rows)


def main(pred_csv, truth_csv):
    y_pred = pd.read_csv(pred_csv)
    y_true = pd.read_csv(truth_csv)
    protein_cols = [c for c in y_pred.columns if c not in ("t_src", "t_tgt")]

    print(f"Loaded {len(y_pred)} cells, {len(protein_cols)} proteins")
    res = per_protein_r_ccc(y_true, y_pred, protein_cols)

    mean_r = res["R"].mean()
    mean_ccc = res["CCC"].mean()
    n_valid = res["R"].notna().sum()

    print(f"\nValid proteins (non-zero variance both sides): {n_valid}/{len(protein_cols)}")
    print(f"Mean per-protein R:   {mean_r:.4f}")
    print(f"Mean per-protein CCC: {mean_ccc:.4f}")

    print(f"\nWorst 5 CCC:")
    print(res.sort_values("CCC").head(5).to_string(index=False))
    print(f"\nBest 5 CCC:")
    print(res.sort_values("CCC", ascending=False).head(5).to_string(index=False))

    res.to_csv("myeloid_per_protein_R_CCC.csv", index=False)
    print(f"\nSaved full per-protein table to myeloid_per_protein_R_CCC.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--truth_csv", type=str, required=True)
    args = parser.parse_args()
    main(args.pred_csv, args.truth_csv)
