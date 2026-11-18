"""
zeroshot_per_protein_check.py
────────────────────────────────
Closes TODO #5: recomputes per-protein R for the zero-shot ibrutinib
predictions, directly from the already-saved prediction/truth CSVs,
no retraining needed. Confirms/updates which channels are top-ranked
(the old CD38/BLNK claim was computed on the pre-fix checkpoint).

Usage:
    python zeroshot_per_protein_check.py --dir new_results/zeroshot_ibrutinib_leakfix
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    pred_path = os.path.join(args.dir, "joint_y_pred_t6to180.csv")
    truth_path = os.path.join(args.dir, "joint_y_truth_t6to180.csv")
    pred_df = pd.read_csv(pred_path)
    truth_df = pd.read_csv(truth_path)

    meta_cols = ["t_src", "t_tgt"]
    protein_names = [c for c in pred_df.columns if c not in meta_cols]
    pred_vals = pred_df[protein_names].values
    truth_vals = truth_df[protein_names].values

    rows = []
    for i, name in enumerate(protein_names):
        yt, yp = truth_vals[:, i], pred_vals[:, i]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            r = float("nan")
        else:
            r, _ = pearsonr(yt, yp)
        rows.append({"protein": name, "R": r})

    df = pd.DataFrame(rows).sort_values("R", ascending=False)
    n_positive = (df["R"] > 0.1).sum()
    n_negative = (df["R"] < -0.1).sum()

    print(f"Total channels: {len(df)}")
    print(f"Meaningfully positive (R > 0.1): {n_positive}")
    print(f"Meaningfully negative (R < -0.1): {n_negative}")
    print(f"\nTop {args.top_n} channels by R:")
    print(df.head(args.top_n).to_string(index=False))

    for gene in ["CD38", "BLNK"]:
        row = df[df["protein"] == gene]
        if len(row) > 0:
            rank = df.reset_index(drop=True).index[df.reset_index(drop=True)["protein"] == gene][0] + 1
            print(f"\n{gene}: R={row['R'].values[0]:.4f}, rank {rank}/{len(df)}")
        else:
            print(f"\n{gene}: not found in this dataset's protein panel")

    out_path = os.path.join(args.dir, "per_protein_R_recomputed.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
