"""
verify_final_model_pooled_vs_perprotein.py

Purpose: Same verification as verify_pooled_vs_perprotein.py, but for the
final model's predictions, which are split across 10 separate files (one
per timepoint pair) rather than one combined file. This script
concatenates all 10 pairs' predictions and truths together (stacking
cells from every pair into one big array), then computes pooled vs.
per-protein CCC/R on the combined set -- matching how the final model's
overall reported mean R/CCC figures are computed elsewhere in the thesis
(e.g. averaged across all ten timepoint pairs).

Expects a directory containing files named like:
    joint_y_pred_t0to2.csv, joint_y_truth_t0to2.csv,
    joint_y_pred_t0to4.csv, joint_y_truth_t0to4.csv,
    ... (10 pairs total)

Usage:
    python verify_final_model_pooled_vs_perprotein.py --dir final_model_corrloss_only_leakfix/
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def ccc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    covariance = np.mean((x - mean_x) * (y - mean_y))
    return (2 * covariance) / (var_x + var_y + (mean_x - mean_y) ** 2)


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def cosine_similarity(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return np.nan
    return np.dot(x, y) / denom


LABEL_COLUMNS = {"t_src", "t_tgt"}


def drop_index_col(df, name):
    first_col = df.columns[0]
    looks_like_index = (
        df[first_col].dtype == object
        or "id" in first_col.lower()
        or "barcode" in first_col.lower()
        or "unnamed" in first_col.lower()
        or first_col.strip() == ""
    )
    if looks_like_index:
        print(f"  Dropping likely index column '{first_col}' from {name}")
        df = df.drop(columns=[first_col])

    label_cols_present = [c for c in df.columns if c.strip().lower() in LABEL_COLUMNS]
    if label_cols_present:
        print(f"  Dropping timepoint label column(s) {label_cols_present} from {name}")
        df = df.drop(columns=label_cols_present)
    return df


def main(directory, per_pair_report):
    pred_files = sorted(glob.glob(os.path.join(directory, "*y_pred_t*.csv")))
    if not pred_files:
        raise FileNotFoundError(f"No files matching '*y_pred_t*.csv' found in {directory}")

    pred_dfs = []
    truth_dfs = []
    pair_labels = []

    for pred_path in pred_files:
        # Derive matching truth path by replacing 'y_pred' with 'y_truth'
        truth_path = pred_path.replace("y_pred", "y_truth")
        if not os.path.exists(truth_path):
            print(f"WARNING: no matching truth file for {pred_path}, skipping.")
            continue

        pair_label = re.search(r"t(\d+to\d+)", os.path.basename(pred_path))
        pair_label = pair_label.group(1) if pair_label else os.path.basename(pred_path)

        pred = pd.read_csv(pred_path)
        truth = pd.read_csv(truth_path)
        pred = drop_index_col(pred, f"pred ({pair_label})")
        truth = drop_index_col(truth, f"truth ({pair_label})")

        common_cols = [c for c in pred.columns if c in truth.columns]
        pred = pred[common_cols]
        truth = truth[common_cols]

        if pred.shape[0] != truth.shape[0]:
            print(f"WARNING: row mismatch for pair {pair_label} ({pred.shape[0]} vs {truth.shape[0]}), skipping.")
            continue

        pred_dfs.append(pred)
        truth_dfs.append(truth)
        pair_labels.append(pair_label)

        if per_pair_report:
            pred_arr = pred.values.astype(float)
            truth_arr = truth.values.astype(float)
            p_r = pearson_r(truth_arr.flatten(), pred_arr.flatten())
            p_ccc = ccc(truth_arr.flatten(), pred_arr.flatten())
            per_r = [pearson_r(truth[c].values, pred[c].values) for c in pred.columns]
            per_ccc = [ccc(truth[c].values, pred[c].values) for c in pred.columns]
            print(f"Pair {pair_label}: n={pred.shape[0]} cells, "
                  f"pooled R={p_r:.4f} CCC={p_ccc:.4f}, "
                  f"per-protein mean R={np.nanmean(per_r):.4f} CCC={np.nanmean(per_ccc):.4f}")

    # Require identical columns across all pairs before concatenating
    common_cols = set(pred_dfs[0].columns)
    for df in pred_dfs[1:]:
        common_cols &= set(df.columns)
    common_cols = sorted(common_cols)
    print(f"\nUsing {len(common_cols)} protein channels common to all {len(pred_dfs)} pairs.")

    pred_all = pd.concat([df[common_cols] for df in pred_dfs], axis=0, ignore_index=True)
    truth_all = pd.concat([df[common_cols] for df in truth_dfs], axis=0, ignore_index=True)

    print(f"\nCombined shape across all {len(pred_dfs)} timepoint pairs: "
          f"{pred_all.shape[0]} total cell-rows x {pred_all.shape[1]} protein channels\n")

    pred_arr = pred_all.values.astype(float)
    truth_arr = truth_all.values.astype(float)

    pred_flat = pred_arr.flatten()
    truth_flat = truth_arr.flatten()
    pooled_r = pearson_r(truth_flat, pred_flat)
    pooled_ccc = ccc(truth_flat, pred_flat)
    pooled_cosine = cosine_similarity(truth_flat, pred_flat)

    print("=== POOLED METRICS (all pairs concatenated, then flattened) ===")
    print(f"Pooled Pearson R:        {pooled_r:.4f}")
    print(f"Pooled CCC:              {pooled_ccc:.4f}")
    print(f"Pooled cosine similarity:{pooled_cosine:.4f}")

    per_protein_r, per_protein_ccc = [], []
    for col in common_cols:
        per_protein_r.append(pearson_r(truth_all[col].values, pred_all[col].values))
        per_protein_ccc.append(ccc(truth_all[col].values, pred_all[col].values))

    mean_r = np.nanmean(per_protein_r)
    mean_ccc = np.nanmean(per_protein_ccc)

    print(f"\n=== PER-PROTEIN METRICS (per channel across all pairs combined, then averaged) ===")
    print(f"Mean per-protein R:      {mean_r:.4f}")
    print(f"Mean per-protein CCC:    {mean_ccc:.4f}")

    print(f"\n=== COMPARISON ===")
    print(f"Pooled CCC:       {pooled_ccc:.4f}")
    print(f"Per-protein CCC:  {mean_ccc:.4f}")
    print(f"Gap (pooled - per-protein): {pooled_ccc - mean_ccc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the 10 joint_y_pred_t*.csv / joint_y_truth_t*.csv file pairs")
    parser.add_argument("--per_pair_report", action="store_true", help="Also print pooled/per-protein metrics for each individual pair before combining")
    args = parser.parse_args()
    main(args.dir, args.per_pair_report)