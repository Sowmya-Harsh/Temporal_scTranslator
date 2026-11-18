"""
verify_pooled_vs_perprotein.py

Purpose: Independently recompute pooled vs. per-protein Pearson correlation
and CCC directly from raw y_pred.csv / y_truth.csv files, to verify the
numbers reported in Table 5.1 (Pooled vs. per-protein CCC) from scratch,
rather than trusting a previously-computed result.

Expects y_pred.csv and y_truth.csv to have the same shape: rows = cells,
columns = proteins/channels, with matching column names/order between
the two files. If your files have an index/ID column (e.g. cell barcode),
it will be auto-detected and dropped.

Usage:
    python verify_pooled_vs_perprotein.py \
        --pred_path y_pred.csv \
        --truth_path y_truth.csv
"""
import argparse

import numpy as np
import pandas as pd


def ccc(x, y):
    """Concordance Correlation Coefficient between two 1D arrays."""
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


def load_and_align(pred_path, truth_path):
    pred = pd.read_csv(pred_path)
    truth = pd.read_csv(truth_path)

    # Auto-detect and drop a likely index/ID column (non-numeric first column)
    for df, name in [(pred, "pred"), (truth, "truth")]:
        first_col = df.columns[0]
        if df[first_col].dtype == object or "id" in first_col.lower() or "barcode" in first_col.lower():
            print(f"Dropping likely index column '{first_col}' from {name}")
            df.drop(columns=[first_col], inplace=True)

    # Keep only columns present in both, in matching order
    common_cols = [c for c in pred.columns if c in truth.columns]
    if len(common_cols) == 0:
        raise ValueError(
            "No matching column names found between pred and truth files. "
            "Check that both files have the same protein/channel column headers, "
            "or edit this script to align columns positionally instead."
        )
    dropped_pred = set(pred.columns) - set(common_cols)
    dropped_truth = set(truth.columns) - set(common_cols)
    if dropped_pred:
        print(f"Columns in pred but not truth (dropped): {sorted(dropped_pred)}")
    if dropped_truth:
        print(f"Columns in truth but not pred (dropped): {sorted(dropped_truth)}")

    pred = pred[common_cols]
    truth = truth[common_cols]

    if pred.shape[0] != truth.shape[0]:
        raise ValueError(
            f"Row count mismatch: pred has {pred.shape[0]} rows, truth has {truth.shape[0]} rows. "
            "Files must have one row per cell, in matching order."
        )

    return pred, truth


def main(pred_path, truth_path):
    pred, truth = load_and_align(pred_path, truth_path)
    print(f"\nAligned shape: {pred.shape[0]} cells x {pred.shape[1]} protein channels\n")

    pred_arr = pred.values.astype(float)
    truth_arr = truth.values.astype(float)

    # --- Pooled metrics: flatten everything into one long list ---
    pred_flat = pred_arr.flatten()
    truth_flat = truth_arr.flatten()
    pooled_r = pearson_r(truth_flat, pred_flat)
    pooled_ccc = ccc(truth_flat, pred_flat)
    pooled_cosine = cosine_similarity(truth_flat, pred_flat)

    print("=== POOLED METRICS (flattened cell x protein) ===")
    print(f"Pooled Pearson R:        {pooled_r:.4f}")
    print(f"Pooled CCC:              {pooled_ccc:.4f}")
    print(f"Pooled cosine similarity:{pooled_cosine:.4f}")

    # --- Per-protein metrics: compute separately per column, then average ---
    per_protein_r = []
    per_protein_ccc = []
    per_protein_names = []
    for col in pred.columns:
        r = pearson_r(truth[col].values, pred[col].values)
        c = ccc(truth[col].values, pred[col].values)
        per_protein_r.append(r)
        per_protein_ccc.append(c)
        per_protein_names.append(col)

    per_protein_df = pd.DataFrame({
        "protein": per_protein_names,
        "R": per_protein_r,
        "CCC": per_protein_ccc,
    })

    mean_r = np.nanmean(per_protein_r)
    mean_ccc = np.nanmean(per_protein_ccc)
    n_nan_r = np.sum(np.isnan(per_protein_r))
    n_nan_ccc = np.sum(np.isnan(per_protein_ccc))

    print(f"\n=== PER-PROTEIN METRICS (computed per channel, then averaged) ===")
    print(f"Mean per-protein R:      {mean_r:.4f}  ({n_nan_r} channels excluded as undefined)")
    print(f"Mean per-protein CCC:    {mean_ccc:.4f}  ({n_nan_ccc} channels excluded as undefined)")

    print(f"\n=== COMPARISON ===")
    print(f"Pooled CCC:       {pooled_ccc:.4f}")
    print(f"Per-protein CCC:  {mean_ccc:.4f}")
    print(f"Gap (pooled - per-protein): {pooled_ccc - mean_ccc:.4f}")

    per_protein_df.to_csv("per_protein_breakdown.csv", index=False)
    print(f"\nSaved full per-protein breakdown to per_protein_breakdown.csv")
    print("\nLowest-scoring proteins (by CCC):")
    print(per_protein_df.sort_values("CCC").head(10).to_string(index=False))
    print("\nHighest-scoring proteins (by CCC):")
    print(per_protein_df.sort_values("CCC", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--truth_path", type=str, required=True)
    args = parser.parse_args()
    main(args.pred_path, args.truth_path)
