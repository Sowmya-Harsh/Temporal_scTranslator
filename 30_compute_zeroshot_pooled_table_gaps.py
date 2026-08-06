"""
30_compute_zeroshot_pooled_table_gaps.py

Purpose: Fill in the missing per-protein CCC / R values for scTranslator's
native zero-shot rows (CITE-seq own validation data, and zero-shot on
QuRIE-seq), matching the same corrected metric convention already used
for the fine-tuned rows in the same table.

Flexible loader: tries .npy first (assumes paired y_pred.npy / y_true.npy
in the same directory), then falls back to .csv with the same
t_src,t_tgt,<protein columns> format used throughout this thesis. Adjust
the loader if your saved files use a different format/naming.

Usage:
    python 30_compute_zeroshot_pooled_table_gaps.py \
        --pred_path <path to scTranslator's saved zero-shot predictions> \
        --truth_path <path to matching ground truth> \
        --label "Zero-shot, QuRIE-seq"
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def concordance_correlation_coefficient(y_true, y_pred):
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    return (2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12)


def load_array(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        protein_cols = [c for c in df.columns if c not in ("t_src", "t_tgt")]
        return df[protein_cols].values
    else:
        raise ValueError(f"Unrecognized file type: {path.suffix}. "
                          f"Adjust load_array() for your actual saved format.")


def main(pred_path, truth_path, label):
    y_pred = load_array(pred_path)
    y_true = load_array(truth_path)

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: pred {y_pred.shape} vs truth {y_true.shape}. "
                          f"Check both files correspond to the same cells/proteins.")

    print(f"Loaded {label}: {y_pred.shape[0]} cells, {y_pred.shape[1]} proteins")

    rs, cccs = [], []
    for j in range(y_pred.shape[1]):
        yt, yp = y_true[:, j], y_pred[:, j]
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            r, _ = pearsonr(yt, yp)
            if not np.isnan(r):
                rs.append(r)
                cccs.append(concordance_correlation_coefficient(yt, yp))

    print(f"\n{'='*50}")
    print(f"RESULT: {label}")
    print(f"{'='*50}")
    print(f"Valid proteins: {len(rs)}/{y_pred.shape[1]}")
    print(f"Mean per-protein R:   {np.mean(rs):.4f}")
    print(f"Mean per-protein CCC: {np.mean(cccs):.4f}")

    pooled_pred = y_pred.flatten()
    pooled_truth = y_true.flatten()
    pooled_ccc = concordance_correlation_coefficient(pooled_truth, pooled_pred)
    pooled_cosine = np.dot(pooled_truth, pooled_pred) / (
        np.linalg.norm(pooled_truth) * np.linalg.norm(pooled_pred) + 1e-12)
    print(f"Pooled CCC:              {pooled_ccc:.4f}")
    print(f"Pooled cosine similarity: {pooled_cosine:.4f}")

    print(f"\nAdd to table: pooled CCC={pooled_ccc:.3f}, pooled cosine={pooled_cosine:.3f}, "
          f"per-protein CCC={np.mean(cccs):.3f}, per-protein R={np.mean(rs):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--truth_path", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    args = parser.parse_args()
    main(args.pred_path, args.truth_path, args.label)