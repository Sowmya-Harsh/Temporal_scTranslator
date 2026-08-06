"""
29_verify_pooled_artefact_table.py

Purpose: Directly verify Table 5.1's claim for "This work, final model":
pooled CCC (and/or pooled cosine similarity) approx 1.000, versus the
real per-protein mean R approx 0.319. This was flagged earlier as an
unverified number -- this script computes both pooled and per-protein
metrics from the SAME actual saved predictions, so the table can be
confirmed or corrected with real numbers rather than trusted blindly.

Usage:
    python 29_verify_pooled_artefact_table.py \
        --run_dir run_results/correlationlosswt_test_deg1000_filmpre
"""
import argparse
import glob

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def concordance_correlation_coefficient(y_true, y_pred):
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    return (2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12)


def main(run_dir):
    pred_files = sorted(glob.glob(f"{run_dir}/joint_y_pred_t*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No joint_y_pred_t*.csv files found in {run_dir}")

    all_pred, all_truth = [], []
    per_protein_rs, per_protein_cccs = [], []

    for pred_path in pred_files:
        truth_path = pred_path.replace("y_pred", "y_truth")
        y_pred = pd.read_csv(pred_path)
        y_true = pd.read_csv(truth_path)
        protein_cols = [c for c in y_pred.columns if c not in ("t_src", "t_tgt")]

        for p in protein_cols:
            yt, yp = y_true[p].values, y_pred[p].values
            if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
                r, _ = pearsonr(yt, yp)
                if not np.isnan(r):
                    per_protein_rs.append(r)
                    per_protein_cccs.append(concordance_correlation_coefficient(yt, yp))

        all_pred.append(y_pred[protein_cols].values.flatten())
        all_truth.append(y_true[protein_cols].values.flatten())

    pooled_pred = np.concatenate(all_pred)
    pooled_truth = np.concatenate(all_truth)

    pooled_r, _ = pearsonr(pooled_truth, pooled_pred)
    pooled_ccc = concordance_correlation_coefficient(pooled_truth, pooled_pred)
    pooled_cosine = np.dot(pooled_truth, pooled_pred) / (
        np.linalg.norm(pooled_truth) * np.linalg.norm(pooled_pred) + 1e-12)

    mean_per_protein_r = np.mean(per_protein_rs)
    mean_per_protein_ccc = np.mean(per_protein_cccs)

    print(f"{'='*60}")
    print("VERIFICATION: This work, final model")
    print(f"{'='*60}")
    print(f"Pooled Pearson R:        {pooled_r:.4f}")
    print(f"Pooled CCC:              {pooled_ccc:.4f}")
    print(f"Pooled cosine similarity: {pooled_cosine:.4f}")
    print(f"\nPer-protein mean R:   {mean_per_protein_r:.4f}")
    print(f"Per-protein mean CCC: {mean_per_protein_ccc:.4f}")
    print(f"\n{'='*60}")
    print("COMPARISON TO TABLE 5.1 CLAIM ('1.000  1.000  0.319')")
    print(f"{'='*60}")
    print(f"  If table's first two columns are pooled R / pooled CCC:")
    print(f"    Table claims: 1.000, 1.000  |  Actual: {pooled_r:.3f}, {pooled_ccc:.3f}")
    print(f"  If table's first two columns are pooled cosine / pooled CCC:")
    print(f"    Table claims: 1.000, 1.000  |  Actual: {pooled_cosine:.3f}, {pooled_ccc:.3f}")
    print(f"  Third column (per-protein R): Table claims 0.319 | Actual: {mean_per_protein_r:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.run_dir)
