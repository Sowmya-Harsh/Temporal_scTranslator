"""
29_verify_pooled_artefact_table.py

Purpose: Directly verify Table 5.1's row "This work, final model":
  pooled CCC = 0.994, pooled cosine similarity = 0.995, per-protein CCC = 0.206.
This computes all three quantities from the SAME actual saved predictions,
so the table can be confirmed or corrected with real numbers rather than
trusted blindly.

--- FIXES IN THIS VERSION ---
The original version of this script compared the wrong pair of numbers:
its final printout compared the computed per-protein mean R against a
placeholder "0.319", which is not a value that appears anywhere in
Table 5.1 at all (Table 5.1's third column is per-protein CCC, not R;
0.319/0.321 is the final model's mean per-protein R from a DIFFERENT
table, Table 5.6). This meant the script's "verification" was checking
a number against a value it was never claiming to match, while never
actually printing the one comparison that matters for THIS table:
computed per-protein CCC vs. the table's reported 0.206.

This version:
  1. Compares CCC against CCC and cosine against cosine (not R against
     CCC), against the table's ACTUAL reported values (0.994, 0.995,
     0.206), not generic 1.000/1.000/0.319 placeholders.
  2. Reports per-protein mean R separately, clearly labelled as
     belonging to a different table (Table 5.6), not compared against
     anything in Table 5.1.
  3. Makes the pooling convention explicit: predictions and truths are
     flattened and concatenated across ALL ten timepoint pairs before
     computing the pooled metrics, i.e. one single pooled figure across
     the entire dataset, not ten per-pair pooled figures averaged
     together. If Table 5.1's original 0.994/0.995 were computed under
     a different pooling convention (e.g. per-pair pooled, then
     averaged across pairs), this script's numbers will not match even
     though the underlying predictions are identical -- check this
     explicitly if a mismatch appears before assuming an error in the
     table.

Usage:
    python 29_verify_pooled_artefact_table.py \
        --run_dir run_results/correlationlosswt_test_deg1000_filmpre
"""
import argparse
import glob

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# Table 5.1's actual reported values for "This work, final model"
# (Section~\ref{sec:results-pooled-artefact}), hardcoded here so the
# comparison printout checks against the real numbers, not placeholders.
TABLE_POOLED_CCC = 0.994
TABLE_POOLED_COSINE = 0.995
TABLE_PER_PROTEIN_CCC = 0.206

# For reference only -- NOT part of Table 5.1, belongs to Table 5.6
# (tab:final-per-pair), included here so the two aren't accidentally
# conflated when reading the printout.
TABLE_5_6_PER_PROTEIN_R_SINGLE_SPLIT = 0.321
TABLE_5_6_PER_PROTEIN_R_5FOLD = 0.292


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

    # Pooled across ALL ten timepoint pairs simultaneously -- see note
    # in the module docstring on why this specific pooling convention
    # matters and should be checked against how Table 5.1 was produced.
    pooled_pred = np.concatenate(all_pred)
    pooled_truth = np.concatenate(all_truth)

    pooled_r, _ = pearsonr(pooled_truth, pooled_pred)
    pooled_ccc = concordance_correlation_coefficient(pooled_truth, pooled_pred)
    pooled_cosine = np.dot(pooled_truth, pooled_pred) / (
        np.linalg.norm(pooled_truth) * np.linalg.norm(pooled_pred) + 1e-12)

    mean_per_protein_r = np.mean(per_protein_rs)
    mean_per_protein_ccc = np.mean(per_protein_cccs)

    print(f"{'='*60}")
    print("COMPUTED FROM SAVED PREDICTIONS: This work, final model")
    print(f"{'='*60}")
    print(f"Pooled Pearson R:         {pooled_r:.4f}   (not reported in Table 5.1)")
    print(f"Pooled CCC:               {pooled_ccc:.4f}")
    print(f"Pooled cosine similarity: {pooled_cosine:.4f}")
    print(f"Per-protein mean R:       {mean_per_protein_r:.4f}   (compare to Table 5.6, not Table 5.1 -- see below)")
    print(f"Per-protein mean CCC:     {mean_per_protein_ccc:.4f}")

    print(f"\n{'='*60}")
    print("COMPARISON TO TABLE 5.1 ROW: 'This work, final model'")
    print("(reported: pooled CCC=0.994, pooled cosine=0.995, per-protein CCC=0.206)")
    print(f"{'='*60}")

    def _check(name, table_val, actual_val, tol=0.01):
        gap = abs(table_val - actual_val)
        flag = "OK" if gap <= tol else f"*** MISMATCH (gap={gap:.4f} > tol={tol}) ***"
        print(f"  {name:<28} table {table_val:.3f}  |  actual {actual_val:.3f}   {flag}")

    _check("Pooled CCC:", TABLE_POOLED_CCC, pooled_ccc)
    _check("Pooled cosine similarity:", TABLE_POOLED_COSINE, pooled_cosine)
    _check("Per-protein CCC:", TABLE_PER_PROTEIN_CCC, mean_per_protein_ccc)

    print(f"\n  (For reference, NOT part of Table 5.1 -- Table 5.6 reports mean per-protein R "
          f"of {TABLE_5_6_PER_PROTEIN_R_SINGLE_SPLIT:.3f} single-split / "
          f"{TABLE_5_6_PER_PROTEIN_R_5FOLD:.3f} 5-fold. "
          f"This script's computed per-protein mean R above ({mean_per_protein_r:.3f}) "
          f"should be checked against THAT table if relevant, not against anything in Table 5.1.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.run_dir)