"""
compute_pooled_metrics.py
────────────────────────────
Recomputes POOLED CCC and pooled cosine similarity -- the same
methodology used by the native scTranslator pipeline, and the same
"pooled" convention this thesis's Table pooled-artefact demonstrates
as misleading -- directly from a model's saved per-pair prediction/
truth CSVs (joint_y_pred_t*.csv / joint_y_truth_t*.csv), produced by
evaluate_per_pair() in train_joint.py.

"Pooled" here means: flatten ALL cells across ALL timepoint pairs AND
ALL protein channels into one single long vector for predictions, and
one single long vector for ground truth, then compute ONE cosine
similarity and ONE CCC on those two flat vectors. This is exactly the
computation the pooled-metric artefact is about -- it is NOT the
per-protein-then-averaged convention used everywhere else in this
thesis.

Usage:
    python compute_pooled_metrics.py --dir new_results/final_model_corrloss_only_leakfix
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def concordance_correlation_coefficient(y_true, y_pred):
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return 0.0 if denom == 0 else float((2 * covariance) / denom)


def cosine_similarity(y_true, y_pred):
    num = np.dot(y_true, y_pred)
    denom = np.linalg.norm(y_true) * np.linalg.norm(y_pred)
    return 0.0 if denom == 0 else float(num / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                         help="Dir containing joint_y_pred_t*.csv / joint_y_truth_t*.csv "
                              "(produced by evaluate_per_pair() with outdir set)")
    args = parser.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.dir, "joint_y_pred_t*.csv")))
    if not pred_files:
        raise FileNotFoundError(f"No joint_y_pred_t*.csv files found in {args.dir}")

    all_pred, all_truth = [], []
    for pred_path in pred_files:
        truth_path = pred_path.replace("y_pred", "y_truth")
        if not os.path.exists(truth_path):
            print(f"  WARNING: no matching truth file for {pred_path}, skipping")
            continue

        pred_df = pd.read_csv(pred_path)
        truth_df = pd.read_csv(truth_path)

        # Drop the t_src/t_tgt metadata columns, keep only protein channel columns
        meta_cols = ["t_src", "t_tgt"]
        pred_vals = pred_df.drop(columns=[c for c in meta_cols if c in pred_df.columns]).values
        truth_vals = truth_df.drop(columns=[c for c in meta_cols if c in truth_df.columns]).values

        print(f"  {os.path.basename(pred_path)}: {pred_vals.shape[0]} cells x {pred_vals.shape[1]} proteins")
        all_pred.append(pred_vals.flatten())
        all_truth.append(truth_vals.flatten())

    pred_flat = np.concatenate(all_pred)
    truth_flat = np.concatenate(all_truth)
    print(f"\nTotal pooled vector length: {len(pred_flat)} (all cells x all proteins x all pairs)")

    pooled_ccc = concordance_correlation_coefficient(truth_flat, pred_flat)
    pooled_cos = cosine_similarity(truth_flat, pred_flat)
    pooled_r = float(np.corrcoef(truth_flat, pred_flat)[0, 1])

    print(f"\n{'='*50}")
    print(f"Pooled CCC:              {pooled_ccc:.4f}")
    print(f"Pooled cosine similarity: {pooled_cos:.4f}")
    print(f"Pooled Pearson R:        {pooled_r:.4f}  (for reference, not in the thesis table)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
