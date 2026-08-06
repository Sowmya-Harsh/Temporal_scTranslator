"""
10_check_collapse.py

Purpose: Reusable diagnostic for scTranslator's saved y_pred.csv /
y_truth.csv outputs (from stage3_inference_without_finetune.py or
stage3_fine-tune.py). Checks for mean-collapse and, more importantly,
whether predictions carry any real per-protein correlation with truth
-- distinguishing "safe but useless" (flat, near-zero correlation)
from "actively wrong" (compressed variance AND negative correlation,
often a sign of overfitting on too few training samples) from
"working as intended" (reasonable variance, positive correlation).

Usage:
    python 10_check_collapse.py \
        --pred result/test/new_data/y_pred.csv \
        --truth result/test/new_data/y_truth.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


def concordance_correlation_coefficient(y_true, y_pred):
    """Lin's CCC -- the REAL concordance correlation coefficient, not to be
    confused with cosine similarity (which scTranslator's own code calls
    'ccc' but is a different, much more lenient metric -- see discussion).
    Penalizes scale/offset mismatch that Pearson r and cosine similarity
    are both blind to."""
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return 0.0 if denom == 0 else float((2 * covariance) / denom)


def cosine_similarity_per_row(y_true, y_pred, eps=1e-8):
    """What scTranslator's own train()/test() actually compute and call
    'ccc' -- raw cosine similarity between prediction and truth vectors,
    NOT Lin's CCC. Computed here for direct side-by-side comparison."""
    num = (y_true * y_pred).sum(axis=1)
    denom = (np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)) + eps
    return num / denom


def load_preds_truth(pred_path, truth_path, protein_names_path=None):
    if pred_path.endswith(".npy"):
        pred_vals = np.load(pred_path)
        truth_vals = np.load(truth_path)
        if protein_names_path and os.path.exists(protein_names_path):
            protein_names = [l.strip() for l in open(protein_names_path)]
        else:
            protein_names = [f"protein_{i}" for i in range(pred_vals.shape[1])]
        return pred_vals, truth_vals, protein_names
    else:
        pred = pd.read_csv(pred_path, index_col=0)
        truth = pd.read_csv(truth_path, index_col=0)
        return pred.values, truth.values, pred.columns.tolist()


def main(pred_path, truth_path, protein_names_path=None):
    pred_vals, truth_vals, protein_names = load_preds_truth(pred_path, truth_path, protein_names_path)

    print(f"Pred shape: {pred_vals.shape}   Truth shape: {truth_vals.shape}")
    if pred_vals.shape != truth_vals.shape:
        print("WARNING: shape mismatch between pred and truth files!")

    if pred_vals.shape[0] < 30:
        print(f"\n*** CAUTION: only {pred_vals.shape[0]} samples. Correlation and "
              f"variance estimates below are UNSTABLE with this few samples "
              f"-- treat as a rough signal to investigate further, not a "
              f"reliable conclusion. Aim for a larger held-out test set "
              f"before drawing firm conclusions. ***\n")

    pred_std = pred_vals.std(axis=0)
    truth_std = truth_vals.std(axis=0)

    print(f"\nMean std across proteins -- pred: {pred_std.mean():.4f}   "
          f"truth: {truth_std.mean():.4f}")
    print(f"Variance compression ratio (pred_std/truth_std, overall): "
          f"{pred_std.mean() / (truth_std.mean() + 1e-8):.3f}")

    per_protein_r = []
    per_protein_names_used = []
    for i, name in enumerate(protein_names):
        if truth_std[i] > 1e-6 and pred_std[i] > 1e-6:
            r = np.corrcoef(pred_vals[:, i], truth_vals[:, i])[0, 1]
            per_protein_r.append(r)
            per_protein_names_used.append(name)

    per_protein_r = np.array(per_protein_r)
    print(f"\nPer-protein Pearson r (n={len(per_protein_r)} proteins with variance):")
    print(f"  mean={per_protein_r.mean():.4f}  median={np.median(per_protein_r):.4f}  "
          f"min={per_protein_r.min():.4f}  max={per_protein_r.max():.4f}")

    n_positive = (per_protein_r > 0.1).sum()
    n_negative = (per_protein_r < -0.1).sum()
    n_near_zero = len(per_protein_r) - n_positive - n_negative
    print(f"\nProteins with r > 0.1 (meaningfully positive): {n_positive}")
    print(f"Proteins with r < -0.1 (meaningfully negative): {n_negative}")
    print(f"Proteins with |r| <= 0.1 (near zero / flat):    {n_near_zero}")

    # --- REAL metrics: per-protein CCC (correct) vs pooled/flattened CCC
    #     and cosine similarity (both misleading -- see discussion) ---
    per_protein_ccc = []
    for i in range(truth_vals.shape[1]):
        if np.std(truth_vals[:, i]) > 1e-8 and np.std(pred_vals[:, i]) > 1e-8:
            c = concordance_correlation_coefficient(truth_vals[:, i], pred_vals[:, i])
            per_protein_ccc.append(c)
    mean_per_protein_ccc = float(np.mean(per_protein_ccc)) if per_protein_ccc else float("nan")

    pooled_ccc = concordance_correlation_coefficient(truth_vals.flatten(), pred_vals.flatten())
    cos_sim_per_cell = cosine_similarity_per_row(truth_vals, pred_vals)
    mean_cos_sim = float(np.nanmean(cos_sim_per_cell))

    print("\n" + "=" * 60)
    print("PER-PROTEIN CCC (correct) vs POOLED CCC vs COSINE SIMILARITY")
    print("=" * 60)
    print(f"  Mean per-protein CCC (CORRECT metric):  {mean_per_protein_ccc:.4f}")
    print(f"  Pooled/flattened CCC (misleading):       {pooled_ccc:.4f}")
    print(f"  Mean per-cell cosine similarity (scTranslator's 'ccc', also misleading): {mean_cos_sim:.4f}")
    print(f"\n  NOTE: pooled CCC and cosine similarity are both inflated here by")
    print(f"  BETWEEN-protein structure (which channels are generally high vs low")
    print(f"  after per-cell normalization) rather than WITHIN-protein, cell-to-cell")
    print(f"  predictive accuracy. Per-protein CCC (averaged) is the metric that")
    print(f"  actually reflects whether the model ranks cells correctly for each")
    print(f"  protein -- use THIS number, not the pooled ones, for real reporting.")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    var_ratio = pred_std.mean() / (truth_std.mean() + 1e-8)
    mean_r = per_protein_r.mean()

    if var_ratio < 0.3 and abs(mean_r) < 0.1:
        print("Pattern: TRUE MEAN COLLAPSE. Low prediction variance AND near-zero "
              "correlation -- model is predicting close to a constant regardless "
              "of input. Likely weak/absent usable signal from RNA, or the model "
              "found the 'safe' MSE-minimizing constant.")
    elif var_ratio < 0.6 and mean_r < -0.05:
        print("Pattern: COMPRESSED + MISCORRELATED. Variance is reduced AND "
              "correlation is negative/near-zero -- NOT simple mean collapse. "
              "More consistent with overfitting on too few training samples "
              "(model fit noise in a small training set, generalizes poorly), "
              "or a genuine data/pipeline issue upstream (bad OT targets, "
              "label leakage, etc). Investigate training set size and OT "
              "target quality before concluding the architecture can't work.")
    elif mean_r > 0.2:
        print("Pattern: WORKING AS INTENDED. Reasonable variance and positive "
              "mean correlation -- model appears to be learning real signal.")
    else:
        print("Pattern: AMBIGUOUS / WEAK SIGNAL. Neither clearly collapsed nor "
              "clearly working. Consider more test samples, and check the "
              "shuffle_rna / shuffle_time controls to isolate the cause.")

    print(f"\nPer-protein correlations (sorted, worst to best):")
    order = np.argsort(per_protein_r)
    for idx in order:
        print(f"  {per_protein_names_used[idx]:<20} r={per_protein_r[idx]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help=".csv or .npy")
    parser.add_argument("--truth", required=True, help=".csv or .npy")
    parser.add_argument("--protein_names", default=None,
                         help="optional text file, one protein name per line "
                              "(only used when --pred/--truth are .npy)")
    args = parser.parse_args()
    main(args.pred, args.truth, args.protein_names)