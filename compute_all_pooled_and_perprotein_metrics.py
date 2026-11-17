"""
compute_all_pooled_and_perprotein_metrics.py

Purpose: Single script to compute pooled AND per-protein Pearson R, CCC,
and cosine similarity, for ANY saved y_pred.csv / y_truth.csv pair. Run
this once per row you need for Table 5.1 (native scTranslator QuRIE-seq
fine-tune, native scTranslator CITE-seq fine-tune, this thesis's own
final model), so every row in the table is computed with the exact same
script and convention, rather than mixing self-reported numbers with
independently-recomputed ones inconsistently.

This extends verify_pooled_vs_perprotein.py by ALSO reporting pooled and
per-protein PEARSON R (not just CCC and cosine similarity), since R was
previously missing from the Table 5.1 comparison entirely.

Two pooling conventions are both computed and clearly labelled, since
they are NOT mathematically equivalent for a nonlinear statistic:
  1. "whole-set pooled": every cell x every protein in the ENTIRE input
     file is flattened into one long vector, and ONE correlation is
     computed over that whole vector.
  2. "batch-averaged pooled": the input is split into fixed-size
     minibatches (matching the native scTranslator pipeline's own
     evaluation loop, see utils.py's test() function), a correlation is
     computed on each batch's flattened cell x protein values, and the
     per-batch values are averaged. This is what the native scTranslator
     pipeline's own self-reported "ccc" actually computes -- NOT a
     single whole-set pooled statistic.
Reporting both side by side is what lets Table 5.1 explain why a
self-reported number and an independently-recomputed "pooled" number for
the same underlying predictions can differ (e.g. 0.937 vs 0.969):
they are two different pooling conventions, not the same calculation
computed twice.

Usage:
    python compute_all_pooled_and_perprotein_metrics.py \
        --pred_path y_pred.csv \
        --truth_path y_truth.csv \
        --label "QuRIE-seq, native scTranslator fine-tune" \
        --batch_size 8 \
        --outdir results/table_5_1_metrics

Run once per row needed, e.g.:
    python compute_all_pooled_and_perprotein_metrics.py --pred_path sctranslator_qurie_finetune_pred.csv --truth_path sctranslator_qurie_finetune_truth.csv --label "QuRIE-seq_native_finetune" --batch_size 8 --outdir results/table_5_1_metrics
    python compute_all_pooled_and_perprotein_metrics.py --pred_path sctranslator_cite_finetune_pred.csv --truth_path sctranslator_cite_finetune_truth.csv --label "CITEseq_native_finetune" --batch_size 8 --outdir results/table_5_1_metrics
    python compute_all_pooled_and_perprotein_metrics.py --pred_path final_model_pred.csv --truth_path final_model_truth.csv --label "This_work_final_model" --batch_size 32 --outdir results/table_5_1_metrics

After running for every row, all results are also appended to a single
combined CSV (table_5_1_all_rows.csv) in --outdir, so the full table can
be assembled directly from one file.
"""
import argparse
import os

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

    for df, name in [(pred, "pred"), (truth, "truth")]:
        first_col = df.columns[0]
        if df[first_col].dtype == object or "id" in first_col.lower() or "barcode" in first_col.lower():
            print(f"Dropping likely index column '{first_col}' from {name}")
            df.drop(columns=[first_col], inplace=True)

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


def batch_averaged_pooled(pred_arr, truth_arr, batch_size):
    """
    Replicates the native scTranslator pipeline's own evaluation
    convention (utils.py's test() function): split into fixed-size
    minibatches (by ROW / cell, matching how their DataLoader batches),
    flatten each batch's cell x protein values, compute one metric per
    batch, then average across batches. NOT the same as a single
    whole-set pooled statistic -- see module docstring.
    """
    n = pred_arr.shape[0]
    r_vals, ccc_vals, cos_vals = [], [], []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        p_batch = pred_arr[start:end].flatten()
        t_batch = truth_arr[start:end].flatten()
        if len(p_batch) < 2:
            continue
        r_vals.append(pearson_r(t_batch, p_batch))
        ccc_vals.append(ccc(t_batch, p_batch))
        cos_vals.append(cosine_similarity(t_batch, p_batch))
    return (
        float(np.nanmean(r_vals)) if r_vals else float("nan"),
        float(np.nanmean(ccc_vals)) if ccc_vals else float("nan"),
        float(np.nanmean(cos_vals)) if cos_vals else float("nan"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--truth_path", type=str, required=True)
    parser.add_argument("--label", type=str, required=True,
                         help="Row label for this run, e.g. 'QuRIE-seq_native_finetune'.")
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Batch size to replicate for the 'batch-averaged pooled' "
                              "convention. Use 8 to match the native scTranslator "
                              "pipeline's own default batch size when reproducing their "
                              "self-reported number; use whatever batch size is relevant "
                              "for other rows (e.g. this thesis's own model).")
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pred, truth = load_and_align(args.pred_path, args.truth_path)
    print(f"\n[{args.label}] Aligned shape: {pred.shape[0]} cells x {pred.shape[1]} protein channels\n")

    pred_arr = pred.values.astype(float)
    truth_arr = truth.values.astype(float)

    # --- Convention 1: whole-set pooled (entire file flattened into one vector) ---
    pred_flat = pred_arr.flatten()
    truth_flat = truth_arr.flatten()
    whole_pooled_r = pearson_r(truth_flat, pred_flat)
    whole_pooled_ccc = ccc(truth_flat, pred_flat)
    whole_pooled_cosine = cosine_similarity(truth_flat, pred_flat)

    # --- Convention 2: batch-averaged pooled (matches native scTranslator eval loop) ---
    batch_r, batch_ccc, batch_cosine = batch_averaged_pooled(pred_arr, truth_arr, args.batch_size)

    # --- Per-protein: computed separately per column, then averaged ---
    per_protein_r, per_protein_ccc, per_protein_names = [], [], []
    for col in pred.columns:
        per_protein_r.append(pearson_r(truth[col].values, pred[col].values))
        per_protein_ccc.append(ccc(truth[col].values, pred[col].values))
        per_protein_names.append(col)

    mean_per_protein_r = np.nanmean(per_protein_r)
    mean_per_protein_ccc = np.nanmean(per_protein_ccc)
    n_nan_r = int(np.sum(np.isnan(per_protein_r)))
    n_nan_ccc = int(np.sum(np.isnan(per_protein_ccc)))

    print("=== WHOLE-SET POOLED (entire file flattened, one statistic) ===")
    print(f"Pooled R:        {whole_pooled_r:.4f}")
    print(f"Pooled CCC:      {whole_pooled_ccc:.4f}")
    print(f"Pooled cosine:   {whole_pooled_cosine:.4f}")

    print(f"\n=== BATCH-AVERAGED POOLED (batch_size={args.batch_size}, matches native scTranslator eval loop) ===")
    print(f"Batch-avg R:     {batch_r:.4f}")
    print(f"Batch-avg CCC:   {batch_ccc:.4f}")
    print(f"Batch-avg cosine:{batch_cosine:.4f}")

    print(f"\n=== PER-PROTEIN (computed per channel, then averaged) ===")
    print(f"Mean per-protein R:   {mean_per_protein_r:.4f}  ({n_nan_r} channels excluded as undefined)")
    print(f"Mean per-protein CCC: {mean_per_protein_ccc:.4f}  ({n_nan_ccc} channels excluded as undefined)")

    # --- Save per-protein breakdown for this row ---
    per_protein_df = pd.DataFrame({
        "protein": per_protein_names, "R": per_protein_r, "CCC": per_protein_ccc,
    })
    per_protein_path = os.path.join(args.outdir, f"per_protein_breakdown_{args.label}.csv")
    per_protein_df.to_csv(per_protein_path, index=False)
    print(f"\nSaved per-protein breakdown to {per_protein_path}")

    # --- Append this row's summary to the combined table CSV ---
    combined_path = os.path.join(args.outdir, "table_5_1_all_rows.csv")
    row = {
        "label": args.label,
        "n_cells": pred.shape[0], "n_proteins": pred.shape[1],
        "whole_pooled_R": whole_pooled_r, "whole_pooled_CCC": whole_pooled_ccc,
        "whole_pooled_cosine": whole_pooled_cosine,
        "batch_avg_R": batch_r, "batch_avg_CCC": batch_ccc, "batch_avg_cosine": batch_cosine,
        "batch_size_used": args.batch_size,
        "per_protein_mean_R": mean_per_protein_r, "per_protein_mean_CCC": mean_per_protein_ccc,
    }
    if os.path.exists(combined_path):
        combined_df = pd.read_csv(combined_path)
        combined_df = combined_df[combined_df["label"] != args.label]  # replace if rerun
        combined_df = pd.concat([combined_df, pd.DataFrame([row])], ignore_index=True)
    else:
        combined_df = pd.DataFrame([row])
    combined_df.to_csv(combined_path, index=False)
    print(f"Appended this row to combined table: {combined_path}")

    print("\n=== SUMMARY FOR THIS ROW ===")
    for k, v in row.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
