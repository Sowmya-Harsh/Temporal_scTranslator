"""
21_check_myeloid_truth_variance.py

Purpose: Precisely determine whether the myeloid y_truth CSV genuinely
has near-zero variance across cells (real mean-collapse) or just
appeared that way from limited-precision eyeballing. Reports actual
per-protein standard deviation, not just a visual impression.

Usage:
    python 21_check_myeloid_truth_variance.py \
        --truth_csv myeloid_finetune_results/joint_y_truth_t0to10080.csv
"""
import argparse

import pandas as pd
import numpy as np


def main(truth_csv):
    df = pd.read_csv(truth_csv)
    protein_cols = [c for c in df.columns if c not in ("t_src", "t_tgt")]
    print(f"Loaded {len(df)} cells, {len(protein_cols)} protein columns from {truth_csv}")

    stds = df[protein_cols].std()
    means = df[protein_cols].mean()

    print(f"\nPer-protein std across cells:")
    print(f"  Mean of stds: {stds.mean():.6f}")
    print(f"  Median of stds: {stds.median():.6f}")
    print(f"  Min std (most collapsed protein): {stds.min():.6f} ({stds.idxmin()})")
    print(f"  Max std (most varied protein): {stds.max():.6f} ({stds.idxmax()})")

    n_near_zero = (stds < 1e-4).sum()
    print(f"\nProteins with std < 1e-4 (genuinely ~constant across cells): {n_near_zero}/{len(protein_cols)}")

    print(f"\nFirst 5 cells, first 5 proteins (raw values, full precision):")
    print(df[protein_cols[:5]].head(5).to_string())

    if n_near_zero > len(protein_cols) * 0.5:
        print("\nVERDICT: Majority of proteins genuinely near-constant across cells -- "
              "real mean-collapse confirmed, not a display artefact. Root cause is NOT "
              "the OT ot_reg setting (already ruled out) -- check target construction "
              "logic itself, or whether the wrong/train-mean fallback path is being hit.")
    else:
        print("\nVERDICT: Real per-cell variance exists -- NOT genuine mean-collapse. "
              "Earlier visual impression was likely a precision/display issue, not a "
              "real problem with the data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth_csv", type=str, required=True)
    args = parser.parse_args()
    main(args.truth_csv)
