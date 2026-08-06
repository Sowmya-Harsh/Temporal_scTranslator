"""
18_aggregate_kfold_gated.py

Purpose: After running the 5-fold SLURM array job (slurm_kfold_gated.sh),
aggregate each fold's joint_results_summary.csv into one kfold_summary.csv
with mean +/- std per timepoint pair, matching the exact format already
used for your FiLM-pre headline result (kfold_final_deg1000_filmpre/
kfold_summary.csv), so the two are directly comparable.

Usage:
    python 18_aggregate_kfold_gated.py \
        --kfold_dir run_results/kfold_gated_corr_ccc \
        --n_folds 5 \
        --out kfold_summary.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def main(kfold_dir, n_folds, out_path):
    kfold_dir = Path(kfold_dir)
    all_folds = []

    for fold in range(1, n_folds + 1):
        csv_path = kfold_dir / f"fold_{fold}" / "joint_results_summary.csv"
        if not csv_path.exists():
            print(f"  [warn] {csv_path} not found -- skipping fold {fold}")
            continue
        df = pd.read_csv(csv_path)
        df["fold"] = fold
        all_folds.append(df)
        print(f"  Loaded fold {fold}: {len(df)} pairs")

    if not all_folds:
        raise FileNotFoundError(f"No fold results found under {kfold_dir}. "
                                 f"Check that slurm_kfold_gated.sh has finished running.")

    combined = pd.concat(all_folds, ignore_index=True)

    summary = combined.groupby(["t_src", "t_tgt"]).agg(
        n_folds=("fold", "count"),
        mean_pearson_r_avg=("mean_pearson_r", "mean"),
        mean_pearson_r_std=("mean_pearson_r", "std"),
        ccc_avg=("ccc", "mean"),
        ccc_std=("ccc", "std"),
        mse_avg=("mse", "mean"),
        mse_std=("mse", "std"),
    ).reset_index()

    summary = summary.sort_values(["t_src", "t_tgt"])
    summary.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("K-FOLD SUMMARY (mean +/- std across folds)")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    print(f"\nOverall mean R across all pairs:   {summary['mean_pearson_r_avg'].mean():.4f}")
    print(f"Overall mean CCC across all pairs: {summary['ccc_avg'].mean():.4f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--out", type=str, default="kfold_summary.csv")
    args = parser.parse_args()
    main(args.kfold_dir, args.n_folds, args.out)