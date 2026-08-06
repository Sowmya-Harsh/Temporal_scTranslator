"""
combine_sweep_results.py

Purpose: After run_full_sweep.sh finishes, combine every run's
joint_results_summary.csv (one per gene_strategy/n_genes/model combo,
each containing per-timepoint-pair rows) into a single master CSV --
tagged with gene_strategy, n_genes, model as extra columns -- so you
can sort/filter/pivot the whole 36-run x 10-pair table at once.

Usage:
    python combine_sweep_results.py --dir run_results/full_sweep
"""

import argparse
import glob
import os
import re

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.dir, "master_sweep_results.csv")

    summary_files = glob.glob(os.path.join(args.dir, "*", "joint_results_summary.csv"))
    if not summary_files:
        print(f"No joint_results_summary.csv files found under {args.dir}/*/")
        return

    print(f"Found {len(summary_files)} run summaries.")
    all_rows = []
    for f in summary_files:
        tag = os.path.basename(os.path.dirname(f))
        m = re.match(r"(hvg|deg)_(\w+)_(mlp|film_pre|film_post)$", tag)
        if not m:
            print(f"WARNING: could not parse tag '{tag}' from path {f} -- skipping.")
            continue
        gene_strategy, n_genes, model = m.groups()

        df = pd.read_csv(f)
        df.insert(0, "gene_strategy", gene_strategy)
        df.insert(1, "n_genes", n_genes)
        df.insert(2, "model", model)
        all_rows.append(df)

    if not all_rows:
        print("No valid runs parsed. Check folder naming matches {gene_strategy}_{n_genes}_{model}.")
        return

    master = pd.concat(all_rows, ignore_index=True)
    master.to_csv(out_path, index=False)
    print(f"Saved master table ({len(master)} rows, {len(summary_files)} runs) to {out_path}")

    print("\n--- Top 10 rows by mean_pearson_r ---")
    if "mean_pearson_r" in master.columns:
        print(master.sort_values("mean_pearson_r", ascending=False).head(10).to_string(index=False))

    print("\n--- Best config per (t_src, t_tgt) pair, by mean_pearson_r ---")
    if {"t_src", "t_tgt", "mean_pearson_r"}.issubset(master.columns):
        best_per_pair = master.loc[master.groupby(["t_src", "t_tgt"])["mean_pearson_r"].idxmax()]
        print(best_per_pair[["t_src", "t_tgt", "gene_strategy", "n_genes", "model", "mean_pearson_r", "ccc"]]
              .to_string(index=False))

    print("\n--- Average mean_pearson_r per (gene_strategy, n_genes, model) config, across all pairs ---")
    if "mean_pearson_r" in master.columns:
        agg = master.groupby(["gene_strategy", "n_genes", "model"])["mean_pearson_r"].mean().sort_values(ascending=False)
        print(agg.to_string())


if __name__ == "__main__":
    main()