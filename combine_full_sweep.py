"""
combine_full_sweep.py
────────────────────────
Reads all 36 run_results/full_sweep/{gene_strategy}_{n_genes}_{model}/
joint_results_summary.csv files (each with val+test rows per pair,
post-leakage-fix), and produces:

  1. full_sweep_master.csv   -- every row from every config, tagged
     with gene_strategy/n_genes/model, both val and test.
  2. full_sweep_pooled.csv   -- one row per config (36 rows), pooled
     mean R/CCC over the 10 pairs, val and test as separate columns.
     This replaces Table gene-selection-comparison and the data
     behind Figure film-necessity-sweep.

Usage:
    python combine_full_sweep.py --dir run_results/full_sweep --out results/full_sweep_combined
"""
import argparse
import os
import glob
import re

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Base dir containing {tag}/joint_results_summary.csv")
    parser.add_argument("--out", required=True, help="Output dir for combined CSVs")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(args.dir, "*", "joint_results_summary.csv")))
    print(f"Found {len(csv_paths)} config result files under {args.dir}")

    all_rows = []
    pooled_rows = []
    for path in csv_paths:
        tag = os.path.basename(os.path.dirname(path))
        # tag format: {gene_strategy}_{n_genes}_{model}, e.g. "deg_1000_film_pre" or "hvg_all_mlp"
        m = re.match(r"^(hvg|deg)_(\d+|all)_(mlp|film_pre|film_post|gated|gated_post)$", tag)
        if not m:
            print(f"  WARNING: could not parse config from tag '{tag}', skipping")
            continue
        gene_strategy, n_genes, model = m.groups()

        df = pd.read_csv(path)
        df["gene_strategy"] = gene_strategy
        df["n_genes"] = n_genes
        df["model"] = model
        df["tag"] = tag
        all_rows.append(df)

        for split in ["val", "test"]:
            sub = df[df["split"] == split]
            if len(sub) == 0:
                continue
            pooled_rows.append({
                "tag": tag, "gene_strategy": gene_strategy, "n_genes": n_genes, "model": model,
                "split": split, "n_pairs": len(sub),
                "mean_pearson_r": sub["mean_pearson_r"].mean(),
                "mean_ccc": sub["ccc"].mean(),
                "mean_mse": sub["mse"].mean(),
            })

    master = pd.concat(all_rows, ignore_index=True)
    master_path = os.path.join(args.out, "full_sweep_master.csv")
    master.to_csv(master_path, index=False)
    print(f"Saved {len(master)} rows ({len(csv_paths)} configs x ~10 pairs x 2 splits) to {master_path}")

    pooled = pd.DataFrame(pooled_rows)
    pooled_path = os.path.join(args.out, "full_sweep_pooled.csv")
    pooled.to_csv(pooled_path, index=False)
    print(f"Saved {len(pooled)} pooled rows (config x split) to {pooled_path}")

    # --- Quick console summary: best configs, DEG vs HVG at matched sizes ---
    # Uses VALIDATION, not test -- config selection/comparison must never be
    # based on test data (see select_best_config.py for the full selection
    # workflow with val-vs-test discrepancy checking). This summary here is
    # just a quick eyeball view using the same correct split.
    val_pooled = pooled[pooled["split"] == "val"].copy()
    val_pooled = val_pooled.sort_values("mean_pearson_r", ascending=False)
    print(f"\n{'='*70}\nTOP 5 CONFIGS BY VALIDATION R (pooled over 10 pairs)\n{'='*70}")
    print(val_pooled[["tag", "mean_pearson_r", "mean_ccc"]].head(5).to_string(index=False))

    print(f"\n{'='*70}\nDEG vs HVG at matched panel size (FiLM-post, VALIDATION split)\n{'='*70}")
    fp = val_pooled[val_pooled["model"] == "film_post"]
    order = ["50", "100", "200", "500", "1000", "all"]
    fp_pivot = fp.pivot(index="n_genes", columns="gene_strategy", values="mean_pearson_r").reindex(order)
    print(fp_pivot.to_string())

    with open(os.path.join(args.out, "full_sweep_config.json"), "w") as f:
        import json
        json.dump({"n_configs": len(csv_paths), "source_dir": args.dir}, f, indent=2)

    print(f"\nAll outputs saved to {args.out}/")


if __name__ == "__main__":
    main()

# --- ADDENDUM: val-based selection (run separately, appended below) ---
