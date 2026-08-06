"""
11_per_protein_bar_charts.py

Purpose: Load saved per-pair prediction/truth CSVs (format confirmed from
your actual output: columns = t_src, t_tgt, <protein names...>, rows =
cells), compute per-protein R and CCC (matching Section 4.8.1's
per-protein-averaged convention), and produce ranked bar charts of the
highest-predicted proteins.

Works for both:
  - In-distribution results (pass --pairs to average across specific
    timepoint pairs, or --pool_all to pool every pair together)
  - Zero-shot ibrutinib results (point --run_dir at wherever that eval
    saved its own y_pred/y_true CSVs)

Expects, for each pair, TWO files in run_dir:
    joint_y_pred_t{src}to{tgt}.csv
    joint_y_true_t{src}to{tgt}.csv
(adjust TRUE_FILE_PATTERN below if your true-value file uses a different
name -- I'm inferring it from the y_pred naming convention you showed me,
not confirmed directly.)

Usage:
    # Average per-protein R/CCC across ALL pairs in a run directory:
    python 11_per_protein_bar_charts.py \
        --run_dir run_results/correlationlosswt_test_deg1000_filmpre \
        --pool_all \
        --top_n 15 \
        --out_prefix final_model

    # Just one pair (e.g. zero-shot, which likely has only one file):
    python 11_per_protein_bar_charts.py \
        --run_dir run_results/zeroshot_ibrutinib \
        --pairs 6:180 \
        --top_n 15 \
        --out_prefix zeroshot_ibrutinib
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

TRUE_FILE_PATTERN = "joint_y_truth_t{src}to{tgt}.csv"
PRED_FILE_PATTERN = "joint_y_pred_t{src}to{tgt}.csv"
# Zero-shot runs (e.g. zero_shot_ibrutinib/) use a DIFFERENT, non-per-pair
# naming convention with no t{src}to{tgt} suffix, e.g.:
#   zeroshot_ibru_y_pred.csv, zeroshot_ibru_y_truth.csv
# Use --single_file_prefix instead of --pool_all/--pairs for these.


def per_protein_r_ccc(y_true: pd.DataFrame, y_pred: pd.DataFrame, protein_cols):
    rows = []
    for p in protein_cols:
        yt, yp = y_true[p].values, y_pred[p].values
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            rows.append({"protein": p, "R": np.nan, "CCC": np.nan})
            continue
        r, _ = pearsonr(yt, yp)
        mean_t, mean_p = yt.mean(), yp.mean()
        var_t, var_p = yt.var(), yp.var()
        cov = np.mean((yt - mean_t) * (yp - mean_p))
        ccc = (2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12)
        rows.append({"protein": p, "R": r, "CCC": ccc})
    return pd.DataFrame(rows)


def discover_pairs(run_dir: Path):
    pred_files = sorted(glob.glob(str(run_dir / "joint_y_pred_t*to*.csv")))
    pairs = []
    for f in pred_files:
        m = re.search(r"joint_y_pred_t(\d+)to(\d+)\.csv", f)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def main(run_dir, run_dirs_multi, pairs_arg, pool_all, single_file_prefix, top_n, out_prefix, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_dirs_multi:
        # Multi-fold mode: compute per-protein R/CCC within each fold directory
        # separately (pooling all pairs within that fold), then average ACROSS
        # folds -- matches the same aggregation logic as kfold_summary.csv's
        # mean_pearson_r_avg, applied at the per-protein level instead of the
        # per-pair level.
        fold_results = []
        for fold_dir in run_dirs_multi:
            fold_dir = Path(fold_dir)
            pairs = discover_pairs(fold_dir)
            if not pairs:
                print(f"  [warn] no pairs found in {fold_dir}, skipping this fold")
                continue
            fold_all = []
            for t_src, t_tgt in pairs:
                pred_path = fold_dir / PRED_FILE_PATTERN.format(src=t_src, tgt=t_tgt)
                true_path = fold_dir / TRUE_FILE_PATTERN.format(src=t_src, tgt=t_tgt)
                if not pred_path.exists() or not true_path.exists():
                    continue
                y_pred = pd.read_csv(pred_path)
                y_true = pd.read_csv(true_path)
                protein_cols = [c for c in y_pred.columns if c not in ("t_src", "t_tgt")]
                res = per_protein_r_ccc(y_true, y_pred, protein_cols)
                fold_all.append(res)
            if not fold_all:
                print(f"  [warn] no usable pairs in {fold_dir}, skipping this fold")
                continue
            fold_combined = pd.concat(fold_all, ignore_index=True)
            fold_avg = fold_combined.groupby("protein")[["R", "CCC"]].mean().reset_index()
            fold_avg["fold"] = fold_dir.name
            fold_results.append(fold_avg)
            print(f"  {fold_dir.name}: {len(pairs)} pairs pooled")

        if not fold_results:
            raise ValueError("No usable fold directories found -- check --run_dirs paths and that "
                              "14_kfold_export_predictions.py has been run for each fold first.")

        all_folds = pd.concat(fold_results, ignore_index=True)
        avg = all_folds.groupby("protein")[["R", "CCC"]].mean().reset_index()
        avg = avg.sort_values("R", ascending=False)
        print(f"\nAveraged per-protein R/CCC across {len(fold_results)} folds.")

        csv_out = out_dir / f"{out_prefix}_per_protein_R_CCC.csv"
        avg.to_csv(csv_out, index=False)
        print(f"Saved full per-protein table to {csv_out}")

        top = avg.head(top_n)
        print(f"\nTop {top_n} proteins by mean R (averaged across folds):")
        print(top.to_string(index=False))

        _plot_bar_chart(top, top_n, out_prefix, out_dir)
        return

    run_dir = Path(run_dir)

    if single_file_prefix:
        # Zero-shot-style: single pred/truth file, no per-pair suffix,
        # e.g. zeroshot_ibru_y_pred.csv / zeroshot_ibru_y_truth.csv
        pred_path = run_dir / f"{single_file_prefix}_y_pred.csv"
        true_path = run_dir / f"{single_file_prefix}_y_truth.csv"
        if not pred_path.exists() or not true_path.exists():
            raise FileNotFoundError(
                f"Expected {pred_path} and {true_path}. Check the actual filenames in {run_dir} "
                f"(e.g. `ls {run_dir}`) and adjust --single_file_prefix to match."
            )
        y_pred = pd.read_csv(pred_path)
        y_true = pd.read_csv(true_path)
        protein_cols = [c for c in y_pred.columns if c not in ("t_src", "t_tgt")]
        res = per_protein_r_ccc(y_true, y_pred, protein_cols)
        all_results = [res]
        print(f"  loaded {len(y_pred)} cells, {len(protein_cols)} proteins from {pred_path.name}")
    else:
        if pool_all:
            pairs = discover_pairs(run_dir)
            print(f"Discovered pairs: {pairs}")
        else:
            pairs = [tuple(int(x) for x in p.split(":")) for p in pairs_arg]

        if not pairs:
            raise ValueError(f"No pairs found/specified in {run_dir}. "
                              f"Check TRUE_FILE_PATTERN/PRED_FILE_PATTERN match your actual filenames "
                              f"(run `ls {run_dir}` first to confirm).")

        all_results = []
        for t_src, t_tgt in pairs:
            pred_path = run_dir / PRED_FILE_PATTERN.format(src=t_src, tgt=t_tgt)
            true_path = run_dir / TRUE_FILE_PATTERN.format(src=t_src, tgt=t_tgt)
            if not pred_path.exists():
                print(f"  [skip] {pred_path} not found")
                continue
            if not true_path.exists():
                print(f"  [skip] {true_path} not found -- run `ls {run_dir}` to confirm actual naming")
                continue

            y_pred = pd.read_csv(pred_path)
            y_true = pd.read_csv(true_path)
            protein_cols = [c for c in y_pred.columns if c not in ("t_src", "t_tgt")]

            res = per_protein_r_ccc(y_true, y_pred, protein_cols)
            res["t_src"], res["t_tgt"] = t_src, t_tgt
            all_results.append(res)
            print(f"  {t_src}->{t_tgt}: loaded {len(y_pred)} cells, {len(protein_cols)} proteins")

    combined = pd.concat(all_results, ignore_index=True)
    avg = combined.groupby("protein")[["R", "CCC"]].mean().reset_index()
    avg = avg.sort_values("R", ascending=False)

    csv_out = out_dir / f"{out_prefix}_per_protein_R_CCC.csv"
    avg.to_csv(csv_out, index=False)
    print(f"\nSaved full per-protein table to {csv_out}")

    top = avg.head(top_n)
    print(f"\nTop {top_n} proteins by mean R:")
    print(top.to_string(index=False))

    _plot_bar_chart(top, top_n, out_prefix, out_dir)


def _plot_bar_chart(top, top_n, out_prefix, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.35 * top_n)))
    for ax, metric, color in zip(axes, ["R", "CCC"], ["tab:blue", "tab:orange"]):
        sorted_top = top.sort_values(metric, ascending=True)
        ax.barh(sorted_top["protein"], sorted_top[metric], color=color)
        ax.set_xlabel(f"Mean {metric}", color="black")
        ax.set_title(f"Top {top_n} proteins by {metric}", color="black")
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")

    fig.suptitle(f"Highest-predicted proteins: {out_prefix}", color="black", fontsize=13)
    fig.tight_layout()
    png_out = out_dir / f"{out_prefix}_top_proteins.png"
    fig.savefig(png_out, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {png_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None,
                         help="Single run directory (ignored if --run_dirs is given)")
    parser.add_argument("--run_dirs", type=str, nargs="+", default=None,
                         help="Multiple fold directories to average per-protein R/CCC ACROSS "
                              "(e.g. --run_dirs run_results/kfold_final_deg1000_filmpre/fold_1 "
                              "... fold_5), matching how kfold_summary.csv's headline metric is "
                              "computed. Use this instead of --run_dir for k-fold results.")
    parser.add_argument("--pairs", type=str, nargs="*", default=[],
                         help="e.g. --pairs 0:2 2:6 6:180 (ignored if --pool_all or --run_dirs)")
    parser.add_argument("--pool_all", action="store_true",
                         help="Auto-discover and average across ALL pairs found in run_dir")
    parser.add_argument("--single_file_prefix", type=str, default=None,
                         help="For zero-shot-style runs with a single pred/truth file pair, "
                              "e.g. --single_file_prefix zeroshot_ibru (expects "
                              "zeroshot_ibru_y_pred.csv / zeroshot_ibru_y_truth.csv in run_dir)")
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--out_prefix", type=str, default="model")
    parser.add_argument("--out_dir", type=str, default="protein_ranking_figures")
    args = parser.parse_args()
    if not args.run_dir and not args.run_dirs:
        parser.error("Must specify either --run_dir or --run_dirs")
    main(args.run_dir, args.run_dirs, args.pairs, args.pool_all, args.single_file_prefix,
         args.top_n, args.out_prefix, args.out_dir)