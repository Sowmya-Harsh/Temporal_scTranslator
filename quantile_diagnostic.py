"""
quantile_diagnostic.py
───────────────────────────
Diagnostic: for each protein, split cells into quantile bins by TRUE
abundance (e.g. 4 bins: lowest 25%, 25-50%, 50-75%, highest 25%), and
compute R separately WITHIN each bin.

Purpose: a single overall R per protein can hide WHERE the model is
failing. Two proteins can have the same overall R for very different
reasons:
  - uniformly noisy across the whole range, or
  - accurate in the middle of the range but failing specifically at
    the extremes (a signature of mean-collapse -- a flat/near-constant
    prediction is, by construction, furthest from the truth exactly
    where true values are most extreme).

This computes per-quantile R for every protein, pooling cells across
ALL timepoint pairs for that protein (consistent with wanting enough
cells per bin to get a stable R), and reports:
  1. A CSV: one row per protein x quantile bin, with R and n cells.
  2. A summary print: proteins with the LARGEST gap between their
     best-scoring and worst-scoring quantile bin -- these are the
     proteins whose overall R most understates/overstates how they
     perform in different parts of their range.
  3. A faceted figure: for a handful of example proteins (highest
     overall R, lowest overall R, and largest quantile-gap), a
     scatter of predicted vs. true, colored by which quantile bin
     each cell falls into, so the pattern is visible directly.

Usage:
    python quantile_diagnostic.py \
        --dir new_results/final_model_corrloss_only_leakfix \
        --n_quantiles 4 \
        --out_csv quantile_diagnostic.csv \
        --out_fig quantile_diagnostic_examples.png
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUANTILE_PALETTE = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00"]  # low -> high, colorblind-safe


def discover_pairs(run_dir):
    pred_paths = sorted(glob.glob(os.path.join(run_dir, "joint_y_pred_t*.csv")))
    pairs = []
    for p in pred_paths:
        m = re.search(r"joint_y_pred_t(.+)\.csv$", os.path.basename(p))
        if m:
            pairs.append(m.group(1))
    return pairs


def load_all_pairs(run_dir, pairs):
    """Pool (true, pred) across ALL pairs, per protein, so each protein
    has enough cells total to split into stable quantile bins."""
    protein_names = None
    all_true, all_pred = None, None
    for pair in pairs:
        pred_df = pd.read_csv(os.path.join(run_dir, f"joint_y_pred_t{pair}.csv"))
        truth_df = pd.read_csv(os.path.join(run_dir, f"joint_y_truth_t{pair}.csv"))
        meta_cols = ["t_src", "t_tgt"]
        names = [c for c in pred_df.columns if c not in meta_cols]
        if protein_names is None:
            protein_names = names
            all_true = truth_df[names].values
            all_pred = pred_df[names].values
        else:
            all_true = np.concatenate([all_true, truth_df[names].values], axis=0)
            all_pred = np.concatenate([all_pred, pred_df[names].values], axis=0)
    return protein_names, all_true, all_pred


def quantile_r_for_protein(y_true, y_pred, n_quantiles, min_cells_per_bin=10):
    """Split cells into n_quantiles bins by TRUE value, compute R within
    each bin. Returns a list of dicts (one per bin) with bin label,
    R, and n. Skips bins with too few cells or no variance to avoid
    unstable/undefined R."""
    if np.std(y_true) < 1e-8:
        return []
    try:
        bin_labels = pd.qcut(y_true, q=n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return []  # not enough distinct values to form the requested bins

    results = []
    n_bins_actual = int(np.nanmax(bin_labels)) + 1 if len(bin_labels) else 0
    for b in range(n_bins_actual):
        mask = bin_labels == b
        n = int(mask.sum())
        if n < min_cells_per_bin:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            r = np.nan
        else:
            r, _ = pearsonr(yt, yp)
        results.append({"quantile_bin": b, "n_cells": n, "R": r,
                         "true_range": f"[{yt.min():.3f}, {yt.max():.3f}]"})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory with joint_y_pred_t*.csv / joint_y_truth_t*.csv")
    parser.add_argument("--n_quantiles", type=int, default=4)
    parser.add_argument("--min_cells_per_bin", type=int, default=10)
    parser.add_argument("--n_example_proteins", type=int, default=6,
                         help="How many example proteins to show in the figure (split across highest-R, lowest-R, largest-gap)")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_fig", required=True)
    args = parser.parse_args()

    pairs = discover_pairs(args.dir)
    if not pairs:
        raise FileNotFoundError(f"No joint_y_pred_t*.csv files found in {args.dir}")
    print(f"Discovered {len(pairs)} pair(s): {pairs}")

    protein_names, all_true, all_pred = load_all_pairs(args.dir, pairs)
    print(f"Pooled {all_true.shape[0]} cells x {len(protein_names)} proteins across all pairs "
          f"(pooling across pairs ONLY to get enough cells per quantile bin -- "
          f"this is a diagnostic, not the thesis's reported per-pair-then-averaged R).")

    rows = []
    overall_r = {}
    for i, name in enumerate(protein_names):
        yt, yp = all_true[:, i], all_pred[:, i]
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            overall_r[name] = pearsonr(yt, yp)[0]
        else:
            overall_r[name] = np.nan

        bin_results = quantile_r_for_protein(yt, yp, args.n_quantiles, args.min_cells_per_bin)
        for br in bin_results:
            rows.append({"protein": name, "overall_R": overall_r[name], **br})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv}")

    # --- Summary: proteins with the largest gap between best and worst quantile bin ---
    gap_rows = []
    for name, g in df.groupby("protein"):
        g_valid = g.dropna(subset=["R"])
        if len(g_valid) < 2:
            continue
        gap = g_valid["R"].max() - g_valid["R"].min()
        worst_bin = g_valid.loc[g_valid["R"].idxmin(), "quantile_bin"]
        best_bin = g_valid.loc[g_valid["R"].idxmax(), "quantile_bin"]
        gap_rows.append({
            "protein": name, "overall_R": overall_r[name],
            "quantile_gap": gap,
            "best_bin": best_bin, "best_bin_R": g_valid["R"].max(),
            "worst_bin": worst_bin, "worst_bin_R": g_valid["R"].min(),
        })
    gap_df = pd.DataFrame(gap_rows).sort_values("quantile_gap", ascending=False)

    print("\n=== Proteins with the LARGEST gap between best and worst quantile bin ===")
    print("(large gap + worst bin = highest-value bin is the signature of underprediction")
    print(" at high true values, consistent with mean-collapse)")
    print(gap_df.head(15).to_string(index=False))

    n_high_bin_worst = (gap_df["worst_bin"] == args.n_quantiles - 1).sum()
    n_low_bin_worst = (gap_df["worst_bin"] == 0).sum()
    print(f"\nOf {len(gap_df)} proteins with a defined gap:")
    print(f"  {n_high_bin_worst} ({100*n_high_bin_worst/len(gap_df):.0f}%) score WORST in the "
          f"highest quantile bin (underprediction at high true values)")
    print(f"  {n_low_bin_worst} ({100*n_low_bin_worst/len(gap_df):.0f}%) score WORST in the "
          f"lowest quantile bin (overprediction at low true values)")

    # --- Figure: example proteins showing the pattern directly ---
    by_overall = gap_df.sort_values("overall_R", ascending=False)
    highest_r = by_overall.head(max(1, args.n_example_proteins // 3))["protein"].tolist()
    lowest_r = by_overall.tail(max(1, args.n_example_proteins // 3))["protein"].tolist()
    largest_gap = gap_df.head(max(1, args.n_example_proteins - len(highest_r) - len(lowest_r)))["protein"].tolist()
    example_proteins = list(dict.fromkeys(highest_r + largest_gap + lowest_r))[:args.n_example_proteins]

    ncols = min(3, len(example_proteins))
    nrows = int(np.ceil(len(example_proteins) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax_i, name in enumerate(example_proteins):
        ax = axes[ax_i]
        idx = protein_names.index(name)
        yt, yp = all_true[:, idx], all_pred[:, idx]
        try:
            bin_labels = pd.qcut(yt, q=args.n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            bin_labels = np.zeros_like(yt, dtype=int)
        for b in sorted(pd.unique(bin_labels)):
            mask = bin_labels == b
            color = QUANTILE_PALETTE[int(b) % len(QUANTILE_PALETTE)]
            ax.scatter(yt[mask], yp[mask], s=10, alpha=0.5, color=color, edgecolors="none",
                       label=f"Q{int(b)+1}")
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "--", color="black", linewidth=1, alpha=0.6)
        ax.set_title(f"{name} (overall R={overall_r[name]:.2f})", fontsize=10)
        ax.set_xlabel("True", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.tick_params(labelsize=8)

    for ax_i in range(len(example_proteins), len(axes)):
        axes[ax_i].axis("off")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=QUANTILE_PALETTE[b],
                          label=f"Quantile {b+1} ({'lowest' if b == 0 else 'highest' if b == args.n_quantiles-1 else 'mid'} true values)")
               for b in range(args.n_quantiles)]
    fig.legend(handles=handles, loc="lower center", ncol=args.n_quantiles, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Predicted vs. true, colored by true-value quantile bin\n"
                 "(examples: highest overall R, lowest overall R, largest within-protein gap)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    plt.savefig(args.out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {args.out_fig}")


if __name__ == "__main__":
    main()
