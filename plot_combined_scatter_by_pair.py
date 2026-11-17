"""
plot_combined_scatter_by_pair.py
────────────────────────────────────
All 10 timepoint pairs' predictions on ONE scatter plot, points
colored by which pair they belong to. Auto-discovers pairs (same
generic approach as plot_top_proteins.py), one distinct color per pair.

The legend shows each pair's own mean-per-protein R (matching the
thesis's correct convention), NOT a single pooled R across everything
-- combining colors visually is fine, but reporting one aggregate
number across all pairs+proteins would reproduce the pooled artefact.

Usage:
    python plot_combined_scatter_by_pair.py \
        --dir new_results/final_model_corrloss_only_leakfix \
        --out Figures/combined_scatter_by_pair.png
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
from matplotlib.colors import LinearSegmentedColormap

# Distinct, colorblind-safer palette, one per pair (up to 10)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442",
           "#56B4E9", "#E69F00", "#000000", "#999999", "#8B4513"]


def discover_pairs(run_dir):
    pred_paths = sorted(glob.glob(os.path.join(run_dir, "joint_y_pred_t*.csv")))
    pairs = []
    for p in pred_paths:
        m = re.search(r"joint_y_pred_t(.+)\.csv$", os.path.basename(p))
        if m:
            pairs.append(m.group(1))
    return pairs


def load_pair(run_dir, pair):
    pred_df = pd.read_csv(os.path.join(run_dir, f"joint_y_pred_t{pair}.csv"))
    truth_df = pd.read_csv(os.path.join(run_dir, f"joint_y_truth_t{pair}.csv"))
    meta_cols = ["t_src", "t_tgt"]
    names = [c for c in pred_df.columns if c not in meta_cols]
    return pred_df[names].values, truth_df[names].values, names


def mean_per_protein_r(pred_vals, truth_vals):
    rs = []
    for k in range(pred_vals.shape[1]):
        yt, yp = truth_vals[:, k], pred_vals[:, k]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            continue
        r, _ = pearsonr(yt, yp)
        rs.append(r)
    return np.mean(rs) if rs else float("nan")


def plot_faceted(run_dir, pairs, out_path):
    """One small density-weighted panel per pair -- darker regions have
    more overlapping points, so within-pair density structure is
    visible instead of washing out into flat transparent dots."""
    ncols = 5
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).flatten()

    global_min, global_max = None, None
    pair_arrays = {}
    for pair in pairs:
        pred_vals, truth_vals, _ = load_pair(run_dir, pair)
        flat_t, flat_p = truth_vals.flatten(), pred_vals.flatten()
        pair_arrays[pair] = (flat_t, flat_p, mean_per_protein_r(pred_vals, truth_vals))
        lo, hi = min(flat_t.min(), flat_p.min()), max(flat_t.max(), flat_p.max())
        global_min = lo if global_min is None else min(global_min, lo)
        global_max = hi if global_max is None else max(global_max, hi)

    for i, pair in enumerate(pairs):
        ax = axes[i]
        flat_t, flat_p, mean_r = pair_arrays[pair]
        color = PALETTE[i % len(PALETTE)]
        # White -> pair's own color colormap: dark/saturated = many
        # overlapping points in that region, light = sparse.
        cmap = LinearSegmentedColormap.from_list(f"pair{i}", ["#FFFFFF", color])
        ax.hexbin(flat_t, flat_p, gridsize=35, cmap=cmap, mincnt=1,
                  extent=[global_min, global_max, global_min, global_max])
        ax.plot([global_min, global_max], [global_min, global_max],
                color="black", linestyle="--", linewidth=1, alpha=0.7)
        pair_label = pair.replace("to", "→")
        ax.set_title(f"{pair_label}  (R={mean_r:.2f})", fontsize=11, color="black")
        ax.tick_params(colors="black", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("black")

    for j in range(len(pairs), len(axes)):
        axes[j].axis("off")

    fig.text(0.5, 0.02, "True protein abundance", ha="center", color="black")
    fig.text(0.01, 0.5, "Predicted protein abundance", va="center", rotation=90, color="black")
    fig.suptitle("Predicted vs. true, per timepoint pair (darker = more overlapping points)",
                 color="black", fontsize=13)
    plt.tight_layout(rect=[0.02, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_points_per_pair", type=int, default=3000,
                         help="Subsample large pairs so no single pair visually dominates (overlay mode only).")
    parser.add_argument("--faceted", action="store_true",
                         help="One density-weighted (hexbin) panel per pair instead of one "
                              "overlaid scatter -- darker regions show more overlapping points.")
    args = parser.parse_args()

    pairs = discover_pairs(args.dir)
    if not pairs:
        raise FileNotFoundError(f"No joint_y_pred_t*.csv files found in {args.dir}")
    print(f"Discovered {len(pairs)} pair(s): {pairs}")

    if args.faceted:
        plot_faceted(args.dir, pairs, args.out)
        return

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    rng = np.random.default_rng(0)
    all_vals = []

    for i, pair in enumerate(pairs):
        pred_path = os.path.join(args.dir, f"joint_y_pred_t{pair}.csv")
        truth_path = os.path.join(args.dir, f"joint_y_truth_t{pair}.csv")
        pred_df = pd.read_csv(pred_path)
        truth_df = pd.read_csv(truth_path)
        meta_cols = ["t_src", "t_tgt"]
        names = [c for c in pred_df.columns if c not in meta_cols]
        pred_vals = pred_df[names].values
        truth_vals = truth_df[names].values

        # Per-pair mean-per-protein R (correct convention), for the legend only
        per_protein_r = []
        for k in range(len(names)):
            yt, yp = truth_vals[:, k], pred_vals[:, k]
            if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
                continue
            r, _ = pearsonr(yt, yp)
            per_protein_r.append(r)
        mean_r = np.mean(per_protein_r) if per_protein_r else float("nan")

        flat_truth = truth_vals.flatten()
        flat_pred = pred_vals.flatten()
        all_vals.append(flat_truth)
        all_vals.append(flat_pred)

        if len(flat_truth) > args.max_points_per_pair:
            idx = rng.choice(len(flat_truth), args.max_points_per_pair, replace=False)
            flat_truth, flat_pred = flat_truth[idx], flat_pred[idx]

        color = PALETTE[i % len(PALETTE)]
        pair_label = pair.replace("to", "→")
        ax.scatter(flat_truth, flat_pred, s=4, alpha=0.35, color=color,
                   edgecolors="none", label=f"{pair_label} (mean per-protein R={mean_r:.2f})")

    all_vals = np.concatenate(all_vals)
    lims = [all_vals.min(), all_vals.max()]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.2, label="perfect prediction", zorder=0)

    ax.set_xlabel("True protein abundance", color="black")
    ax.set_ylabel("Predicted protein abundance", color="black")
    ax.set_title("Predicted vs. true, all timepoint pairs, colored by pair", color="black")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    legend = ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    for text in legend.get_texts():
        text.set_color("black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
