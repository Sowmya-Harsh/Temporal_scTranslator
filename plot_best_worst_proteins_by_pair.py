"""
plot_best_worst_proteins_by_pair.py
───────────────────────────────────────
Faceted scatter: one panel per protein (the top-N highest-R and
bottom-N lowest-R proteins), each panel showing that protein's
predicted-vs-true points colored by SOURCE timepoint (0, 2, 4, 6, 60,
180 -- read directly from each CSV's t_src column), not by the full
10-pair identity. Several pairs share a source timepoint (e.g.
0->2, 0->4, 0->6, 0->60, 0->180 all have t_src=0), so this gives a
simpler 6-color legend instead of 10, while still showing temporal
structure. Ranking uses the correct per-pair-then-averaged R
convention (never pools cells across pairs before computing R).

Usage:
    python plot_best_worst_proteins_by_pair.py \
        --dir new_results/final_model_corrloss_only_leakfix \
        --top_n 5 --bottom_n 5 \
        --out Figures/best_worst_proteins_by_pair.png
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

# One color per real source timepoint (0, 2, 4, 6, 60, 180)
TIMEPOINTS = [0, 2, 4, 6, 60, 180]
TIMEPOINT_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000"]


def discover_pairs(run_dir):
    pred_paths = sorted(glob.glob(os.path.join(run_dir, "joint_y_pred_t*.csv")))
    pairs = []
    for p in pred_paths:
        m = re.search(r"joint_y_pred_t(.+)\.csv$", os.path.basename(p))
        if m:
            pairs.append(m.group(1))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--bottom_n", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pairs = discover_pairs(args.dir)
    if not pairs:
        raise FileNotFoundError(f"No joint_y_pred_t*.csv files found in {args.dir}")
    print(f"Discovered {len(pairs)} pair(s): {pairs}")

    # Load all pairs once, keep t_src alongside each pair's values
    pair_data = {}
    protein_names = None
    for pair in pairs:
        pred_df = pd.read_csv(os.path.join(args.dir, f"joint_y_pred_t{pair}.csv"))
        truth_df = pd.read_csv(os.path.join(args.dir, f"joint_y_truth_t{pair}.csv"))
        meta_cols = ["t_src", "t_tgt"]
        names = [c for c in pred_df.columns if c not in meta_cols]
        if protein_names is None:
            protein_names = names
        t_src_val = float(pred_df["t_src"].iloc[0]) if "t_src" in pred_df.columns else None
        pair_data[pair] = (pred_df[names].values, truth_df[names].values, t_src_val)

    # Correct convention: R per protein per pair, then averaged across pairs
    rows = []
    for i, name in enumerate(protein_names):
        per_pair_r = []
        for pair in pairs:
            pred_vals, truth_vals, _ = pair_data[pair]
            yt, yp = truth_vals[:, i], pred_vals[:, i]
            if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
                continue
            r, _ = pearsonr(yt, yp)
            per_pair_r.append(r)
        if per_pair_r:
            rows.append({"protein": name, "idx": i, "R": float(np.mean(per_pair_r))})

    df = pd.DataFrame(rows).sort_values("R", ascending=False)
    best = df.head(args.top_n)
    worst = df.tail(args.bottom_n).iloc[::-1]

    ncols = max(args.top_n, args.bottom_n)
    fig, axes = plt.subplots(2, ncols, figsize=(3.0 * ncols, 6.4))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    tp_colors = {tp: TIMEPOINT_PALETTE[j] for j, tp in enumerate(TIMEPOINTS)}

    def plot_one(ax, row):
        i = row["idx"]
        all_t_list, all_p_list = [], []
        for pair in pairs:
            pred_vals, truth_vals, t_src_val = pair_data[pair]
            yt, yp = truth_vals[:, i], pred_vals[:, i]
            color = tp_colors.get(t_src_val, "#CCCCCC")
            ax.scatter(yt, yp, s=10, alpha=0.5, color=color, edgecolors="none")
            all_t_list.append(yt)
            all_p_list.append(yp)
        all_t = np.concatenate(all_t_list)
        all_p = np.concatenate(all_p_list)
        lims = [min(all_t.min(), all_p.min()), max(all_t.max(), all_p.max())]
        ax.plot(lims, lims, "--", color="black", linewidth=1, alpha=0.6)
        ax.set_title(f"{row['protein']}\n(R={row['R']:.2f})", fontsize=10, color="black")
        ax.tick_params(colors="black", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("black")

    for c in range(ncols):
        if c < len(best):
            plot_one(axes[0, c], best.iloc[c])
        else:
            axes[0, c].axis("off")
        if c < len(worst):
            plot_one(axes[1, c], worst.iloc[c])
        else:
            axes[1, c].axis("off")

    axes[0, 0].set_ylabel("Highest R proteins\nPredicted", fontsize=10, color="black")
    axes[1, 0].set_ylabel("Lowest R proteins\nPredicted", fontsize=10, color="black")

    # Shared legend, now just 6 source-timepoint colors instead of 10 pairs
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=tp_colors[tp],
                          label=f"t={tp} min") for tp in TIMEPOINTS]
    fig.legend(handles=handles, loc="lower center", ncol=len(TIMEPOINTS),
               fontsize=9, bbox_to_anchor=(0.5, -0.06), frameon=False)

    fig.text(0.5, 0.02, "True protein abundance", ha="center", color="black", fontsize=11)
    fig.suptitle("Best and worst predicted proteins, colored by source timepoint", color="black", fontsize=13)
    plt.tight_layout(rect=[0.02, 0.1, 1, 0.95])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out}")
    print("\nBest:")
    print(best[["protein", "R"]].to_string(index=False))
    print("\nWorst:")
    print(worst[["protein", "R"]].to_string(index=False))


if __name__ == "__main__":
    main()
