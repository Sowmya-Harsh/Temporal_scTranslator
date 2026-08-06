"""
13_prediction_vs_truth_scatter.py

Purpose: Faceted scatter plots of predicted vs. true protein abundance,
for a chosen set of important proteins (e.g. the highest-ranked ones from
11_per_protein_bar_charts.py, or specific biologically-motivated ones like
BLNK, CD38, XBP1), using the same joint_y_pred/joint_y_true CSV format.

Each protein gets its own small panel with a y=x reference line, so you
can see calibration (not just correlation) -- whether predictions are
systematically over/under the true range, not just correlated with it.

Usage:
    python 13_prediction_vs_truth_scatter.py \
        --run_dir run_results/correlationlosswt_test_deg1000_filmpre \
        --pairs 0:180 6:180 60:180 \
        --proteins BLNK CD38 XBP1 CD19 p-p65 GAPDH \
        --out scatter_important_proteins.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PRED_FILE_PATTERN = "joint_y_pred_t{src}to{tgt}.csv"
TRUE_FILE_PATTERN = "joint_y_truth_t{src}to{tgt}.csv"


FOLD_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
               "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


def main(run_dir, run_dirs_multi, pairs_arg, proteins, out_path):
    pairs = [tuple(int(x) for x in p.split(":")) for p in pairs_arg]

    run_dirs = run_dirs_multi if run_dirs_multi else [run_dir]
    fold_color = {Path(rd).name: FOLD_COLORS[i % len(FOLD_COLORS)] for i, rd in enumerate(run_dirs)}

    # Pool cells across the requested pairs (and, if given, across multiple
    # fold directories too) for each protein -- this pools raw cells rather
    # than averaging a scalar, since a scatter plot needs actual points, not
    # a summary statistic. Each point retains which fold it came from, so it
    # can be colour-coded rather than pooled into one indistinguishable blob.
    pooled = {p: {"true": [], "pred": [], "fold": []} for p in proteins}

    for rd in run_dirs:
        rd = Path(rd)
        for t_src, t_tgt in pairs:
            pred_path = rd / PRED_FILE_PATTERN.format(src=t_src, tgt=t_tgt)
            true_path = rd / TRUE_FILE_PATTERN.format(src=t_src, tgt=t_tgt)
            if not pred_path.exists() or not true_path.exists():
                print(f"  [skip] {rd.name} {t_src}->{t_tgt}: files not found")
                continue
            y_pred = pd.read_csv(pred_path)
            y_true = pd.read_csv(true_path)
            for p in proteins:
                if p not in y_pred.columns:
                    print(f"  [warn] protein '{p}' not found in {pred_path.name}, skipping for this pair")
                    continue
                pooled[p]["true"].extend(y_true[p].values)
                pooled[p]["pred"].extend(y_pred[p].values)
                pooled[p]["fold"].extend([rd.name] * len(y_true))

    n = len(proteins)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, p in enumerate(proteins):
        ax = axes[i]
        yt = np.array(pooled[p]["true"])
        yp = np.array(pooled[p]["pred"])
        folds = np.array(pooled[p]["fold"])
        if len(yt) == 0:
            ax.set_title(f"{p} (no data found)", color="black")
            ax.axis("off")
            continue

        if run_dirs_multi:
            for fold_name in sorted(set(folds)):
                mask = folds == fold_name
                ax.scatter(yt[mask], yp[mask], s=10, alpha=0.4,
                           color=fold_color[fold_name], label=fold_name)
        else:
            ax.scatter(yt, yp, s=10, alpha=0.4, color="tab:blue")

        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y = x")

        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            r, _ = pearsonr(yt, yp)
            ax.set_title(f"{p} (R = {r:.3f}, n = {len(yt)})", color="black")
        else:
            ax.set_title(f"{p} (n = {len(yt)})", color="black")

        ax.set_xlabel("True", color="black")
        ax.set_ylabel("Predicted", color="black")
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
        if run_dirs_multi and i == 0:
            legend = ax.legend(fontsize=7, loc="upper left")
            for text in legend.get_texts():
                text.set_color("black")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    title_suffix = f"pooled across {len(run_dirs)} fold(s) x shown pairs" if run_dirs_multi else "pooled across shown pairs"
    fig.suptitle(f"Predicted vs.\ true protein abundance, selected proteins ({title_suffix})",
                 color="black", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None,
                         help="Single run directory (ignored if --run_dirs is given)")
    parser.add_argument("--run_dirs", type=str, nargs="+", default=None,
                         help="Multiple fold directories to POOL cells across "
                              "(e.g. --run_dirs run_results/kfold_final_deg1000_filmpre/fold_1 "
                              "... fold_5). Use this instead of --run_dir for k-fold results.")
    parser.add_argument("--pairs", type=str, nargs="+", required=True,
                         help="e.g. --pairs 0:180 6:180 60:180")
    parser.add_argument("--proteins", type=str, nargs="+", required=True,
                         help="e.g. --proteins BLNK CD38 XBP1")
    parser.add_argument("--out", type=str, default="scatter_important_proteins.png")
    args = parser.parse_args()
    if not args.run_dir and not args.run_dirs:
        parser.error("Must specify either --run_dir or --run_dirs")
    main(args.run_dir, args.run_dirs, args.pairs, args.proteins, args.out)