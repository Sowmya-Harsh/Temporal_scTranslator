"""
12_plot_pred_vs_truth.py

Purpose: Generate one PNG per timepoint pair showing predicted vs true
protein abundance -- a scatter plot with the perfect-prediction
diagonal line, colored/faceted so you can see how well (or poorly)
predictions track truth for that specific transition.

Auto-discovers all joint_y_pred_t{src}to{tgt}.csv / joint_y_truth_t{src}to{tgt}.csv
pairs in a directory (the output format of train_joint.py / infer_joint_checkpoint.py)
and plots each one.

Usage:
    python 12_plot_pred_vs_truth.py --dir run_results/joint_csv_recovery_soft
    python 12_plot_pred_vs_truth.py --dir run_results/joint_csv_recovery_soft --prefix joint
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt


def find_pairs(directory, prefix):
    pred_files = glob.glob(os.path.join(directory, f"{prefix}_y_pred_t*to*.csv"))
    pairs = []
    for pf in pred_files:
        m = re.search(r"t(\-?\d+)to(\-?\d+)\.csv$", pf)
        if not m:
            continue
        t_src, t_tgt = int(m.group(1)), int(m.group(2))
        truth_file = pf.replace("_y_pred_", "_y_truth_")
        if os.path.exists(truth_file):
            pairs.append((t_src, t_tgt, pf, truth_file))
    return sorted(pairs)


def plot_pair(t_src, t_tgt, pred_path, truth_path, outdir):
    pred_df = pd.read_csv(pred_path)
    truth_df = pd.read_csv(truth_path)

    protein_cols = [c for c in pred_df.columns if c not in ("t_src", "t_tgt")]
    pred_vals = pred_df[protein_cols].values.flatten()
    truth_vals = truth_df[protein_cols].values.flatten()

    r = np.corrcoef(truth_vals, pred_vals)[0, 1] if np.std(pred_vals) > 1e-8 else float("nan")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(truth_vals, pred_vals, s=4, alpha=0.25, color="#1a365d", edgecolors="none")

    lims = [min(truth_vals.min(), pred_vals.min()), max(truth_vals.max(), pred_vals.max())]
    ax.plot(lims, lims, color="#c53030", linestyle="--", linewidth=1.5, label="perfect prediction")

    ax.set_xlabel("True protein abundance")
    ax.set_ylabel("Predicted protein abundance")
    ax.set_title(f"t{t_src} \u2192 t{t_tgt}  (n_cells={len(pred_df)}, pooled r={r:.3f})")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()

    out_path = os.path.join(outdir, f"pred_vs_truth_t{t_src}to{t_tgt}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}  (pooled r={r:.3f})")
    return out_path


def plot_pair_faceted(t_src, t_tgt, pred_path, truth_path, outdir, n_show=12):
    """Facets by individual protein -- shows the REAL within-protein,
    cell-to-cell prediction quality, not the between-protein clustering
    artifact that inflates the pooled scatter plot."""
    pred_df = pd.read_csv(pred_path)
    truth_df = pd.read_csv(truth_path)
    protein_cols = [c for c in pred_df.columns if c not in ("t_src", "t_tgt")]

    # rank proteins by truth variance (most informative to look at first)
    variances = truth_df[protein_cols].var().sort_values(ascending=False)
    show_proteins = variances.index[:n_show].tolist()

    ncols = 4
    nrows = int(np.ceil(len(show_proteins) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, protein in enumerate(show_proteins):
        ax = axes[i]
        t = truth_df[protein].values
        p = pred_df[protein].values
        r = np.corrcoef(t, p)[0, 1] if np.std(p) > 1e-8 and np.std(t) > 1e-8 else float("nan")
        ax.scatter(t, p, s=8, alpha=0.5, color="#1a365d", edgecolors="none")
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, color="#c53030", linestyle="--", linewidth=1)
        ax.set_title(f"{protein}  r={r:.3f}", fontsize=9)
        ax.set_xlabel("truth", fontsize=8)
        ax.set_ylabel("pred", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(show_proteins), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"t{t_src} \u2192 t{t_tgt}: per-protein prediction quality "
                 f"(top {n_show} by truth variance)", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(outdir, f"pred_vs_truth_FACETED_t{t_src}to{t_tgt}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved faceted plot: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="directory containing the y_pred/y_truth CSVs")
    parser.add_argument("--prefix", default="joint", help="filename prefix (default 'joint', "
                         "matches train_joint.py's output naming)")
    parser.add_argument("--outdir", default=None, help="where to save PNGs (default: same as --dir)")
    parser.add_argument("--faceted", action="store_true",
                         help="also generate per-protein faceted plots (top 12 by truth variance) "
                              "-- shows real within-protein prediction quality, not the pooled "
                              "scatter's between-protein clustering artifact")
    args = parser.parse_args()

    outdir = args.outdir or args.dir
    os.makedirs(outdir, exist_ok=True)

    pairs = find_pairs(args.dir, args.prefix)
    if not pairs:
        print(f"No matching {args.prefix}_y_pred_t*to*.csv / {args.prefix}_y_truth_t*to*.csv "
              f"pairs found in {args.dir}. Check --prefix if your files use a different naming "
              f"(e.g. 'rp' for train_joint_rna_protein_input.py, or 'stage1_frozen'/"
              f"'stage2_unfrozen' for train_joint_pretrained_encoder.py).")
        return

    print(f"Found {len(pairs)} timepoint pairs. Generating plots...")
    for t_src, t_tgt, pred_path, truth_path in pairs:
        plot_pair(t_src, t_tgt, pred_path, truth_path, outdir)
        if args.faceted:
            plot_pair_faceted(t_src, t_tgt, pred_path, truth_path, outdir)

    print(f"\nDone. {len(pairs)} PNGs saved to {outdir}/")


if __name__ == "__main__":
    main()