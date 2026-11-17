"""
plot_top_proteins.py
─────────────────────────
Best protein predictions figure. Auto-discovers whichever timepoint
pairs actually exist in --dir (works for the 10-pair final model run,
the single-pair zero-shot ibrutinib run, and the 3-pair myeloid
external run alike -- no hardcoded pair list).

Computes R and CCC SEPARATELY within each discovered pair, then
averages those per-pair values across pairs -- matches the thesis's
per-protein-averaged convention exactly. Never concatenates cells
across pairs before correlating (that reproduces the same artefact
Table pooled-artefact warns about, just with "pair" playing the role
"protein" played there). Pair ORDER doesn't matter for this figure
(unlike the heatmap) since we're only ever averaging across pairs.

Usage:
    python plot_top_proteins.py \
        --dir new_results/final_model_corrloss_only_leakfix \
        --top_n 15 \
        --title "Top 15 proteins by mean R (final model)" \
        --out Figures/final_model_top_proteins.png
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


def concordance_correlation_coefficient(y_true, y_pred):
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return 0.0 if denom == 0 else float((2 * covariance) / denom)


def discover_pairs(run_dir, prefix="joint"):
    """Finds every {prefix}_y_pred_t{pair}.csv / {prefix}_y_truth_t{pair}.csv
    pair present in run_dir, whatever the actual timepoint values are.
    Default prefix 'joint' matches train_joint.py's evaluate_per_pair()
    convention; pass --prefix zeroshot for 23_zeroshot_ibrutinib.py's
    output, which uses a different prefix (zeroshot_y_pred_t*.csv)."""
    pred_paths = sorted(glob.glob(os.path.join(run_dir, f"{prefix}_y_pred_t*.csv")))
    pairs = []
    for p in pred_paths:
        m = re.search(rf"{prefix}_y_pred_t(.+)\.csv$", os.path.basename(p))
        if m:
            pairs.append(m.group(1))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--prefix", default="joint",
                         help="Filename prefix before '_y_pred_t*.csv' / '_y_truth_t*.csv'. "
                              "Default 'joint' matches train_joint.py's output. Use 'zeroshot' "
                              "for 23_zeroshot_ibrutinib.py's output, which uses a different prefix.")
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--title", default=None, help="Custom chart title (default auto-generated)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pairs = discover_pairs(args.dir, prefix=args.prefix)
    if not pairs:
        raise FileNotFoundError(f"No {args.prefix}_y_pred_t*.csv files found in {args.dir}")
    print(f"Discovered {len(pairs)} pair(s): {pairs}")

    pair_data = []
    protein_names = None
    for pair in pairs:
        pred_path = os.path.join(args.dir, f"{args.prefix}_y_pred_t{pair}.csv")
        truth_path = os.path.join(args.dir, f"{args.prefix}_y_truth_t{pair}.csv")
        pred_df = pd.read_csv(pred_path)
        truth_df = pd.read_csv(truth_path)
        meta_cols = ["t_src", "t_tgt"]
        names = [c for c in pred_df.columns if c not in meta_cols]
        if protein_names is None:
            protein_names = names
        pair_data.append((pred_df[names].values, truth_df[names].values))

    rows = []
    for i, name in enumerate(protein_names):
        per_pair_r, per_pair_ccc = [], []
        for pred, truth in pair_data:
            yt, yp = truth[:, i], pred[:, i]
            if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
                continue
            r, _ = pearsonr(yt, yp)
            ccc = concordance_correlation_coefficient(yt, yp)
            per_pair_r.append(r)
            per_pair_ccc.append(ccc)
        if not per_pair_r:
            continue
        rows.append({"protein": name, "R": float(np.mean(per_pair_r)), "CCC": float(np.mean(per_pair_ccc))})

    df = pd.DataFrame(rows).sort_values("R", ascending=False).head(args.top_n)
    df = df.iloc[::-1]  # highest R appears at the TOP of the horizontal chart

    fig, ax = plt.subplots(figsize=(7.5, max(5, args.top_n * 0.4)))
    y_pos = np.arange(len(df))
    bar_h = 0.36

    ax.barh(y_pos + bar_h / 2, df["R"], height=bar_h, color="#0072B2", label="R")
    ax.barh(y_pos - bar_h / 2, df["CCC"], height=bar_h, color="#D55E00", label="CCC")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["protein"], fontsize=10, color="black")
    ax.set_xlabel("Score", fontsize=12, color="black")
    title = args.title or f"Top {args.top_n} proteins by mean R ({len(pairs)} pair(s) averaged)"
    ax.set_title(title, fontsize=13, color="black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")
    ax.axvline(0, color="black", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("black")
    legend = ax.legend(loc="lower right", frameon=True)
    for text in legend.get_texts():
        text.set_color("black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out}")
    print(df.iloc[::-1].to_string(index=False))


if __name__ == "__main__":
    main()