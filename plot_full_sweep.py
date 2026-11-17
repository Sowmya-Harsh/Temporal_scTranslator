"""
plot_full_sweep.py
─────────────────────
Generates Figures/film_necessity_sweep.png from full_sweep_pooled.csv
(output of combine_full_sweep.py). Uses VALIDATION R only, per the
selection-leakage fix -- this is the figure behind Table
gene-selection-comparison and the "FiLM-necessity" discussion, and
must be built from the same split used to actually pick the winning
config (see select_best_config.py).

Colour-coded by model type, using the Okabe-Ito colourblind-safe
palette (the standard choice for scientific figures -- distinguishable
under all common forms of colour vision deficiency, and still fine
in greyscale printing):
    MLP        -> blue       #0072B2
    FiLM-pre   -> vermillion #D55E00
    FiLM-post  -> green      #009E73
Two subplots side by side (HVG left, DEG right), same y-axis scale
for direct visual comparison, x-axis = gene panel size (log-spaced
categorical: 50/100/200/500/1000/all).

Usage:
    python plot_full_sweep.py \
        --csv new_results/full_sweep_combined/full_sweep_pooled.csv \
        --out Figures/film_necessity_sweep.png
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette
COLORS = {
    "mlp": "#0072B2",        # blue
    "film_pre": "#D55E00",   # vermillion
    "film_post": "#009E73",  # green
}
LABELS = {
    "mlp": "MLP (no conditioning)",
    "film_pre": "FiLM-pre",
    "film_post": "FiLM-post",
}
PANEL_ORDER = ["50", "100", "200", "500", "1000", "all"]
X_POS = {v: i for i, v in enumerate(PANEL_ORDER)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"],
                         help="Which split to plot. Default 'val' -- this is the "
                              "methodologically correct choice; 'test' is provided "
                              "only for sanity-check/debugging, never for the thesis figure.")
    args = parser.parse_args()

    if args.split == "test":
        print("WARNING: plotting TEST split. This is NOT appropriate for the thesis "
              "figure -- config comparison must use validation. Proceeding anyway "
              "since --split test was explicitly requested.")

    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split].copy()
    df["n_genes"] = df["n_genes"].astype(str)
    df["x"] = df["n_genes"].map(X_POS)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    y_max = df["mean_pearson_r"].max() * 1.15

    for ax, strategy, title in zip(axes, ["hvg", "deg"], ["HVG selection", "DEG selection"]):
        sub_strategy = df[df["gene_strategy"] == strategy]
        for model in ["mlp", "film_pre", "film_post"]:
            sub = sub_strategy[sub_strategy["model"] == model].sort_values("x")
            if len(sub) == 0:
                continue
            ax.plot(sub["x"], sub["mean_pearson_r"], marker="o", markersize=6,
                     linewidth=2, color=COLORS[model], label=LABELS[model])
        ax.set_xticks(range(len(PANEL_ORDER)))
        ax.set_xticklabels(PANEL_ORDER)
        ax.set_xlabel("Gene panel size")
        ax.set_title(title)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(f"Mean per-protein R ({args.split})")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), frameon=False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {args.out}")

    # Also save the underlying pivoted numbers next to the figure, for the
    # thesis table (Table gene-selection-comparison replacement)
    table_path = os.path.splitext(args.out)[0] + "_filmpost_table.csv"
    fp = df[df["model"] == "film_post"].pivot(index="n_genes", columns="gene_strategy",
                                               values="mean_pearson_r").reindex(PANEL_ORDER)
    fp.to_csv(table_path)
    print(f"Saved matched-size FiLM-post table to {table_path}")


if __name__ == "__main__":
    main()
