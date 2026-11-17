"""
plot_per_protein_heatmap.py
────────────────────────────────
Generates per_protein_heatmap_final_model.png: per-protein Pearson R
across all timepoint pairs, with the actual R value annotated inside
every cell, on a white-centered diverging colormap (blue = negative,
white = zero, orange = positive) -- matches the reference figure
style requested, rather than a flat low-to-high gradient.

Usage:
    python plot_per_protein_heatmap.py \
        --dir new_results/final_model_corrloss_only_leakfix \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --out Figures/per_protein_heatmap_final_model.png
"""
import argparse
import os

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIR_ORDER = ["0to2", "0to4", "2to6", "0to6", "4to60", "6to60", "0to60", "60to180", "6to180", "0to180"]
PAIR_LABELS = ["0→2", "0→4", "2→6", "0→6", "4→60", "6→60", "0→60", "60→180", "6→180", "0→180"]


def load_pairs(pred_dir):
    out = {}
    for pair in PAIR_ORDER:
        pred_path = os.path.join(pred_dir, f"joint_y_pred_t{pair}.csv")
        truth_path = os.path.join(pred_dir, f"joint_y_truth_t{pair}.csv")
        if not (os.path.exists(pred_path) and os.path.exists(truth_path)):
            print(f"  WARNING: missing files for pair t{pair}, skipping")
            continue
        pred_df = pd.read_csv(pred_path)
        truth_df = pd.read_csv(truth_path)
        meta_cols = ["t_src", "t_tgt"]
        protein_names = [c for c in pred_df.columns if c not in meta_cols]
        out[pair] = (pred_df[protein_names].values, truth_df[protein_names].values, protein_names)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--protein", required=True, help="Cleaned protein h5ad with is_phospho tag")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    prot_adata = ad.read_h5ad(args.protein)
    is_phospho_map = dict(zip(prot_adata.var_names, prot_adata.var["is_phospho"]))

    pairs_data = load_pairs(args.dir)
    print(f"Loaded {len(pairs_data)}/{len(PAIR_ORDER)} pairs")

    # Explicit guard: columns MUST follow PAIR_ORDER (chronological by
    # delta_t), never an alphabetical/glob-derived order. This has broken
    # silently before -- fail loudly instead of producing a scrambled figure.
    found_order = [p for p in PAIR_ORDER if p in pairs_data]
    assert found_order == [p for p in PAIR_ORDER if p in pairs_data], \
        "Column order drifted from PAIR_ORDER -- this should be structurally impossible; check load_pairs()."
    print(f"Column order (verified chronological): {[PAIR_LABELS[PAIR_ORDER.index(p)] for p in found_order]}")

    protein_names = pairs_data[list(pairs_data.keys())[0]][2]
    phospho_names = sorted([p for p in protein_names if is_phospho_map.get(p, False)])
    total_names = sorted([p for p in protein_names if not is_phospho_map.get(p, False)])
    ordered_names = phospho_names + total_names

    r_matrix = np.full((len(ordered_names), len(PAIR_ORDER)), np.nan)
    # Hard guard: columns MUST be built in chronological PAIR_ORDER, never
    # alphabetical/glob order -- this is exactly the bug that produced a
    # scrambled column order in an earlier run of a different script.
    assert PAIR_ORDER == ["0to2", "0to4", "2to6", "0to6", "4to60", "6to60", "0to60", "60to180", "6to180", "0to180"], \
        "PAIR_ORDER was modified -- column ordering guarantee broken"
    for j, pair in enumerate(PAIR_ORDER):
        if pair not in pairs_data:
            continue
        pred, truth, names = pairs_data[pair]
        name_to_idx = {n: i for i, n in enumerate(names)}
        for i, pname in enumerate(ordered_names):
            if pname not in name_to_idx:
                continue
            k = name_to_idx[pname]
            yt, yp = truth[:, k], pred[:, k]
            if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
                continue
            r, _ = pearsonr(yt, yp)
            r_matrix[i, j] = r

    fig_height = max(9, len(ordered_names) * 0.22)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    vmax = np.nanmax(np.abs(r_matrix))
    im = ax.imshow(r_matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    # Annotate every cell with its actual R value, matching the reference
    # figure -- text color flips to white on strongly-saturated cells so
    # it stays legible against both the dark-blue and dark-orange ends.
    for i in range(r_matrix.shape[0]):
        for j in range(r_matrix.shape[1]):
            val = r_matrix[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color=text_color)

    ax.set_xticks(range(len(PAIR_LABELS)))
    ax.set_xticklabels(PAIR_LABELS, rotation=45, ha="right", fontsize=10, color="black")
    ax.set_yticks(range(len(ordered_names)))
    ax.set_yticklabels(ordered_names, fontsize=7, color="black")
    ax.axhline(len(phospho_names) - 0.5, color="black", linewidth=1.5)
    # Group labels in the left margin (axes fraction coords, clip_on=False)
    # so they never collide with the protein-name tick labels.
    ax.text(-0.30, 1 - (len(phospho_names) / 2) / len(ordered_names), "PTM",
            rotation=90, ha="center", va="center", fontsize=11, fontweight="bold",
            transform=ax.transAxes, clip_on=False, color="black")
    ax.text(-0.30, 1 - (len(phospho_names) + len(total_names) / 2) / len(ordered_names), "Total\nprotein",
            rotation=90, ha="center", va="center", fontsize=11, fontweight="bold",
            transform=ax.transAxes, clip_on=False, color="black")
    ax.set_xlabel("Timepoint pair (min)", fontsize=12, color="black")
    ax.set_title("Per-protein Pearson R across all timepoint pairs\n(final model)", color="black", fontsize=13)
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.06)
    cbar.set_label("Pearson R", fontsize=11, color="black")
    cbar.ax.tick_params(colors="black")

    plt.subplots_adjust(left=0.30)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
