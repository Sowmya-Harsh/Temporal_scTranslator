"""
plot_final_model_figures.py
──────────────────────────────
Regenerates the four final-model figures from real, leak-fixed data:
  1. loss_curves.png            -- from loss_history.csv
  2. per_protein_heatmap_final_model.png -- from saved per-pair predictions
  3. scatter_final_model.png    -- from saved per-pair predictions
  4. final_model_top_proteins.png -- from saved per-pair predictions

Usage:
    python plot_final_model_figures.py \
        --dir new_results/final_model_corrloss_only_leakfix \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --out_dir Figures
"""
import argparse
import glob
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


def concordance_correlation_coefficient(y_true, y_pred):
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return 0.0 if denom == 0 else float((2 * covariance) / denom)


def load_pairs(pred_dir):
    """Returns dict[pair] -> (pred_df, truth_df, protein_names)"""
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


def plot_loss_curves(loss_history_path, out_path):
    df = pd.read_csv(loss_history_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["epoch"], df["train_loss"], color="#0072B2", linewidth=1.8, label="Train loss")
    ax.plot(df["epoch"], df["val_loss"], color="#D55E00", linewidth=1.8, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_per_protein_heatmap(pairs_data, is_phospho_map, out_path):
    protein_names = pairs_data[PAIR_ORDER[0]][2]
    phospho_names = [p for p in protein_names if is_phospho_map.get(p, False)]
    total_names = [p for p in protein_names if not is_phospho_map.get(p, False)]
    ordered_names = phospho_names + total_names

    r_matrix = np.full((len(ordered_names), len(PAIR_ORDER)), np.nan)
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

    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(r_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.7)
    ax.set_xticks(range(len(PAIR_LABELS)))
    ax.set_xticklabels(PAIR_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(ordered_names)))
    ax.set_yticklabels(ordered_names, fontsize=7)
    ax.axhline(len(phospho_names) - 0.5, color="black", linewidth=1.2)
    ax.set_xlabel("Timepoint pair")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pearson R")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_scatter(pairs_data, out_path, n_proteins=6):
    # Pick n_proteins by highest mean R across pairs, for a representative faceted scatter
    pred_all, truth_all = [], []
    protein_names = pairs_data[PAIR_ORDER[0]][2]
    for pair in PAIR_ORDER:
        if pair not in pairs_data:
            continue
        pred, truth, _ = pairs_data[pair]
        pred_all.append(pred)
        truth_all.append(truth)
    pred_all = np.concatenate(pred_all, axis=0)
    truth_all = np.concatenate(truth_all, axis=0)

    rs = []
    for i, name in enumerate(protein_names):
        yt, yp = truth_all[:, i], pred_all[:, i]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            rs.append(-999)
            continue
        r, _ = pearsonr(yt, yp)
        rs.append(r)
    top_idx = np.argsort(rs)[::-1][:n_proteins]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, i in zip(axes.flat, top_idx):
        ax.scatter(truth_all[:, i], pred_all[:, i], s=8, alpha=0.5, color="#0072B2")
        lims = [min(truth_all[:, i].min(), pred_all[:, i].min()),
                max(truth_all[:, i].max(), pred_all[:, i].max())]
        ax.plot(lims, lims, "--", color="#888780", linewidth=1)
        ax.set_title(f"{protein_names[i]} (R={rs[i]:.2f})", fontsize=11)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_top_proteins(pairs_data, out_path, top_n=15):
    protein_names = pairs_data[PAIR_ORDER[0]][2]
    pred_all, truth_all = [], []
    for pair in PAIR_ORDER:
        if pair not in pairs_data:
            continue
        pred, truth, _ = pairs_data[pair]
        pred_all.append(pred)
        truth_all.append(truth)
    pred_all = np.concatenate(pred_all, axis=0)
    truth_all = np.concatenate(truth_all, axis=0)

    rows = []
    for i, name in enumerate(protein_names):
        yt, yp = truth_all[:, i], pred_all[:, i]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            continue
        r, _ = pearsonr(yt, yp)
        ccc = concordance_correlation_coefficient(yt, yp)
        rows.append({"protein": name, "R": r, "CCC": ccc})
    df = pd.DataFrame(rows).sort_values("R", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["R"], color="#0072B2", height=0.7, label="R")
    ax.barh(y_pos, df["CCC"], color="#D55E00", height=0.35, label="CCC")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["protein"])
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Dir with joint_y_pred_t*.csv, joint_y_truth_t*.csv, loss_history.csv")
    parser.add_argument("--protein", required=True, help="Cleaned protein h5ad with is_phospho tag")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prot_adata = ad.read_h5ad(args.protein)
    is_phospho_map = dict(zip(prot_adata.var_names, prot_adata.var["is_phospho"]))

    pairs_data = load_pairs(args.dir)
    print(f"Loaded {len(pairs_data)}/{len(PAIR_ORDER)} pairs")

    loss_history_path = os.path.join(args.dir, "loss_history.csv")
    if os.path.exists(loss_history_path):
        plot_loss_curves(loss_history_path, os.path.join(args.out_dir, "loss_curves.png"))
    else:
        print(f"  WARNING: {loss_history_path} not found -- skipping loss curve plot. "
              f"Rerun train_joint_improved.py (now patched to save loss_history.csv) first.")

    plot_per_protein_heatmap(pairs_data, is_phospho_map,
                              os.path.join(args.out_dir, "per_protein_heatmap_final_model.png"))
    plot_scatter(pairs_data, os.path.join(args.out_dir, "scatter_final_model.png"))
    plot_top_proteins(pairs_data, os.path.join(args.out_dir, "final_model_top_proteins.png"))


if __name__ == "__main__":
    main()
