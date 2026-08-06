"""
05_dataset_figures.py

Purpose: Produce the remaining Chapter 3 (Dataset) figures:

  1. Cells per timepoint x condition (Table 3.1 as a grouped bar chart)
  2. QC metric distributions (n_genes_by_counts, pct_counts_mt) BEFORE vs
     AFTER cleaning, as violin plots — visually backs the 200-gene /
     20%-mito thresholds from Section 3.6.
  3. Protein/PTM channel composition: total-protein/other markers (36)
     vs phosphorylation-specific markers (22 retained after dropping the
     22 zero-signal channels) vs the 22 dropped zero-signal channels —
     as a single stacked/grouped bar.

All panels styled consistently: black text throughout, no colored axis
labels, real (non-log-scientific) timepoint ticks where applicable.

Usage:
    python 05_dataset_figures.py \
        --rna_raw Final_Dataset/qurie_RNA.h5ad \
        --rna_clean_aIg Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --rna_clean_aIg_Ibru Final_Dataset/qurie_RNA_clean_aIg_Ibru.h5ad \
        --prot_raw Final_Dataset/qurie_Protein.h5ad \
        --prot_clean_aIg Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --out_dir dataset_figures
"""
import argparse
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

TIMEPOINTS = [0, 2, 4, 6, 60, 180]
BLACK = "black"


def style_axes_black(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=BLACK, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=BLACK)
    if ylabel:
        ax.set_ylabel(ylabel, color=BLACK)
    ax.tick_params(axis="x", colors=BLACK)
    ax.tick_params(axis="y", colors=BLACK)
    for spine in ax.spines.values():
        spine.set_color(BLACK)


# ---------------------------------------------------------------------
# 1. Cells per timepoint x condition (Table 3.1 as a figure)
# ---------------------------------------------------------------------
def fig_cells_per_timepoint(rna_clean_aIg_path, rna_clean_aIg_Ibru_path, out_path):
    aIg = ad.read_h5ad(rna_clean_aIg_path, backed="r")
    ibru = ad.read_h5ad(rna_clean_aIg_Ibru_path, backed="r")

    aIg_counts = aIg.obs["time_min"].value_counts().sort_index()
    ibru_counts = ibru.obs["time_min"].value_counts().sort_index()

    all_times = sorted(set(aIg_counts.index) | set(ibru_counts.index))
    aIg_vals = [aIg_counts.get(t, 0) for t in all_times]
    ibru_vals = [ibru_counts.get(t, 0) for t in all_times]

    x = np.arange(len(all_times))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, aIg_vals, width, label="aIg", color="tab:blue")
    ax.bar(x + width / 2, ibru_vals, width, label="aIg+Ibru", color="tab:red")

    for xi, v in zip(x - width / 2, aIg_vals):
        if v > 0:
            ax.text(xi, v + 10, str(v), ha="center", fontsize=8, color=BLACK)
    for xi, v in zip(x + width / 2, ibru_vals):
        if v > 0:
            ax.text(xi, v + 10, str(v), ha="center", fontsize=8, color=BLACK)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in all_times], color=BLACK)
    style_axes_black(ax, title="Cells per timepoint and condition (post-cleaning)",
                      xlabel="Time (min)", ylabel="Number of cells")
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color(BLACK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# 2. QC metric distributions before vs after cleaning
# ---------------------------------------------------------------------
def fig_qc_distributions(rna_raw_path, rna_clean_path, out_path, condition="aIg"):
    raw = ad.read_h5ad(rna_raw_path)
    clean = ad.read_h5ad(rna_clean_path)

    # Match conditions: "before" must be the SAME condition subset as "after,"
    # otherwise a difference between conditions (not the cleaning filter itself)
    # can masquerade as a cleaning effect.
    if "condition" in raw.obs.columns:
        n_before_all = raw.n_obs
        raw = raw[raw.obs["condition"] == condition].copy()
        print(f"[condition-match] raw: kept {raw.n_obs}/{n_before_all} cells with condition == '{condition}'")

    # IMPORTANT: the real cleaning pipeline computed pct_counts_mt using ONLY
    # the single gene MT-CO2 (confirmed via 10_diagnose_mito_pct.py -- clean
    # file's var['mt'] flags exactly one gene, and recomputing with only that
    # gene reproduces the clean file's stored pct_counts_mt almost exactly:
    # MAE=0.065, corr=0.975). Using the standard 13-gene MT- prefix panel here
    # instead would NOT match what was actually filtered on, and silently
    # reproduces the earlier before/after mismatch. Do not "fix" this to the
    # full panel without also noting it as a pipeline limitation (Section 6.7).
    if "n_genes_by_counts" not in raw.obs.columns or "pct_counts_mt" not in raw.obs.columns:
        raw.var["mt"] = raw.var_names == "MT-CO2"
        sc.pp.calculate_qc_metrics(raw, qc_vars=["mt"], percent_top=None,
                                    log1p=False, inplace=True)

    # Diagnostic: flag if "after cleaning" still has cells below the stated
    # thresholds — this would mean n_genes_by_counts/pct_counts_mt in the
    # clean file were recomputed AFTER gene-level filtering (which can push
    # cells back under the per-cell threshold) rather than being the values
    # actually used at filtering time. Doesn't silently hide the issue.
    n_below_200_after = int((clean.obs["n_genes_by_counts"] < 200).sum())
    n_above_20mt_after = int((clean.obs["pct_counts_mt"] > 20).sum())
    if n_below_200_after > 0 or n_above_20mt_after > 0:
        print(f"[WARNING] 'After cleaning' data still has {n_below_200_after} cells with "
              f"n_genes_by_counts < 200 and {n_above_20mt_after} cells with pct_counts_mt > 20. "
              f"This usually means QC metrics were recomputed after gene-level filtering "
              f"(which changes n_genes_by_counts per cell) rather than reflecting the values "
              f"used at the time cells were actually filtered. Check your cleaning script's "
              f"ordering: QC metrics should be computed and thresholded BEFORE any gene removal.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    data_genes = [raw.obs["n_genes_by_counts"].values, clean.obs["n_genes_by_counts"].values]
    axes[0].violinplot(data_genes, showmedians=True)
    axes[0].axhline(200, color="tab:red", linestyle="--", label="Cutoff (200 genes)")
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Before cleaning", "After cleaning"], color=BLACK)
    style_axes_black(axes[0], title="Genes detected per cell", ylabel="n_genes_by_counts")
    leg0 = axes[0].legend()
    for t in leg0.get_texts():
        t.set_color(BLACK)

    data_mt = [raw.obs["pct_counts_mt"].values, clean.obs["pct_counts_mt"].values]
    axes[1].violinplot(data_mt, showmedians=True)
    axes[1].axhline(20, color="tab:red", linestyle="--", label="Cutoff (20%, MT-CO2-based)")
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Before cleaning", "After cleaning"], color=BLACK)
    style_axes_black(axes[1], title="Mitochondrial read percentage", ylabel="pct_counts_mt")
    leg1 = axes[1].legend()
    for t in leg1.get_texts():
        t.set_color(BLACK)

    fig.suptitle("QC metric distributions before vs. after cleaning", color=BLACK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# 3. Protein/PTM channel composition
# ---------------------------------------------------------------------
def fig_channel_composition(prot_raw_path, prot_clean_path, out_path):
    raw = ad.read_h5ad(prot_raw_path)
    clean = ad.read_h5ad(prot_clean_path)

    n_total_raw = raw.n_vars
    n_retained = clean.n_vars
    n_dropped = n_total_raw - n_retained

    if "is_phospho" in clean.var.columns:
        n_phospho = int(clean.var["is_phospho"].sum())
        n_total_protein = n_retained - n_phospho
    else:
        # fall back to the numbers already documented in the thesis (Section 3.4)
        n_phospho, n_total_protein = 22, 36

    categories = ["Total-protein /\nother markers", "Phosphorylation-\nspecific markers", "Dropped\n(zero-signal)"]
    values = [n_total_protein, n_phospho, n_dropped]
    colors = ["tab:blue", "tab:orange", "tab:gray"]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars = ax.bar(categories, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center", color=BLACK, fontsize=11)

    style_axes_black(ax, title=f"Protein/PTM channel composition (of {n_total_raw} total)",
                      ylabel="Number of channels")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_cells_per_timepoint(
        args.rna_clean_aIg, args.rna_clean_aIg_Ibru,
        out_dir / "cells_per_timepoint.png",
    )
    fig_qc_distributions(
        args.rna_raw, args.rna_clean_aIg,
        out_dir / "qc_distributions.png",
        condition="aIg",
    )
    fig_channel_composition(
        args.prot_raw, args.prot_clean_aIg,
        out_dir / "channel_composition.png",
    )
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_raw", type=str, required=True)
    parser.add_argument("--rna_clean_aIg", type=str, required=True)
    parser.add_argument("--rna_clean_aIg_Ibru", type=str, required=True)
    parser.add_argument("--prot_raw", type=str, required=True)
    parser.add_argument("--prot_clean_aIg", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="dataset_figures")
    args = parser.parse_args()
    main(args)