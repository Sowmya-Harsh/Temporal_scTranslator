"""
24_ptm_breakdown.py

Purpose: Splits an existing per-protein R/CCC results table (from
11_per_protein_bar_charts.py's output CSV) into phosphorylation-
specific (PTM) channels vs. total-protein/other channels, using the
is_phospho column already present in your protein h5ad's .var. Pure
post-processing of results you already have -- no new training.

Addresses the "did we actually answer the PTM part of the research
question" gap: the thesis's stated research question explicitly
includes PTM state, but no dedicated PTM-vs-total-protein performance
breakdown exists yet.

Usage:
    python 24_ptm_breakdown.py \
        --per_protein_csv protein_ranking_figures/final_model_per_protein_R_CCC.csv \
        --protein_h5ad Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --out_prefix final_model
"""
import argparse

import anndata as ad
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(per_protein_csv, protein_h5ad, out_prefix):
    res = pd.read_csv(per_protein_csv)
    prot = ad.read_h5ad(protein_h5ad, backed="r")

    if "is_phospho" not in prot.var.columns:
        raise ValueError(f"'is_phospho' column not found in {protein_h5ad}.var -- "
                          f"available columns: {list(prot.var.columns)}")

    phospho_map = dict(zip(prot.var_names, prot.var["is_phospho"]))
    res["is_phospho"] = res["protein"].map(phospho_map)

    n_missing = res["is_phospho"].isna().sum()
    if n_missing > 0:
        print(f"[warn] {n_missing} proteins in the results CSV not found in the h5ad's "
              f"var_names -- check for naming mismatches. These rows excluded from breakdown.")
    res = res.dropna(subset=["is_phospho"])

    ptm = res[res["is_phospho"] == True]  # noqa: E712
    total_protein = res[res["is_phospho"] == False]  # noqa: E712

    print(f"PTM (phospho-specific) channels: {len(ptm)}")
    print(f"  Mean R:   {ptm['R'].mean():.4f}")
    print(f"  Mean CCC: {ptm['CCC'].mean():.4f}")
    print(f"\nTotal-protein/other channels: {len(total_protein)}")
    print(f"  Mean R:   {total_protein['R'].mean():.4f}")
    print(f"  Mean CCC: {total_protein['CCC'].mean():.4f}")

    print(f"\nTop 5 PTM channels by R:")
    print(ptm.sort_values("R", ascending=False).head(5)[["protein", "R", "CCC"]].to_string(index=False))
    print(f"\nTop 5 total-protein channels by R:")
    print(total_protein.sort_values("R", ascending=False).head(5)[["protein", "R", "CCC"]].to_string(index=False))

    out_csv = f"{out_prefix}_ptm_breakdown.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nSaved full annotated table to {out_csv}")

    # Simple grouped bar comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, metric in zip(axes, ["R", "CCC"]):
        means = [total_protein[metric].mean(), ptm[metric].mean()]
        stds = [total_protein[metric].std(), ptm[metric].std()]
        ax.bar(["Total-protein\n(non-phospho)", "PTM\n(phospho-specific)"], means, yerr=stds,
               color=["tab:blue", "tab:orange"], capsize=5)
        ax.set_ylabel(f"Mean {metric}", color="black")
        ax.set_title(f"Mean {metric} by channel type", color="black")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")

    fig.suptitle(f"PTM vs. total-protein performance: {out_prefix}", color="black", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_ptm_breakdown.png", dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_prefix}_ptm_breakdown.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_protein_csv", type=str, required=True)
    parser.add_argument("--protein_h5ad", type=str, required=True)
    parser.add_argument("--out_prefix", type=str, default="model")
    args = parser.parse_args()
    main(args.per_protein_csv, args.protein_h5ad, args.out_prefix)
