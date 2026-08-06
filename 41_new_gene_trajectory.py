"""
04_gene_protein_trajectories.py

Purpose: For genes with a real, matched RNA<->protein identity (the 33 of 36
gene-mapped protein channels confirmed in Section 3.5 to share an identifier
with their corresponding RNA gene), plot mean expression trajectories across
the six aIg timepoints (0, 2, 4, 6, 60, 180 min), RNA in blue and protein in
orange, on a shared time axis with two y-axes (since raw scales differ).
Each trajectory is now shown with a shaded band of +/- 1 standard deviation
across cells at that timepoint, alongside the mean line.

"Highly varying / important" selection: ranks candidate genes by a combined
importance score = z-scored RNA trajectory range + z-scored protein trajectory
range + a bonus if the gene is one of the phosphorylation-specific /
signalling-relevant panel entries (BCR pathway markers), so the selection
favors biologically meaningful, dynamic genes over incidentally noisy ones.

Produces a grid of small-multiple subplots (one panel per gene), not one
crowded plot, so trajectories stay readable.

Usage:
    python 04_gene_protein_trajectories.py \
        --rna_path Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --prot_path Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --mapping_csv Final_Dataset/qurie_Protein_myId_mapping_table.csv \
        --top_n 9 \
        --out_png gene_protein_trajectories.png \
        --out_csv gene_protein_trajectory_data.csv
"""
import argparse

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIMEPOINTS = [0, 2, 4, 6, 60, 180]

# A short list of BCR-signalling-relevant gene symbols (from the thesis's own
# biological framing, Sections 1.3 / 3.2 / 6.5) used as a small relevance
# bonus in the importance score, so selection isn't purely variance-driven.
SIGNALLING_RELEVANT = {
    "CD19", "CD38", "BLNK", "RELA", "RPS6", "CD79A", "CD79B", "PIK3CA",
    "PRKCB", "NFKB1", "STAT3", "MAPK1", "MAPK3", "PLCG2",
}


def normalize_col(x, eps=1e-8):
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin + eps)


def main(rna_path, prot_path, mapping_csv, top_n, out_png, out_csv):
    rna = ad.read_h5ad(rna_path)
    prot = ad.read_h5ad(prot_path)
    mapping = pd.read_csv(mapping_csv)

    print("Mapping table columns:", list(mapping.columns))
    # Expecting something like protein_channel_name, gene_symbol, my_Id, is_phospho —
    # adjust the column names below to match your actual CSV if they differ.
    # We only want the 33 confirmed gene-mapped (non-phospho, real gene identity)
    # channels, per Section 3.5.
    id_col_candidates = ["gene_symbol", "feature", "symbol", "gene"]
    id_col = next((c for c in id_col_candidates if c in mapping.columns), None)
    if id_col is None:
        raise ValueError(
            f"Could not find a gene-symbol column in {mapping_csv}. "
            f"Available columns: {list(mapping.columns)} — edit id_col_candidates."
        )

    if "is_phospho" in mapping.columns:
        gene_mapped = mapping[mapping["is_phospho"] == False]  # noqa: E712
    else:
        gene_mapped = mapping

    protein_channel_col = "protein_channel" if "protein_channel" in mapping.columns else mapping.columns[0]

    rna_gene_symbols = rna.var["feature"].astype(str).values
    rna_X = rna.X.toarray() if hasattr(rna.X, "toarray") else np.asarray(rna.X)
    prot_X = prot.X.toarray() if hasattr(prot.X, "toarray") else np.asarray(prot.X)
    prot_channel_names = list(prot.var_names)

    times = rna.obs["time_min"].values.astype(float)
    times_p = prot.obs["time_min"].values.astype(float)

    candidates = []
    for _, row in gene_mapped.iterrows():
        gene_symbol = str(row[id_col])
        protein_channel = str(row[protein_channel_col])
        if protein_channel not in prot_channel_names:
            continue
        rna_matches = np.where(rna_gene_symbols == gene_symbol)[0]
        if len(rna_matches) == 0:
            continue
        rna_idx = rna_matches[0]
        prot_idx = prot_channel_names.index(protein_channel)

        rna_traj, prot_traj = [], []
        rna_std, prot_std = [], []  # NEW: per-timepoint std across cells
        for t in TIMEPOINTS:
            rna_mask = np.isclose(times, t, atol=1e-3)
            prot_mask = np.isclose(times_p, t, atol=1e-3)
            if rna_mask.sum() == 0 or prot_mask.sum() == 0:
                rna_traj.append(np.nan)
                prot_traj.append(np.nan)
                rna_std.append(np.nan)
                prot_std.append(np.nan)
                continue
            rna_vals = rna_X[rna_mask, rna_idx]
            prot_vals = prot_X[prot_mask, prot_idx]
            rna_traj.append(float(rna_vals.mean()))
            prot_traj.append(float(prot_vals.mean()))
            # ddof=1 for sample std; if a timepoint has only 1 cell this is NaN,
            # which matplotlib's fill_between will simply skip.
            rna_std.append(float(rna_vals.std(ddof=1)) if len(rna_vals) > 1 else 0.0)
            prot_std.append(float(prot_vals.std(ddof=1)) if len(prot_vals) > 1 else 0.0)

        rna_traj = np.array(rna_traj)
        prot_traj = np.array(prot_traj)
        rna_std = np.array(rna_std)
        prot_std = np.array(prot_std)
        if np.all(np.isnan(rna_traj)) or np.all(np.isnan(prot_traj)):
            continue

        rna_range = np.nanmax(rna_traj) - np.nanmin(rna_traj)
        prot_range = np.nanmax(prot_traj) - np.nanmin(prot_traj)

        candidates.append({
            "gene_symbol": gene_symbol,
            "protein_channel": protein_channel,
            "rna_traj": rna_traj,
            "prot_traj": prot_traj,
            "rna_std": rna_std,      # NEW
            "prot_std": prot_std,    # NEW
            "rna_range": rna_range,
            "prot_range": prot_range,
            "relevant": gene_symbol in SIGNALLING_RELEVANT,
        })

    if not candidates:
        raise RuntimeError(
            "No matched gene/protein pairs found — check id_col / protein_channel_col "
            "detection above against your actual mapping CSV columns."
        )

    df = pd.DataFrame(candidates)
    df["rna_range_z"] = (df["rna_range"] - df["rna_range"].mean()) / (df["rna_range"].std() + 1e-8)
    df["prot_range_z"] = (df["prot_range"] - df["prot_range"].mean()) / (df["prot_range"].std() + 1e-8)
    df["importance_score"] = df["rna_range_z"] + df["prot_range_z"] + df["relevant"].astype(float) * 0.5
    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)

    top = df.head(top_n)
    print("\nSelected genes (by importance score):")
    print(top[["gene_symbol", "protein_channel", "rna_range", "prot_range", "relevant", "importance_score"]])

    # Save the underlying trajectory data (long format) for reuse / re-plotting,
    # now including per-timepoint std alongside mean.
    long_rows = []
    for _, r in top.iterrows():
        for t, rv, pv, rs, ps in zip(
            TIMEPOINTS, r["rna_traj"], r["prot_traj"], r["rna_std"], r["prot_std"]
        ):
            long_rows.append({
                "gene_symbol": r["gene_symbol"], "protein_channel": r["protein_channel"],
                "time_min": t, "mean_rna": rv, "mean_protein": pv,
                "std_rna": rs, "std_protein": ps,
            })
    pd.DataFrame(long_rows).to_csv(out_csv, index=False)
    print(f"Saved trajectory data (with std) to {out_csv}")

    # Small-multiples plot, RNA blue / protein orange, dual y-axis per panel,
    # each line now with a +/- 1 std shaded band.
    n = len(top)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, (_, r) in enumerate(top.iterrows()):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        t = np.array(TIMEPOINTS, dtype=float)
        rna_traj, rna_std = r["rna_traj"], r["rna_std"]
        prot_traj, prot_std = r["prot_traj"], r["prot_std"]

        ax1.plot(t, rna_traj, color="tab:blue", marker="o", label="RNA")
        ax1.fill_between(
            t, rna_traj - rna_std, rna_traj + rna_std,
            color="tab:blue", alpha=0.2, linewidth=0,
        )

        ax2.plot(t, prot_traj, color="tab:orange", marker="s", label="Protein")
        ax2.fill_between(
            t, prot_traj - prot_std, prot_traj + prot_std,
            color="tab:orange", alpha=0.2, linewidth=0,
        )

        ax1.set_xscale("symlog", linthresh=1)
        ax1.set_xticks(TIMEPOINTS)
        ax1.set_xticklabels([str(tp) for tp in TIMEPOINTS])
        ax1.set_title(f"{r['gene_symbol']} ({r['protein_channel']})", fontsize=11, color="black")
        ax1.set_xlabel("Time (min)", color="black")
        ax1.set_ylabel("Mean RNA expr.", color="black")
        ax2.set_ylabel("Mean protein abund.", color="black")
        ax1.tick_params(axis="x", labelcolor="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax2.tick_params(axis="y", labelcolor="black")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Single shared legend
    handles = [
        plt.Line2D([0], [0], color="tab:blue", marker="o", label="RNA (mean $\\pm$ 1 std)"),
        plt.Line2D([0], [0], color="tab:orange", marker="s", label="Protein (mean $\\pm$ 1 std)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), labelcolor="black")
    fig.suptitle("RNA vs. protein trajectories, highly-varying signalling genes", y=1.05, fontsize=13, color="black")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", type=str, required=True)
    parser.add_argument("--prot_path", type=str, required=True)
    parser.add_argument("--mapping_csv", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=9)
    parser.add_argument("--out_png", type=str, default="gene_protein_trajectories_new.png")
    parser.add_argument("--out_csv", type=str, default="gene_protein_trajectory_data_new.csv")
    args = parser.parse_args()
    main(args.rna_path, args.prot_path, args.mapping_csv, args.top_n, args.out_png, args.out_csv)