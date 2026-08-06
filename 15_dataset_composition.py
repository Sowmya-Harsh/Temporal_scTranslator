"""
02_dataset_composition.py

Purpose: Characterise the composition and variance structure of the QuRIE-seq
dataset across all provided .h5ad variants (raw / mapped / cleaned / condition-split),
to (a) sanity-check the cleaning pipeline numbers already in the thesis (Table 3.1),
and (b) produce the raw stats needed for dataset composition figures.

Run this ON THE CLUSTER where the files live (not here) — it only needs
anndata/scanpy + pandas. Outputs a JSON summary + a few CSVs you can hand off
for plotting.

Usage:
    python 02_dataset_composition.py --data_dir Final_Dataset --out_dir dataset_summary
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad


FILES = {
    "protein_raw": "qurie_Protein.h5ad",
    "protein_mapped": "qurie_Protein_mapped.h5ad",
    "protein_clean_aIg": "qurie_Protein_clean_aIg.h5ad",
    "protein_clean_aIg_Ibru": "qurie_Protein_clean_aIg_Ibru.h5ad",
    "protein_clean_aIg_t180": "qurie_Protein_clean_aIg_t180.h5ad",
    "rna_raw": "qurie_RNA.h5ad",
    "rna_mapped": "qurie_RNA_mapped.h5ad",
    "rna_clean_aIg": "qurie_RNA_clean_aIg.h5ad",
    "rna_clean_aIg_Ibru": "qurie_RNA_clean_aIg_Ibru.h5ad",
    "rna_clean_aIg_t180": "qurie_RNA_clean_aIg_t180.h5ad",
}

# Candidate obs column names for timepoint / condition — adjust if your
# actual column names differ (script will print available obs columns
# for each file so you can confirm/adjust on first run).
TIME_COL_CANDIDATES = ["time_min", "time", "timepoint", "Time", "time_point", "minutes", "t"]
COND_COL_CANDIDATES = ["condition", "Condition", "treatment", "group"]


def find_col(obs_columns, candidates):
    for c in candidates:
        if c in obs_columns:
            return c
    return None


def summarize_h5ad(path: Path) -> dict:
    a = ad.read_h5ad(path, backed="r")  # backed mode: don't load X into memory
    obs_cols = list(a.obs.columns)
    time_col = find_col(obs_cols, TIME_COL_CANDIDATES)
    cond_col = find_col(obs_cols, COND_COL_CANDIDATES)

    summary = {
        "file": path.name,
        "n_obs": a.n_obs,
        "n_vars": a.n_vars,
        "obs_columns": obs_cols,
        "var_columns": list(a.var.columns),
        "detected_time_col": time_col,
        "detected_condition_col": cond_col,
        "layers": list(a.layers.keys()),
    }

    if time_col is not None:
        summary["cells_per_timepoint"] = (
            a.obs[time_col].value_counts().sort_index().to_dict()
        )
    if cond_col is not None:
        summary["cells_per_condition"] = a.obs[cond_col].value_counts().to_dict()
    if time_col is not None and cond_col is not None:
        ct = pd.crosstab(a.obs[time_col], a.obs[cond_col])
        summary["cells_per_timepoint_x_condition"] = ct.to_dict()

    # Sparsity / zero-signal channel check (protein files only, cheap enough
    # to load fully since var count is small — e.g. 58-80 channels)
    if "protein" in path.name.lower() and a.n_vars < 500:
        a_full = ad.read_h5ad(path)  # load fully, protein panels are tiny
        X = a_full.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        col_sum = X.sum(axis=0)
        col_max = X.max(axis=0)
        col_nonzero = (X != 0).sum(axis=0)
        zero_channels = int(((col_sum == 0) & (col_max == 0) & (col_nonzero == 0)).sum())
        summary["n_protein_channels"] = a_full.n_vars
        summary["n_zero_signal_channels"] = zero_channels
        summary["zero_signal_channel_names"] = [
            a_full.var_names[i]
            for i in range(a_full.n_vars)
            if col_sum[i] == 0 and col_max[i] == 0 and col_nonzero[i] == 0
        ]

    # Gene/protein-level variance summary (for HVG-vs-DEG style composition plots)
    if a.n_vars < 5000:
        a_full = ad.read_h5ad(path)
        X = a_full.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        per_feature_var = X.var(axis=0)
        summary["mean_per_feature_variance"] = float(np.mean(per_feature_var))
        summary["median_per_feature_variance"] = float(np.median(per_feature_var))

    return summary


def main(data_dir: str, out_dir: str):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for key, fname in FILES.items():
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"[skip] {fname} not found in {data_dir}")
            continue
        print(f"[reading] {fname}")
        try:
            all_summaries[key] = summarize_h5ad(fpath)
        except Exception as e:
            print(f"[error] {fname}: {e}")
            all_summaries[key] = {"file": fname, "error": str(e)}

    with open(out_dir / "dataset_composition_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    # Flatten cells-per-timepoint-per-condition into one tidy CSV for plotting,
    # for the three "clean" files specifically (these map to Table 3.1)
    rows = []
    for key in ["protein_clean_aIg", "protein_clean_aIg_Ibru", "protein_clean_aIg_t180"]:
        s = all_summaries.get(key, {})
        cpt = s.get("cells_per_timepoint")
        if cpt:
            for tp, n in cpt.items():
                rows.append({"file": key, "timepoint": tp, "n_cells": n})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "cells_per_timepoint_tidy.csv", index=False)

    print(f"\nDone. Summary written to {out_dir/'dataset_composition_summary.json'}")
    print("If detected_time_col / detected_condition_col came back None for any file,")
    print("open the JSON, check 'obs_columns', and re-run with TIME_COL_CANDIDATES /")
    print("COND_COL_CANDIDATES edited to match your actual column names.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Final_Dataset")
    parser.add_argument("--out_dir", type=str, default="dataset_summary")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)