"""
25_aggregate_myeloid_multiseed.py

Purpose: Aggregate all completed myeloid multi-seed runs (FiLM-pre and/or
gated) into one summary: mean +/- std R (from each seed's results.json)
and mean +/- std CCC (computed fresh from each seed's saved pred/truth
CSVs, per-protein convention, averaged across all 3 pairs per seed).

Usage:
    python 25_aggregate_myeloid_multiseed.py \
        --seed_dirs myeloid_finetune_seed1 myeloid_finetune_seed2 myeloid_finetune_seed3 myeloid_finetune_seed4 myeloid_finetune_seed5 \
        --model_label "FiLM-pre" \
        --out_csv myeloid_filmpre_multiseed_summary.csv

    python 25_aggregate_myeloid_multiseed.py \
        --seed_dirs myeloid_finetune_gated_seed1 myeloid_finetune_gated_seed2 myeloid_finetune_gated_seed3 myeloid_finetune_gated_seed4 myeloid_finetune_gated_seed5 \
        --model_label "Gated+CCC" \
        --out_csv myeloid_gated_multiseed_summary.csv
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def per_protein_r_ccc_mean(pred_csv, truth_csv):
    y_pred = pd.read_csv(pred_csv)
    y_true = pd.read_csv(truth_csv)
    protein_cols = [c for c in y_pred.columns if c not in ("t_src", "t_tgt")]
    rs, cccs = [], []
    for p in protein_cols:
        yt, yp = y_true[p].values, y_pred[p].values
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            continue
        r, _ = pearsonr(yt, yp)
        mean_t, mean_p = yt.mean(), yp.mean()
        var_t, var_p = yt.var(), yp.var()
        cov = np.mean((yt - mean_t) * (yp - mean_p))
        ccc = (2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12)
        if not np.isnan(r):
            rs.append(r)
            cccs.append(ccc)
    return (float(np.mean(rs)) if rs else np.nan,
            float(np.mean(cccs)) if cccs else np.nan)


def main(seed_dirs, model_label, out_csv):
    rows = []
    for sd in seed_dirs:
        sd = Path(sd)
        results_json = sd / "results.json"
        if not results_json.exists():
            print(f"  [skip] {sd}: no results.json found")
            continue
        with open(results_json) as f:
            res = json.load(f)
        test_r = res.get("test_R", res.get("final_test_R", np.nan))

        pred_files = sorted(glob.glob(str(sd / "joint_y_pred_t*.csv")))
        pair_rs, pair_cccs = [], []
        for pred_path in pred_files:
            truth_path = pred_path.replace("y_pred", "y_truth")
            if not Path(truth_path).exists():
                continue
            r, ccc = per_protein_r_ccc_mean(pred_path, truth_path)
            pair_rs.append(r)
            pair_cccs.append(ccc)

        avg_ccc = float(np.mean(pair_cccs)) if pair_cccs else np.nan
        avg_r_from_pairs = float(np.mean(pair_rs)) if pair_rs else np.nan

        rows.append({
            "seed_dir": sd.name,
            "test_R_reported": test_r,
            "avg_R_from_pairs": avg_r_from_pairs,
            "avg_CCC_from_pairs": avg_ccc,
            "n_pairs_found": len(pair_rs),
        })
        print(f"  {sd.name}: reported test_R={test_r:.4f} | recomputed avg R={avg_r_from_pairs:.4f} "
              f"| avg CCC={avg_ccc:.4f} (from {len(pair_rs)} pairs)")

    if not rows:
        raise FileNotFoundError("No valid seed directories with results.json found.")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_label} ({len(rows)} seeds)")
    print(f"{'='*60}")
    print(f"Mean R (reported, test_R):     {df['test_R_reported'].mean():.4f} +/- {df['test_R_reported'].std():.4f}")
    print(f"Mean R (recomputed from pairs): {df['avg_R_from_pairs'].mean():.4f} +/- {df['avg_R_from_pairs'].std():.4f}")
    print(f"Mean CCC:                       {df['avg_CCC_from_pairs'].mean():.4f} +/- {df['avg_CCC_from_pairs'].std():.4f}")
    print(f"\nSaved per-seed detail to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--model_label", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()
    main(args.seed_dirs, args.model_label, args.out_csv)
