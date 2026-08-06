"""
03_baselines_v2.py

Drop-in replacement for 03_baselines.py — uses your real DataStoreOT
pipeline (qurie_dataset_ot.py) instead of guessing at saved-array file
names, so baselines are evaluated against EXACTLY the same OT-matched
targets your main model is evaluated against (Section 4.8), on the
same train/val/test cell splits (Section 4.9).

Three baselines, no temporal conditioning, no correlation loss, no
mixup, no OT reweighting for the baselines' own predictions (only used
to build the shared ground-truth targets, same as your main pipeline):

  1. Persistence — y_hat = the source cell's OWN real protein vector
     (normalized), i.e. "nothing changes between t_src and t_tgt."
     If this beats or matches your model, temporal modelling isn't
     adding anything.
  2. Population mean — y_hat = train-split mean protein vector at
     t_tgt (this is literally self.train_mean_prot[t_tgt] already
     computed inside DataStoreOT — reused directly, not recomputed).
  3. Linear regression — per-protein-channel linear model fit on
     (train RNA, train OT-matched targets), predicted on test RNA.
     No FiLM, no nonlinearity, no explicit time conditioning at all
     (time is implicit only via which t_tgt-specific mean/OT-target
     was used at fit time).

Ground truth for all three = the SAME test_targets your CurriculumDatasetOT
test split serves to the real model (built via DataStoreOT._get_targets,
i.e. soft/Sinkhorn OT against the full train_tgt population).

Usage (run in the same environment / same directory as your training script,
since it imports qurie_dataset_ot.py directly):

    python 03_baselines_v2.py \
        --rna_path dataset/qurie_RNA_clean_aIg.h5ad \
        --prot_path dataset/qurie_Protein_clean_aIg.h5ad \
        --ot_reg 0.01 \
        --out_csv baselines_results.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from qurie_dataset_ot import DataStoreOT, CURRICULUM, normalize


def per_protein_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_proteins = y_true.shape[1]
    rs = []
    for p in range(n_proteins):
        yt, yp = y_true[:, p], y_pred[:, p]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            continue
        r, _ = pearsonr(yt, yp)
        if not np.isnan(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def per_protein_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_proteins = y_true.shape[1]
    cccs = []
    for p in range(n_proteins):
        yt, yp = y_true[:, p], y_pred[:, p]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            continue
        mean_t, mean_p = np.mean(yt), np.mean(yp)
        var_t, var_p = np.var(yt), np.var(yp)
        cov = np.mean((yt - mean_t) * (yp - mean_p))
        ccc = (2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12)
        cccs.append(ccc)
    return float(np.mean(cccs)) if cccs else float("nan")


def main(rna_path, prot_path, ot_reg, ot_max_genes, out_csv, seed):
    # use_mixup=False: baselines should see the same real cells everyone
    # else sees, not the mixup-augmented training set — mixup is a training-time
    # trick for the real model, not something a "does temporal modelling help
    # at all" baseline should benefit or suffer from.
    store = DataStoreOT(
        rna_path=rna_path,
        protein_path=prot_path,
        condition="aIg",
        use_ot=True,
        use_mixup=False,
        ot_reg=ot_reg,
        ot_max_genes=ot_max_genes,
        seed=seed,
    )

    rows = []
    for t_src, t_tgt in CURRICULUM:
        print(f"\n=== {t_src} -> {t_tgt} ===")
        train_src = store.train_cells[t_src]
        train_tgt = store.train_cells[t_tgt]
        test_src = store.test_cells[t_src]

        # Ground truth: same OT-matched targets the real model is scored against
        train_targets = store._get_targets(train_src, train_tgt, "train")
        test_targets = store._get_targets(test_src, train_tgt, "test")

        rna_train = np.array([normalize(store.rna[c]) for c in train_src])
        rna_test = np.array([normalize(store.rna[c]) for c in test_src])

        # 1. Persistence: source cell's OWN real (normalized) protein vector
        persistence_pred = np.array([normalize(store.prot[c]) for c in test_src])
        r_persist = per_protein_r(test_targets, persistence_pred)
        ccc_persist = per_protein_ccc(test_targets, persistence_pred)

        # 2. Population mean at t_tgt (reuses DataStoreOT's own precomputed mean)
        mean_pred = np.tile(store.train_mean_prot[t_tgt], (len(test_src), 1))
        r_mean = per_protein_r(test_targets, mean_pred)
        ccc_mean = per_protein_ccc(test_targets, mean_pred)

        # 3. Linear regression: RNA -> OT-matched target, no time conditioning
        lr = LinearRegression()
        lr.fit(rna_train, train_targets)
        lr_pred = lr.predict(rna_test)
        r_lr = per_protein_r(test_targets, lr_pred)
        ccc_lr = per_protein_ccc(test_targets, lr_pred)

        print(f"  persistence R={r_persist:.3f} | pop-mean R={r_mean:.3f} | linreg R={r_lr:.3f}")

        rows.append({
            "tsrc": t_src, "ttgt": t_tgt, "delta_t": t_tgt - t_src,
            "n_test_cells": len(test_src),
            "R_persistence": r_persist, "CCC_persistence": ccc_persist,
            "R_population_mean": r_mean, "CCC_population_mean": ccc_mean,
            "R_linear_regression": r_lr, "CCC_linear_regression": ccc_lr,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print("\nOverall means (compare directly to Table 5.7's final-model mean R=0.359):")
    print(df[["R_persistence", "R_population_mean", "R_linear_regression"]].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", type=str, required=True)
    parser.add_argument("--prot_path", type=str, required=True)
    parser.add_argument("--ot_reg", type=float, default=0.01,
                         help="Match whatever ot_reg your final model config used (Section 4.3.1: 0.01 default, 0.005 for longest interval)")
    parser.add_argument("--ot_max_genes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1105,
                         help="Match DataStoreOT's default seed so splits align with your main run if possible")
    parser.add_argument("--out_csv", type=str, default="baselines_results.csv")
    args = parser.parse_args()
    main(args.rna_path, args.prot_path, args.ot_reg, args.ot_max_genes, args.out_csv, args.seed)