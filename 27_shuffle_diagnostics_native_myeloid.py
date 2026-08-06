"""
27_shuffle_diagnostics_native_myeloid.py

Purpose: Run the three shuffle diagnostics (prediction-shuffle, time-
shuffle, RNA-shuffle) on the native-myeloid-trained checkpoint from
26_train_native_myeloid.py. Reuses that script's exact data loading and
train/val/test split logic (same seed, same gene selection), since
posthoc_shuffle_tests.py is QuRIE-seq-specific (expects a 'condition'
column myeloid doesn't have) and cannot be reused directly here.

Usage:
    python 27_shuffle_diagnostics_native_myeloid.py \
        --rna_path myeloid_geo_data/myeloid_RNA_standardized.h5ad \
        --prot_path myeloid_geo_data/myeloid_Protein_standardized.h5ad \
        --gene_panel_size 1000 \
        --model gated \
        --checkpoint results/native_myeloid_gated/native_myeloid_best.pt \
        --seed 42 \
        --outdir results/shuffle_native_myeloid
"""
import argparse
import json
import os
import sys

import numpy as np
import anndata as ad
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_distances

try:
    import ot
except ImportError:
    raise SystemExit("POT required: pip install pot --break-system-packages")

sys.path.append(os.path.dirname(__file__))
from models import build_model


def normalize(x, eps=1e-8):
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        return np.full_like(x, 1e-8, dtype=np.float32)
    return (1e-8 + (x - xmin) / (xmax - xmin + eps)).astype(np.float32)


def select_deg_genes(rna_adata, time_col, panel_size):
    a = rna_adata.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.obs["_time_group"] = a.obs[time_col].astype(str)
    sc.tl.rank_genes_groups(a, groupby="_time_group", method="wilcoxon")
    best_rank = {}
    for g in a.obs["_time_group"].unique():
        names = a.uns["rank_genes_groups"]["names"][g]
        for rank, gene in enumerate(names):
            if gene not in best_rank or rank < best_rank[gene]:
                best_rank[gene] = rank
    return sorted(best_rank.keys(), key=lambda g: best_rank[g])[:panel_size]


def compute_ot_targets(rna_src, rna_tgt, prot_tgt, reg=0.01, max_genes=2000):
    n_src, n_tgt = len(rna_src), len(rna_tgt)
    gene_var = rna_src.var(axis=0) + rna_tgt.var(axis=0)
    top_genes = np.argsort(gene_var)[-max_genes:]
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)
    a = np.ones(n_src) / n_src
    b = np.ones(n_tgt) / n_tgt
    T = ot.sinkhorn(a, b, C, reg=reg)
    T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-8)
    targets = (T_norm @ prot_tgt).astype(np.float32)
    return np.array([normalize(p) for p in targets])


class PairDS(Dataset):
    def __init__(self, rna_src, targets, t_src, t_tgt):
        self.rna_src, self.targets, self.t_src, self.t_tgt = rna_src, targets, t_src, t_tgt

    def __len__(self):
        return len(self.rna_src)

    def __getitem__(self, idx):
        x = normalize(self.rna_src[idx])
        y = self.targets[idx]
        t_info = np.array([np.log1p(self.t_src), np.log1p(self.t_tgt - self.t_src)], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y), torch.tensor(t_info)


def within_pair_r(pred, true, min_cells=5):
    all_rs = []
    for p, t in zip(pred, true):
        if len(p) < min_cells:
            continue
        for j in range(p.shape[1]):
            if p[:, j].std() > 1e-6 and t[:, j].std() > 1e-6:
                r, _ = pearsonr(p[:, j], t[:, j])
                if not np.isnan(r):
                    all_rs.append(r)
    return float(np.mean(all_rs)) if all_rs else 0.0


def run_inference(model, test_ds_by_pair, device, shuffle_rna=False, shuffle_time=False, seed=42):
    rng = np.random.default_rng(seed)
    model.eval()
    preds, trues = [], []
    for pair, ds in test_ds_by_pair.items():
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        p_all, t_all = [], []
        with torch.no_grad():
            for x, y, t_info in loader:
                if shuffle_rna:
                    perm = torch.randperm(x.shape[0])
                    x = x[perm]
                if shuffle_time:
                    fake_src = torch.tensor(rng.uniform(0, 30240, size=x.shape[0]), dtype=torch.float32)
                    fake_delta = torch.tensor(rng.uniform(0, 30240, size=x.shape[0]), dtype=torch.float32)
                    t_info = torch.stack([torch.log1p(fake_src), torch.log1p(fake_delta)], dim=1)
                x, t_info = x.to(device), t_info.to(device)
                p_all.append(model(x, t_info).cpu().numpy())
                t_all.append(y.numpy())
        preds.append(np.concatenate(p_all))
        trues.append(np.concatenate(t_all))
    return preds, trues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", required=True)
    parser.add_argument("--prot_path", required=True)
    parser.add_argument("--gene_panel_size", type=int, default=1000)
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post", "gated", "gated_post"], required=True)
    parser.add_argument("--ot_reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print("Reconstructing native myeloid test sets (same seed/config as training)...")
    rna = ad.read_h5ad(args.rna_path)
    prot = ad.read_h5ad(args.prot_path)

    deg_genes = select_deg_genes(rna, "time_min", args.gene_panel_size)
    deg_genes = [g for g in deg_genes if g in rna.var_names]
    rna_X = rna[:, deg_genes].X
    rna_X = (rna_X.toarray() if hasattr(rna_X, "toarray") else rna_X).astype(np.float32)
    prot_X = (prot.X.toarray() if hasattr(prot.X, "toarray") else prot.X).astype(np.float32)
    protein_names = list(prot.var_names)

    times = sorted(rna.obs["time_min"].unique().tolist())
    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, test_idx = {}, {}, {}
    for t in times:
        idx = np.where(np.isclose(rna.obs["time_min"].values.astype(float), t))[0]
        rng.shuffle(idx)
        n_tr, n_va = int(0.8 * len(idx)), int(0.1 * len(idx))
        train_idx[t], val_idx[t], test_idx[t] = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    pairs = [(t1, t2) for i, t1 in enumerate(times) for t2 in times[i + 1:]]
    test_ds_by_pair = {}
    for t_src, t_tgt in pairs:
        te_targets = compute_ot_targets(rna_X[test_idx[t_src]], rna_X[train_idx[t_tgt]],
                                         prot_X[train_idx[t_tgt]], reg=args.ot_reg)
        test_ds_by_pair[(t_src, t_tgt)] = PairDS(rna_X[test_idx[t_src]], te_targets, t_src, t_tgt)

    model = build_model(args.model, n_genes=len(deg_genes), n_proteins=len(protein_names)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (val_loss={ckpt.get('val_loss', '?')})")

    print("\n=== REAL (baseline) ===")
    pred_real, true_real = run_inference(model, test_ds_by_pair, device)
    real_r = within_pair_r(pred_real, true_real)
    print(f"Real within-pair R: {real_r:.4f}")

    # Save per-cell prediction/truth CSVs, same format as everywhere else
    # in the thesis, so you can inspect the actual numbers directly.
    import pandas as pd
    pair_list = list(test_ds_by_pair.keys())
    for (t_src, t_tgt), pred_arr, true_arr in zip(pair_list, pred_real, true_real):
        pred_df = pd.DataFrame(pred_arr, columns=protein_names)
        pred_df.insert(0, "t_tgt", t_tgt)
        pred_df.insert(0, "t_src", t_src)
        truth_df = pd.DataFrame(true_arr, columns=protein_names)
        truth_df.insert(0, "t_tgt", t_tgt)
        truth_df.insert(0, "t_src", t_src)
        pred_path = os.path.join(args.outdir, f"native_y_pred_t{int(t_src)}to{int(t_tgt)}.csv")
        truth_path = os.path.join(args.outdir, f"native_y_truth_t{int(t_src)}to{int(t_tgt)}.csv")
        pred_df.to_csv(pred_path, index=False)
        truth_df.to_csv(truth_path, index=False)
        print(f"  Saved {os.path.basename(pred_path)} and {os.path.basename(truth_path)} ({len(pred_arr)} cells)")

    print("\n=== TEST 1: PREDICTION SHUFFLE ===")
    rng2 = np.random.default_rng(args.seed)
    pred_shuf = [p[rng2.permutation(len(p))] for p in pred_real]
    r_pred = within_pair_r(pred_shuf, true_real)
    print(f"Real R={real_r:.4f} | Shuffled R={r_pred:.4f} | Diff={real_r - r_pred:.4f}")
    print("VERDICT:", "GENUINE (real >> shuffled)" if real_r - r_pred > 0.1 else "SUSPICIOUS -- shuffling did not hurt much")

    print("\n=== TEST 2: TIME SHUFFLE ===")
    pred_tshuf, _ = run_inference(model, test_ds_by_pair, device, shuffle_time=True, seed=args.seed)
    r_time = within_pair_r(pred_tshuf, true_real)
    print(f"Real R={real_r:.4f} | Time-shuffled R={r_time:.4f} | Diff={real_r - r_time:.4f}")
    print("VERDICT:", "model GENUINELY uses temporal information" if real_r - r_time > 0.05 else "model may NOT be using time meaningfully")

    print("\n=== TEST 3: RNA SHUFFLE ===")
    pred_rshuf, _ = run_inference(model, test_ds_by_pair, device, shuffle_rna=True, seed=args.seed)
    r_rna = within_pair_r(pred_rshuf, true_real)
    print(f"Real R={real_r:.4f} | RNA-shuffled R={r_rna:.4f} | Diff={real_r - r_rna:.4f}")
    print("VERDICT:", "model GENUINELY uses RNA signal" if real_r - r_rna > 0.1 else "SUSPICIOUS -- RNA shuffling did not hurt much")

    results = {"real_r": real_r, "pred_shuffle_r": r_pred, "time_shuffle_r": r_time, "rna_shuffle_r": r_rna}
    with open(os.path.join(args.outdir, "shuffle_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.outdir}/shuffle_results.json")


if __name__ == "__main__":
    main()
