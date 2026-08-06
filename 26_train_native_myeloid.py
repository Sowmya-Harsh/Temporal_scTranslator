"""
26_train_native_myeloid.py

Purpose: Train the gated (or film_pre) architecture NATIVELY on myeloid
data -- random initialization, myeloid's own train/val/test split, no
QuRIE-seq checkpoint involved at all. This is a DIFFERENT, complementary
test to the adapter-based fine-tuning already done (06_generalized_
finetune.py): it asks whether the near-zero CCC on myeloid is specific
to TRANSFERRING a QuRIE-seq-tuned representation, or whether the
architecture itself struggles with myeloid's data regardless of where
training started.

Usage:
    python 26_train_native_myeloid.py \
        --rna_path myeloid_geo_data/myeloid_RNA_standardized.h5ad \
        --prot_path myeloid_geo_data/myeloid_Protein_standardized.h5ad \
        --gene_panel_size 1000 \
        --model gated \
        --correlation_loss_weight 0.3 --ccc_loss_weight 0.3 \
        --epochs 100 --seed 42 \
        --outdir results/native_myeloid_gated
"""
import argparse
import json
import os
import sys

import numpy as np
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_distances

try:
    import ot
except ImportError:
    raise SystemExit("POT required: pip install pot --break-system-packages")

sys.path.append(os.path.dirname(__file__))
from models import build_model
from training_improvements import correlation_loss, ccc_loss


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
        self.rna_src = rna_src
        self.targets = targets
        self.t_src = t_src
        self.t_tgt = t_tgt

    def __len__(self):
        return len(self.rna_src)

    def __getitem__(self, idx):
        x = normalize(self.rna_src[idx])
        y = self.targets[idx]
        t_info = np.array([np.log1p(self.t_src), np.log1p(self.t_tgt - self.t_src)], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y), torch.tensor(t_info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", required=True)
    parser.add_argument("--prot_path", required=True)
    parser.add_argument("--gene_panel_size", type=int, default=1000)
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post", "gated", "gated_post"], required=True)
    parser.add_argument("--correlation_loss_weight", type=float, default=0.0)
    parser.add_argument("--ccc_loss_weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ot_reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading myeloid data natively (no QuRIE-seq checkpoint)...")
    rna = ad.read_h5ad(args.rna_path)
    prot = ad.read_h5ad(args.prot_path)

    deg_genes = select_deg_genes(rna, "time_min", args.gene_panel_size)
    deg_genes = [g for g in deg_genes if g in rna.var_names]
    rna_X = rna[:, deg_genes].X
    rna_X = (rna_X.toarray() if hasattr(rna_X, "toarray") else rna_X).astype(np.float32)
    prot_X = (prot.X.toarray() if hasattr(prot.X, "toarray") else prot.X).astype(np.float32)
    protein_names = list(prot.var_names)

    times = sorted(rna.obs["time_min"].unique().tolist())
    print(f"Timepoints: {times}, genes: {len(deg_genes)}, proteins: {len(protein_names)}")

    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, test_idx = {}, {}, {}
    for t in times:
        idx = np.where(np.isclose(rna.obs["time_min"].values.astype(float), t))[0]
        rng.shuffle(idx)
        n_tr, n_va = int(0.8 * len(idx)), int(0.1 * len(idx))
        train_idx[t], val_idx[t], test_idx[t] = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    pairs = [(t1, t2) for i, t1 in enumerate(times) for t2 in times[i + 1:]]
    train_ds_list, val_ds_by_pair, test_ds_by_pair = [], {}, {}
    for t_src, t_tgt in pairs:
        tgt_train = train_idx[t_tgt]
        tr_targets = compute_ot_targets(rna_X[train_idx[t_src]], rna_X[tgt_train], prot_X[tgt_train], reg=args.ot_reg)
        va_targets = compute_ot_targets(rna_X[val_idx[t_src]], rna_X[tgt_train], prot_X[tgt_train], reg=args.ot_reg)
        te_targets = compute_ot_targets(rna_X[test_idx[t_src]], rna_X[tgt_train], prot_X[tgt_train], reg=args.ot_reg)
        train_ds_list.append(PairDS(rna_X[train_idx[t_src]], tr_targets, t_src, t_tgt))
        val_ds_by_pair[(t_src, t_tgt)] = PairDS(rna_X[val_idx[t_src]], va_targets, t_src, t_tgt)
        test_ds_by_pair[(t_src, t_tgt)] = PairDS(rna_X[test_idx[t_src]], te_targets, t_src, t_tgt)
        print(f"  {t_src}->{t_tgt}: train={len(train_idx[t_src])} val={len(val_idx[t_src])} test={len(test_idx[t_src])}")

    model = build_model(args.model, n_genes=len(deg_genes), n_proteins=len(protein_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    train_loader = DataLoader(ConcatDataset(train_ds_list), batch_size=args.batch_size, shuffle=True)
    ckpt_path = os.path.join(args.outdir, "native_myeloid_best.pt")
    best_val = float("inf")

    print("\nTraining natively on myeloid...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, t_info in train_loader:
            x, y, t_info = x.to(device), y.to(device), t_info.to(device)
            optimizer.zero_grad()
            y_hat = model(x, t_info)
            loss = mse(y_hat, y)
            if args.correlation_loss_weight > 0:
                loss = loss + args.correlation_loss_weight * correlation_loss(y_hat, y)
            if args.ccc_loss_weight > 0:
                loss = loss + args.ccc_loss_weight * ccc_loss(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for pair, ds in val_ds_by_pair.items():
                if len(ds) == 0:
                    continue
                loader = DataLoader(ds, batch_size=args.batch_size)
                for x, y, t_info in loader:
                    x, y, t_info = x.to(device), y.to(device), t_info.to(device)
                    val_losses.append(mse(model(x, t_info), y).item())
        val_loss = np.mean(val_losses) if val_losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(), "val_loss": val_loss}, ckpt_path)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: val_loss={val_loss:.4f}")

    print(f"\nLoading best checkpoint (val_loss={best_val:.4f})...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("\n=== FINAL: native myeloid model evaluated per pair ===")
    all_rs, all_cccs = [], []
    for pair, ds in test_ds_by_pair.items():
        if len(ds) < 5:
            continue
        loader = DataLoader(ds, batch_size=args.batch_size)
        preds, trues = [], []
        with torch.no_grad():
            for x, y, t_info in loader:
                x, t_info = x.to(device), t_info.to(device)
                preds.append(model(x, t_info).cpu().numpy())
                trues.append(y.numpy())
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        rs, cccs = [], []
        for j in range(preds.shape[1]):
            if np.std(trues[:, j]) > 1e-8 and np.std(preds[:, j]) > 1e-8:
                r, _ = pearsonr(trues[:, j], preds[:, j])
                mean_t, mean_p = trues[:, j].mean(), preds[:, j].mean()
                var_t, var_p = trues[:, j].var(), preds[:, j].var()
                cov = np.mean((trues[:, j] - mean_t) * (preds[:, j] - mean_p))
                ccc = (2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12)
                rs.append(r)
                cccs.append(ccc)
        pair_r, pair_ccc = np.mean(rs), np.mean(cccs)
        all_rs.append(pair_r)
        all_cccs.append(pair_ccc)
        print(f"  {pair[0]}->{pair[1]}: R={pair_r:.4f} CCC={pair_ccc:.4f} (n={len(ds)})")

    print(f"\nOverall: mean R={np.mean(all_rs):.4f}, mean CCC={np.mean(all_cccs):.4f}")
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump({"mean_R": float(np.mean(all_rs)), "mean_CCC": float(np.mean(all_cccs))}, f, indent=2)


if __name__ == "__main__":
    main()
