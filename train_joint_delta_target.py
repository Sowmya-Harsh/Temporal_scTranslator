"""
train_joint_delta_target.py
──────────────────────────────
Predicts the CHANGE in protein abundance (protein_t_tgt - protein_t_src)
instead of absolute protein_t_tgt. Hypothesis: isolating what actually
CHANGED due to the biological transition is an easier target than
re-deriving each protein's whole baseline level from scratch every
time -- especially since baseline levels vary a lot between proteins
(some near 0, some near 1 after normalization) while the CHANGE may be
a more homogeneous, learnable quantity.

Requires protein_t_src (the source cell's own real protein reading --
same requirement as train_joint_rna_protein_input.py) to compute the
delta target, and to reconstruct the absolute prediction at eval time
(predicted_abundance = protein_t_src + predicted_delta).

Same joint (all-pairs, one model) structure as train_joint.py. Not run
yet -- review before using.

Usage:
    python train_joint_delta_target.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy hvg --n_genes 100 \
        --ot_type soft --ot_reg 0.01 \
        --model film_post \
        --epochs 100 --outdir results/joint_delta_target
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from models import build_model
from train_joint import (
    CURRICULUM, split_cells_per_timepoint, get_ot_targets,
    concordance_correlation_coefficient, normalize,
)


class DeltaPairDataset(Dataset):
    """Input = normalized RNA_t_src. Target = OT-matched protein_t_tgt
    MINUS the source cell's own real protein_t_src (the delta). Also
    keeps protein_t_src around (unnormalized reference scale) so the
    absolute prediction can be reconstructed at eval time."""
    def __init__(self, rna_src, prot_src_real, ot_matched_targets, t_src, t_tgt):
        self.rna = rna_src.astype(np.float32)
        self.prot_src_norm = np.array([normalize(p) for p in prot_src_real.astype(np.float32)])
        self.delta_targets = (ot_matched_targets - self.prot_src_norm).astype(np.float32)
        self.t_src = t_src
        self.delta_t = t_tgt - t_src

    def __len__(self):
        return len(self.rna)

    def __getitem__(self, idx):
        x = normalize(self.rna[idx])
        y_delta = self.delta_targets[idx]
        prot_src = self.prot_src_norm[idx]
        t_info = np.array([np.log1p(self.t_src), np.log1p(self.delta_t)], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y_delta), torch.tensor(prot_src), torch.tensor(t_info)


def build_pair_dataset_delta(rna_selected, prot_full, train_cells, val_cells, test_cells,
                              t_src, t_tgt, ot_type, ot_reg):
    if t_src not in train_cells or t_tgt not in train_cells:
        return None, None, None
    train_src, train_tgt = train_cells[t_src], train_cells[t_tgt]
    val_src, test_src = val_cells[t_src], test_cells[t_src]
    if len(train_src) < 2 or len(train_tgt) < 2:
        return None, None, None

    print(f"  Building {t_src}->{t_tgt} (DELTA target): train_src={len(train_src)} train_tgt={len(train_tgt)}")
    train_ot = get_ot_targets(rna_selected[train_src], rna_selected[train_tgt], prot_full[train_tgt], ot_type, ot_reg)
    val_ot = get_ot_targets(rna_selected[val_src], rna_selected[train_tgt], prot_full[train_tgt], ot_type, ot_reg) if len(val_src) > 1 else None
    test_ot = get_ot_targets(rna_selected[test_src], rna_selected[train_tgt], prot_full[train_tgt], ot_type, ot_reg) if len(test_src) > 1 else None

    train_ds = DeltaPairDataset(rna_selected[train_src], prot_full[train_src], train_ot, t_src, t_tgt)
    val_ds = DeltaPairDataset(rna_selected[val_src], prot_full[val_src], val_ot, t_src, t_tgt) if val_ot is not None else None
    test_ds = DeltaPairDataset(rna_selected[test_src], prot_full[test_src], test_ot, t_src, t_tgt) if test_ot is not None else None
    return train_ds, val_ds, test_ds


def to_delta_range(sigmoid_out):
    """Rescales the model's raw sigmoid output ([0,1]) to a valid delta
    range ([-1,1]), since delta targets (difference of two [0,1]-bounded
    values) can be negative, which a raw sigmoid output cannot represent."""
    return 2.0 * sigmoid_out - 1.0


def evaluate_per_pair_delta(model, test_ds_by_pair, device, protein_names, outdir):
    """Reconstructs ABSOLUTE predictions (prot_src + predicted_delta) before
    computing metrics -- so results are comparable to the non-delta scripts."""
    model.eval()
    results = {}
    for (t_src, t_tgt), ds in test_ds_by_pair.items():
        if ds is None or len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        pred_abs_all, truth_abs_all = [], []
        with torch.no_grad():
            for x, y_delta, prot_src, t_info in loader:
                x, t_info = x.to(device), t_info.to(device)
                pred_delta = to_delta_range(model(x, t_info)).cpu().numpy()
                pred_abs = np.clip(prot_src.numpy() + pred_delta, 0, 1)  # stay in valid [0,1] range
                truth_abs = np.clip(prot_src.numpy() + y_delta.numpy(), 0, 1)
                pred_abs_all.append(pred_abs)
                truth_abs_all.append(truth_abs)
        pred_abs_all = np.concatenate(pred_abs_all, axis=0)
        truth_abs_all = np.concatenate(truth_abs_all, axis=0)

        pred_df = pd.DataFrame(pred_abs_all, columns=protein_names)
        pred_df.insert(0, "t_src", t_src); pred_df.insert(1, "t_tgt", t_tgt)
        pred_df.to_csv(os.path.join(outdir, f"delta_y_pred_t{int(t_src)}to{int(t_tgt)}.csv"), index=False)

        truth_df = pd.DataFrame(truth_abs_all, columns=protein_names)
        truth_df.insert(0, "t_src", t_src); truth_df.insert(1, "t_tgt", t_tgt)
        truth_df.to_csv(os.path.join(outdir, f"delta_y_truth_t{int(t_src)}to{int(t_tgt)}.csv"), index=False)

        mse = float(np.mean((pred_abs_all - truth_abs_all) ** 2))
        per_r, per_ccc = [], []
        for i in range(truth_abs_all.shape[1]):
            if np.std(truth_abs_all[:, i]) > 1e-8 and np.std(pred_abs_all[:, i]) > 1e-8:
                r, _ = pearsonr(truth_abs_all[:, i], pred_abs_all[:, i])
                per_r.append(r)
                per_ccc.append(concordance_correlation_coefficient(truth_abs_all[:, i], pred_abs_all[:, i]))
        mean_r = float(np.nanmean(per_r)) if per_r else float("nan")
        ccc = float(np.mean(per_ccc)) if per_ccc else float("nan")
        results[(t_src, t_tgt)] = {"mse": mse, "mean_pearson_r": mean_r, "ccc": ccc, "n_test": len(ds)}
        print(f"    t{int(t_src)}->t{int(t_tgt)}: mse={mse:.4f} mean_r={mean_r:.4f} ccc={ccc:.4f} (n={len(ds)})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--condition", default="aIg")
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], required=True)
    parser.add_argument("--n_genes", required=True)
    parser.add_argument("--ot_type", choices=["soft", "hard", "hard_assignment"], required=True)
    parser.add_argument("--ot_reg", type=float, default=0.1)
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Config: DELTA TARGET, gene_strategy={args.gene_strategy} n_genes={n_genes_arg} "
          f"ot_type={args.ot_type} model={args.model}")

    rna_adata = ad.read_h5ad(args.rna)
    prot_adata = ad.read_h5ad(args.protein)
    mask = rna_adata.obs["condition"] == args.condition
    rna_adata = rna_adata[mask].copy()
    prot_adata = prot_adata[mask].copy()

    rna_full = rna_adata.layers["counts"]
    prot_full = prot_adata.layers["counts"]
    rna_full = (rna_full.toarray() if hasattr(rna_full, "toarray") else rna_full).astype(np.float32)
    prot_full = (prot_full.toarray() if hasattr(prot_full, "toarray") else prot_full).astype(np.float32)
    protein_names = list(prot_adata.var_names)
    obs_time = rna_adata.obs["time_min"].values.astype(float)

    gene_idx = select_genes(rna_adata, args.gene_strategy, n_genes_arg, groupby="time_min")
    rna_selected = rna_full[:, gene_idx]

    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)
    available_pairs = [(s, t) for s, t in CURRICULUM if s in train_cells and t in train_cells]
    print(f"Available pairs: {available_pairs}")

    train_ds_by_pair, val_ds_by_pair, test_ds_by_pair = {}, {}, {}
    for (t_src, t_tgt) in available_pairs:
        tr, va, te = build_pair_dataset_delta(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                               t_src, t_tgt, args.ot_type, args.ot_reg)
        if tr is not None:
            train_ds_by_pair[(t_src, t_tgt)] = tr
            val_ds_by_pair[(t_src, t_tgt)] = va
            test_ds_by_pair[(t_src, t_tgt)] = te

    full_train = ConcatDataset(list(train_ds_by_pair.values()))
    full_val = ConcatDataset([v for v in val_ds_by_pair.values() if v is not None])
    # NOTE: DeltaPairDataset yields 4 items (x, y_delta, prot_src, t_info),
    # so the model still only sees (x, t_info) -- prot_src is only used for
    # reconstructing absolute predictions at eval time, not fed to the model.
    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(full_val, batch_size=args.batch_size, shuffle=False)

    n_proteins = prot_full.shape[1]
    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=n_proteins).to(device)

    # FIX (previously flagged as an open bug): build_model's output is
    # sigmoid-bounded to [0,1], but delta targets (difference of two
    # [0,1]-bounded values) can be negative, spanning [-1,1]. Rescale via
    # to_delta_range() (module-level, defined above) wherever the model's
    # raw output is used as a predicted delta -- now applied consistently
    # in training, validation, AND final evaluation (previously only the
    # first two, a real inconsistency that has been fixed here).

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    ckpt_path = os.path.join(args.outdir, "delta_model_best.pt")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y_delta, prot_src, t_info in train_loader:
            x, y_delta, t_info = x.to(device), y_delta.to(device), t_info.to(device)
            optimizer.zero_grad()
            pred_delta = to_delta_range(model(x, t_info))
            loss = criterion(pred_delta, y_delta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_delta, prot_src, t_info in val_loader:
                x, y_delta, t_info = x.to(device), y_delta.to(device), t_info.to(device)
                val_loss += criterion(to_delta_range(model(x, t_info)), y_delta).item()
        val_loss /= max(1, len(val_loader))

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}, ckpt_path)
        if epoch % 10 == 0 or epoch == 1 or improved:
            flag = " *improved*" if improved else ""
            print(f"  Epoch {epoch:>4}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}{flag}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n=== FINAL: delta-target joint model, evaluated per pair (absolute reconstructed) ===")
    results = evaluate_per_pair_delta(model, test_ds_by_pair, device, protein_names, args.outdir)

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({"config": vars(args), "results": {f"{s}->{t}": v for (s, t), v in results.items()}}, f, indent=2)
    print(f"\nSaved to {args.outdir}/. Compare against train_joint.py's absolute-target results.")


if __name__ == "__main__":
    main()
