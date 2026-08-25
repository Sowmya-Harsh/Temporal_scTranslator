"""
train_joint.py
────────────────
Trains ONE model across ALL timepoint pairs (not one model per pair).
Same weights carried through the entire run -- no ensembling, no
separate models to combine afterward.

Two modes, controlled by --curriculum:
  --curriculum off (default): all pairs mixed together every epoch,
      from the start. Simple, standard.
  --curriculum on: pairs introduced progressively in the order defined
      by CURRICULUM (short delta_t first, then longer). Same model
      throughout -- curriculum only controls what data it sees when,
      not which model. Each stage trains for --epochs_per_stage
      epochs before the next (harder) pair is added to the mix.

Uses the SAME gene selection + OT target computation as train.py, just
applied once per pair up front, then all pairs' (rna, target, t_info)
tuples get pooled into one big dataset (or added stage-by-stage for
curriculum mode).

Usage:
    # joint, no curriculum
    python train_joint.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 100 \
        --ot_type soft --ot_reg 0.01 \
        --model film_post \
        --epochs 100 --outdir results/joint

    # curriculum
    python train_joint.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 100 \
        --ot_type soft --ot_reg 0.01 \
        --model film_post \
        --curriculum --epochs_per_stage 30 --outdir results/joint_curriculum
"""

import argparse
import os
import sys
import json
import csv

import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from ot_matching import compute_ot_hard
from models import build_model

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from qurie_dataset_ot import compute_ot_targets as compute_ot_soft, normalize
except ImportError:
    print("WARNING: qurie_dataset_ot.py not importable. Soft OT unavailable.")
    compute_ot_soft = None

    def normalize(x, eps=1e-8):
        xmin, xmax = float(x.min()), float(x.max())
        if xmin == xmax:
            return np.full_like(x, 1e-8, dtype=np.float32)
        return (1e-8 + (x - xmin) / (xmax - xmin + eps)).astype(np.float32)

# Same ordering as qurie_dataset_ot.py's CURRICULUM, sorted by delta_t
CURRICULUM = [
    (0, 2), (0, 4), (2, 6), (0, 6), (4, 60),
    (6, 60), (0, 60), (60, 180), (6, 180), (0, 180),
]


def concordance_correlation_coefficient(y_true, y_pred):
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return 0.0 if denom == 0 else float((2 * covariance) / denom)


class PairDataset(Dataset):
    """RNA + OT-matched targets + t_info for ONE (t_src,t_tgt) pair."""
    def __init__(self, rna, targets, t_src, t_tgt, shuffle_time=False, seed=1105):
        self.rna = rna.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.t_src = t_src
        self.delta_t = t_tgt - t_src
        self.shuffle_time = shuffle_time
        if shuffle_time:
            # CONTROL: assign each sample a random (fake) t_src/delta_t,
            # independent of the real transition -- if performance holds up
            # with this on, time conditioning isn't contributing meaningfully.
            rng = np.random.default_rng(seed)
            self._fake_t_src = rng.uniform(0, 180, size=len(self.rna)).astype(np.float32)
            self._fake_delta_t = rng.uniform(0, 180, size=len(self.rna)).astype(np.float32)

    def __len__(self):
        return len(self.rna)

    def __getitem__(self, idx):
        x = normalize(self.rna[idx])
        y = self.targets[idx]
        if self.shuffle_time:
            t_info = np.array([np.log1p(self._fake_t_src[idx]),
                                np.log1p(self._fake_delta_t[idx])], dtype=np.float32)
        else:
            t_info = np.array([np.log1p(self.t_src), np.log1p(self.delta_t)], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y), torch.tensor(t_info)


def split_cells_per_timepoint(obs_time, train_ratio=0.8, val_ratio=0.1, seed=1105):
    rng = np.random.default_rng(seed)
    timepoints = sorted(np.unique(obs_time).tolist())
    train_cells, val_cells, test_cells = {}, {}, {}
    for t in timepoints:
        cells = np.where(np.isclose(obs_time, t, atol=1e-3))[0]
        rng.shuffle(cells)
        n_train = int(len(cells) * train_ratio)
        n_val = int(len(cells) * val_ratio)
        train_cells[t] = cells[:n_train]
        val_cells[t] = cells[n_train:n_train + n_val]
        test_cells[t] = cells[n_train + n_val:]
    return train_cells, val_cells, test_cells


def get_ot_targets(rna_src, rna_tgt, prot_tgt, ot_type, reg, max_genes=2000):
    if ot_type == "soft":
        if compute_ot_soft is None:
            raise RuntimeError("Soft OT requested but qurie_dataset_ot.py not importable.")
        return compute_ot_soft(rna_src, rna_tgt, prot_tgt, reg=reg, max_genes=max_genes)
    elif ot_type == "hard":
        return compute_ot_hard(rna_src, rna_tgt, prot_tgt, mode="exact", max_genes=max_genes)
    elif ot_type == "hard_assignment":
        return compute_ot_hard(rna_src, rna_tgt, prot_tgt, mode="assignment", max_genes=max_genes)
    else:
        raise ValueError(f"Unknown ot_type: {ot_type}")


def build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                       t_src, t_tgt, ot_type, ot_reg, shuffle_rna=False,
                       shuffle_time=False, seed=1105):
    """Build train/test PairDataset for one (t_src,t_tgt) transition.

    If val_cells is provided, also builds a validation dataset.
    If val_cells is None, only train and test datasets are built.
    """

    if t_src not in train_cells or t_tgt not in train_cells:
        print(f"  SKIP {t_src}->{t_tgt}: timepoint not in data.")
        return None, None, None

    train_src = train_cells[t_src]
    train_tgt = train_cells[t_tgt]

    test_src = test_cells[t_src]

    # Validation is optional.
    if val_cells is not None:
        val_src = val_cells[t_src]
    else:
        val_src = None

    if len(train_src) < 2 or len(train_tgt) < 2:
        print(f"  SKIP {t_src}->{t_tgt}: too few cells.")
        return None, None, None

    print(
        f"  Building {t_src}->{t_tgt}: "
        f"train_src={len(train_src)} "
        f"train_tgt={len(train_tgt)}"
        + (
            f" val_src={len(val_src)}"
            if val_src is not None
            else ""
        )
        + f" test_src={len(test_src)}"
    )

    # ---------------------------------------------------------
    # Training OT targets
    # ---------------------------------------------------------
    train_targets = get_ot_targets(
        rna_selected[train_src],
        rna_selected[train_tgt],
        prot_full[train_tgt],
        ot_type,
        ot_reg
    )

    # ---------------------------------------------------------
    # Validation OT targets -- only if validation exists
    # ---------------------------------------------------------
    if val_src is not None and len(val_src) > 1:
        val_targets = get_ot_targets(
            rna_selected[val_src],
            rna_selected[train_tgt],
            prot_full[train_tgt],
            ot_type,
            ot_reg
        )
    else:
        val_targets = None

    # ---------------------------------------------------------
    # Test OT targets
    # ---------------------------------------------------------
    test_targets = (
        get_ot_targets(
            rna_selected[test_src],
            rna_selected[train_tgt],
            prot_full[train_tgt],
            ot_type,
            ot_reg
        )
        if len(test_src) > 1
        else None
    )

    # ---------------------------------------------------------
    # Training RNA
    # ---------------------------------------------------------
    # shuffle_rna is a control applied ONLY to training data.
    # Targets remain unchanged.
    # Validation/test data remain untouched.
    train_rna = rna_selected[train_src]

    if shuffle_rna:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train_rna))
        train_rna = train_rna[perm]

        print(
            f"    *** shuffle_rna applied to "
            f"{t_src}->{t_tgt} training data ***"
        )

    # ---------------------------------------------------------
    # Construct datasets
    # ---------------------------------------------------------
    train_ds = PairDataset(
        train_rna,
        train_targets,
        t_src,
        t_tgt,
        shuffle_time=shuffle_time,
        seed=seed
    )

    if val_targets is not None:
        val_ds = PairDataset(
            rna_selected[val_src],
            val_targets,
            t_src,
            t_tgt
        )
    else:
        val_ds = None

    if test_targets is not None:
        test_ds = PairDataset(
            rna_selected[test_src],
            test_targets,
            t_src,
            t_tgt
        )
    else:
        test_ds = None

    return train_ds, val_ds, test_ds


def run_epoch(model, loader, device, optimizer=None, criterion=None, variance_penalty=0.0, grad_clip=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for x, y, t_info in loader:
            x, y, t_info = x.to(device), y.to(device), t_info.to(device)
            if is_train:
                optimizer.zero_grad()
            y_hat = model(x, t_info)
            loss = criterion(y_hat, y)
            if is_train and variance_penalty > 0:
                var_true = y.var(dim=0)
                var_pred = y_hat.var(dim=0)
                var_gap = torch.clamp(var_true - var_pred, min=0).mean()
                loss = loss + variance_penalty * var_gap
            if is_train:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate_per_pair(model, test_datasets_by_pair, device, protein_names, outdir=None):
    """Evaluate the ONE joint model separately on each pair's held-out test
    set, so you can see how the single model performs across the full
    range of delta_t -- not to compare separate models, just to inspect
    the joint model's per-transition performance."""
    model.eval()
    results = {}
    for (t_src, t_tgt), ds in test_datasets_by_pair.items():
        if ds is None or len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        y_hat_all, y_all = [], []
        with torch.no_grad():
            for x, y, t_info in loader:
                x, t_info = x.to(device), t_info.to(device)
                y_hat = model(x, t_info).cpu().numpy()
                y_hat_all.append(y_hat)
                y_all.append(y.numpy())
        y_hat_all = np.concatenate(y_hat_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        if outdir is not None:
            pred_df = pd.DataFrame(y_hat_all, columns=protein_names)
            pred_df.insert(0, "t_src", t_src)
            pred_df.insert(1, "t_tgt", t_tgt)
            pred_df.to_csv(os.path.join(outdir, f"joint_y_pred_t{int(t_src)}to{int(t_tgt)}.csv"), index=False)

            truth_df = pd.DataFrame(y_all, columns=protein_names)
            truth_df.insert(0, "t_src", t_src)
            truth_df.insert(1, "t_tgt", t_tgt)
            truth_df.to_csv(os.path.join(outdir, f"joint_y_truth_t{int(t_src)}to{int(t_tgt)}.csv"), index=False)

        mse = float(np.mean((y_hat_all - y_all) ** 2))
        per_r = []
        for i in range(y_all.shape[1]):
            if np.std(y_all[:, i]) > 1e-8 and np.std(y_hat_all[:, i]) > 1e-8:
                r, _ = pearsonr(y_all[:, i], y_hat_all[:, i])
                per_r.append(r)
        mean_r = float(np.nanmean(per_r)) if per_r else float("nan")
        per_ccc = []
        for i in range(y_all.shape[1]):
            if np.std(y_all[:, i]) > 1e-8 and np.std(y_hat_all[:, i]) > 1e-8:
                per_ccc.append(concordance_correlation_coefficient(y_all[:, i], y_hat_all[:, i]))
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
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post", "gated", "gated_post"], required=True)
    parser.add_argument("--curriculum", action="store_true",
                         help="Progressive: train on short delta_t pairs first, add "
                              "longer ones stage by stage. Same model throughout.")
    parser.add_argument("--epochs", type=int, default=100,
                         help="total epochs (non-curriculum mode)")
    parser.add_argument("--epochs_per_stage", type=int, default=20,
                         help="epochs per curriculum stage (curriculum mode)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--variance_penalty", type=float, default=0.0,
                         help="Weight on variance-gap penalty (0=off). Applied "
                              "during training only, not validation/test loss.")
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--shuffle_rna", action="store_true",
                         help="CONTROL: shuffle RNA input across training cells (targets "
                              "fixed), applied to every pair. If performance barely drops "
                              "vs the real run, the model isn't actually using RNA content.")
    parser.add_argument("--shuffle_time", action="store_true",
                         help="CONTROL: randomize t_info per training sample, every pair. "
                              "If performance barely drops vs the real run, time "
                              "conditioning isn't doing anything.")
    parser.add_argument("--time_embed", choices=["raw", "sinusoidal"], default="raw",
                         help="'sinusoidal': multi-frequency sin/cos embedding of "
                              "(log_t_src, log_delta_t) instead of the two raw scalars, "
                              "in the style of transformer positional encodings.")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Config: gene_strategy={args.gene_strategy} n_genes={n_genes_arg} "
          f"ot_type={args.ot_type} ot_reg={args.ot_reg} model={args.model} "
          f"curriculum={args.curriculum} shuffle_rna={args.shuffle_rna} "
          f"shuffle_time={args.shuffle_time}")

    # --- Load data ---
    rna_adata = ad.read_h5ad(args.rna)
    prot_adata = ad.read_h5ad(args.protein)
    mask = rna_adata.obs["condition"] == args.condition
    rna_adata = rna_adata[mask].copy()
    prot_adata = prot_adata[mask].copy()

    if "counts" not in rna_adata.layers or "counts" not in prot_adata.layers:
        raise RuntimeError("Expected '.layers[\"counts\"]' in both h5ad files.")
    rna_full = rna_adata.layers["counts"]
    prot_full = prot_adata.layers["counts"]
    rna_full = (rna_full.toarray() if hasattr(rna_full, "toarray") else rna_full).astype(np.float32)
    prot_full = (prot_full.toarray() if hasattr(prot_full, "toarray") else prot_full).astype(np.float32)
    protein_names = list(prot_adata.var_names)
    obs_time = rna_adata.obs["time_min"].values.astype(float)

    # --- Split FIRST, then select genes using ONLY the train-split cells ---
    # (previously gene selection ran on the full population before this split,
    # so val/test cells' RNA leaked into which genes got chosen)
    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)

    train_idx_all = np.concatenate([idx for idx in train_cells.values() if len(idx) > 0])
    rna_adata_train = rna_adata[train_idx_all].copy()
    gene_idx = select_genes(rna_adata_train, args.gene_strategy, n_genes_arg, groupby="time_min")
    print(f"Selected {len(gene_idx)} genes via {args.gene_strategy} "
          f"(train-split cells only: {len(train_idx_all)}/{rna_adata.n_obs}).")
    rna_selected = rna_full[:, gene_idx]

    available_pairs = [(s, t) for s, t in CURRICULUM if s in train_cells and t in train_cells]
    print(f"Available timepoint pairs in this data: {available_pairs}")

    # --- Build datasets for every pair up front ---
    print("\nBuilding datasets for all pairs...")
    train_ds_by_pair, val_ds_by_pair, test_ds_by_pair = {}, {}, {}
    for (t_src, t_tgt) in available_pairs:
        tr, va, te = build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                         t_src, t_tgt, args.ot_type, args.ot_reg,
                                         shuffle_rna=args.shuffle_rna, shuffle_time=args.shuffle_time,
                                         seed=args.seed)
        if tr is not None:
            train_ds_by_pair[(t_src, t_tgt)] = tr
            val_ds_by_pair[(t_src, t_tgt)] = va
            test_ds_by_pair[(t_src, t_tgt)] = te

    # --- Build ONE model ---
    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names),
                         time_embed=args.time_embed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    ckpt_path = os.path.join(args.outdir, "joint_model_best.pt")
    best_val_loss = float("inf")

    def val_loss_over_all_pairs():
        losses = []
        for pair, va in val_ds_by_pair.items():
            if va is None or len(va) == 0:
                continue
            loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)
            losses.append(run_epoch(model, loader, device, optimizer=None, criterion=criterion))
        return float(np.mean(losses)) if losses else float("inf")

    def save_ckpt(epoch, val_loss, stage_pairs=None):
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss,
            "stage_pairs": stage_pairs,
        }, ckpt_path)

    if not args.curriculum:
        # --- JOINT MODE: pool every pair's train set into one big dataset ---
        print("\n=== JOINT TRAINING (all pairs mixed from epoch 1) ===")
        full_train = ConcatDataset(list(train_ds_by_pair.values()))
        train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True)

        loss_history = []  # every epoch, for plotting
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, device, optimizer=optimizer,
                                    criterion=criterion, variance_penalty=args.variance_penalty,
                                    grad_clip=1.0)
            val_loss = val_loss_over_all_pairs()
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                save_ckpt(epoch, val_loss)
            loss_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "improved": improved})
            if epoch % 10 == 0 or epoch == 1 or improved:
                flag = " *improved, checkpoint saved*" if improved else ""
                print(f"  Epoch {epoch:>4}: train_loss={train_loss:.4f} val_loss(avg over pairs)={val_loss:.4f}{flag}")

        pd.DataFrame(loss_history).to_csv(os.path.join(args.outdir, "loss_history.csv"), index=False)
        print(f"Saved loss history ({len(loss_history)} epochs) to {args.outdir}/loss_history.csv")

    else:
        # --- CURRICULUM MODE: progressively add pairs, same model throughout ---
        print("\n=== CURRICULUM TRAINING (progressive, short delta_t first) ===")
        active_pairs = []
        global_epoch = 0
        for stage_i, pair in enumerate(available_pairs, 1):
            active_pairs.append(pair)
            print(f"\n--- Stage {stage_i}/{len(available_pairs)}: now training on {active_pairs} ---")
            stage_train = ConcatDataset([train_ds_by_pair[p] for p in active_pairs])
            train_loader = DataLoader(stage_train, batch_size=args.batch_size, shuffle=True)

            for e in range(1, args.epochs_per_stage + 1):
                global_epoch += 1
                train_loss = run_epoch(model, train_loader, device, optimizer=optimizer,
                                        criterion=criterion, variance_penalty=args.variance_penalty,
                                        grad_clip=1.0)
                if np.isnan(train_loss):
                    print(f"    WARNING: NaN loss detected at stage {stage_i} epoch {e} "
                          f"(global {global_epoch}). Reverting to last good checkpoint and "
                          f"halving LR for the optimizer before continuing.")
                    if os.path.exists(ckpt_path):
                        ckpt = torch.load(ckpt_path, map_location=device)
                        model.load_state_dict(ckpt["model_state_dict"])
                    for g in optimizer.param_groups:
                        g["lr"] = g["lr"] * 0.5
                    continue
                val_loss = val_loss_over_all_pairs()
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    save_ckpt(global_epoch, val_loss, stage_pairs=active_pairs)
                if e % 10 == 0 or e == 1 or improved:
                    flag = " *improved, checkpoint saved*" if improved else ""
                    print(f"    Stage {stage_i} epoch {e:>3} (global {global_epoch}): "
                          f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}{flag}")

    # --- Load best checkpoint, evaluate the ONE model on val AND test sets ---
    print(f"\nLoading best checkpoint (val_loss={best_val_loss:.4f}) for final evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n=== FINAL: single joint model evaluated per timepoint pair (VALIDATION) ===")
    val_pair_results = evaluate_per_pair(model, val_ds_by_pair, device, protein_names, outdir=None)

    print("\n=== FINAL: single joint model evaluated per timepoint pair (TEST) ===")
    per_pair_results = evaluate_per_pair(model, test_ds_by_pair, device, protein_names, outdir=args.outdir)

    # --- Save: one row per (split, t_src, t_tgt), so val and test sit side
    # by side in the same file. Model/config selection across a sweep should
    # be done using the 'val' rows only -- the 'test' rows exist purely for
    # final reporting once a config is already chosen on validation. ---
    summary_path = os.path.join(args.outdir, "joint_results_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "t_src", "t_tgt", "n", "mse", "mean_pearson_r", "ccc"])
        for (t_src, t_tgt), res in val_pair_results.items():
            writer.writerow(["val", t_src, t_tgt, res["n_test"], res["mse"], res["mean_pearson_r"], res["ccc"]])
        for (t_src, t_tgt), res in per_pair_results.items():
            writer.writerow(["test", t_src, t_tgt, res["n_test"], res["mse"], res["mean_pearson_r"], res["ccc"]])

    # --- Pooled scalars (mean across pairs) for quick cross-config comparison
    # during sweeps -- this is the number a config-selection script should
    # read, using the val_* fields only, never the test_* fields. ---
    def _pool(results_dict, key):
        vals = [r[key] for r in results_dict.values() if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    val_mean_r = _pool(val_pair_results, "mean_pearson_r")
    val_mean_ccc = _pool(val_pair_results, "ccc")
    test_mean_r = _pool(per_pair_results, "mean_pearson_r")
    test_mean_ccc = _pool(per_pair_results, "ccc")
    print(f"\nPooled (mean over pairs): val_R={val_mean_r:.4f} val_CCC={val_mean_ccc:.4f} "
          f"| test_R={test_mean_r:.4f} test_CCC={test_mean_ccc:.4f}")

    with open(os.path.join(args.outdir, "joint_config.json"), "w") as f:
        json.dump({"config": vars(args), "n_genes_actual": len(gene_idx),
                    "best_val_loss": best_val_loss, "checkpoint_path": ckpt_path,
                    "available_pairs": available_pairs,
                    "val_mean_pearson_r": val_mean_r, "val_mean_ccc": val_mean_ccc,
                    "test_mean_pearson_r": test_mean_r, "test_mean_ccc": test_mean_ccc},
                  f, indent=2)

    print(f"\nSaved per-pair val+test results to {summary_path}")
    print(f"Saved single joint model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()