"""
train.py
─────────
Single entrypoint for the gene-selection x OT-type x model ablation
grid. Called from a .sh file with specific parameters — one call =
one (gene_strategy, n_genes, ot_type, model) configuration.

DESIGN NOTE: this does NOT reuse qurie_dataset_ot.py's DataStoreOT /
CurriculumDatasetOT classes directly, because those are built around
scTranslator's tokenized full-transcriptome input (my_Id, padding to
enc_max_seq_len=20000, etc.) — overkill and not the right shape for
these simple MLP/FiLM baselines, which take a fixed, small selected-
gene vector as input. Instead this script:
  1. Loads RNA + Protein h5ad (same cleaned/mapped files)
  2. Selects top-N genes via gene_selection.py (hvg or deg)
  3. Splits cells per-timepoint (train/val/test), same ratios/logic
     as DataStoreOT, for consistency with your existing splits
  4. Computes OT targets between t_src and t_tgt train cells using
     EITHER the soft (Sinkhorn, averaging) OT from qurie_dataset_ot.py
     OR the hard (exact/assignment) OT from ot_matching.py
  5. Trains the selected model (mlp / film_pre / film_post)
  6. Evaluates with MSE, Pearson r (per protein + mean), and CCC
     (concordance correlation coefficient)
  7. Saves metrics + predictions to --outdir, named by config

Usage (see run_train.sh for the batch-driven version):
    python train.py \
        --rna qurie_RNA_clean_aIg.h5ad \
        --protein qurie_Protein_clean_aIg.h5ad \
        --t_src 0 --t_tgt 2 \
        --gene_strategy deg --n_genes 500 \
        --ot_type soft \
        --model film_post \
        --epochs 100 --outdir results/
"""

import argparse
import os
import sys
import json

import numpy as np
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from ot_matching import compute_ot_hard
from models import build_model

# reuse the soft-OT implementation from your existing script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from qurie_dataset_ot import compute_ot_targets as compute_ot_soft, normalize
except ImportError:
    print("WARNING: could not import qurie_dataset_ot.py. Make sure it's on the "
          "path (same directory or added via sys.path). Soft OT will be unavailable.")
    compute_ot_soft = None

    def normalize(x, eps=1e-8):
        xmin, xmax = float(x.min()), float(x.max())
        if xmin == xmax:
            return np.full_like(x, 1e-8, dtype=np.float32)
        return (1e-8 + (x - xmin) / (xmax - xmin + eps)).astype(np.float32)


def concordance_correlation_coefficient(y_true, y_pred):
    """CCC per output dim, then averaged. Standard metric for this
    kind of continuous biomarker prediction — see discussion below."""
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom == 0:
        return 0.0
    return (2 * covariance) / denom


class SimpleGeneDataset(Dataset):
    def __init__(self, rna_selected, prot, t_src, t_tgt, shuffle_time=False, seed=1105):
        self.rna = rna_selected.astype(np.float32)
        self.prot = prot.astype(np.float32)
        self.t_src = t_src
        self.delta_t = t_tgt - t_src
        self.shuffle_time = shuffle_time
        if shuffle_time:
            # CONTROL: assign each sample a random (fake) t_src/delta_t drawn
            # from a plausible range, independent of the real transition.
            # If the model's performance is roughly unchanged with this on,
            # time conditioning isn't contributing meaningfully.
            rng = np.random.default_rng(seed)
            self._fake_t_src = rng.uniform(0, 180, size=len(self.rna)).astype(np.float32)
            self._fake_delta_t = rng.uniform(0, 180, size=len(self.rna)).astype(np.float32)

    def __len__(self):
        return len(self.rna)

    def __getitem__(self, idx):
        x = normalize(self.rna[idx])
        y = self.prot[idx]
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


def get_ot_targets(rna_src, rna_tgt, prot_tgt, ot_type, max_genes=2000, reg=0.1):
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


def train_model(model, train_loader, val_loader, device, epochs, lr=1e-3,
                 ckpt_path=None, variance_penalty=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, t_info in train_loader:
            x, y, t_info = x.to(device), y.to(device), t_info.to(device)
            optimizer.zero_grad()
            y_hat = model(x, t_info)
            loss = criterion(y_hat, y)
            if variance_penalty > 0:
                # Penalize the model's predictions having much lower variance
                # than the targets -- a common "mean-collapse" failure mode
                # when the input signal is weak (model just predicts avg).
                var_true = y.var(dim=0)
                var_pred = y_hat.var(dim=0)
                var_gap = torch.clamp(var_true - var_pred, min=0).mean()
                loss = loss + variance_penalty * var_gap
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, t_info in val_loader:
                x, y, t_info = x.to(device), y.to(device), t_info.to(device)
                y_hat = model(x, t_info)
                val_loss += criterion(y_hat, y).item()
        val_loss /= max(1, len(val_loader))

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                }, ckpt_path)

        if epoch % 10 == 0 or epoch == 1 or improved:
            flag = " *improved, checkpoint saved*" if improved else ""
            print(f"  Epoch {epoch:>4}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{flag}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate(model, loader, device, protein_names):
    model.eval()
    y_hat_all, y_all = [], []
    with torch.no_grad():
        for x, y, t_info in loader:
            x, t_info = x.to(device), t_info.to(device)
            y_hat = model(x, t_info).cpu().numpy()
            y_hat_all.append(y_hat)
            y_all.append(y.numpy())
    y_hat_all = np.concatenate(y_hat_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    mse = float(np.mean((y_hat_all - y_all) ** 2))
    per_protein_r = []
    for i in range(y_all.shape[1]):
        if np.std(y_all[:, i]) < 1e-8 or np.std(y_hat_all[:, i]) < 1e-8:
            per_protein_r.append(np.nan)
        else:
            r, _ = pearsonr(y_all[:, i], y_hat_all[:, i])
            per_protein_r.append(r)
    mean_r = float(np.nanmean(per_protein_r))

    # Per-protein CCC, averaged -- NOT pooled/flattened CCC. Pooled CCC on
    # flattened (cells x proteins) arrays is inflated by BETWEEN-protein
    # structure (which channels are generally high vs low after per-cell
    # normalization), not WITHIN-protein cell-to-cell predictive accuracy.
    # See conversation history / 10_check_collapse.py for the diagnosis.
    per_protein_ccc = []
    for i in range(y_all.shape[1]):
        if np.std(y_all[:, i]) > 1e-8 and np.std(y_hat_all[:, i]) > 1e-8:
            per_protein_ccc.append(concordance_correlation_coefficient(y_all[:, i], y_hat_all[:, i]))
    ccc = float(np.mean(per_protein_ccc)) if per_protein_ccc else float("nan")

    return {
        "mse": mse,
        "mean_pearson_r": mean_r,
        "ccc": float(ccc),
        "per_protein_pearson_r": dict(zip(protein_names, [float(x) if not np.isnan(x) else None for x in per_protein_r])),
    }, y_hat_all, y_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--condition", default="aIg")
    parser.add_argument("--t_src", type=float, required=True)
    parser.add_argument("--t_tgt", type=float, required=True)
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], required=True)
    parser.add_argument("--n_genes", required=True,
                         help="int, or 'all'")
    parser.add_argument("--ot_type", choices=["soft", "hard", "hard_assignment"], required=True)
    parser.add_argument("--ot_reg", type=float, default=0.1,
                         help="Sinkhorn regularization for soft OT. Tune per-pair: "
                              "smaller (e.g. 0.005-0.01) for long delta_t, larger "
                              "(e.g. 0.1) fine for short delta_t. See 09_sweep_ot_reg.py.")
    parser.add_argument("--model", choices=["mlp", "film_pre", "film_post"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--variance_penalty", type=float, default=0.0,
                         help="Weight on variance-gap penalty (0=off). Discourages "
                              "the model from collapsing to predicting the mean "
                              "protein vector for every cell.")
    parser.add_argument("--shuffle_rna", action="store_true",
                         help="CONTROL: shuffle RNA input across cells (targets fixed). "
                              "If performance barely drops vs the real run, the model "
                              "isn't actually using RNA content.")
    parser.add_argument("--shuffle_time", action="store_true",
                         help="CONTROL: randomize t_info per sample. If performance "
                              "barely drops vs the real run, time conditioning isn't "
                              "doing anything.")
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Config: gene_strategy={args.gene_strategy} n_genes={n_genes_arg} "
          f"ot_type={args.ot_type} model={args.model} t_src={args.t_src} t_tgt={args.t_tgt}")

    # --- Load data ---
    rna_adata = ad.read_h5ad(args.rna)
    prot_adata = ad.read_h5ad(args.protein)
    mask = rna_adata.obs["condition"] == args.condition
    rna_adata = rna_adata[mask].copy()
    prot_adata = prot_adata[mask].copy()

    # Use RAW COUNTS (preserved in .layers['counts'] by 03_clean_qurie_data.py),
    # not the already log1p/arcsinh-transformed .X — SimpleGeneDataset applies
    # its own per-cell normalize() below, matching scTranslator's own
    # convention of a single normalization pass on raw counts. Using .X here
    # would stack log1p -> min-max, an unintended double-normalization.
    if "counts" not in rna_adata.layers or "counts" not in prot_adata.layers:
        raise RuntimeError("Expected '.layers[\"counts\"]' in both RNA and Protein "
                            "h5ad files (written by 03_clean_qurie_data.py). Re-run "
                            "cleaning if this is missing.")
    rna_full = rna_adata.layers["counts"]
    prot_full = prot_adata.layers["counts"]
    rna_full = (rna_full.toarray() if hasattr(rna_full, "toarray") else rna_full).astype(np.float32)
    prot_full = (prot_full.toarray() if hasattr(prot_full, "toarray") else prot_full).astype(np.float32)
    protein_names = list(prot_adata.var_names)
    obs_time = rna_adata.obs["time_min"].values.astype(float)

    # --- Gene selection (computed over ALL cells in this condition) ---
    gene_idx = select_genes(rna_adata, args.gene_strategy, n_genes_arg, groupby="time_min")
    print(f"Selected {len(gene_idx)} genes via {args.gene_strategy}.")
    rna_selected = rna_full[:, gene_idx]

    # --- Split cells per timepoint ---
    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)

    if args.t_src not in train_cells or args.t_tgt not in train_cells:
        print(f"ERROR: t_src={args.t_src} or t_tgt={args.t_tgt} not found in data. "
              f"Available: {sorted(train_cells.keys())}")
        return

    train_src, train_tgt = train_cells[args.t_src], train_cells[args.t_tgt]
    val_src = val_cells[args.t_src]
    test_src = test_cells[args.t_src]

    # --- Compute OT targets (train uses OT to t_tgt train cells; val/test
    #     evaluated against OT-matched targets too, for a consistent metric
    #     across splits — swap to real same-cell targets if you have them) ---
    print("Computing train targets...")
    train_targets = get_ot_targets(rna_selected[train_src], rna_selected[train_tgt], prot_full[train_tgt], args.ot_type, reg=args.ot_reg)
    print("Computing val targets...")
    val_targets = get_ot_targets(rna_selected[val_src], rna_selected[train_tgt], prot_full[train_tgt], args.ot_type, reg=args.ot_reg)
    print("Computing test targets...")
    test_targets = get_ot_targets(rna_selected[test_src], rna_selected[train_tgt], prot_full[train_tgt], args.ot_type, reg=args.ot_reg)

    rna_train_input = rna_selected[train_src]
    if args.shuffle_rna:
        print("*** CONTROL: shuffling RNA input across training cells (targets unchanged) ***")
        perm = np.random.default_rng(args.seed).permutation(len(rna_train_input))
        rna_train_input = rna_train_input[perm]

    train_ds = SimpleGeneDataset(rna_train_input, train_targets, args.t_src, args.t_tgt,
                                  shuffle_time=args.shuffle_time, seed=args.seed)
    val_ds = SimpleGeneDataset(rna_selected[val_src], val_targets, args.t_src, args.t_tgt,
                                shuffle_time=args.shuffle_time, seed=args.seed)
    test_ds = SimpleGeneDataset(rna_selected[test_src], test_targets, args.t_src, args.t_tgt,
                                 shuffle_time=args.shuffle_time, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # --- Build + train model ---
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.gene_strategy}_{n_genes_arg}_{args.ot_type}_{args.model}_t{int(args.t_src)}to{int(args.t_tgt)}"
    if args.shuffle_rna:
        tag += "_SHUFFLE-RNA"
    if args.shuffle_time:
        tag += "_SHUFFLE-TIME"
    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{tag}_best.pt")

    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)
    model, best_val_loss = train_model(model, train_loader, val_loader, device, args.epochs,
                                        lr=args.lr, ckpt_path=ckpt_path,
                                        variance_penalty=args.variance_penalty)

    # --- Evaluate ---
    test_metrics, y_hat, y_true = evaluate(model, test_loader, device, protein_names)
    print(f"TEST: mse={test_metrics['mse']:.4f} mean_pearson_r={test_metrics['mean_pearson_r']:.4f} "
          f"ccc={test_metrics['ccc']:.4f}")

    # --- Save ---
    with open(os.path.join(args.outdir, f"{tag}_metrics.json"), "w") as f:
        json.dump({
            "t_src": args.t_src,
            "t_tgt": args.t_tgt,
            "gene_strategy": args.gene_strategy,
            "n_genes_requested": n_genes_arg,
            "n_genes_actual": len(gene_idx),
            "ot_type": args.ot_type,
            "ot_reg": args.ot_reg,
            "model": args.model,
            "variance_penalty": args.variance_penalty,
            "shuffle_rna": args.shuffle_rna,
            "shuffle_time": args.shuffle_time,
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
            "checkpoint_path": ckpt_path,
            "full_config": vars(args),
        }, f, indent=2)
    np.save(os.path.join(args.outdir, f"{tag}_y_pred.npy"), y_hat)
    np.save(os.path.join(args.outdir, f"{tag}_y_true.npy"), y_true)
    print(f"Saved results to {args.outdir}/{tag}_*  (checkpoint: {ckpt_path})")

    # --- Append to shared summary CSV (one row per run, across all configs) ---
    import csv
    summary_path = os.path.join(args.outdir, "results_summary.csv")
    row = {
        "t_src": args.t_src,
        "t_tgt": args.t_tgt,
        "gene_strategy": args.gene_strategy,
        "n_genes": n_genes_arg,
        "n_genes_actual": len(gene_idx),
        "ot_type": args.ot_type,
        "ot_reg": args.ot_reg,
        "model": args.model,
        "variance_penalty": args.variance_penalty,
        "shuffle_rna": args.shuffle_rna,
        "shuffle_time": args.shuffle_time,
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "test_mse": test_metrics["mse"],
        "test_mean_pearson_r": test_metrics["mean_pearson_r"],
        "test_ccc": test_metrics["ccc"],
        "checkpoint_path": ckpt_path,
    }
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended run to {summary_path}")


if __name__ == "__main__":
    main()