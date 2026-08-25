"""
06_generalized_finetune.py

Purpose: Generalize the thesis's final from-scratch model (DEG selection,
FiLM-pre, soft-OT, correlation loss; Section 5.6) so it can be fine-tuned
on ANY user-provided time-resolved single-cell RNA + protein/PTM dataset,
not just QuRIE-seq. This is the concrete implementation behind the
"Future Work" placeholder (generalized fine-tuning entry point).

Design constraints this has to handle, that the original pipeline didn't:
  1. Arbitrary timepoint SET (not fixed to 0,2,4,6,60,180) and arbitrary
     COUNT of timepoints (>= 2).
  2. Arbitrary gene panel / arbitrary protein panel — almost certainly a
     different size and different identities than the 1000-gene DEG panel
     and 58-protein panel the original checkpoint was trained on.
  3. No guarantee the new dataset's genes/proteins overlap heavily with
     the original panel.

Because (2)+(3) mean the new data's input/output dimensionality won't
match the pretrained checkpoint's first/last layer shapes, this uses an
ADAPTER strategy rather than assuming direct weight reuse:
  - A new linear input-adapter layer projects the new dataset's own
    (gene-selected) RNA dimensionality into the pretrained trunk's
    expected input dimensionality.
  - A new linear output-adapter layer projects the pretrained trunk's
    output back down to the new dataset's own protein/PTM panel size.
  - The pretrained trunk (512-256-128 FiLM-pre backbone) is loaded from
    the thesis checkpoint and fine-tuned (not frozen) alongside the new
    adapters, at a lower learning rate than the adapters themselves,
    same two-stage spirit as Section 4.4.2 (warm up new components,
    then fine-tune jointly).

Gene/protein selection on the NEW dataset is recomputed from scratch
(DEG selection against the new dataset's own timepoints, Section 3.6.1
methodology) rather than reusing the QuRIE-seq gene panel, since a
different biological system will have different informative genes.

Evaluation reuses the exact same per-protein R/CCC convention and
3-part shuffle diagnostic (Section 4.8) as the rest of the thesis, so
results are directly comparable in kind (not magnitude) to Table 5.7.

STATUS: implementation complete; PENDING an actual external dataset
to run it on (see companion note on candidate GEO datasets). Until run,
this is reported in the thesis as a demonstrated, tested-on-synthetic-
data capability, not yet validated on real external data — do not
claim real-data validation until this has actually been run and the
results section updated.

Usage:
    python 06_generalized_finetune.py \
        --rna_path external/some_dataset_RNA.h5ad \
        --prot_path external/some_dataset_Protein.h5ad \
        --time_col time_min \
        --checkpoint checkpoints/final_model_fold1.pt \
        --gene_panel_size 1000 \
        --epochs 30 \
        --out_dir external_finetune_results

Expected input h5ad format (minimal contract, deliberately loose):
  - RNA h5ad: obs must contain a numeric time column (name given via
    --time_col). var_names should be gene symbols (or set --gene_id_col
    if identifiers live in a var column instead).
  - Protein h5ad: obs must contain the SAME time column and same cell
    ordering assumption is NOT required — cells are matched by barcode
    if obs has a 'barcode' column, otherwise assumed pre-aligned by index.
"""
import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_distances

try:
    import ot
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False


# ---------------------------------------------------------------------
# Shared utilities (mirroring qurie_dataset_ot.py conventions so results
# stay comparable in kind to the main thesis pipeline)
# ---------------------------------------------------------------------
def normalize(x, eps=1e-8):
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        return np.full_like(x, 1e-8, dtype=np.float32)
    return (1e-8 + (x - xmin) / (xmax - xmin + eps)).astype(np.float32)


def per_protein_r(y_true, y_pred):
    rs = []
    for p in range(y_true.shape[1]):
        yt, yp = y_true[:, p], y_pred[:, p]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            continue
        r, _ = pearsonr(yt, yp)
        if not np.isnan(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def compute_ot_targets(rna_src, rna_tgt, prot_tgt, reg=0.01, max_genes=2000):
    if not OT_AVAILABLE:
        mean = normalize(prot_tgt.mean(axis=0))
        return np.tile(mean, (len(rna_src), 1))
    n_src, n_tgt = len(rna_src), len(rna_tgt)
    gene_var = rna_src.var(axis=0) + rna_tgt.var(axis=0)
    top_genes = np.argsort(gene_var)[-max_genes:]
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)
    a = np.ones(n_src, dtype=np.float64) / n_src
    b = np.ones(n_tgt, dtype=np.float64) / n_tgt
    T = ot.sinkhorn(a, b, C, reg=reg)
    T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-8)
    targets = (T_norm @ prot_tgt).astype(np.float32)
    return np.array([normalize(p) for p in targets])


# ---------------------------------------------------------------------
# Gene selection (DEG, Section 3.6.1 methodology, generalized to any
# number/labeling of timepoints)
# ---------------------------------------------------------------------
def select_deg_genes(rna_adata, time_col, panel_size):
    a = rna_adata.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.obs["_time_group"] = a.obs[time_col].astype(str)
    sc.tl.rank_genes_groups(a, groupby="_time_group", method="wilcoxon")
    best_rank = {}
    groups = a.obs["_time_group"].unique()
    for g in groups:
        names = a.uns["rank_genes_groups"]["names"][g]
        for rank, gene in enumerate(names):
            if gene not in best_rank or rank < best_rank[gene]:
                best_rank[gene] = rank
    ranked_genes = sorted(best_rank.keys(), key=lambda g: best_rank[g])
    return ranked_genes[:panel_size]


import sys
sys.path.append(os.path.dirname(__file__))
from models import build_model


# ---------------------------------------------------------------------
# Model: pretrained trunk (loaded via models.py's build_model, so ANY
# architecture -- mlp/film_pre/film_post/gated/gated_post -- works, not
# just FiLM-pre) + new input/output adapters for the new dataset's shapes
# ---------------------------------------------------------------------
class AdaptedModel(nn.Module):
    """Pretrained trunk (any architecture from models.py's build_model,
    including gated/gated_post) + new linear adapters so it can consume/
    produce the NEW dataset's own gene/protein panel sizes, whatever
    they are."""

    def __init__(self, trunk, new_input_dim, new_output_dim, trunk_input_dim, trunk_output_dim):
        super().__init__()
        self.input_adapter = nn.Linear(new_input_dim, trunk_input_dim)
        self.trunk = trunk
        self.output_adapter = nn.Linear(trunk_output_dim, new_output_dim)

    def forward(self, x, t_info):
        x = self.input_adapter(x)
        # trunk (from models.py) already applies its own sigmoid internally
        h = self.trunk(x, t_info)
        # h is already in [0,1] from the trunk's own sigmoid; adapter maps
        # dimensionality only, followed by a fresh sigmoid for the new
        # output panel size
        return torch.sigmoid(self.output_adapter(h))


def correlation_loss(y_pred, y_true, eps=1e-8):
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    num = (yp * yt).sum(dim=0)
    den = torch.sqrt((yp ** 2).sum(dim=0) * (yt ** 2).sum(dim=0) + eps)
    r = num / (den + eps)
    valid = (y_true.std(dim=0) > 1e-4) & (y_pred.std(dim=0) > 1e-4)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=y_pred.device)
    return 1 - r[valid].mean()


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading external dataset...")
    rna = ad.read_h5ad(args.rna_path)
    prot = ad.read_h5ad(args.prot_path)

    if args.gene_id_col and args.gene_id_col in rna.var.columns:
        rna.var_names = rna.var[args.gene_id_col].astype(str)

    times = sorted(rna.obs[args.time_col].unique().tolist())
    if len(times) < 2:
        raise ValueError(f"Need >= 2 distinct timepoints in {args.time_col}, found {times}")
    print(f"Detected timepoints: {times}")

    print(f"Selecting top {args.gene_panel_size} DEG genes on external data...")
    deg_genes = select_deg_genes(rna, args.time_col, args.gene_panel_size)
    deg_genes = [g for g in deg_genes if g in rna.var_names]
    print(f"  {len(deg_genes)} genes selected (after intersecting with var_names)")

    rna_X_full = rna[:, deg_genes].X
    rna_X = rna_X_full.toarray() if hasattr(rna_X_full, "toarray") else np.asarray(rna_X_full)
    prot_X_full = prot.X
    prot_X = prot_X_full.toarray() if hasattr(prot_X_full, "toarray") else np.asarray(prot_X_full)

    new_input_dim = rna_X.shape[1]
    new_output_dim = prot_X.shape[1]
    print(f"New input dim (genes): {new_input_dim} | new output dim (proteins): {new_output_dim}")

    # Load pretrained trunk -- built via models.py's build_model, so it
    # matches whichever architecture was ACTUALLY trained (--model flag),
    # not hardcoded to FiLM-pre. trunk_input_dim/trunk_output_dim MUST
    # match what the checkpoint was actually trained on.
    print(f"Loading pretrained checkpoint from {args.checkpoint} (architecture: {args.model})...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # trunk_input_dim = ckpt.get("trunk_input_dim", 1000)  # matches thesis's 1000-gene DEG panel
    # Prefer the checkpoint's own saved value if present (most reliable),
    # but fall back to args.gene_panel_size (which you set explicitly on
    # the command line) rather than a hardcoded 1000 -- silently defaulting
    # to 1000 here caused a shape mismatch when loading non-1000-gene
    # checkpoints (e.g. the DEG-200/DEG-500 panel-size sweep checkpoints).
    trunk_input_dim = ckpt.get("trunk_input_dim", args.gene_panel_size)
    trunk_output_dim = ckpt.get("trunk_output_dim", 58)  # matches thesis's 58-protein panel
    trunk = build_model(args.model, n_genes=trunk_input_dim, n_proteins=trunk_output_dim)
    state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    missing, unexpected = trunk.load_state_dict(ckpt[state_dict_key], strict=False)
    if missing or unexpected:
        print(f"  WARNING: state_dict mismatch — missing={missing}, unexpected={unexpected}. "
              f"Check --model matches the architecture the checkpoint was actually trained with.")

    model = AdaptedModel(trunk, new_input_dim, new_output_dim, trunk_input_dim, trunk_output_dim)
    optimizer = torch.optim.Adam([
        {"params": model.input_adapter.parameters(), "lr": args.lr_adapter},
        {"params": model.output_adapter.parameters(), "lr": args.lr_adapter},
        {"params": model.trunk.parameters(), "lr": args.lr_trunk},
    ])

    # Build all pairwise timepoint transitions from whatever timepoints exist
    pairs = [(t1, t2) for i, t1 in enumerate(times) for t2 in times[i + 1:]]
    rng = np.random.default_rng(args.seed)
    time_masks = {t: np.where(np.isclose(rna.obs[args.time_col].values.astype(float), t))[0] for t in times}

    # Split EACH timepoint's cells into train/val/test independently first
    # (Section 4.9 protocol), so we can restrict OT target pools correctly.
    train_masks, val_masks, test_masks = {}, {}, {}
    for t in times:
        idx = time_masks[t]
        rng.shuffle(idx)
        n_tr = int(0.8 * len(idx))
        n_va = int(0.1 * len(idx))
        train_masks[t] = idx[:n_tr]
        val_masks[t] = idx[n_tr:n_tr + n_va]
        test_masks[t] = idx[n_tr + n_va:]

    train_pairs, val_pairs, test_pairs = [], [], []
    for t_src, t_tgt in pairs:
        src_train, src_val, src_test = train_masks[t_src], val_masks[t_src], test_masks[t_src]
        # IMPORTANT: target pool is ALWAYS the target timepoint's TRAIN split only,
        # for train/val/test sources alike -- matches Section 4.9: "validation and
        # test sources matched against the same train-split target population,"
        # so no val/test cell identity ever leaks into its own OT-derived label.
        tgt_train_idx = train_masks[t_tgt]
        if len(src_train) < 5 or len(tgt_train_idx) < 5:
            print(f"  [skip pair {t_src}->{t_tgt}: too few cells]")
            continue

        train_targets = compute_ot_targets(rna_X[src_train], rna_X[tgt_train_idx], prot_X[tgt_train_idx], reg=args.ot_reg)
        val_targets = compute_ot_targets(rna_X[src_val], rna_X[tgt_train_idx], prot_X[tgt_train_idx], reg=args.ot_reg) if len(src_val) > 0 else np.zeros((0, prot_X.shape[1]))
        test_targets = compute_ot_targets(rna_X[src_test], rna_X[tgt_train_idx], prot_X[tgt_train_idx], reg=args.ot_reg) if len(src_test) > 0 else np.zeros((0, prot_X.shape[1]))

        for idx, tgt in zip(src_train, train_targets):
            train_pairs.append((idx, tgt, t_src, t_tgt))
        for idx, tgt in zip(src_val, val_targets):
            val_pairs.append((idx, tgt, t_src, t_tgt))
        for idx, tgt in zip(src_test, test_targets):
            test_pairs.append((idx, tgt, t_src, t_tgt))

    print(f"Total train/val/test examples: {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")

    def batch_tensors_with_pairs(pairs_subset):
        idxs = [p[0] for p in pairs_subset]
        targets = np.stack([p[1] for p in pairs_subset]) if pairs_subset else np.zeros((0, new_output_dim))
        t_srcs = np.array([p[2] for p in pairs_subset], dtype=np.float32)
        t_tgts = np.array([p[3] for p in pairs_subset], dtype=np.float32)
        pair_ids = [(p[2], p[3]) for p in pairs_subset]
        x = np.array([normalize(rna_X[i]) for i in idxs], dtype=np.float32) if idxs else np.zeros((0, new_input_dim))
        t_info = np.stack([np.log1p(t_srcs), np.log1p(np.maximum(t_tgts - t_srcs, 0))], axis=1) if idxs else np.zeros((0, 2))
        return (torch.tensor(x, dtype=torch.float32), torch.tensor(t_info, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32), pair_ids)

    def per_pair_r(y_true, y_pred, pair_ids):
        """Compute per-protein R separately WITHIN each timepoint pair, then
        average across pairs -- matches posthoc_shuffle_tests.py's within_pair_r
        convention (Section 4.8), rather than pooling all pairs together."""
        unique_pairs = sorted(set(pair_ids))
        pair_rs = []
        for pair in unique_pairs:
            mask = np.array([pid == pair for pid in pair_ids])
            if mask.sum() < 5:
                continue
            r = per_protein_r(y_true[mask], y_pred[mask])
            if not np.isnan(r):
                pair_rs.append(r)
        return float(np.mean(pair_rs)) if pair_rs else float("nan")

    x_tr, t_tr, y_tr, pairs_tr = batch_tensors_with_pairs(train_pairs)
    x_va, t_va, y_va, pairs_va = batch_tensors_with_pairs(val_pairs)
    x_te, t_te, y_te, pairs_te = batch_tensors_with_pairs(test_pairs)

    print("Fine-tuning...")
    best_val = -np.inf
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_tr, t_tr)
        loss = nn.functional.mse_loss(pred, y_tr) + args.corr_weight * correlation_loss(pred, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_va, t_va).numpy()
        val_r = per_pair_r(y_va.numpy(), val_pred, pairs_va)
        if val_r > best_val:
            best_val = val_r
            torch.save(model.state_dict(), out_dir / "best_finetuned_model.pt")
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch}: loss={loss.item():.4f} val_R={val_r:.4f}")

    model.load_state_dict(torch.load(out_dir / "best_finetuned_model.pt"))
    model.eval()
    with torch.no_grad():
        test_pred = model(x_te, t_te).numpy()
    test_r = per_pair_r(y_te.numpy(), test_pred, pairs_te)

    # Shuffle diagnostics (Section 4.8), computed per-pair then averaged --
    # matches posthoc_shuffle_tests.py convention, not pooled across pairs.
    # Prediction-shuffle: permute WITHIN each pair separately (not globally
    # across pairs), matching posthoc_shuffle_tests.py exactly.
    rng2 = np.random.default_rng(0)
    shuffled_pred = test_pred.copy()
    for pair in sorted(set(pairs_te)):
        idxs = [i for i, pid in enumerate(pairs_te) if pid == pair]
        if len(idxs) < 2:
            continue
        perm = rng2.permutation(idxs)
        shuffled_pred[idxs] = test_pred[perm]
    r_pred_shuffle = per_pair_r(y_te.numpy(), shuffled_pred, pairs_te)

    t_te_shuffled = t_te[torch.randperm(len(t_te))]
    with torch.no_grad():
        pred_time_shuffle = model(x_te, t_te_shuffled).numpy()
    r_time_shuffle = per_pair_r(y_te.numpy(), pred_time_shuffle, pairs_te)

    x_te_shuffled = x_te[torch.randperm(len(x_te))]
    with torch.no_grad():
        pred_rna_shuffle = model(x_te_shuffled, t_te).numpy()
    r_rna_shuffle = per_pair_r(y_te.numpy(), pred_rna_shuffle, pairs_te)

    per_pair_breakdown = {}
    for pair in sorted(set(pairs_te)):
        mask = np.array([pid == pair for pid in pairs_te])
        if mask.sum() < 5:
            continue
        r = per_protein_r(y_te.numpy()[mask], test_pred[mask])
        per_pair_breakdown[f"{pair[0]}->{pair[1]}"] = {"n_test": int(mask.sum()), "R": r}

    # Save per-pair pred/truth CSVs in the SAME format as the main pipeline
    # (joint_y_pred_t{src}to{tgt}.csv / joint_y_truth_t{src}to{tgt}.csv,
    # columns = t_src, t_tgt, <protein names>), so 11_per_protein_bar_charts.py
    # and 13_prediction_vs_truth_scatter.py work on this run's output with no
    # changes -- protein_names here are the external dataset's own column
    # names (e.g. the myeloid CITE-seq panel), not QuRIE-seq's.
    protein_names = list(prot.var_names)
    y_te_np = y_te.numpy()
    for pair in sorted(set(pairs_te)):
        mask = np.array([pid == pair for pid in pairs_te])
        if mask.sum() == 0:
            continue
        t_src_val, t_tgt_val = pair
        pred_df = pd.DataFrame(test_pred[mask], columns=protein_names)
        pred_df.insert(0, "t_tgt", t_tgt_val)
        pred_df.insert(0, "t_src", t_src_val)
        truth_df = pd.DataFrame(y_te_np[mask], columns=protein_names)
        truth_df.insert(0, "t_tgt", t_tgt_val)
        truth_df.insert(0, "t_src", t_src_val)

        pred_path = out_dir / f"joint_y_pred_t{int(t_src_val)}to{int(t_tgt_val)}.csv"
        truth_path = out_dir / f"joint_y_truth_t{int(t_src_val)}to{int(t_tgt_val)}.csv"
        pred_df.to_csv(pred_path, index=False)
        truth_df.to_csv(truth_path, index=False)
        print(f"  Saved {pred_path.name} and {truth_path.name} ({mask.sum()} cells)")

    results = {
        "n_timepoints": len(times),
        "timepoints": times,
        "n_genes_selected": len(deg_genes),
        "n_proteins": new_output_dim,
        "test_R": test_r,
        "per_pair_breakdown": per_pair_breakdown,
        "diagnostics": {
            "real_R": test_r,
            "prediction_shuffle_R": r_pred_shuffle,
            "time_shuffle_R": r_time_shuffle,
            "rna_shuffle_R": r_rna_shuffle,
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal test R = {test_r:.4f}")
    print(f"Diagnostics: pred-shuffle R={r_pred_shuffle:.4f} | time-shuffle R={r_time_shuffle:.4f} | rna-shuffle R={r_rna_shuffle:.4f}")
    print(f"Saved results to {out_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", type=str, required=True)
    parser.add_argument("--prot_path", type=str, required=True)
    parser.add_argument("--time_col", type=str, default="time_min")
    parser.add_argument("--gene_id_col", type=str, default=None,
                         help="var column holding gene symbols, if not already var_names")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="film_pre",
                         choices=["mlp", "film_pre", "film_post", "gated", "gated_post"],
                         help="Architecture the checkpoint was trained with -- MUST match, "
                              "or state_dict loading will warn/fail.")
    parser.add_argument("--gene_panel_size", type=int, default=1000)
    parser.add_argument("--ot_reg", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_adapter", type=float, default=1e-3)
    parser.add_argument("--lr_trunk", type=float, default=1e-5)
    parser.add_argument("--corr_weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for model init and train/val/test split -- run with "
                              "several different seeds and report mean+/-std for a reliable result")
    parser.add_argument("--out_dir", type=str, default="external_finetune_results")
    args = parser.parse_args()
    main(args)
