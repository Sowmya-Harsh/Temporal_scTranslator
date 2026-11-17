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
        if np.std(yt) < 1e-8:
            continue  # no real signal in this channel's truth
        if np.std(yp) < 1e-8:
            rs.append(0.0)  # constant predictor -> no correlation, not NaN
            continue
        r, _ = pearsonr(yt, yp)
        if not np.isnan(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def per_protein_ccc(y_true, y_pred):
    """Matches the convention used everywhere else in this pipeline
    (train_joint.py, 14_baselines.py) -- was previously MISSING here
    despite the module docstring claiming CCC was computed. This gap
    meant Table myeloid-multiseed's CCC column had no script behind it."""
    cccs = []
    for p in range(y_true.shape[1]):
        yt, yp = y_true[:, p], y_pred[:, p]
        if np.std(yt) < 1e-8:
            continue
        mean_t, mean_p = np.mean(yt), np.mean(yp)
        var_t, var_p = np.var(yt), np.var(yp)
        cov = np.mean((yt - mean_t) * (yp - mean_p))
        ccc = (2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12)
        cccs.append(ccc)
    return float(np.mean(cccs)) if cccs else float("nan")


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


# ---------------------------------------------------------------------
# Model: pretrained trunk (must match the architecture of the checkpoint
# being loaded) + new input/output adapters for the new dataset's shapes
# ---------------------------------------------------------------------
class FiLMGenerator(nn.Module):
    """Maps t_info -> (gamma, beta) of size `feature_dim` each.
    Matches models.py's FiLMGenerator exactly (module name/shape must
    line up with the saved checkpoint)."""
    def __init__(self, t_dim, feature_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim * 2)
        )
        self.feature_dim = feature_dim

    def forward(self, t_info):
        out = self.net(t_info)
        gamma, beta = out[:, :self.feature_dim], out[:, self.feature_dim:]
        return 1.0 + gamma, beta


class FiLMPreTrunk(nn.Module):
    """Reconstructs the REAL FiLMPreMLP architecture from models.py
    (film_in -> trunk -> out), so the pretrained checkpoint
    (correlationlosswt_test_deg1000_filmpre/joint_model_best.pt) loads
    with matching module names. trunk_input_dim here = n_genes the
    checkpoint was actually trained on (1000, the DEG panel size);
    trunk_output_dim = n_proteins it was trained on (58)."""

    def __init__(self, trunk_input_dim, trunk_output_dim, hidden_dims=(512, 256, 128),
                 dropout=0.2, t_dim=2):
        super().__init__()
        self.film_in = FiLMGenerator(t_dim, trunk_input_dim)
        dims = [trunk_input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], trunk_output_dim)
        self.trunk_input_dim = trunk_input_dim
        self.trunk_output_dim = trunk_output_dim

    def forward(self, x, t_info):
        gamma, beta = self.film_in(t_info)
        x_mod = gamma * x + beta
        h = self.trunk(x_mod)
        return torch.sigmoid(self.out(h))


class GatedTimeInjectionExternal(nn.Module):
    """Exact copy of models.py's GatedTimeInjection -- module names
    (shift, gate) and init (gate bias=-2.0) must match exactly for the
    gated checkpoint's state_dict to load correctly."""
    def __init__(self, n_genes, t_dim=2, hidden=64):
        super().__init__()
        self.shift = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_genes),
        )
        self.gate = nn.Sequential(
            nn.Linear(t_dim, n_genes),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -2.0)

    def forward(self, x, t_info):
        shift = self.shift(t_info)
        gate = self.gate(t_info)
        return x + gate * shift


class GatedTrunk(nn.Module):
    """Reconstructs models.py's GatedInjectionMLP (gated_in -> trunk -> out)
    for loading the new final-model checkpoint (gated architecture,
    DEG-3000, combined correlation+CCC loss). Module names must match
    GatedInjectionMLP exactly so the checkpoint's state_dict loads
    without remapping."""

    def __init__(self, trunk_input_dim, trunk_output_dim, hidden_dims=(512, 256, 128),
                 dropout=0.2, t_dim=2):
        super().__init__()
        self.gated_in = GatedTimeInjectionExternal(trunk_input_dim, t_dim=t_dim)
        dims = [trunk_input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], trunk_output_dim)
        self.trunk_input_dim = trunk_input_dim
        self.trunk_output_dim = trunk_output_dim

    def forward(self, x, t_info):
        x_mod = self.gated_in(x, t_info)
        h = self.trunk(x_mod)
        return torch.sigmoid(self.out(h))


class AdaptedModel(nn.Module):
    """Pretrained trunk + new linear adapters so it can consume/produce
    the NEW dataset's own gene/protein panel sizes, whatever they are."""

    def __init__(self, trunk: FiLMPreTrunk, new_input_dim, new_output_dim):
        super().__init__()
        self.input_adapter = nn.Linear(new_input_dim, trunk.trunk_input_dim)
        self.trunk = trunk
        self.output_adapter = nn.Linear(trunk.trunk_output_dim, new_output_dim)

    def forward(self, x, t_info):
        x = self.input_adapter(x)
        h = self.trunk(x, t_info)
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

    # --- Split ONCE, here, before anything touches gene selection or OT ---
    # (previously select_deg_genes ran on the full `rna` before any split
    # existed, so val/test cells' RNA leaked into which genes got chosen).
    # This single split is reused below for gene selection, OT target pools,
    # and train/val/test dataset construction -- no second split is computed
    # later, so there is no risk of the two silently diverging.
    time_masks = {t: np.where(np.isclose(rna.obs[args.time_col].values.astype(float), t))[0]
                  for t in times}
    rng = np.random.default_rng(args.seed)
    train_masks, val_masks, test_masks = {}, {}, {}
    for t in times:
        idx = time_masks[t].copy()
        rng.shuffle(idx)
        n_tr = int(0.8 * len(idx))
        n_va = int(0.1 * len(idx))
        train_masks[t] = idx[:n_tr]
        val_masks[t] = idx[n_tr:n_tr + n_va]
        test_masks[t] = idx[n_tr + n_va:]

    train_idx_all = np.concatenate([train_masks[t] for t in times if len(train_masks[t]) > 0])
    print(f"Selecting top {args.gene_panel_size} DEG genes on external data "
          f"(train-split cells only: {len(train_idx_all)}/{rna.n_obs})...")
    rna_train_for_genes = rna[train_idx_all].copy()
    deg_genes = select_deg_genes(rna_train_for_genes, args.time_col, args.gene_panel_size)
    deg_genes = [g for g in deg_genes if g in rna.var_names]
    print(f"  {len(deg_genes)} genes selected (after intersecting with var_names)")

    rna_X_full = rna[:, deg_genes].X
    rna_X = rna_X_full.toarray() if hasattr(rna_X_full, "toarray") else np.asarray(rna_X_full)
    prot_X_full = prot.X
    prot_X = prot_X_full.toarray() if hasattr(prot_X_full, "toarray") else np.asarray(prot_X_full)

    new_input_dim = rna_X.shape[1]
    new_output_dim = prot_X.shape[1]
    print(f"New input dim (genes): {new_input_dim} | new output dim (proteins): {new_output_dim}")

    # Load pretrained trunk
    print(f"Loading pretrained checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    sd = ckpt[state_dict_key]

    # Infer trunk_input_dim/trunk_output_dim directly from the checkpoint's
    # own tensor shapes -- NEVER trust ckpt.get("trunk_input_dim", 1000),
    # since train_joint_improved.py's saved checkpoints never actually
    # contain that key (only epoch/model_state_dict/optimizer_state_dict/
    # val_loss), so that .get() call always silently fell back to the
    # hardcoded default of 1000 regardless of the real training config.
    # This worked by coincidence for the original DEG-1000 checkpoint and
    # broke silently/loudly for any other panel size (e.g. DEG-3000).
    # trunk.0 is always the first Linear layer of the shared trunk
    # (nn.Linear(trunk_input_dim, 512, ...)) in BOTH FiLMPreTrunk and
    # GatedTrunk, so its second dimension is trunk_input_dim regardless
    # of architecture -- this is simpler and correct for both, unlike
    # inferring from the architecture-specific conditioning module.
    trunk_input_dim = sd["trunk.0.weight"].shape[1]
    trunk_output_dim = sd["out.weight"].shape[0]
    print(f"Inferred trunk dims from checkpoint: input={trunk_input_dim} output={trunk_output_dim}")

    if args.model == "gated":
        trunk = GatedTrunk(trunk_input_dim, trunk_output_dim)
    else:
        trunk = FiLMPreTrunk(trunk_input_dim, trunk_output_dim)
    missing, unexpected = trunk.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  WARNING: state_dict mismatch — missing={missing}, unexpected={unexpected}. "
              f"Check FiLMPreTrunk architecture matches your real checkpoint's saved class exactly.")

    model = AdaptedModel(trunk, new_input_dim, new_output_dim)
    optimizer = torch.optim.Adam([
        {"params": model.input_adapter.parameters(), "lr": args.lr_adapter},
        {"params": model.output_adapter.parameters(), "lr": args.lr_adapter},
        {"params": model.trunk.parameters(), "lr": args.lr_trunk},
    ])

    # Build all pairwise timepoint transitions from whatever timepoints exist.
    # (train_masks/val_masks/test_masks/time_masks/rng were already built
    # above, before gene selection -- reused here as-is, not recomputed.)
    pairs = [(t1, t2) for i, t1 in enumerate(times) for t2 in times[i + 1:]]

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

    def per_pair_ccc(y_true, y_pred, pair_ids):
        """Same convention as per_pair_r above, but CCC. Was previously
        entirely missing from this script -- Table myeloid-multiseed's
        CCC column had no computation behind it before this fix."""
        unique_pairs = sorted(set(pair_ids))
        pair_cccs = []
        for pair in unique_pairs:
            mask = np.array([pid == pair for pid in pair_ids])
            if mask.sum() < 5:
                continue
            ccc = per_protein_ccc(y_true[mask], y_pred[mask])
            if not np.isnan(ccc):
                pair_cccs.append(ccc)
        return float(np.mean(pair_cccs)) if pair_cccs else float("nan")

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
    test_ccc = per_pair_ccc(y_te.numpy(), test_pred, pairs_te)

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
    ccc_pred_shuffle = per_pair_ccc(y_te.numpy(), shuffled_pred, pairs_te)

    t_te_shuffled = t_te[torch.randperm(len(t_te))]
    with torch.no_grad():
        pred_time_shuffle = model(x_te, t_te_shuffled).numpy()
    r_time_shuffle = per_pair_r(y_te.numpy(), pred_time_shuffle, pairs_te)
    ccc_time_shuffle = per_pair_ccc(y_te.numpy(), pred_time_shuffle, pairs_te)

    x_te_shuffled = x_te[torch.randperm(len(x_te))]
    with torch.no_grad():
        pred_rna_shuffle = model(x_te_shuffled, t_te).numpy()
    r_rna_shuffle = per_pair_r(y_te.numpy(), pred_rna_shuffle, pairs_te)
    ccc_rna_shuffle = per_pair_ccc(y_te.numpy(), pred_rna_shuffle, pairs_te)

    per_pair_breakdown = {}
    for pair in sorted(set(pairs_te)):
        mask = np.array([pid == pair for pid in pairs_te])
        if mask.sum() < 5:
            continue
        r = per_protein_r(y_te.numpy()[mask], test_pred[mask])
        ccc = per_protein_ccc(y_te.numpy()[mask], test_pred[mask])
        per_pair_breakdown[f"{pair[0]}->{pair[1]}"] = {"n_test": int(mask.sum()), "R": r, "CCC": ccc}

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
        "test_CCC": test_ccc,
        "per_pair_breakdown": per_pair_breakdown,
        "diagnostics": {
            "real_R": test_r, "real_CCC": test_ccc,
            "prediction_shuffle_R": r_pred_shuffle, "prediction_shuffle_CCC": ccc_pred_shuffle,
            "time_shuffle_R": r_time_shuffle, "time_shuffle_CCC": ccc_time_shuffle,
            "rna_shuffle_R": r_rna_shuffle, "rna_shuffle_CCC": ccc_rna_shuffle,
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal test R = {test_r:.4f} | Final test CCC = {test_ccc:.4f}")
    print(f"Diagnostics R:   pred-shuffle={r_pred_shuffle:.4f} | time-shuffle={r_time_shuffle:.4f} | rna-shuffle={r_rna_shuffle:.4f}")
    print(f"Diagnostics CCC: pred-shuffle={ccc_pred_shuffle:.4f} | time-shuffle={ccc_time_shuffle:.4f} | rna-shuffle={ccc_rna_shuffle:.4f}")
    print(f"Saved results to {out_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", type=str, required=True)
    parser.add_argument("--prot_path", type=str, required=True)
    parser.add_argument("--time_col", type=str, default="time_min")
    parser.add_argument("--gene_id_col", type=str, default=None,
                         help="var column holding gene symbols, if not already var_names")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="film_pre", choices=["film_pre", "gated"],
                         help="Trunk architecture matching the checkpoint being loaded. "
                              "Defaults to film_pre for backward compatibility with the "
                              "original correlation-only/DEG-1000 checkpoint; use 'gated' "
                              "for the new gated/DEG-3000/combined-loss final model.")
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