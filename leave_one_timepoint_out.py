"""
leave_one_timepoint_out.py
─────────────────────────────
Leave-one-timepoint-out interpolation/extrapolation experiment.

Question: can the model generalize its RNA->protein mapping to a
TIMEPOINT it never saw ANY supervision for, using only the continuous
(log_t_src, log_delta_t) time-conditioning to extrapolate/interpolate?
This is a stronger generalization test than the normal held-out-CELLS
test/val split, which always includes cells from every timepoint during
training -- here, one entire timepoint's cells are excluded from
training completely (never used as a training source OR training
target, for ANY pair).

Held-out timepoints available in CURRICULUM: {0, 2, 4, 6, 60, 180}.
  - Holding out an INTERIOR timepoint (2, 4, 6, 60) is a genuine
    interpolation test: the model is trained on pairs both earlier
    and later than the held-out point, then evaluated on pairs that
    involve it.
  - Holding out an ENDPOINT (0 or 180) is EXTRAPOLATION, not
    interpolation, since 0 only ever appears as a source and 180 only
    ever appears as a target in CURRICULUM -- holding either out
    removes a large fraction of all pairs and tests a different
    question (generalizing beyond the observed time range, not within
    it). The script runs either way but labels the holdout type in
    its output so this distinction isn't lost.

For the held-out timepoint's cells: since they are NEVER used to
update model weights (excluded from every training pair), ALL of
them (not just a train-designated slice) are used for evaluation --
there is no need to reserve a train/val portion of a timepoint that
was never trained on.

Gene selection is recomputed using ONLY the cells belonging to
timepoints that remain in the training curriculum (the held-out
timepoint's cells never touch gene ranking, consistent with the
leakage-avoidance rule applied everywhere else).

Usage:
    python leave_one_timepoint_out.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --ot_type soft --ot_reg 0.01 \
        --model film_pre --correlation_loss_weight 0.3 \
        --holdout_time 6 \
        --epochs 100 --outdir results/leave_one_timepoint_out_t6
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(os.path.dirname(__file__))
import train_joint as tj
from gene_selection import select_genes
from models import build_model
from training_improvements import correlation_loss, ccc_loss


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
    parser.add_argument("--correlation_loss_weight", type=float, default=0.0)
    parser.add_argument("--ccc_loss_weight", type=float, default=0.0)
    parser.add_argument("--holdout_time", type=float, required=True,
                         help="Timepoint (in minutes) to hold out ENTIRELY from training. "
                              "Interior values {2,4,6,60} test interpolation; "
                              "endpoints {0,180} test extrapolation instead.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)

    holdout_type = "interpolation" if args.holdout_time in (2, 4, 6, 60) else \
                   ("extrapolation" if args.holdout_time in (0, 180) else "unknown")
    print(f"Leave-one-timepoint-out: holdout_time={args.holdout_time} ({holdout_type})")

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

    # Normal per-cell split, all timepoints (holdout's cells get a
    # train/val/test label too, but its TRAIN pairs are never built --
    # see below. Its train_cells slice is still usable as an OT target
    # pool when it's used as a target in an eval pair, val/test slices
    # aren't needed since nothing is scored against them here.)
    train_cells, val_cells, test_cells = tj.split_cells_per_timepoint(obs_time, seed=args.seed)

    # --- Training curriculum: exclude ANY pair touching the holdout ---
    train_pairs = [(s, t) for s, t in tj.CURRICULUM
                   if s != args.holdout_time and t != args.holdout_time
                   and s in train_cells and t in train_cells]
    eval_pairs = [(s, t) for s, t in tj.CURRICULUM
                  if (s == args.holdout_time or t == args.holdout_time)
                  and s in train_cells and t in train_cells]
    print(f"Training pairs (holdout excluded): {train_pairs}")
    print(f"Eval pairs (involve holdout timepoint): {eval_pairs}")
    if not eval_pairs:
        raise ValueError(f"No CURRICULUM pairs involve holdout_time={args.holdout_time} -- check the value.")

    # --- Gene selection: cells from training-curriculum timepoints only,
    # holdout's cells excluded regardless of split (train/val/test) ---
    train_timepoints = sorted(set([s for s, t in train_pairs] + [t for s, t in train_pairs]))
    train_idx_all = np.concatenate([train_cells[t] for t in train_timepoints if len(train_cells[t]) > 0])
    rna_adata_train = rna_adata[train_idx_all].copy()
    gene_idx = select_genes(rna_adata_train, args.gene_strategy, n_genes_arg, groupby="time_min")
    rna_selected = rna_full[:, gene_idx]
    print(f"Selected {len(gene_idx)} genes (train-curriculum cells only, holdout excluded: "
          f"{len(train_idx_all)}/{rna_adata.n_obs})")

    # --- Build TRAINING datasets: standard train/val split, holdout never appears ---
    train_ds_by_pair, val_ds_by_pair = {}, {}
    for (t_src, t_tgt) in train_pairs:
        tr, va, _ = tj.build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                           t_src, t_tgt, args.ot_type, args.ot_reg, seed=args.seed)
        if tr is not None:
            train_ds_by_pair[(t_src, t_tgt)] = tr
            val_ds_by_pair[(t_src, t_tgt)] = va

    # --- Build EVAL datasets: pairs touching the holdout timepoint ---
    # If holdout is the TARGET: source = normal test-split cells of t_src
    # (consistent with the standard protocol elsewhere); target pool =
    # ALL of holdout's cells (not just its train slice -- nothing here
    # updates model weights, so there's no reason to withhold any of
    # holdout's cells from being the OT target pool).
    # If holdout is the SOURCE: source = ALL of holdout's cells (novel
    # input distribution, none were used in training); target pool =
    # t_tgt's TRAIN split (matches the leakage-avoidance rule used
    # everywhere else in the pipeline: target pool = target timepoint's
    # train split).
    all_cells_by_time = {}
    for t in sorted(np.unique(obs_time).tolist()):
        all_cells_by_time[t] = np.where(np.isclose(obs_time, t, atol=1e-3))[0]

    eval_ds_by_pair = {}
    for (t_src, t_tgt) in eval_pairs:
        if t_tgt == args.holdout_time:
            src_idx = test_cells[t_src]
            tgt_pool_idx = all_cells_by_time[args.holdout_time]
        else:  # t_src == holdout
            src_idx = all_cells_by_time[args.holdout_time]
            tgt_pool_idx = train_cells[t_tgt]

        if len(src_idx) < 2 or len(tgt_pool_idx) < 2:
            print(f"  SKIP eval {t_src}->{t_tgt}: too few cells (src={len(src_idx)}, tgt_pool={len(tgt_pool_idx)}).")
            continue

        eval_targets = tj.get_ot_targets(rna_selected[src_idx], rna_selected[tgt_pool_idx],
                                          prot_full[tgt_pool_idx], args.ot_type, args.ot_reg)
        eval_ds_by_pair[(t_src, t_tgt)] = tj.PairDataset(rna_selected[src_idx], eval_targets, t_src, t_tgt)
        print(f"  Eval pair {t_src}->{t_tgt}: n_src={len(src_idx)} n_tgt_pool={len(tgt_pool_idx)}")

    # --- Train the model, holdout never involved ---
    model = build_model(args.model, n_genes=len(gene_idx), n_proteins=len(protein_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    ckpt_path = os.path.join(args.outdir, "model_best.pt")
    best_val_loss = float("inf")

    full_train = ConcatDataset(list(train_ds_by_pair.values()))
    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True)

    def val_loss_over_all_pairs():
        losses = []
        for pair, va in val_ds_by_pair.items():
            if va is None or len(va) == 0:
                continue
            loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)
            losses.append(tj.run_epoch(model, loader, device, optimizer=None, criterion=criterion))
        return float(np.mean(losses)) if losses else float("inf")

    print(f"\n=== Training (holdout timepoint {args.holdout_time} fully excluded) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, t_info in train_loader:
            x, y, t_info = x.to(device), y.to(device), t_info.to(device)
            optimizer.zero_grad()
            y_hat = model(x, t_info)
            loss = criterion(y_hat, y)
            if args.correlation_loss_weight > 0:
                loss = loss + args.correlation_loss_weight * correlation_loss(y_hat, y)
            if args.ccc_loss_weight > 0:
                loss = loss + args.ccc_loss_weight * ccc_loss(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss = val_loss_over_all_pairs()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ckpt_path)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}: val_loss={val_loss:.4f} (best={best_val_loss:.4f})")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"\n=== Evaluating on pairs involving holdout timepoint {args.holdout_time} ({holdout_type}) ===")
    eval_results = tj.evaluate_per_pair(model, eval_ds_by_pair, device, protein_names, outdir=args.outdir)

    rows = []
    for (t_src, t_tgt), res in eval_results.items():
        rows.append({"holdout_time": args.holdout_time, "holdout_type": holdout_type,
                      "t_src": t_src, "t_tgt": t_tgt, "n_eval": res["n_test"],
                      "mse": res["mse"], "mean_pearson_r": res["mean_pearson_r"], "ccc": res["ccc"]})
    df = pd.DataFrame(rows)
    out_path = os.path.join(args.outdir, "leave_one_timepoint_out_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nSaved to {out_path}")

    with open(os.path.join(args.outdir, "leave_one_timepoint_out_config.json"), "w") as f:
        json.dump({"config": vars(args), "holdout_type": holdout_type,
                    "train_pairs": train_pairs, "eval_pairs": eval_pairs,
                    "n_genes_actual": len(gene_idx), "best_val_loss": best_val_loss}, f, indent=2)


if __name__ == "__main__":
    main()
