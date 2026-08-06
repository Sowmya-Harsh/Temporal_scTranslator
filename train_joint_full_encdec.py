"""
train_joint_full_encdec.py
──────────────────────────────
Uses the FULL scTranslator pipeline (encoder -> translator -> decoder),
NOT the mean-pooled-encoder-only approach in train_joint_pretrained_encoder.py.

This directly fixes the likely bottleneck in that earlier script:
mean-pooling over per-gene encoder embeddings discarded exactly the
fine-grained structure the transformer was built to capture. Here,
the translator does the RNA-space -> protein-space dimensionality
reduction itself (as it was pretrained to), and time is injected via
PostTranslatorFiLM (temporal_gated.py) -- the shortest possible
gradient path: MSE -> decoder -> FiLM -> HERE.

Uses our own OT-matched, all-pairs-joint data pipeline (same as
train_joint.py: CURRICULUM, build_pair_dataset, soft/hard OT), not the
reference script's SimultaneousDataStore -- so gene selection, OT
tuning, and per-protein CCC/Pearson r metrics all stay consistent with
the rest of this project.

Two stages, same as train_joint_pretrained_encoder.py:
  Stage 1: freeze base (enc+translator+dec), train only PostFiLM (~33k params)
  Stage 2: unfreeze everything, fine-tune end-to-end at low LR

Usage:
    python train_joint_full_encdec.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --ot_type soft --ot_reg 0.01 \
        --pretrain_checkpoint checkpoint/scTranslator_2M.pt \
        --stage1_epochs 50 --stage2_epochs 50 \
        --outdir results/joint_full_encdec
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
from train_joint import (
    CURRICULUM, split_cells_per_timepoint, build_pair_dataset,
    concordance_correlation_coefficient,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "temporal_models"))
from temporal_gated import TemporalPostFiLMModel, TemporalGatedModel

# The checkpoint is a full pickled model object (not a state_dict), so
# torch.load needs to import performer_enc_dec.py to reconstruct the
# scPerformerEncDec class -- make sure that module is importable.
# Everything is flat in the repo root: this file, temporal_models/, and
# code/model/ are all direct children of the same root directory.
sys.path.append(os.path.join(os.path.dirname(__file__), "code", "model"))


class WithGeneAndProteinIDs(Dataset):
    """Wraps a train_joint.PairDataset (x, y, t_info) to also carry the
    shared RNA gene_ids and protein_ids arrays needed by the real
    decoder (which needs to know WHICH protein each output slot is).

    IMPORTANT: BOTH the translator (fixed enc_max_seq_len=20000 input)
    AND the decoder (fixed dec_max_seq_len=1000 output length) are
    fixed-size, unlike the standalone encoder. RNA values/gene_ids are
    zero-padded to enc_max_seq_len; protein_ids are zero-padded to
    dec_max_seq_len. The real proteins are always the FIRST n_proteins
    positions (padding trails), so the decoder's output can simply be
    sliced back to [:n_proteins] before comparing against the real
    (unpadded) target -- no need to pad the target itself."""
    def __init__(self, base_ds, gene_ids, protein_ids, enc_max_seq_len=20000, dec_max_seq_len=1000):
        self.base = base_ds
        n_genes = len(gene_ids)
        if n_genes > enc_max_seq_len:
            raise ValueError(f"Selected {n_genes} genes exceeds enc_max_seq_len={enc_max_seq_len}.")
        n_proteins = len(protein_ids)
        if n_proteins > dec_max_seq_len:
            raise ValueError(f"{n_proteins} proteins exceeds dec_max_seq_len={dec_max_seq_len}.")

        self.enc_max_seq_len = enc_max_seq_len
        self.n_genes = n_genes
        padded_gene_ids = np.zeros(enc_max_seq_len, dtype=np.int64)
        padded_gene_ids[:n_genes] = gene_ids
        self.gene_ids = torch.tensor(padded_gene_ids)

        self.n_proteins = n_proteins
        padded_protein_ids = np.zeros(dec_max_seq_len, dtype=np.int64)
        padded_protein_ids[:n_proteins] = protein_ids
        self.protein_ids = torch.tensor(padded_protein_ids)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, t_info = self.base[idx]
        x_padded = torch.zeros(self.enc_max_seq_len, dtype=x.dtype)
        x_padded[:self.n_genes] = x
        log_t, log_dt = t_info[0], t_info[1]
        return x_padded, self.gene_ids, y, self.protein_ids, log_t, log_dt


def run_stage(model, train_loader, val_loader, device, epochs, lr, stage_name, ckpt_path, n_proteins):
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  [{stage_name}] Trainable params: {n_trainable:,}")
    optimizer = torch.optim.Adam(trainable, lr=lr, amsgrad=True)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_vals, gene_ids, y, prot_ids, log_t, log_dt in train_loader:
            x_vals, gene_ids = x_vals.to(device), gene_ids.to(device)
            y, prot_ids = y.to(device), prot_ids.to(device)
            log_t, log_dt = log_t.to(device), log_dt.to(device)

            optimizer.zero_grad()
            B = x_vals.shape[0]
            y_hat = model(x_vals, gene_ids, prot_ids, log_t, log_dt)
            y_hat = y_hat.reshape(B, -1)  # robust to any internal squeeze/dim quirks
            y_hat = y_hat[:, :n_proteins]  # decoder output is padded to dec_max_seq_len; slice to real proteins
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_vals, gene_ids, y, prot_ids, log_t, log_dt in val_loader:
                x_vals, gene_ids = x_vals.to(device), gene_ids.to(device)
                y, prot_ids = y.to(device), prot_ids.to(device)
                log_t, log_dt = log_t.to(device), log_dt.to(device)
                y_hat = model(x_vals, gene_ids, prot_ids, log_t, log_dt)
                B = x_vals.shape[0]
                y_hat = y_hat.reshape(B, -1)
                y_hat = y_hat[:, :n_proteins]
                val_loss += criterion(y_hat, y).item()
                n_val += 1
        val_loss /= max(1, n_val)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state_dict": best_state, "val_loss": val_loss}, ckpt_path)
        if epoch % 10 == 0 or epoch == 1 or improved:
            flag = " *improved*" if improved else ""
            print(f"  [{stage_name}] Epoch {epoch:>4}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}{flag}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate_per_pair(model, test_ds_by_pair, device, protein_names, outdir, tag, n_proteins):
    model.eval()
    results = {}
    for (t_src, t_tgt), ds in test_ds_by_pair.items():
        if ds is None or len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        y_hat_all, y_all = [], []
        with torch.no_grad():
            for x_vals, gene_ids, y, prot_ids, log_t, log_dt in loader:
                x_vals, gene_ids = x_vals.to(device), gene_ids.to(device)
                prot_ids = prot_ids.to(device)
                log_t, log_dt = log_t.to(device), log_dt.to(device)
                y_hat = model(x_vals, gene_ids, prot_ids, log_t, log_dt)
                B = x_vals.shape[0]
                y_hat = y_hat.reshape(B, -1)
                y_hat = y_hat[:, :n_proteins]
                y_hat_all.append(y_hat.cpu().numpy())
                y_all.append(y.numpy())
        y_hat_all = np.concatenate(y_hat_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        pred_df = pd.DataFrame(y_hat_all, columns=protein_names)
        pred_df.insert(0, "t_src", t_src)
        pred_df.insert(1, "t_tgt", t_tgt)
        pred_df.to_csv(os.path.join(outdir, f"{tag}_y_pred_t{int(t_src)}to{int(t_tgt)}.csv"), index=False)

        truth_df = pd.DataFrame(y_all, columns=protein_names)
        truth_df.insert(0, "t_src", t_src)
        truth_df.insert(1, "t_tgt", t_tgt)
        truth_df.to_csv(os.path.join(outdir, f"{tag}_y_truth_t{int(t_src)}to{int(t_tgt)}.csv"), index=False)

        mse = float(np.mean((y_hat_all - y_all) ** 2))
        per_r, per_ccc = [], []
        for i in range(y_all.shape[1]):
            if np.std(y_all[:, i]) > 1e-8 and np.std(y_hat_all[:, i]) > 1e-8:
                r, _ = pearsonr(y_all[:, i], y_hat_all[:, i])
                per_r.append(r)
                per_ccc.append(concordance_correlation_coefficient(y_all[:, i], y_hat_all[:, i]))
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
    parser.add_argument("--pretrain_checkpoint", required=True)
    parser.add_argument("--architecture", choices=["postfilm", "gated"], default="postfilm")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--freeze_mode", choices=["both", "frozen_only", "unfrozen_only"], default="both")
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument("--stage1_lr", type=float, default=2e-4)
    parser.add_argument("--stage2_lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stage2_batch_size", type=int, default=None,
                         help="Batch size for Stage 2 (full unfrozen backprop through 117M "
                              "params needs much more memory than Stage 1's frozen forward "
                              "pass). Defaults to batch_size // 2 if not set. Lower this "
                              "further if you still hit CUDA OOM at Stage 2.")
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Config: FULL enc-translator-dec, architecture={args.architecture} "
          f"gene_strategy={args.gene_strategy} n_genes={n_genes_arg} "
          f"ot_type={args.ot_type} ot_reg={args.ot_reg}")

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
    print(f"Selected {len(gene_idx)} genes via {args.gene_strategy}.")
    rna_selected = rna_full[:, gene_idx]
    gene_my_ids = rna_adata.var["my_Id"].astype(int).values[gene_idx]

    if "my_Id" not in prot_adata.var.columns:
        raise RuntimeError("Protein h5ad missing 'my_Id' column -- required for the real "
                            "decoder to know which protein each output slot corresponds to.")
    protein_my_ids = prot_adata.var["my_Id"].astype(int).values
    if gene_my_ids.max() >= 85500 or protein_my_ids.max() >= 85500:
        print("ERROR: a gene or protein my_Id exceeds the pretrained embedding table size. Aborting.")
        return

    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)
    available_pairs = [(s, t) for s, t in CURRICULUM if s in train_cells and t in train_cells]
    print(f"Available timepoint pairs: {available_pairs}")

    train_ds_by_pair, val_ds_by_pair, test_ds_by_pair = {}, {}, {}
    for (t_src, t_tgt) in available_pairs:
        tr, va, te = build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                         t_src, t_tgt, args.ot_type, args.ot_reg)
        if tr is not None:
            train_ds_by_pair[(t_src, t_tgt)] = WithGeneAndProteinIDs(tr, gene_my_ids, protein_my_ids)
            val_ds_by_pair[(t_src, t_tgt)] = WithGeneAndProteinIDs(va, gene_my_ids, protein_my_ids) if va else None
            test_ds_by_pair[(t_src, t_tgt)] = WithGeneAndProteinIDs(te, gene_my_ids, protein_my_ids) if te else None

    full_train = ConcatDataset(list(train_ds_by_pair.values()))
    full_val = ConcatDataset([v for v in val_ds_by_pair.values() if v is not None])
    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(full_val, batch_size=args.batch_size, shuffle=False)

    print(f"\nLoading base model from {args.pretrain_checkpoint}...")
    base_model = torch.load(args.pretrain_checkpoint, map_location="cpu")

    ModelClass = TemporalPostFiLMModel if args.architecture == "postfilm" else TemporalGatedModel
    all_results = {}
    model = None

    n_proteins = len(protein_names)

    if args.freeze_mode in ("both", "frozen_only"):
        print(f"\n=== STAGE 1: FROZEN base, training only {args.architecture} module ===")
        model = ModelClass(base_model=base_model, dim=args.dim, freeze_base=True).to(device)
        ckpt1 = os.path.join(args.outdir, "stage1_frozen_best.pt")
        model, val1 = run_stage(model, train_loader, val_loader, device, args.stage1_epochs,
                                 args.stage1_lr, "stage1-frozen", ckpt1, n_proteins)
        print("\n  Stage 1 per-pair test results:")
        all_results["stage1_frozen"] = evaluate_per_pair(model, test_ds_by_pair, device, protein_names,
                                                           args.outdir, "stage1_frozen", n_proteins)

    if args.freeze_mode in ("both", "unfrozen_only"):
        # Stage 2 backprops through all 117M params (full reversible-layer
        # activations), needing far more memory than Stage 1's frozen
        # forward pass at the same batch size -- both prior runs OOM'd
        # here. Use a smaller batch size and clear the cache first.
        stage2_bs = args.stage2_batch_size if args.stage2_batch_size is not None else max(1, args.batch_size // 2)
        print(f"\nClearing CUDA cache before Stage 2, using batch_size={stage2_bs} "
              f"(Stage 1 used {args.batch_size})...")
        torch.cuda.empty_cache()
        stage2_train_loader = DataLoader(full_train, batch_size=stage2_bs, shuffle=True, drop_last=True)
        stage2_val_loader = DataLoader(full_val, batch_size=stage2_bs, shuffle=False)

        if args.freeze_mode == "both":
            for p in model.parameters():
                p.requires_grad = True
            print("Base unfrozen.")
        else:
            base_model2 = torch.load(args.pretrain_checkpoint, map_location="cpu")
            model = ModelClass(base_model=base_model2, dim=args.dim, freeze_base=False).to(device)

        stage_label = "stage2-unfrozen" if args.freeze_mode == "both" else "unfrozen-from-start"
        lr = args.stage2_lr if args.freeze_mode == "both" else args.stage1_lr
        print(f"\n=== {'STAGE 2' if args.freeze_mode == 'both' else 'UNFROZEN FROM START'}: fine-tuning end-to-end ===")
        ckpt2 = os.path.join(args.outdir, "stage2_unfrozen_best.pt")
        model, val2 = run_stage(model, stage2_train_loader, stage2_val_loader, device, args.stage2_epochs,
                                 lr, stage_label, ckpt2, n_proteins)
        print(f"\n  {stage_label} per-pair test results:")
        all_results["stage2_unfrozen"] = evaluate_per_pair(model, test_ds_by_pair, device, protein_names,
                                                             args.outdir, "stage2_unfrozen", n_proteins)

    print("\n" + "=" * 60)
    print(f"SUMMARY (architecture={args.architecture}, freeze_mode={args.freeze_mode})")
    print("=" * 60)
    for stage_name, res in all_results.items():
        mean_r = np.mean([v["mean_pearson_r"] for v in res.values()])
        mean_ccc = np.mean([v["ccc"] for v in res.values()])
        print(f"  {stage_name}: avg mean_r={mean_r:.4f}  avg ccc={mean_ccc:.4f}")

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({"config": vars(args), "available_pairs": available_pairs,
                    "results": {k: {f"{s}->{t}": v for (s, t), v in res.items()}
                                for k, res in all_results.items()}}, f, indent=2)
    print(f"\nSaved all results to {args.outdir}/")


if __name__ == "__main__":
    main()