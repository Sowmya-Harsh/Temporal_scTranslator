"""
train_joint_pretrained_encoder.py
────────────────────────────────────
ONE model, trained across ALL timepoint pairs (same joint structure as
train_joint.py -- no ensembling, no per-timepoint models), but using
scTranslator's PRETRAINED encoder as the RNA feature extractor instead
of raw input, in two stages:

  Stage 1 (FROZEN):   pretrained encoder weights frozen, train a small
                       head on top across ALL pairs' pooled training data.
  Stage 2 (UNFROZEN):  unfreeze the encoder, continue training the SAME
                       model end-to-end across ALL pairs, low LR.

Same one-model-across-all-pairs structure as train_joint.py: pairs are
OT-matched (soft/hard/hard_assignment, same as train_joint.py), pooled
into one training set per stage, and the single resulting model is
evaluated separately on each pair's held-out test set at the end --
saved as one CSV pair (y_pred/y_truth) PER timepoint pair, each with
t_src/t_tgt columns, exactly like train_joint.py's output.

Usage:
    python train_joint_pretrained_encoder.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy hvg --n_genes 100 \
        --ot_type soft --ot_reg 0.01 \
        --pretrain_checkpoint checkpoint/scTranslator_2M.pt \
        --stage1_epochs 50 --stage2_epochs 50 \
        --outdir results/joint_pretrained_encoder
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
from models import FiLMGenerator, FiLMLayer
from attention_pooling import AttentionPool
# Reuse train_joint.py's pair-building, OT, and split logic directly --
# no duplication of that logic here.
from train_joint import (
    CURRICULUM, split_cells_per_timepoint, build_pair_dataset,
    concordance_correlation_coefficient,
)

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "code", "model"))
from performer_enc_dec import scPerformerLM


class WithGeneIDs(Dataset):
    """Wraps a train_joint.PairDataset (returns x,y,t_info) to also
    inject the shared gene_ids tensor needed by the pretrained encoder."""
    def __init__(self, base_ds, gene_ids):
        self.base = base_ds
        self.gene_ids = torch.tensor(gene_ids.astype(np.int64))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, t_info = self.base[idx]
        return x, self.gene_ids, y, t_info


class PretrainedEncoderHeadMLP(nn.Module):
    """Frozen-or-not pretrained encoder -> pool over genes (mean or
    learned attention) -> plain MLP head (t_info just concatenated,
    no FiLM) -> sigmoid."""
    def __init__(self, encoder, enc_dim, n_proteins, hidden=(128, 64), t_dim=2, dropout=0.2, pooling="mean"):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AttentionPool(enc_dim)
        dims = [enc_dim + t_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.head = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], n_proteins)

    def forward(self, x, gene_ids, t_info):
        encodings = self.encoder(x, gene_ids, return_encodings=True)
        pooled = self.pool(encodings) if self.pooling == "attention" else encodings.mean(dim=1)
        h = torch.cat([pooled, t_info], dim=-1)
        h = self.head(h)
        return torch.sigmoid(self.out(h))

    def set_encoder_trainable(self, trainable: bool):
        for p in self.encoder.parameters():
            p.requires_grad = trainable


class PretrainedEncoderHeadFiLMPre(nn.Module):
    """Pretrained encoder -> pool (mean or attention) -> FiLM applied
    ONCE to the pooled embedding (t_info generates gamma/beta) ->
    plain MLP -> sigmoid."""
    def __init__(self, encoder, enc_dim, n_proteins, hidden=(128, 64), t_dim=2, dropout=0.2, pooling="mean"):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AttentionPool(enc_dim)
        self.film_in = FiLMGenerator(t_dim, enc_dim)
        dims = [enc_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.head = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], n_proteins)

    def forward(self, x, gene_ids, t_info):
        encodings = self.encoder(x, gene_ids, return_encodings=True)
        pooled = self.pool(encodings) if self.pooling == "attention" else encodings.mean(dim=1)
        gamma, beta = self.film_in(t_info)
        pooled_mod = gamma * pooled + beta
        h = self.head(pooled_mod)
        return torch.sigmoid(self.out(h))

    def set_encoder_trainable(self, trainable: bool):
        for p in self.encoder.parameters():
            p.requires_grad = trainable


class PretrainedEncoderHeadFiLMPost(nn.Module):
    """Pretrained encoder -> pool (mean or attention) -> FiLM applied
    at EVERY hidden layer of the head (matches the from-scratch
    film_post architecture, for a consistent, apples-to-apples
    comparison) -> sigmoid."""
    def __init__(self, encoder, enc_dim, n_proteins, hidden=(128, 64), t_dim=2, dropout=0.2, pooling="mean"):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AttentionPool(enc_dim)
        dims = [enc_dim] + list(hidden)
        self.layers = nn.ModuleList([
            FiLMLayer(dims[i], dims[i + 1], t_dim=t_dim, dropout=dropout)
            for i in range(len(dims) - 1)
        ])
        self.out = nn.Linear(dims[-1], n_proteins)

    def forward(self, x, gene_ids, t_info):
        encodings = self.encoder(x, gene_ids, return_encodings=True)
        pooled = self.pool(encodings) if self.pooling == "attention" else encodings.mean(dim=1)
        h = pooled
        for layer in self.layers:
            h = layer(h, t_info)
        return torch.sigmoid(self.out(h))

    def set_encoder_trainable(self, trainable: bool):
        for p in self.encoder.parameters():
            p.requires_grad = trainable


def build_pretrained_head(head_type, encoder, enc_dim, n_proteins, pooling="mean"):
    if head_type == "mlp":
        return PretrainedEncoderHeadMLP(encoder, enc_dim, n_proteins, pooling=pooling)
    elif head_type == "film_pre":
        return PretrainedEncoderHeadFiLMPre(encoder, enc_dim, n_proteins, pooling=pooling)
    elif head_type == "film_post":
        return PretrainedEncoderHeadFiLMPost(encoder, enc_dim, n_proteins, pooling=pooling)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")


def load_pretrained_encoder(checkpoint_path, dim, depth, heads, max_seq_len, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, nn.Module):
        # Checkpoint was saved as the FULL model object (torch.save(model)),
        # not a state_dict. Simplest, most reliable path: just pull its
        # encoder submodule directly rather than reconstructing weights.
        print("Checkpoint is a full model object (not a state_dict). "
              "Extracting its .enc submodule directly.")
        if not hasattr(ckpt, "enc"):
            raise RuntimeError(f"Loaded model object has no '.enc' attribute. "
                                f"Available attributes: {[a for a in dir(ckpt) if not a.startswith('_')]}")
        encoder = ckpt.enc
        return encoder.to(device)

    # Otherwise assume it's a state_dict (possibly wrapped in a dict with
    # 'model_state_dict'/'state_dict' keys) -- original expected path.
    encoder = scPerformerLM(max_seq_len=max_seq_len, dim=dim, depth=depth, heads=heads, num_tokens=1)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    enc_state = {k[len("enc."):]: v for k, v in state_dict.items() if k.startswith("enc.")}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"Loaded pretrained encoder. Missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    if len(enc_state) == 0:
        print("WARNING: no 'enc.' prefixed keys found in checkpoint -- check the "
              "checkpoint format before trusting this run.")
    return encoder.to(device)


def run_stage(model, train_loader, val_loader, device, epochs, lr, stage_name, ckpt_path):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, gene_ids, y, t_info in train_loader:
            x, gene_ids, y, t_info = x.to(device), gene_ids.to(device), y.to(device), t_info.to(device)
            optimizer.zero_grad()
            y_hat = model(x, gene_ids, t_info)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, gene_ids, y, t_info in val_loader:
                x, gene_ids, y, t_info = x.to(device), gene_ids.to(device), y.to(device), t_info.to(device)
                val_loss += criterion(model(x, gene_ids, t_info), y).item()
                n_val_batches += 1
        val_loss /= max(1, n_val_batches)

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


def evaluate_per_pair(model, test_ds_by_pair, device, protein_names, outdir, tag):
    model.eval()
    results = {}
    for (t_src, t_tgt), ds in test_ds_by_pair.items():
        if ds is None or len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        y_hat_all, y_all = [], []
        with torch.no_grad():
            for x, gene_ids, y, t_info in loader:
                x, gene_ids, t_info = x.to(device), gene_ids.to(device), t_info.to(device)
                y_hat_all.append(model(x, gene_ids, t_info).cpu().numpy())
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
    parser.add_argument("--head_type", choices=["mlp", "film_pre", "film_post"], default="mlp",
                         help="conditioning style for the head on top of the pooled "
                              "encoder embedding. 'film_post' matches your best-performing "
                              "from-scratch architecture, for a fair comparison.")
    parser.add_argument("--pooling", choices=["mean", "attention"], default="mean",
                         help="'mean': naive mean-pool over per-gene encoder embeddings "
                              "(original approach). 'attention': learned attention-weighted "
                              "pooling (attention_pooling.py), lets the model learn which "
                              "genes matter most instead of averaging all equally.")
    parser.add_argument("--enc_dim", type=int, default=128)
    parser.add_argument("--enc_depth", type=int, default=2)
    parser.add_argument("--enc_heads", type=int, default=8)
    parser.add_argument("--enc_max_seq_len", type=int, default=20000)
    parser.add_argument("--stage1_epochs", type=int, default=50, help="frozen encoder")
    parser.add_argument("--stage2_epochs", type=int, default=50, help="unfrozen fine-tune")
    parser.add_argument("--stage1_lr", type=float, default=1e-3)
    parser.add_argument("--stage2_lr", type=float, default=1e-5)
    parser.add_argument("--freeze_mode", choices=["both", "frozen_only", "unfrozen_only"],
                         default="both",
                         help="'both' (default): stage1 frozen then stage2 unfrozen. "
                              "'frozen_only': run stage1 only, encoder never unfreezes. "
                              "'unfrozen_only': skip stage1, train encoder+head "
                              "end-to-end from the start (stage1_lr used, not stage2_lr).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Config: gene_strategy={args.gene_strategy} n_genes={n_genes_arg} "
          f"ot_type={args.ot_type} ot_reg={args.ot_reg} -- JOINT training across ALL pairs, "
          f"pretrained encoder (frozen -> unfrozen)")

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
    if gene_my_ids.max() >= 85500:
        print("ERROR: a selected gene's my_Id exceeds the pretrained embedding table size. Aborting.")
        return

    train_cells, val_cells, test_cells = split_cells_per_timepoint(obs_time, seed=args.seed)
    available_pairs = [(s, t) for s, t in CURRICULUM if s in train_cells and t in train_cells]
    print(f"Available timepoint pairs: {available_pairs}")

    # --- Build datasets per pair (reuses train_joint.py's OT logic), then
    #     wrap each with gene_ids for the pretrained encoder ---
    train_ds_by_pair, val_ds_by_pair, test_ds_by_pair = {}, {}, {}
    for (t_src, t_tgt) in available_pairs:
        tr, va, te = build_pair_dataset(rna_selected, prot_full, train_cells, val_cells, test_cells,
                                         t_src, t_tgt, args.ot_type, args.ot_reg)
        if tr is not None:
            train_ds_by_pair[(t_src, t_tgt)] = WithGeneIDs(tr, gene_my_ids)
            val_ds_by_pair[(t_src, t_tgt)] = WithGeneIDs(va, gene_my_ids) if va else None
            test_ds_by_pair[(t_src, t_tgt)] = WithGeneIDs(te, gene_my_ids) if te else None

    full_train = ConcatDataset(list(train_ds_by_pair.values()))
    full_val = ConcatDataset([v for v in val_ds_by_pair.values() if v is not None])
    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(full_val, batch_size=args.batch_size, shuffle=False)

    encoder = load_pretrained_encoder(args.pretrain_checkpoint, args.enc_dim, args.enc_depth,
                                       args.enc_heads, args.enc_max_seq_len, device)
    model = build_pretrained_head(args.head_type, encoder, args.enc_dim, len(protein_names), pooling=args.pooling).to(device)

    all_results = {}

    if args.freeze_mode in ("both", "frozen_only"):
        print("\n=== STAGE 1: FROZEN encoder, training head only (ALL pairs pooled) ===")
        model.set_encoder_trainable(False)
        ckpt1 = os.path.join(args.outdir, "stage1_frozen_best.pt")
        model, val1 = run_stage(model, train_loader, val_loader, device, args.stage1_epochs,
                                 args.stage1_lr, "stage1-frozen", ckpt1)
        print("\n  Stage 1 per-pair test results:")
        all_results["stage1_frozen"] = evaluate_per_pair(model, test_ds_by_pair, device, protein_names,
                                                           args.outdir, f"stage1_frozen_{args.head_type}")
    else:
        print("\n=== SKIPPING Stage 1 (freeze_mode=unfrozen_only) ===")

    if args.freeze_mode in ("both", "unfrozen_only"):
        stage_label = "stage2-unfrozen" if args.freeze_mode == "both" else "unfrozen-from-start"
        lr = args.stage2_lr if args.freeze_mode == "both" else args.stage1_lr
        print(f"\n=== {'STAGE 2' if args.freeze_mode == 'both' else 'UNFROZEN FROM START'}: "
              f"encoder trainable, fine-tuning end-to-end (ALL pairs pooled) ===")
        model.set_encoder_trainable(True)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"  Trainable params this stage: {n_trainable:,} total, "
              f"of which {n_encoder_trainable:,} are encoder params "
              f"({'encoder IS included' if n_encoder_trainable > 0 else 'WARNING: encoder NOT included, requires_grad flags did not take effect'})")
        encoder_weights_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}

        ckpt2 = os.path.join(args.outdir, "stage2_unfrozen_best.pt")
        model, val2 = run_stage(model, train_loader, val_loader, device, args.stage2_epochs,
                                 lr, stage_label, ckpt2)

        # Verify encoder weights actually changed, not just nominally trainable
        total_change = 0.0
        for k, v_before in encoder_weights_before.items():
            v_after = model.encoder.state_dict()[k]
            total_change += (v_after - v_before).abs().sum().item()
        print(f"  Total absolute change in encoder weights after Stage 2: {total_change:.6f} "
              f"({'encoder weights DID move -- genuinely fine-tuned' if total_change > 1e-4 else 'WARNING: encoder weights barely moved -- effectively still frozen in practice, likely lr too small or gradients vanishing'})")
        print(f"\n  {stage_label} per-pair test results:")
        all_results["stage2_unfrozen"] = evaluate_per_pair(model, test_ds_by_pair, device, protein_names,
                                                             args.outdir, f"stage2_unfrozen_{args.head_type}")
    else:
        print("\n=== SKIPPING Stage 2 (freeze_mode=frozen_only) ===")

    print("\n" + "=" * 60)
    print(f"SUMMARY (freeze_mode={args.freeze_mode}), averaged across all pairs")
    print("=" * 60)
    if not all_results:
        print("  No stages ran -- check --freeze_mode.")
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