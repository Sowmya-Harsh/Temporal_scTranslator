"""
true_kfold_cv.py
─────────────────
Genuine 5-fold cross-validation for the joint (all-pairs) training
pipeline.

This version implements standard 5-fold cross-validation:

  1. Each timepoint's cells are partitioned ONCE into 5 disjoint folds
     using a single seeded shuffle + np.array_split.

  2. The 5 folds are rotated across 5 runs. In each run:
       - 4 folds (80%) are used for training
       - 1 fold (20%) is held out for testing

     Every cell therefore serves as a held-out test observation
     exactly once across the 5 rotations.

  3. There is NO separate validation fold.

  4. Gene selection (DEG/HVG) is recomputed independently in every
     fold using ONLY that fold's 80% training cells. This prevents
     gene-selection leakage.

  5. OT matching is constructed independently for each fold using
     the corresponding train/test partition.

  6. Each fold trains a freshly initialized model for the predetermined
     number of epochs. Since there is no validation set, no checkpoint
     is selected using held-out performance. The final epoch model is
     evaluated on that fold's held-out test set.

  7. Per-pair R/CCC/MSE are reported for each held-out fold and then
     aggregated across the 5 folds as mean +/- standard deviation.

This is standard 5-fold CV:
    Fold 1: 80% train / 20% test
    Fold 2: 80% train / 20% test
    Fold 3: 80% train / 20% test
    Fold 4: 80% train / 20% test
    Fold 5: 80% train / 20% test

The zero-shot ibrutinib evaluation remains a completely separate
downstream evaluation and is NOT part of this cross-validation.

Usage:
    python true_kfold_cv.py \
        --rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --ot_type soft --ot_reg 0.01 \
        --model film_pre \
        --correlation_loss_weight 0.3 --ccc_loss_weight 0.3 \
        --k_folds 5 --epochs 100 \
        --outdir results/true_kfold_cv
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


def build_kfold_partition(obs_time, k=5, seed=1105):
    """
    Partition each timepoint's cells into k disjoint folds ONCE.

    A single seeded shuffle is performed independently for each
    timepoint, followed by np.array_split.

    Therefore:
      - every cell belongs to exactly one fold
      - folds do not overlap
      - there are no missing cells
      - fold sizes differ by at most one cell
    """
    rng = np.random.default_rng(seed)

    timepoints = sorted(np.unique(obs_time).tolist())
    folds_per_timepoint = {}

    for t in timepoints:
        cells = np.where(
            np.isclose(obs_time, t, atol=1e-3)
        )[0].copy()

        rng.shuffle(cells)

        folds_per_timepoint[t] = [
            f.copy()
            for f in np.array_split(cells, k)
        ]

    return timepoints, folds_per_timepoint


def get_fold_split(
    folds_per_timepoint,
    timepoints,
    fold_idx,
    k=5
):
    """
    Standard k-fold split.

    fold_idx is the held-out test fold.
    All remaining k-1 folds are pooled into training.

    For k=5:
        training = 4/5 = 80%
        test     = 1/5 = 20%

    Returns:
        train_cells
        test_cells
    """
    train_cells = {}
    test_cells = {}

    for t in timepoints:

        folds = folds_per_timepoint[t]

        # Current fold is held out for testing
        test_cells[t] = folds[fold_idx]

        # All other folds are training
        train_idx_list = [
            i for i in range(k)
            if i != fold_idx
        ]

        train_cells[t] = np.concatenate(
            [folds[i] for i in train_idx_list]
        )

    return train_cells, test_cells


def run_one_fold(
    args,
    rna_adata,
    rna_full,
    prot_full,
    protein_names,
    obs_time,
    timepoints,
    folds_per_timepoint,
    fold_idx,
    fold_outdir,
    device
):
    """
    Train and evaluate one 80/20 CV rotation.
    """

    os.makedirs(fold_outdir, exist_ok=True)

    # Reproducible model initialization/training for each fold.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    n_genes_arg = (
        args.n_genes
        if args.n_genes == "all"
        else int(args.n_genes)
    )

    # ---------------------------------------------------------
    # 80/20 split
    # ---------------------------------------------------------
    train_cells, test_cells = get_fold_split(
        folds_per_timepoint,
        timepoints,
        fold_idx,
        k=args.k_folds
    )

    # ---------------------------------------------------------
    # Gene selection:
    # ONLY the 80% training cells
    # ---------------------------------------------------------
    train_idx_all = np.concatenate(
        [
            idx
            for idx in train_cells.values()
            if len(idx) > 0
        ]
    )

    rna_adata_train = rna_adata[
        train_idx_all
    ].copy()

    gene_idx = select_genes(
        rna_adata_train,
        args.gene_strategy,
        n_genes_arg,
        groupby="time_min"
    )

    rna_selected = rna_full[:, gene_idx]

    print(
        f"  Fold {fold_idx + 1}: selected "
        f"{len(gene_idx)} genes "
        f"(training cells only: "
        f"{len(train_idx_all)}/{rna_adata.n_obs})"
    )

    # ---------------------------------------------------------
    # Available timepoint pairs
    # ---------------------------------------------------------
    available_pairs = [
        (s, t)
        for s, t in tj.CURRICULUM
        if s in train_cells and t in train_cells
    ]

    train_ds_by_pair = {}
    test_ds_by_pair = {}

    # ---------------------------------------------------------
    # Build train/test datasets
    # ---------------------------------------------------------
    for (t_src, t_tgt) in available_pairs:

        tr, _, te = tj.build_pair_dataset(
            rna_selected,
            prot_full,
            train_cells,
            None,          # no validation cells
            test_cells,
            t_src,
            t_tgt,
            args.ot_type,
            args.ot_reg,
            seed=args.seed
        )

        if tr is not None:

            train_ds_by_pair[
                (t_src, t_tgt)
            ] = tr

            test_ds_by_pair[
                (t_src, t_tgt)
            ] = te

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = build_model(
        args.model,
        n_genes=len(gene_idx),
        n_proteins=len(protein_names)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    criterion = torch.nn.MSELoss()

    # ---------------------------------------------------------
    # Training dataset
    # ---------------------------------------------------------
    full_train = ConcatDataset(
        list(train_ds_by_pair.values())
    )

    train_loader = DataLoader(
        full_train,
        batch_size=args.batch_size,
        shuffle=True
    )

    # ---------------------------------------------------------
    # Training
    #
    # No validation set.
    # No early stopping.
    # No checkpoint selected using test performance.
    #
    # The number of epochs is predetermined by --epochs.
    # ---------------------------------------------------------
    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        for x, y, t_info in train_loader:

            x = x.to(device)
            y = y.to(device)
            t_info = t_info.to(device)

            optimizer.zero_grad()

            y_hat = model(
                x,
                t_info
            )

            loss = criterion(
                y_hat,
                y
            )

            if args.correlation_loss_weight > 0:

                loss = (
                    loss
                    + args.correlation_loss_weight
                    * correlation_loss(
                        y_hat,
                        y
                    )
                )

            if args.ccc_loss_weight > 0:

                loss = (
                    loss
                    + args.ccc_loss_weight
                    * ccc_loss(
                        y_hat,
                        y
                    )
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

        if (
            epoch % 20 == 0
            or epoch == 1
            or epoch == args.epochs
        ):

            print(
                f"    Fold {fold_idx + 1} "
                f"epoch {epoch:>4}/{args.epochs}"
            )

    # ---------------------------------------------------------
    # Evaluate ONLY on the held-out 20% test fold
    # ---------------------------------------------------------
    test_results = tj.evaluate_per_pair(
        model,
        test_ds_by_pair,
        device,
        protein_names,
        outdir=None
    )

    # Save final model for this CV fold.
    # This is NOT selected using test performance.
    ckpt_path = os.path.join(
        fold_outdir,
        "model_final.pt"
    )

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "fold_idx": fold_idx,
            "n_genes": len(gene_idx),
            "gene_idx": gene_idx.tolist()
            if hasattr(gene_idx, "tolist")
            else list(gene_idx)
        },
        ckpt_path
    )

    return (
        test_results,
        len(gene_idx)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rna",
        required=True
    )

    parser.add_argument(
        "--protein",
        required=True
    )

    parser.add_argument(
        "--condition",
        default="aIg"
    )

    parser.add_argument(
        "--gene_strategy",
        choices=["hvg", "deg"],
        required=True
    )

    parser.add_argument(
        "--n_genes",
        required=True
    )

    parser.add_argument(
        "--ot_type",
        choices=[
            "soft",
            "hard",
            "hard_assignment"
        ],
        required=True
    )

    parser.add_argument(
        "--ot_reg",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--model",
        choices=[
            "mlp",
            "film_pre",
            "film_post",
            "gated",
            "gated_post"
        ],
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--correlation_loss_weight",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--ccc_loss_weight",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--k_folds",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1105,
        help=(
            "Fixed seed used for the one-time fold partition "
            "and model initialization. The same partition is "
            "reused across all rotations."
        )
    )

    parser.add_argument(
        "--outdir",
        required=True
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    print(
        f"True {args.k_folds}-fold CV: "
        f"80% train / 20% test"
    )

    print(
        f"gene_strategy={args.gene_strategy} "
        f"n_genes={args.n_genes} "
        f"ot_type={args.ot_type} "
        f"model={args.model} "
        f"correlation_loss_weight="
        f"{args.correlation_loss_weight} "
        f"ccc_loss_weight="
        f"{args.ccc_loss_weight}"
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    rna_adata = ad.read_h5ad(
        args.rna
    )

    prot_adata = ad.read_h5ad(
        args.protein
    )

    mask = (
        rna_adata.obs["condition"]
        == args.condition
    )

    rna_adata = rna_adata[
        mask
    ].copy()

    prot_adata = prot_adata[
        mask
    ].copy()

    rna_full = rna_adata.layers[
        "counts"
    ]

    prot_full = prot_adata.layers[
        "counts"
    ]

    rna_full = (
        rna_full.toarray()
        if hasattr(
            rna_full,
            "toarray"
        )
        else rna_full
    ).astype(np.float32)

    prot_full = (
        prot_full.toarray()
        if hasattr(
            prot_full,
            "toarray"
        )
        else prot_full
    ).astype(np.float32)

    protein_names = list(
        prot_adata.var_names
    )

    obs_time = (
        rna_adata.obs["time_min"]
        .values
        .astype(float)
    )

    # ---------------------------------------------------------
    # Build the fixed 5-fold partition ONCE
    # ---------------------------------------------------------
    timepoints, folds_per_timepoint = (
        build_kfold_partition(
            obs_time,
            k=args.k_folds,
            seed=args.seed
        )
    )

    print(
        "\nFixed fold partition:"
    )

    for t in timepoints:

        sizes = [
            len(f)
            for f in folds_per_timepoint[t]
        ]

        total = int(
            np.sum(
                np.isclose(
                    obs_time,
                    t,
                    atol=1e-3
                )
            )
        )

        print(
            f"  t={t}: "
            f"{total} cells -> "
            f"fold sizes {sizes}"
        )

    # ---------------------------------------------------------
    # Run all 5 CV rotations
    # ---------------------------------------------------------
    all_test_results = []

    for fold_idx in range(
        args.k_folds
    ):

        test_fold = fold_idx
        train_folds = [
            i
            for i in range(args.k_folds)
            if i != test_fold
        ]

        print(
            f"\n{'=' * 60}\n"
            f"FOLD {fold_idx + 1}/"
            f"{args.k_folds} "
            f"(train folds="
            f"{train_folds}, "
            f"test fold="
            f"{test_fold})\n"
            f"{'=' * 60}"
        )

        fold_outdir = os.path.join(
            args.outdir,
            f"fold_{fold_idx + 1}"
        )

        test_res, n_genes_actual = (
            run_one_fold(
                args,
                rna_adata,
                rna_full,
                prot_full,
                protein_names,
                obs_time,
                timepoints,
                folds_per_timepoint,
                fold_idx,
                fold_outdir,
                device
            )
        )

        all_test_results.append(
            test_res
        )

    # ---------------------------------------------------------
    # Aggregate results across held-out folds
    # ---------------------------------------------------------
    def aggregate(
        all_results
    ):

        all_pairs = set()

        for res in all_results:
            all_pairs.update(
                res.keys()
            )

        rows = []

        for pair in sorted(
            all_pairs
        ):

            rs = [
                res[pair][
                    "mean_pearson_r"
                ]
                for res in all_results
                if pair in res
            ]

            cccs = [
                res[pair]["ccc"]
                for res in all_results
                if pair in res
            ]

            mses = [
                res[pair]["mse"]
                for res in all_results
                if pair in res
            ]

            rows.append(
                {
                    "t_src": pair[0],
                    "t_tgt": pair[1],
                    "n_folds": len(rs),

                    "mean_pearson_r_avg":
                        np.nanmean(rs),

                    "mean_pearson_r_std":
                        np.nanstd(rs),

                    "ccc_avg":
                        np.nanmean(cccs),

                    "ccc_std":
                        np.nanstd(cccs),

                    "mse_avg":
                        np.nanmean(mses),

                    "mse_std":
                        np.nanstd(mses),
                }
            )

        return rows

    rows = aggregate(
        all_test_results
    )

    df = pd.DataFrame(
        rows
    )

    out_path = os.path.join(
        args.outdir,
        "true_kfold_summary.csv"
    )

    df.to_csv(
        out_path,
        index=False
    )

    print(
        f"\n{'=' * 60}\n"
        f"TRUE {args.k_folds}-FOLD CV SUMMARY\n"
        f"80% TRAIN / 20% HELD-OUT TEST\n"
        f"mean +/- std across folds\n"
        f"{'=' * 60}"
    )

    print(
        df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to {out_path}"
    )

    # ---------------------------------------------------------
    # Overall pooled scalar
    # ---------------------------------------------------------
    r_all = [
        r["mean_pearson_r"]
        for res in all_test_results
        for r in res.values()
    ]

    ccc_all = [
        r["ccc"]
        for res in all_test_results
        for r in res.values()
    ]

    mse_all = [
        r["mse"]
        for res in all_test_results
        for r in res.values()
    ]

    print(
        "\nOverall CV performance:"
    )

    print(
        f"  Mean R   = "
        f"{np.nanmean(r_all):.4f}"
    )

    print(
        f"  Mean CCC = "
        f"{np.nanmean(ccc_all):.4f}"
    )

    print(
        f"  Mean MSE = "
        f"{np.nanmean(mse_all):.4f}"
    )

    # ---------------------------------------------------------
    # Save configuration
    # ---------------------------------------------------------
    with open(
        os.path.join(
            args.outdir,
            "true_kfold_config.json"
        ),
        "w"
    ) as f:

        json.dump(
            {
                "config": vars(args),
                "cv_type": "standard_5_fold",
                "train_fraction": 0.80,
                "test_fraction": 0.20,
                "has_validation_split": False
            },
            f,
            indent=2
        )


if __name__ == "__main__":
    main()