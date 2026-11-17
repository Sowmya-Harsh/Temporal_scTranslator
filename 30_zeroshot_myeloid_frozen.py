"""
30_zeroshot_myeloid_frozen.py
──────────────────────────────
Genuine ZERO-SHOT evaluation of the final QuRIE-seq checkpoint on the
external myeloid CITE-seq dataset -- NO fine-tuning, NO trained adapters,
NO gradient updates of any kind. The only thing that changes between this
script and applying the checkpoint to QuRIE-seq itself is a fixed, static
remapping of which value goes into which of the model's existing 1000
input slots and 58 output slots. Nothing is learned on myeloid data.

--- MATCHING BY GENE SYMBOL, NOT my_Id ---
Earlier versions of this script matched genes/proteins across datasets
using QuRIE-seq's 'my_Id' column, scTranslator's internal embedding-table
identifier (Section on Gene Identifier Mapping). This FAILS on myeloid:
myeloid's .h5ad has no 'my_Id' column at all -- that identifier was only
ever assigned to QuRIE-seq genes, during the scTranslator gene-mapping
step, and myeloid was never run through that mapping. myeloid's .var only
has 'gene_ids' (Ensembl IDs) and 'feature_types'; its genes are indexed
by plain gene SYMBOL in var_names (e.g. 'BTK', 'CD19').

This version matches genes and proteins by SYMBOL STRING instead, which
is more portable across datasets that were never both run through the
same internal-ID mapping pipeline:
  - RNA genes: matched directly on var_names (gene symbol) in both files.
  - Protein/antibody channels: myeloid's panel uses a 'CD-label|gene-
    symbol' format (e.g. 'CD11b|ITGAM'); QuRIE-seq's channel names are
    matched, case-insensitively, against EITHER side of that pair, with a
    leading phospho 'p-' prefix stripped from QuRIE-seq's own name where
    present (a coarse approximation for phospho channels, flagged
    explicitly in the printed output -- myeloid's CITE-seq panel does not
    measure phosphorylation at all, so any such match is a total-protein
    proxy, not a genuine phospho-specific match).

--- WHY THIS IS DIFFERENT FROM THE ADAPTER-BASED FINE-TUNED RESULT ---
The adapter-based pipeline (Section on Generalised Fine-Tuning) introduces
NEW trainable linear layers to project myeloid's own gene/protein panel
into the pretrained trunk's expected dimensionality, then fine-tunes the
whole thing (trunk + adapters) on myeloid data for 100 epochs -- real
training, real gradient updates, real exposure to myeloid's training
split.

This script does none of that. It:
  1. RECONSTRUCTS the QuRIE-seq final model's exact 1000 selected genes
     and 58 output proteins directly from the original QuRIE-seq data,
     using the SAME gene-selection procedure and seed as the original
     training run (split first, select DEG genes on train-split cells
     only -- matching train_joint_improved.py exactly, Section on Gene
     Selection).
  2. Optionally restricts the overlap check to myeloid's OWN DEG-selected
     genes (recomputed on myeloid's own timepoints, matching what the
     fine-tuned pipeline does), rather than myeloid's full raw gene panel
     -- controlled by --restrict_to_myeloid_deg (default: on), so the
     zero-shot and fine-tuned comparisons rest on a consistent notion of
     "myeloid's own informative genes". Pass --no_restrict_to_myeloid_deg
     to instead check overlap against myeloid's FULL gene panel.
  3. Runs the QuRIE-seq checkpoint on the remapped input, completely
     frozen (model.eval(), no optimizer, no backward pass at all).
  4. Scores ONLY the output proteins with a real, matched target value in
     myeloid's own panel; proteins with no match are dropped from scoring
     entirely, not scored against a zero/dummy target.
  5. Reports gene and protein overlap counts explicitly -- a low overlap
     and a low score are an expected, informative combination, not a bug.

OT matching is still used to build the "future" target for scoring, for
the same reason as everywhere else in this thesis: myeloid's own
timepoints are destructively sampled, so no true same-cell correspondence
exists.

IMPORTANT: --qurie_rna, --qurie_protein, --gene_strategy, --n_genes,
--condition, and --qurie_seed MUST exactly match the values used to train
the checkpoint in --qurie_checkpoint, or the reconstructed gene/protein
panel will not match what the model actually expects.

Usage:
    python 30_zeroshot_myeloid_frozen.py \
        --qurie_rna Final_Dataset/qurie_RNA_clean_aIg.h5ad \
        --qurie_protein Final_Dataset/qurie_Protein_clean_aIg.h5ad \
        --gene_strategy deg --n_genes 1000 \
        --qurie_seed 1105 \
        --myeloid_rna myeloid_geo_data/myeloid_RNA_standardized.h5ad \
        --myeloid_protein myeloid_geo_data/myeloid_Protein_standardized.h5ad \
        --myeloid_gene_strategy deg --myeloid_n_genes 1000 \
        --qurie_checkpoint /path/to/joint_model_best.pt \
        --ot_type soft --ot_reg 0.01 \
        --seed 5525 \
        --outdir results/zeroshot_myeloid_frozen
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(__file__))
from gene_selection import select_genes
from models import build_model
from train_joint import split_cells_per_timepoint, build_pair_dataset


def concordance_correlation_coefficient(y_true, y_pred):
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    return (2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12)


def build_gene_overlap_mask(qurie_gene_symbols, myeloid_symbol_to_col):
    """
    For each of the QuRIE-seq model's 1000 expected input gene positions
    (identified by gene SYMBOL), check whether that symbol is present in
    myeloid_symbol_to_col -- a dict mapping gene symbol -> column index
    into myeloid's expression matrix. This dict may represent either
    myeloid's full gene panel or just its own DEG-selected genes,
    depending on how the caller built it (see --restrict_to_myeloid_deg).

    Returns:
      - overlap_mask: boolean array, length 1000, True where the symbol
        is present
      - myeloid_col_for_qurie_slot: for each of the 1000 positions, the
        column index into myeloid's expression matrix to pull real values
        from (or -1 if absent, meaning that slot will be zero-filled)
    """
    n_slots = len(qurie_gene_symbols)
    overlap_mask = np.zeros(n_slots, dtype=bool)
    myeloid_col_for_qurie_slot = np.full(n_slots, -1, dtype=int)

    for slot, sym in enumerate(qurie_gene_symbols):
        if sym in myeloid_symbol_to_col:
            overlap_mask[slot] = True
            myeloid_col_for_qurie_slot[slot] = myeloid_symbol_to_col[sym]

    return overlap_mask, myeloid_col_for_qurie_slot


def _normalize_protein_name(name):
    """
    Loosely normalise a protein/antibody channel name for cross-dataset
    matching: strip a leading 'p-' phospho prefix, split on '|' (myeloid
    uses a 'CD-label|gene-symbol' format, e.g. 'CD11b|ITGAM'), uppercase
    each resulting piece. Returns a list of normalised candidate strings
    to check for a match.
    """
    name = name.strip()
    if name.lower().startswith("p-"):
        name = name[2:]
    return [p.strip().upper() for p in name.split("|") if p.strip()]


def build_protein_overlap_mask(qurie_protein_names, myeloid_prot_adata):
    """
    Match the 58 QuRIE-seq output protein channels against myeloid's
    protein panel by normalised name, since neither dataset shares a
    common numeric identifier for the protein/antibody panel.
    """
    myeloid_name_variants = {}  # normalised variant string -> column index
    for i, raw_name in enumerate(myeloid_prot_adata.var_names):
        for variant in _normalize_protein_name(raw_name):
            myeloid_name_variants[variant] = i

    n_slots = len(qurie_protein_names)
    overlap_mask = np.zeros(n_slots, dtype=bool)
    myeloid_col_for_qurie_slot = np.full(n_slots, -1, dtype=int)
    matched_via_phospho_strip = []

    for slot, raw_name in enumerate(qurie_protein_names):
        was_phospho = raw_name.strip().lower().startswith("p-")
        for variant in _normalize_protein_name(raw_name):
            if variant in myeloid_name_variants:
                overlap_mask[slot] = True
                myeloid_col_for_qurie_slot[slot] = myeloid_name_variants[variant]
                if was_phospho:
                    matched_via_phospho_strip.append(raw_name)
                break

    if matched_via_phospho_strip:
        print(f"NOTE: {len(matched_via_phospho_strip)} phospho-specific QuRIE-seq channels "
              f"matched myeloid only after stripping the 'p-' prefix, i.e. against a "
              f"TOTAL-PROTEIN channel in myeloid (myeloid's CITE-seq panel does not measure "
              f"phosphorylation at all): {matched_via_phospho_strip}. These matches are a "
              f"coarse proxy at best; interpret with this caveat if included in results.")

    return overlap_mask, myeloid_col_for_qurie_slot


def remap_rna_to_qurie_slots(myeloid_rna_matrix, myeloid_col_for_qurie_slot):
    """
    Build a [n_cells, 1000] matrix matching the QuRIE-seq model's expected
    input shape. Slots with a myeloid match get real values; slots with
    no match get zero for every cell.
    """
    n_cells = myeloid_rna_matrix.shape[0]
    n_slots = len(myeloid_col_for_qurie_slot)
    remapped = np.zeros((n_cells, n_slots), dtype=np.float32)

    for slot, myeloid_col in enumerate(myeloid_col_for_qurie_slot):
        if myeloid_col >= 0:
            remapped[:, slot] = myeloid_rna_matrix[:, myeloid_col]

    return remapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qurie_rna", required=True)
    parser.add_argument("--qurie_protein", required=True)
    parser.add_argument("--condition", default="aIg")
    parser.add_argument("--gene_strategy", choices=["hvg", "deg"], required=True,
                         help="MUST match the strategy used to train --qurie_checkpoint.")
    parser.add_argument("--n_genes", required=True,
                         help="MUST match the panel size used to train --qurie_checkpoint.")
    parser.add_argument("--qurie_seed", type=int, required=True,
                         help="MUST match the seed used to train --qurie_checkpoint.")
    parser.add_argument("--myeloid_rna", required=True)
    parser.add_argument("--myeloid_protein", required=True)
    parser.add_argument("--restrict_to_myeloid_deg", action="store_true", default=True,
                         help="(default on) Only count a gene as 'overlapping' if it is also "
                              "among myeloid's OWN DEG-selected genes, not merely present "
                              "anywhere in myeloid's raw panel. Matches the fine-tuned "
                              "pipeline's own behaviour for a consistent comparison.")
    parser.add_argument("--no_restrict_to_myeloid_deg", dest="restrict_to_myeloid_deg",
                         action="store_false",
                         help="Check overlap against myeloid's FULL raw gene panel instead.")
    parser.add_argument("--myeloid_gene_strategy", choices=["hvg", "deg"], default="deg")
    parser.add_argument("--myeloid_n_genes", default="1000")
    parser.add_argument("--qurie_checkpoint", required=True)
    parser.add_argument("--model", default="film_pre",
                         choices=["mlp", "film_pre", "film_post", "gated", "gated_post"])
    parser.add_argument("--ot_type", choices=["soft", "hard", "hard_assignment"], default="soft")
    parser.add_argument("--ot_reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=5525,
                         help="Seed for myeloid's train/val/test split, gene selection, and OT matching.")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    n_genes_arg = args.n_genes if args.n_genes == "all" else int(args.n_genes)

    # --- Reconstruct the EXACT gene/protein panel the checkpoint was
    #     trained on, directly from the original QuRIE-seq data, matching
    #     train_joint_improved.py's split-then-select order exactly. ---
    print("Reconstructing the QuRIE-seq gene/protein panel the checkpoint was trained on...")
    qurie_rna_adata = ad.read_h5ad(args.qurie_rna)
    qurie_prot_adata = ad.read_h5ad(args.qurie_protein)
    mask = qurie_rna_adata.obs["condition"] == args.condition
    qurie_rna_adata = qurie_rna_adata[mask].copy()
    qurie_prot_adata = qurie_prot_adata[mask].copy()

    qurie_obs_time = qurie_rna_adata.obs["time_min"].values.astype(float)
    train_cells_qurie, _, _ = split_cells_per_timepoint(qurie_obs_time, seed=args.qurie_seed)
    train_idx_all = np.concatenate([idx for idx in train_cells_qurie.values() if len(idx) > 0])
    qurie_rna_train = qurie_rna_adata[train_idx_all].copy()
    gene_idx = select_genes(qurie_rna_train, args.gene_strategy, n_genes_arg, groupby="time_min")
    print(f"Reconstructed {len(gene_idx)} genes via {args.gene_strategy} "
          f"(train-split cells only: {len(train_idx_all)}/{qurie_rna_adata.n_obs}).")

    # Matched by SYMBOL, not my_Id -- myeloid has no my_Id column.
    qurie_gene_symbols = qurie_rna_adata.var_names[gene_idx].tolist()
    qurie_protein_names = qurie_prot_adata.var_names.tolist()
    n_genes = len(qurie_gene_symbols)
    n_proteins = len(qurie_protein_names)
    print(f"QuRIE-seq checkpoint expects {n_genes} input genes, {n_proteins} output proteins.")

    with open(os.path.join(args.outdir, "reconstructed_qurie_gene_symbols.json"), "w") as f:
        json.dump(qurie_gene_symbols, f)
    with open(os.path.join(args.outdir, "reconstructed_qurie_protein_names.json"), "w") as f:
        json.dump(qurie_protein_names, f)

    myeloid_rna = ad.read_h5ad(args.myeloid_rna)
    myeloid_prot = ad.read_h5ad(args.myeloid_protein)

    myeloid_obs_time = myeloid_rna.obs["time_min"].values.astype(float)
    train_cells_myeloid, val_cells, test_cells = split_cells_per_timepoint(myeloid_obs_time, seed=args.seed)

    if args.restrict_to_myeloid_deg:
        # --- Select myeloid's OWN informative genes (matching what the
        #     fine-tuned pipeline does), and restrict the overlap check
        #     to that set rather than myeloid's full raw panel. ---
        myeloid_n_genes_arg = args.myeloid_n_genes if args.myeloid_n_genes == "all" else int(args.myeloid_n_genes)
        myeloid_train_idx_all = np.concatenate([idx for idx in train_cells_myeloid.values() if len(idx) > 0])
        myeloid_rna_train = myeloid_rna[myeloid_train_idx_all].copy()
        myeloid_selected_gene_idx = select_genes(myeloid_rna_train, args.myeloid_gene_strategy,
                                                  myeloid_n_genes_arg, groupby="time_min")
        print(f"Selected {len(myeloid_selected_gene_idx)} genes on myeloid's OWN data via "
              f"{args.myeloid_gene_strategy} (train-split cells only: "
              f"{len(myeloid_train_idx_all)}/{myeloid_rna.n_obs}).")
        myeloid_symbol_to_col = {
            myeloid_rna.var_names[i]: i for i in myeloid_selected_gene_idx
        }
        overlap_pool_desc = (f"myeloid's own {len(myeloid_selected_gene_idx)} "
                              f"{args.myeloid_gene_strategy.upper()}-selected genes")
    else:
        myeloid_symbol_to_col = {sym: i for i, sym in enumerate(myeloid_rna.var_names)}
        overlap_pool_desc = f"myeloid's full raw panel ({myeloid_rna.n_vars} genes)"

    gene_overlap_mask, gene_slot_to_myeloid_col = build_gene_overlap_mask(
        qurie_gene_symbols, myeloid_symbol_to_col)
    protein_overlap_mask, protein_slot_to_myeloid_col = build_protein_overlap_mask(
        qurie_protein_names, myeloid_prot)

    n_gene_overlap = int(gene_overlap_mask.sum())
    n_protein_overlap = int(protein_overlap_mask.sum())
    print(f"Gene overlap: {n_gene_overlap}/{n_genes} "
          f"({100*n_gene_overlap/n_genes:.1f}%) of QuRIE-seq's input genes are present in "
          f"{overlap_pool_desc}.")
    print(f"Protein overlap: {n_protein_overlap}/{n_proteins} "
          f"({100*n_protein_overlap/n_proteins:.1f}%) of QuRIE-seq's output proteins "
          f"have a real, name-matched target in myeloid.")

    if n_protein_overlap == 0:
        print("ERROR: zero protein overlap -- there is nothing to score against. Aborting.")
        return
    if n_gene_overlap < n_genes * 0.05:
        print(f"WARNING: gene overlap is very low ({n_gene_overlap}/{n_genes}). "
              f"A poor zero-shot result here is expected and should be interpreted as "
              f"'the model had almost no real input signal to work with', not necessarily "
              f"'the pretrained representation does not transfer'.")

    # --- Remap myeloid RNA into the model's fixed 1000-slot input shape ---
    myeloid_rna_full = myeloid_rna.layers["counts"] if "counts" in myeloid_rna.layers else myeloid_rna.X
    myeloid_rna_full = (myeloid_rna_full.toarray() if hasattr(myeloid_rna_full, "toarray")
                         else myeloid_rna_full).astype(np.float32)
    remapped_rna = remap_rna_to_qurie_slots(myeloid_rna_full, gene_slot_to_myeloid_col)

    myeloid_prot_full = myeloid_prot.layers["counts"] if "counts" in myeloid_prot.layers else myeloid_prot.X
    myeloid_prot_full = (myeloid_prot_full.toarray() if hasattr(myeloid_prot_full, "toarray")
                          else myeloid_prot_full).astype(np.float32)

    timepoints = sorted(train_cells_myeloid.keys())
    pairs = [(t_src, t_tgt) for t_src in timepoints for t_tgt in timepoints if t_src < t_tgt]
    print(f"Myeloid timepoint pairs: {pairs}")

    # --- Load the FROZEN QuRIE-seq checkpoint. No optimizer is created,
    #     no parameter ever receives a gradient update in this script. ---
    model = build_model(args.model, n_genes=n_genes, n_proteins=n_proteins).to(device)
    ckpt = torch.load(args.qurie_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded FROZEN checkpoint (epoch={ckpt.get('epoch', '?')}). "
          f"No fine-tuning, no adapters, no gradient updates in this script.")

    results = {}
    for (t_src, t_tgt) in pairs:
        _, _, te = build_pair_dataset(remapped_rna, myeloid_prot_full,
                                       train_cells_myeloid, val_cells, test_cells,
                                       t_src, t_tgt, args.ot_type, args.ot_reg)
        if te is None or len(te) == 0:
            continue

        from torch.utils.data import DataLoader
        loader = DataLoader(te, batch_size=len(te), shuffle=False)
        x_all, y_all, t_info_all = next(iter(loader))
        x_all, t_info_all = x_all.to(device), t_info_all.to(device)
        with torch.no_grad():
            y_hat = model(x_all, t_info_all).cpu().numpy()
        y_true = y_all.numpy()

        per_r, per_ccc = [], []
        for slot in range(n_proteins):
            if not protein_overlap_mask[slot]:
                continue
            myeloid_col = protein_slot_to_myeloid_col[slot]
            yt, yp = y_true[:, myeloid_col], y_hat[:, slot]
            if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
                r, _ = pearsonr(yt, yp)
                if not np.isnan(r):
                    per_r.append(r)
                    per_ccc.append(concordance_correlation_coefficient(yt, yp))

        mean_r = float(np.mean(per_r)) if per_r else float("nan")
        mean_ccc = float(np.mean(per_ccc)) if per_ccc else float("nan")
        results[f"{int(t_src)}->{int(t_tgt)}"] = {
            "n_test": len(te), "mean_r": mean_r, "mean_ccc": mean_ccc,
            "n_proteins_scored": len(per_r),
        }
        print(f"    t{int(t_src)}->t{int(t_tgt)}: mean_r={mean_r:.4f} mean_ccc={mean_ccc:.4f} "
              f"(n={len(te)} cells, {len(per_r)}/{n_protein_overlap} scorable proteins)")

    all_r = [v["mean_r"] for v in results.values() if not np.isnan(v["mean_r"])]
    all_ccc = [v["mean_ccc"] for v in results.values() if not np.isnan(v["mean_ccc"])]
    overall_r = float(np.mean(all_r)) if all_r else float("nan")
    overall_ccc = float(np.mean(all_ccc)) if all_ccc else float("nan")

    print(f"\n{'='*60}")
    print("ZERO-SHOT (FROZEN CHECKPOINT, NO FINE-TUNING) SUMMARY")
    print(f"{'='*60}")
    print(f"Gene overlap:    {n_gene_overlap}/{n_genes} ({100*n_gene_overlap/n_genes:.1f}%)")
    print(f"Protein overlap: {n_protein_overlap}/{n_proteins} ({100*n_protein_overlap/n_proteins:.1f}%)")
    print(f"Overall mean R:   {overall_r:.4f}")
    print(f"Overall mean CCC: {overall_ccc:.4f}")
    print(f"\nCompare against the FINE-TUNED result (adapters, 100 epochs, "
          f"Section on external generalisation): R=0.579, CCC=0.003.")

    with open(os.path.join(args.outdir, "zeroshot_myeloid_results.json"), "w") as f:
        json.dump({
            "config": vars(args),
            "n_gene_overlap": n_gene_overlap, "n_genes_total": n_genes,
            "n_protein_overlap": n_protein_overlap, "n_proteins_total": n_proteins,
            "overall_mean_r": overall_r, "overall_mean_ccc": overall_ccc,
            "per_pair": results,
        }, f, indent=2)
    print(f"\nSaved to {args.outdir}/zeroshot_myeloid_results.json")


if __name__ == "__main__":
    main()