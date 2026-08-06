"""
ot_matching.py
───────────────
HARD optimal transport matching — complements the SOFT (Sinkhorn,
averaging) OT already implemented in qurie_dataset_ot.py's
compute_ot_targets().

Key difference:
  - SOFT OT (existing compute_ot_targets): each source cell's target
    is a WEIGHTED AVERAGE over all target cells, weighted by the
    normalized transport plan row. Smooth, but can blur/average away
    real single-cell heterogeneity.
  - HARD OT (this module): each source cell is assigned to exactly
    ONE target cell (via exact/unregularized OT + row-argmax, or a
    true 1-to-1 assignment when n_src == n_tgt via linear_sum_assignment
    on the exact OT cost). No averaging — the target protein vector
    IS that one matched cell's real measured protein vector.

Two hard variants provided:
  1. ot_hard_targets_exact()  — uses POT's exact EMD solver (ot.emd),
     then argmax per row. Works for any n_src, n_tgt (unbalanced OK).
  2. ot_hard_targets_assignment() — true 1-to-1 bijective assignment
     via scipy linear_sum_assignment on the cost matrix. Requires
     n_src == n_tgt (pads/truncates otherwise — see docstring). Gives
     the strictest possible "hard" matching: every target used exactly
     once (when balanced).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.optimize import linear_sum_assignment

try:
    import ot
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    print("WARNING: POT not installed. Hard OT matching unavailable.")


def _normalize(x, eps=1e-8):
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        return np.full_like(x, 1e-8, dtype=np.float32)
    return (1e-8 + (x - xmin) / (xmax - xmin + eps)).astype(np.float32)


def _select_top_var_genes(rna_src, rna_tgt, max_genes):
    gene_var = rna_src.var(axis=0) + rna_tgt.var(axis=0)
    return np.argsort(gene_var)[-max_genes:]


def ot_hard_targets_exact(rna_src, rna_tgt, prot_tgt, max_genes=2000):
    """
    Exact (unregularized) OT via ot.emd, then hard argmax per source
    cell -> assigns each source cell to its single highest-weight
    target cell. No averaging: returned target vector is the REAL
    protein vector of the matched target cell.

    Works for any n_src, n_tgt (does not require them to be equal).
    Some target cells may be matched multiple times, some never
    (this mirrors real transport mass distribution) — inspect the
    printed match statistics to see the distribution.
    """
    if not OT_AVAILABLE:
        mean = _normalize(prot_tgt.mean(axis=0))
        return np.tile(mean, (len(rna_src), 1)), None

    n_src, n_tgt = len(rna_src), len(rna_tgt)
    top_genes = _select_top_var_genes(rna_src, rna_tgt, max_genes)

    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)
    a = np.ones(n_src, dtype=np.float64) / n_src
    b = np.ones(n_tgt, dtype=np.float64) / n_tgt

    T = ot.emd(a, b, C)  # exact OT plan, sparser than sinkhorn
    match_idx = T.argmax(axis=1)  # hard assignment: 1 target per source

    targets = prot_tgt[match_idx].astype(np.float32)
    targets = np.array([_normalize(p) for p in targets])

    match_counts = np.bincount(match_idx, minlength=n_tgt)
    print(f"  OT-HARD(exact): {n_src} src -> {n_tgt} tgt | "
          f"targets matched >=1x: {(match_counts > 0).sum()}/{n_tgt} "
          f"({100*(match_counts > 0).mean():.1f}%) | "
          f"most-matched target used {match_counts.max()}x")

    return targets, match_idx


def ot_hard_targets_assignment(rna_src, rna_tgt, prot_tgt, max_genes=2000):
    """
    True 1-to-1 bijective hard assignment via the Hungarian algorithm
    (scipy linear_sum_assignment) on the cosine cost matrix. Every
    matched target is used EXACTLY once — strictest possible "hard"
    matching, no reuse, no averaging.

    Requires n_src == n_tgt for a clean bijection. If they differ,
    matches min(n_src, n_tgt) pairs (the assignment algorithm will
    pick the globally lowest-cost subset of that size); leftover
    cells on the larger side are UNMATCHED and excluded from the
    returned arrays — check the returned `matched_src_idx` to know
    which source cells got a target.
    """
    if not OT_AVAILABLE:
        raise RuntimeError("POT not installed; needed for cost matrix consistency "
                            "with the soft/exact variants even though this function "
                            "only uses scipy for the actual assignment.")

    n_src, n_tgt = len(rna_src), len(rna_tgt)
    top_genes = _select_top_var_genes(rna_src, rna_tgt, max_genes)
    C = cosine_distances(rna_src[:, top_genes], rna_tgt[:, top_genes]).astype(np.float64)

    row_idx, col_idx = linear_sum_assignment(C)  # solves min(n_src,n_tgt) matches

    targets = prot_tgt[col_idx].astype(np.float32)
    targets = np.array([_normalize(p) for p in targets])

    n_matched = len(row_idx)
    print(f"  OT-HARD(assignment/bijective): {n_src} src, {n_tgt} tgt | "
          f"{n_matched} 1-to-1 matches made "
          f"({'balanced, all cells matched' if n_src == n_tgt else 'unbalanced, some cells excluded'})")

    return targets, row_idx, col_idx


def compute_ot_hard(rna_src, rna_tgt, prot_tgt, mode="exact", max_genes=2000):
    """
    Dispatcher matching the calling convention of compute_ot_targets()
    in qurie_dataset_ot.py, for easy drop-in swapping soft<->hard.

    mode: 'exact' (default, any n_src/n_tgt) or 'assignment' (strict
          1-to-1 bijection, best when n_src == n_tgt).

    Returns just the targets array (same shape/contract as
    compute_ot_targets), for source cells in their original order.
    If mode='assignment' and n_src != n_tgt, only matched_src_idx
    positions are filled; use ot_hard_targets_assignment directly if
    you need to know which source cells were dropped.
    """
    if mode == "exact":
        targets, _ = ot_hard_targets_exact(rna_src, rna_tgt, prot_tgt, max_genes)
        return targets
    elif mode == "assignment":
        targets, row_idx, _ = ot_hard_targets_assignment(rna_src, rna_tgt, prot_tgt, max_genes)
        if len(row_idx) != len(rna_src):
            print(f"  NOTE: assignment mode matched only {len(row_idx)}/{len(rna_src)} "
                  f"source cells (unbalanced n_src/n_tgt). Returned array is "
                  f"reordered to matched source cells only, length {len(row_idx)} "
                  f"-- NOT aligned 1:1 with the original rna_src order/length. "
                  f"Use ot_hard_targets_assignment() directly and index with "
                  f"row_idx if you need alignment.")
        return targets
    else:
        raise ValueError(f"Unknown hard OT mode: {mode}")