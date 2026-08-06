"""
gene_selection.py
──────────────────
Two gene-selection strategies, each returning gene INDICES (into the
original var order) for a requested top-N size (or 'all'):

  1. get_top_hvg(adata, n)            -> unsupervised, variance-based
  2. get_top_deg(adata, groupby, n)   -> supervised, group-contrast-based
     (groupby='time_min' by default -> DEGs across all timepoints,
     using scanpy's rank_genes_groups with a 'rest' comparison per
     timepoint, then merging by best rank across groups)

Both cache results so repeated calls at different N don't
recompute the underlying scanpy call.

Usage:
    from gene_selection import get_top_hvg, get_top_deg
    idx_hvg_500 = get_top_hvg(rna_adata, 500)
    idx_deg_500 = get_top_deg(rna_adata, 500, groupby='time_min')
"""

import numpy as np
import scanpy as sc

_HVG_CACHE = {}
_DEG_CACHE = {}


def _cache_key(adata):
    # cheap identity key: object id + shape (adata isn't hashable)
    return (id(adata), adata.shape)


def get_top_hvg(adata, n, flavor="seurat"):
    """
    Returns gene indices (into adata.var_names order) for the top-N
    highly variable genes, computed across ALL cells in `adata`
    (i.e. across all timepoints/conditions currently in this AnnData —
    subset before calling if you want HVGs within one condition only).

    n: int, or the string 'all' to return every gene index.
    """
    if n == "all":
        return np.arange(adata.n_vars)

    key = _cache_key(adata)
    if key not in _HVG_CACHE:
        tmp = adata.copy()
        # scanpy's HVG flavor='seurat' checks adata.uns['log1p']['base'] if
        # 'log1p' is present in uns -- but depending on scanpy version, our
        # earlier sc.pp.log1p() call (in 03_clean_qurie_data.py) may have
        # left uns['log1p'] present without a 'base' key, causing a KeyError.
        # Since we used sc.pp.log1p with default settings (natural log,
        # base=None=e), explicitly set this so scanpy's internal check passes.
        if "log1p" in tmp.uns_keys() and "base" not in tmp.uns["log1p"]:
            tmp.uns["log1p"]["base"] = None
        # scanpy's HVG flavor='seurat' expects log-data; our cleaning
        # pipeline already log1p'd RNA, so this is safe as-is.
        sc.pp.highly_variable_genes(tmp, flavor=flavor, n_top_genes=None)
        # dispersions_norm gives a continuous ranking regardless of
        # the internal cutoff scanpy used for the boolean flag
        disp = tmp.var["dispersions_norm"].values
        order = np.argsort(disp)[::-1]  # descending
        _HVG_CACHE[key] = order
        print(f"[gene_selection] Computed HVG ranking over {adata.n_vars} genes "
              f"(cached for this AnnData object).")

    order = _HVG_CACHE[key]
    n = min(n, len(order))
    return order[:n]


def get_top_deg(adata, n, groupby="time_min", method="wilcoxon"):
    """
    Returns gene indices for the top-N differentially expressed genes
    ACROSS the groups in `adata.obs[groupby]` (default: time_min).

    Ranking strategy: for each group, scanpy's rank_genes_groups gives
    a per-gene rank (1 = most DE) in a one-vs-rest test. We take, for
    each gene, its BEST (lowest) rank across all groups — i.e. a gene
    that is strongly DE for at least one timepoint vs the rest ranks
    highly overall. This surfaces genes that mark ANY timepoint
    transition, not just genes that are broadly different everywhere.

    n: int, or the string 'all' to return every gene index.
    """
    if n == "all":
        return np.arange(adata.n_vars)

    key = (_cache_key(adata), groupby, method)
    if key not in _DEG_CACHE:
        tmp = adata.copy()
        tmp.obs[groupby] = tmp.obs[groupby].astype(str).astype("category")
        # Same fix as get_top_hvg above: rank_genes_groups also checks
        # adata.uns['log1p']['base'], which can be missing after our
        # earlier sc.pp.log1p() call depending on scanpy version.
        if "log1p" in tmp.uns_keys() and "base" not in tmp.uns["log1p"]:
            tmp.uns["log1p"]["base"] = None
        sc.tl.rank_genes_groups(tmp, groupby=groupby, method=method)

        n_genes = tmp.n_vars
        gene_names = np.array(tmp.var_names)
        best_rank = np.full(n_genes, np.inf)
        name_to_idx = {g: i for i, g in enumerate(gene_names)}

        groups = tmp.obs[groupby].cat.categories
        for grp in groups:
            ranked_names = tmp.uns["rank_genes_groups"]["names"][grp]
            for rank, gname in enumerate(ranked_names):
                idx = name_to_idx.get(gname)
                if idx is not None and rank < best_rank[idx]:
                    best_rank[idx] = rank

        order = np.argsort(best_rank)  # ascending: best (lowest) rank first
        _DEG_CACHE[key] = order
        print(f"[gene_selection] Computed DEG ranking over {adata.n_vars} genes, "
              f"groupby='{groupby}' ({len(groups)} groups), method='{method}' "
              f"(cached for this AnnData object).")

    order = _DEG_CACHE[key]
    n = min(n, len(order))
    return order[:n]


def select_genes(adata, strategy, n, groupby="time_min"):
    """Convenience dispatcher: strategy in {'hvg', 'deg'}."""
    if strategy == "hvg":
        return get_top_hvg(adata, n)
    elif strategy == "deg":
        return get_top_deg(adata, n, groupby=groupby)
    else:
        raise ValueError(f"Unknown gene selection strategy: {strategy}")