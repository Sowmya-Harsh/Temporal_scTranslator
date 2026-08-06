"""
qurie_dataset_ot.py
────────────────────
Curriculum dataset with:
  1. Optimal Transport (OT) matching for cross-timepoint protein targets
  2. Mixup augmentation to multiply training data

Mixup: for each real cell, create k synthetic cells by interpolating
between two random cells at the same timepoint.
  rna_mix  = λ * rna_i  + (1-λ) * rna_j
  prot_mix = λ * prot_i + (1-λ) * prot_j
  λ ~ Beta(alpha, alpha)

This is biologically valid — interpolating between cells at the same
timepoint creates plausible intermediate cell states.
"""

import numpy as np
import anndata as ad
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_distances

try:
    import ot
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    print('WARNING: POT not installed. Falling back to mean targets.')

# Curriculum stages sorted by delta_t
CURRICULUM = [
    (0,   2),
    (0,   4),
    (2,   6),
    (0,   6),
    (4,   60),
    (6,   60),
    (0,   60),
    (60,  180),
    (6,   180),
    (0,   180),
]


def normalize(x, eps=1e-8):
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        return np.full_like(x, 1e-8, dtype=np.float32)
    return (1e-8 + (x - xmin) / (xmax - xmin + eps)).astype(np.float32)


def mixup_cells(rna: np.ndarray, prot: np.ndarray,
                n_augmented: int, alpha: float = 0.2,
                seed: int = 42) -> tuple:
    """
    Mixup augmentation — creates synthetic cells by interpolating
    between pairs of real cells at the same timepoint.

    Args:
        rna:         [n_cells, n_genes]
        prot:        [n_cells, n_proteins]
        n_augmented: number of synthetic cells to generate
        alpha:       Beta distribution parameter (smaller = closer to real cells)

    Returns:
        rna_aug:  [n_cells + n_augmented, n_genes]
        prot_aug: [n_cells + n_augmented, n_proteins]
    """
    rng = np.random.default_rng(seed)
    n   = len(rna)

    rna_new, prot_new = [], []

    for _ in range(n_augmented):
        # Pick two random cells
        i, j = rng.choice(n, size=2, replace=False)

        # Sample mixing coefficient from Beta distribution
        lam = rng.beta(alpha, alpha)

        # Interpolate
        rna_new.append(lam * rna[i] + (1 - lam) * rna[j])
        prot_new.append(lam * prot[i] + (1 - lam) * prot[j])

    rna_aug  = np.vstack([rna,  np.array(rna_new,  dtype=np.float32)])
    prot_aug = np.vstack([prot, np.array(prot_new, dtype=np.float32)])

    print(f'  Mixup: {n} real + {n_augmented} synthetic = {len(rna_aug)} cells')
    return rna_aug, prot_aug


def compute_ot_targets(rna_src, rna_tgt, prot_tgt,
                       reg=0.1, max_genes=2000):
    if not OT_AVAILABLE:
        mean = normalize(prot_tgt.mean(axis=0))
        return np.tile(mean, (len(rna_src), 1))

    n_src, n_tgt = len(rna_src), len(rna_tgt)

    gene_var = rna_src.var(axis=0) + rna_tgt.var(axis=0)
    top_genes = np.argsort(gene_var)[-max_genes:]

    C = cosine_distances(
        rna_src[:, top_genes], rna_tgt[:, top_genes]
    ).astype(np.float64)

    a = np.ones(n_src, dtype=np.float64) / n_src
    b = np.ones(n_tgt, dtype=np.float64) / n_tgt

    T      = ot.sinkhorn(a, b, C, reg=reg)
    T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-8)
    targets = (T_norm @ prot_tgt).astype(np.float32)
    targets = np.array([normalize(p) for p in targets])

    print(f'  OT: {n_src} src → {n_tgt} tgt | '
          f'entropy: {-(T * np.log(T + 1e-10)).sum():.3f}')
    return targets


class CurriculumDatasetOT(Dataset):
    def __init__(self, rna, cells_src, prot_targets_src,
                 rna_ids, rna_mask, prot_ids, prot_mask,
                 protein_names, n_proteins,
                 t_src, t_tgt, dec_max_seq_len=1000):
        super().__init__()
        self.rna          = rna
        self.cells_src    = cells_src
        self.prot_targets = prot_targets_src
        self.rna_ids      = rna_ids
        self.rna_mask     = rna_mask
        self.prot_ids     = prot_ids
        self.prot_mask    = prot_mask
        self.protein_names   = protein_names
        self.n_proteins      = n_proteins
        self.t_src           = t_src
        self.t_tgt           = t_tgt
        self.delta_t         = t_tgt - t_src
        self.dec_max_seq_len = dec_max_seq_len

    def __len__(self):
        return len(self.cells_src)

    def __getitem__(self, idx):
        cell = self.cells_src[idx]
        rna  = normalize(self.rna[cell])

        x = np.stack([
            rna.astype(np.float32),
            self.rna_ids.astype(np.float32),
            self.rna_mask.astype(np.float32),
        ])

        prot = self.prot_targets[idx]
        pad  = self.dec_max_seq_len - self.n_proteins
        prot_pad = np.concatenate([prot, np.zeros(max(0, pad), dtype=np.float32)])
        y = np.stack([
            prot_pad.astype(np.float32),
            self.prot_ids.astype(np.float32),
            self.prot_mask.astype(np.float32),
        ])

        t_info = np.array([
            np.log1p(self.t_src),
            np.log1p(self.delta_t),
        ], dtype=np.float32)

        return x, y, t_info


class DataStoreOT:
    def __init__(self, rna_path, protein_path,
                 condition='aIg',
                 train_ratio=0.8, val_ratio=0.1,
                 enc_max_seq_len=20000, dec_max_seq_len=1000,
                 ot_reg=0.1, ot_max_genes=2000,
                 use_ot=True,
                 use_mixup=True,
                 mixup_factor=3,     # generate 3x more cells
                 mixup_alpha=0.2,
                 seed=1105):

        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len
        self.train_ratio     = train_ratio
        self.val_ratio       = val_ratio
        self.ot_reg          = ot_reg
        self.ot_max_genes    = ot_max_genes
        self.use_ot          = use_ot and OT_AVAILABLE
        self.use_mixup       = use_mixup
        self.mixup_factor    = mixup_factor
        self.mixup_alpha     = mixup_alpha
        rng = np.random.default_rng(seed)

        print(f'Loading h5ad (OT={self.use_ot}, Mixup={self.use_mixup} '
              f'factor={mixup_factor})...')

        rna_adata  = ad.read_h5ad(rna_path)
        prot_adata = ad.read_h5ad(protein_path)

        mask       = rna_adata.obs['condition'] == condition
        rna_adata  = rna_adata[mask]
        prot_adata = prot_adata[mask]

        rna_times = rna_adata.obs['time_min'].values.astype(float)

        self.rna = (rna_adata.X.toarray() if hasattr(rna_adata.X, 'toarray')
                    else rna_adata.X).astype(np.float32)
        self.prot = (prot_adata.X.toarray() if hasattr(prot_adata.X, 'toarray')
                     else prot_adata.X).astype(np.float32)

        self.protein_names = list(prot_adata.var_names)
        self.n_proteins    = len(self.protein_names)

        rna_ids_raw = rna_adata.var['my_Id'].astype(np.int64).values
        n = min(len(rna_ids_raw), enc_max_seq_len)
        self.rna_ids  = np.zeros(enc_max_seq_len, dtype=np.int64)
        self.rna_mask = np.zeros(enc_max_seq_len, dtype=np.float32)
        self.rna_ids[:n]  = rna_ids_raw[:n]
        self.rna_mask[:n] = 1.0
        self.rna = self.rna[:, :enc_max_seq_len]

        prot_ids_raw = prot_adata.var['my_Id'].astype(np.int64).values
        pad = dec_max_seq_len - self.n_proteins
        self.prot_ids  = np.concatenate(
            [prot_ids_raw, np.zeros(max(0, pad), dtype=np.int64)])
        self.prot_mask = np.concatenate([
            np.ones(self.n_proteins), np.zeros(max(0, pad))
        ]).astype(np.float32)

        # Split ALL timepoints independently
        timepoints = sorted(np.unique(rna_times).tolist())
        self.train_cells = {}
        self.val_cells   = {}
        self.test_cells  = {}
        self.train_mean_prot = {}

        print('Cell-level split per timepoint:')
        for t in timepoints:
            cells = np.where(np.isclose(rna_times, t, atol=1e-3))[0]
            rng.shuffle(cells)
            n_train = int(len(cells) * train_ratio)
            n_val   = int(len(cells) * val_ratio)

            self.train_cells[t] = cells[:n_train]
            self.val_cells[t]   = cells[n_train:n_train + n_val]
            self.test_cells[t]  = cells[n_train + n_val:]
            self.train_mean_prot[t] = normalize(
                self.prot[self.train_cells[t]].mean(axis=0))

            print(f'  t={int(t):>4}: '
                  f'train={len(self.train_cells[t])} '
                  f'val={len(self.val_cells[t])} '
                  f'test={len(self.test_cells[t])}')

        self.timepoints = timepoints

    def _get_targets(self, src_cells, tgt_cells, split_name):
        rna_src  = self.rna[src_cells]
        rna_tgt  = self.rna[tgt_cells]
        prot_tgt = self.prot[tgt_cells]

        if self.use_ot and len(src_cells) > 1 and len(tgt_cells) > 1:
            print(f'  Computing OT targets for {split_name} '
                  f'({len(src_cells)} src → {len(tgt_cells)} tgt)...')
            return compute_ot_targets(
                rna_src, rna_tgt, prot_tgt,
                reg=self.ot_reg, max_genes=self.ot_max_genes,
            )
        else:
            mean = self.train_mean_prot[self._current_t_tgt]
            return np.tile(mean, (len(src_cells), 1))

    def build_datasets(self, t_src, t_tgt):
        self._current_t_tgt = t_tgt

        train_src   = self.train_cells[t_src]
        train_tgt   = self.train_cells[t_tgt]
        val_src     = self.val_cells[t_src]
        test_src    = self.test_cells[t_src]

        print(f'\nt={t_src}→t={t_tgt} (Δt={t_tgt-t_src}):')

        # Apply mixup to training cells BEFORE computing OT targets
        if self.use_mixup and len(train_src) > 1:
            rna_train  = self.rna[train_src]
            prot_train = self.prot[train_src]
            n_aug = int(len(train_src) * self.mixup_factor)
            rna_aug, prot_aug = mixup_cells(
                rna_train, prot_train, n_aug,
                alpha=self.mixup_alpha
            )
            # Store augmented data temporarily
            orig_n = len(self.rna)
            aug_idx = np.arange(orig_n, orig_n + len(rna_aug))

            # Temporarily extend arrays
            self._aug_rna   = rna_aug
            self._aug_prot  = prot_aug
            train_src_aug   = np.arange(len(rna_aug))  # indices into aug arrays

            # Compute OT on augmented src → real tgt
            rna_src_aug  = rna_aug
            prot_tgt_real = prot_aug  # use augmented for self-OT

            if self.use_ot and len(train_tgt) > 1:
                rna_tgt  = self.rna[train_tgt]
                prot_tgt = self.prot[train_tgt]
                print(f'  Computing OT targets for train-aug '
                      f'({len(rna_src_aug)} src → {len(rna_tgt)} tgt)...')
                train_targets = compute_ot_targets(
                    rna_src_aug, rna_tgt, prot_tgt,
                    reg=self.ot_reg, max_genes=self.ot_max_genes,
                )
            else:
                mean = self.train_mean_prot[t_tgt]
                train_targets = np.tile(mean, (len(rna_src_aug), 1))

            # Build augmented train dataset using stored aug arrays
            train_ds = self._build_aug_dataset(
                rna_aug, train_targets, t_src, t_tgt)
        else:
            train_targets = self._get_targets(train_src, train_tgt, 'train')
            train_ds = CurriculumDatasetOT(
                rna=self.rna, cells_src=train_src,
                prot_targets_src=train_targets,
                rna_ids=self.rna_ids, rna_mask=self.rna_mask,
                prot_ids=self.prot_ids, prot_mask=self.prot_mask,
                protein_names=self.protein_names, n_proteins=self.n_proteins,
                t_src=t_src, t_tgt=t_tgt,
                dec_max_seq_len=self.dec_max_seq_len,
            )

        # Val and test — no mixup, use OT with real cells
        val_targets  = self._get_targets(val_src,  train_tgt, 'val')
        test_targets = self._get_targets(test_src, train_tgt, 'test')

        val_ds = CurriculumDatasetOT(
            rna=self.rna, cells_src=val_src,
            prot_targets_src=val_targets,
            rna_ids=self.rna_ids, rna_mask=self.rna_mask,
            prot_ids=self.prot_ids, prot_mask=self.prot_mask,
            protein_names=self.protein_names, n_proteins=self.n_proteins,
            t_src=t_src, t_tgt=t_tgt,
            dec_max_seq_len=self.dec_max_seq_len,
        )
        test_ds = CurriculumDatasetOT(
            rna=self.rna, cells_src=test_src,
            prot_targets_src=test_targets,
            rna_ids=self.rna_ids, rna_mask=self.rna_mask,
            prot_ids=self.prot_ids, prot_mask=self.prot_mask,
            protein_names=self.protein_names, n_proteins=self.n_proteins,
            t_src=t_src, t_tgt=t_tgt,
            dec_max_seq_len=self.dec_max_seq_len,
        )

        print(f'  Pairs → train={len(train_ds)} '
              f'val={len(val_ds)} test={len(test_ds)}')
        return train_ds, val_ds, test_ds

    def _build_aug_dataset(self, rna_aug, targets, t_src, t_tgt):
        """Build dataset from augmented RNA array directly."""
        class AugDataset(Dataset):
            def __init__(self_, rna_aug, targets,
                         rna_ids, rna_mask, prot_ids, prot_mask,
                         protein_names, n_proteins, t_src, t_tgt,
                         dec_max_seq_len):
                self_.rna_aug      = rna_aug
                self_.targets      = targets
                self_.rna_ids      = rna_ids
                self_.rna_mask     = rna_mask
                self_.prot_ids     = prot_ids
                self_.prot_mask    = prot_mask
                self_.protein_names = protein_names
                self_.n_proteins   = n_proteins
                self_.t_src        = t_src
                self_.t_tgt        = t_tgt
                self_.delta_t      = t_tgt - t_src
                self_.dec_max_seq_len = dec_max_seq_len

            def __len__(self_):
                return len(self_.rna_aug)

            def __getitem__(self_, idx):
                rna = normalize(self_.rna_aug[idx])
                x = np.stack([
                    rna.astype(np.float32),
                    self_.rna_ids.astype(np.float32),
                    self_.rna_mask.astype(np.float32),
                ])
                prot = self_.targets[idx]
                pad  = self_.dec_max_seq_len - self_.n_proteins
                prot_pad = np.concatenate([
                    prot, np.zeros(max(0, pad), dtype=np.float32)])
                y = np.stack([
                    prot_pad.astype(np.float32),
                    self_.prot_ids.astype(np.float32),
                    self_.prot_mask.astype(np.float32),
                ])
                t_info = np.array([
                    np.log1p(self_.t_src),
                    np.log1p(self_.delta_t),
                ], dtype=np.float32)
                return x, y, t_info

        return AugDataset(
            rna_aug, targets,
            self.rna_ids, self.rna_mask,
            self.prot_ids, self.prot_mask,
            self.protein_names, self.n_proteins,
            t_src, t_tgt, self.dec_max_seq_len,
        )

    def build_generalization_dataset(self, t_src, t_tgt, split='test'):
        cells_src = {
            'train': self.train_cells,
            'val':   self.val_cells,
            'test':  self.test_cells,
        }[split][t_src]

        # Use OT with all tgt cells for generalization
        all_tgt = np.concatenate([
            self.train_cells[t_tgt],
            self.val_cells[t_tgt],
            self.test_cells[t_tgt],
        ])

        self._current_t_tgt = t_tgt
        if self.use_ot and len(cells_src) > 1 and len(all_tgt) > 1:
            targets = compute_ot_targets(
                self.rna[cells_src], self.rna[all_tgt],
                self.prot[all_tgt],
                reg=self.ot_reg, max_genes=self.ot_max_genes,
            )
        else:
            mean = normalize(self.prot[all_tgt].mean(axis=0))
            targets = np.tile(mean, (len(cells_src), 1))

        return CurriculumDatasetOT(
            rna=self.rna, cells_src=cells_src,
            prot_targets_src=targets,
            rna_ids=self.rna_ids, rna_mask=self.rna_mask,
            prot_ids=self.prot_ids, prot_mask=self.prot_mask,
            protein_names=self.protein_names, n_proteins=self.n_proteins,
            t_src=t_src, t_tgt=t_tgt,
            dec_max_seq_len=self.dec_max_seq_len,
        )