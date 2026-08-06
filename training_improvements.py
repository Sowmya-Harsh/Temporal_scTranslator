"""
training_improvements.py
───────────────────────────
Reusable utilities to drop into any of the train_*.py scripts.

1. EarlyStopper       -- stop training once val loss hasn't improved
                          for `patience` epochs, instead of always
                          running the full --epochs count.
2. build_optimizer     -- AdamW with weight_decay (plain Adam by
                          default fights the overfitting visible in
                          every training curve so far).
3. build_lr_scheduler  -- ReduceLROnPlateau: halves LR once val loss
                          plateaus.
4. correlation_loss    -- differentiable (1 - per-protein Pearson r).
                          Targets R specifically.
5. ccc_loss            -- differentiable (1 - per-protein CCC).
                          Targets CCC specifically -- also penalizes
                          scale/location mismatch, which correlation_loss
                          does not.

Example integration (inside a training loop):

    from training_improvements import EarlyStopper, build_optimizer, build_lr_scheduler, correlation_loss, ccc_loss

    optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-4)
    scheduler = build_lr_scheduler(optimizer, patience=5)
    stopper = EarlyStopper(patience=15)

    for epoch in range(1, max_epochs + 1):
        train_loss = ...  # your existing training loop, but add:
        #   loss = mse_loss + alpha * correlation_loss(y_hat, y) + beta * ccc_loss(y_hat, y)
        val_loss = ...
        scheduler.step(val_loss)
        if stopper.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break
"""

import torch
import torch.nn as nn


class EarlyStopper:
    """Tracks validation loss; call .step(val_loss) each epoch, returns
    True when training should stop (no improvement for `patience` epochs)."""
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_optimizer(model, lr=1e-3, weight_decay=1e-4):
    """AdamW instead of plain Adam -- decoupled weight decay, standard
    fix for the overfitting pattern visible in every training curve so
    far (val loss rising while train loss keeps falling). Try
    weight_decay in [1e-5, 1e-4, 1e-3] -- higher = stronger regularization."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_lr_scheduler(optimizer, patience=5, factor=0.5, min_lr=1e-6):
    """Halves LR once val loss plateaus for `patience` epochs. Call
    scheduler.step(val_loss) once per epoch, AFTER computing val_loss."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
    )


def correlation_loss(y_hat, y, eps=1e-8):
    """Differentiable (1 - per-protein Pearson r), averaged over the
    protein dimension, for one batch. Add to MSE:

        loss = mse_loss(y_hat, y) + alpha * correlation_loss(y_hat, y)

    alpha in [0.1, 0.5] is a reasonable starting range -- too high and
    it can dominate/destabilize training since it's less well-behaved
    than MSE near-zero-variance batches; too low and it does nothing.

    Skips (returns 0 contribution for) any protein column with near-zero
    variance in the batch, since correlation is undefined there.

    IMPORTANT: this targets Pearson R specifically, NOT CCC. R rewards
    correct ranking/direction only and is scale/location invariant --
    it does not penalize a model whose predictions are well-correlated
    but systematically offset or mis-scaled relative to the true range.
    If per-protein CCC lags behind per-protein R (e.g. good R but near-
    zero CCC for some proteins -- a calibration/scale-mismatch pattern),
    correlation_loss alone will not fix that; use ccc_loss below (or
    both together) to specifically target CCC.
    """
    y_hat_centered = y_hat - y_hat.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)

    numerator = (y_hat_centered * y_centered).sum(dim=0)
    denom = (torch.sqrt((y_hat_centered ** 2).sum(dim=0) + eps) *
             torch.sqrt((y_centered ** 2).sum(dim=0) + eps))
    r_per_protein = numerator / denom  # [n_proteins]

    # mask out proteins with ~zero variance in this batch (r undefined/noisy)
    valid = (y_centered.std(dim=0) > 1e-4) & (y_hat_centered.std(dim=0) > 1e-4)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=y.device, requires_grad=True)

    return (1.0 - r_per_protein[valid]).mean()


def ccc_loss(y_hat, y, eps=1e-8):
    """Differentiable (1 - per-protein CCC), averaged over the protein
    dimension, for one batch. Unlike correlation_loss (Pearson r), CCC
    ALSO penalizes variance mismatch and mean/location offset between
    predicted and true values, not just ranking/direction -- this
    directly targets the calibration-mismatch pattern where a protein
    has decent R but poor CCC (predictions correlate with truth but sit
    at the wrong scale or are systematically shifted).

    Add to MSE, optionally alongside correlation_loss:

        loss = mse_loss(y_hat, y) \
             + alpha * correlation_loss(y_hat, y) \
             + beta  * ccc_loss(y_hat, y)

    Start with beta in [0.1, 0.3], same range as alpha -- run a small
    sweep (alpha only / beta only / both) rather than guessing a single
    combination, same spirit as the thesis's existing training-objective
    ablation (Section 4.6.1, Table 5.6).
    """
    mean_true = y.mean(dim=0, keepdim=True)
    mean_pred = y_hat.mean(dim=0, keepdim=True)
    var_true = y.var(dim=0, unbiased=False)
    var_pred = y_hat.var(dim=0, unbiased=False)
    covariance = ((y - mean_true) * (y_hat - mean_pred)).mean(dim=0)

    ccc_per_protein = (2 * covariance) / (
        var_true + var_pred + (mean_true.squeeze(0) - mean_pred.squeeze(0)) ** 2 + eps
    )

    valid = (y.std(dim=0) > 1e-4) & (y_hat.std(dim=0) > 1e-4)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=y.device, requires_grad=True)

    return (1.0 - ccc_per_protein[valid]).mean()