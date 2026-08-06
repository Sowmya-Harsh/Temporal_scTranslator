"""
attention_pooling.py
────────────────────────
Adds a small LEARNED attention-pooling layer, to replace the naive
mean-pooling used in train_joint_pretrained_encoder.py.

Mean-pooling (encodings.mean(dim=1)) treats every one of the 100
selected genes' contextualized embeddings as equally important --
likely erasing exactly the fine-grained, gene-specific structure the
pretrained transformer encoder was built to capture. Attention pooling
lets the model LEARN which genes matter most for this task, weighting
the pooled summary accordingly.

Not wired into train_joint_pretrained_encoder.py yet -- this is a
standalone module. To use it, replace:

    pooled = encodings.mean(dim=1)

with:

    pool = AttentionPool(enc_dim)  # instantiate once, in __init__
    pooled = pool(encodings)        # in forward()

in PretrainedEncoderHeadMLP / FiLMPre / FiLMPost's __init__ and forward
methods (train_joint_pretrained_encoder.py).
"""

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Learns a single query vector; attention-weights the sequence of
    per-gene embeddings against it, producing a weighted-sum pooled
    representation instead of a plain unweighted mean.

    encodings: [batch, n_genes, dim] -> returns [batch, dim]
    """
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, encodings):
        # encodings: [B, n_genes, dim]
        scores = self.score(encodings)               # [B, n_genes, 1]
        weights = torch.softmax(scores, dim=1)        # [B, n_genes, 1] -- sums to 1 over genes
        pooled = (encodings * weights).sum(dim=1)      # [B, dim]
        return pooled

    def forward_with_weights(self, encodings):
        """Same as forward(), but also returns the attention weights
        [B, n_genes] -- useful for inspecting which genes the model
        actually attended to most, a nice interpretability figure for
        your thesis if you want to show which genes drove predictions."""
        scores = self.score(encodings)
        weights = torch.softmax(scores, dim=1)
        pooled = (encodings * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)
