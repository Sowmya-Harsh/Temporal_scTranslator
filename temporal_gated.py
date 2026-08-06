"""
temporal_gated.py
─────────────────
Two improved temporal architectures:

1. GatedTimeInjection — post-encoder gated injection (short gradient path)
   encodings = encodings + gate(t) * shift(t)

2. PostTranslatorFiLM — FiLM after translator, before decoder (shortest path)
   seq_out = FiLM(time, translator(encodings))

Both avoid the gradient vanishing problem of pre-encoder FiLM.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ── Approach 1: Gated Time Injection (post-encoder) ──────────────────────────

class GatedTimeInjection(nn.Module):
    """
    Injects time AFTER encoder via learned gate.
    
    encodings = encodings + gate(t) * shift(t)
    
    gate  : sigmoid → [0,1] per dimension, learns HOW MUCH time matters
    shift : SiLU MLP → learns WHAT the time shift should be
    
    Gradient path: MSE → decoder → translator → HERE (short)
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.shift = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2, dim),
            nn.Sigmoid(),
        )
        # init gate near 0 so it starts as identity
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(self, encodings, log_t, log_delta_t):
        t_vec = torch.stack([log_t, log_delta_t], dim=-1)  # [B, 2]
        shift = self.shift(t_vec).unsqueeze(1)              # [B, 1, dim]
        gate  = self.gate(t_vec).unsqueeze(1)               # [B, 1, dim]
        return encodings + gate * shift                      # [B, N, dim]


class TemporalGatedModel(nn.Module):
    """
    scTranslator + Gated Time Injection after encoder.
    
    Forward:
        encodings = enc(rna, gene_ids)
        encodings = encodings + gate(t) * shift(t)   ← new
        seq_out   = translator(encodings)
        out       = sigmoid(dec(seq_out, prot_ids))
    """
    def __init__(self, base_model, dim=128, freeze_base=True):
        super().__init__()
        self.base = base_model
        self.gate_inject = GatedTimeInjection(dim)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
            print('Base frozen. Only GatedTimeInjection trains.')
        else:
            print('Full model trainable.')

    def forward(self, x_vals, x_ids, y_ids, log_t, log_dt, **kwargs):
        from performer_enc_dec import extract_and_set_enc_dec_kwargs
        enc_kwargs, dec_kwargs, _ = extract_and_set_enc_dec_kwargs(kwargs)

        encodings = self.base.enc(
            x_vals, x_ids, return_encodings=True, **enc_kwargs
        )                                                    # [B, N, dim]

        encodings = self.gate_inject(encodings, log_t, log_dt)  # gated inject

        seq_out = self.base.translator(
            encodings.transpose(1, 2).contiguous()
        ).transpose(1, 2).contiguous()

        out = self.base.dec(seq_out, y_ids, **dec_kwargs)
        out = torch.sigmoid(out)
        return out


# ── Approach 2: Post-Translator FiLM (before decoder) ────────────────────────

class PostTranslatorFiLM(nn.Module):
    """
    FiLM applied AFTER translator, BEFORE decoder.
    
    seq_out = FiLM(time_cond, translator(encodings))
    
    This is the shortest possible gradient path for FiLM:
    MSE → decoder → FiLM → HERE (1 step from loss)
    
    FiLM modulates protein query space directly,
    which is more meaningful than modulating RNA token space.
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
        )
        # identity init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias[:dim], 1.0)   # gamma=1
        nn.init.zeros_(self.net[-1].bias[dim:])            # beta=0

    def forward(self, seq_out, log_t, log_delta_t):
        """
        seq_out: [B, M, dim]  translator output (protein space)
        """
        t_vec  = torch.stack([log_t, log_delta_t], dim=-1)  # [B, 2]
        params = self.net(t_vec)                              # [B, 2*dim]
        gamma, beta = params.chunk(2, dim=-1)                # [B, dim] each
        return gamma.unsqueeze(1) * seq_out + beta.unsqueeze(1)


class TemporalPostFiLMModel(nn.Module):
    """
    scTranslator + FiLM after translator (before decoder).

    Forward:
        encodings = enc(rna, gene_ids)
        seq_out   = translator(encodings)
        seq_out   = FiLM(t, seq_out)              ← new, right before decoder
        out       = sigmoid(dec(seq_out, prot_ids))
    """
    def __init__(self, base_model, dim=128, freeze_base=True):
        super().__init__()
        self.base      = base_model
        self.post_film = PostTranslatorFiLM(dim)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
            print('Base frozen. Only PostTranslatorFiLM trains.')
        else:
            print('Full model trainable.')

    def forward(self, x_vals, x_ids, y_ids, log_t, log_dt, **kwargs):
        from performer_enc_dec import extract_and_set_enc_dec_kwargs
        enc_kwargs, dec_kwargs, _ = extract_and_set_enc_dec_kwargs(kwargs)

        encodings = self.base.enc(
            x_vals, x_ids, return_encodings=True, **enc_kwargs
        )                                                     # [B, N, dim]

        seq_out = self.base.translator(
            encodings.transpose(1, 2).contiguous()
        ).transpose(1, 2).contiguous()                        # [B, M, dim]

        seq_out = self.post_film(seq_out, log_t, log_dt)      # FiLM here ← key

        out = self.base.dec(seq_out, y_ids, **dec_kwargs)
        out = torch.sigmoid(out)
        return out