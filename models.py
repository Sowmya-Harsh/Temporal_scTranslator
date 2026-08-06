"""
models.py
──────────
Four model architectures for RNA(subset) -> protein prediction,
operating directly on the selected gene subset (NOT scTranslator's
tokenized full-transcriptome encoder — these are simple, fast
baselines meant for the gene-selection x OT-type ablation grid).

All four take:
  x:      [batch, n_genes]        selected-gene RNA expression
  t_info: [batch, 2]               [log1p(t_src), log1p(delta_t)]
and output:
  y_hat:  [batch, n_proteins]      predicted protein abundance, in [0,1]

NOTE: all four end in torch.sigmoid() on the output layer. Targets
(OT-matched protein vectors) are normalize()'d to [1e-8, 1] per cell,
so the model's raw output must be bounded to match -- an unconstrained
linear output (no sigmoid) can and does predict negative/out-of-range
values, which showed up as a real scale-mismatch bug when we diagnosed
scTranslator's own fine-tuning output (see conversation history) and
was fixed here proactively before it caused the same issue in our
pipeline.

1. MLPBaseline
   Plain feedforward MLP. Time info is just concatenated to the input
   -- simplest possible baseline, no special time-conditioning
   mechanism.

2. FiLMPreMLP
   FiLM (Feature-wise Linear Modulation) conditioning applied BEFORE
   the main MLP trunk: t_info generates (gamma, beta) that rescale/
   shift the RAW INPUT gene features before they enter the network.
   x' = gamma(t_info) * x + beta(t_info)

3. FiLMPostMLP
   FiLM conditioning applied AFTER each hidden layer of the MLP trunk
   (the more standard FiLM usage): each hidden activation h is
   modulated as h' = gamma(t_info) * h + beta(t_info), letting time
   reshape the network's internal representations at every layer
   rather than only the input.

4. GatedInjectionMLP
   ADDITIVE (not multiplicative) time injection with a learned
   magnitude gate, applied once to the raw input gene vector -- same
   structural position as FiLMPreMLP, for a direct, apples-to-apples
   comparison against FiLM's multiplicative mechanism:
   x' = x + gate(t_info) * shift(t_info)
   gate:  sigmoid in [0,1] per-dimension, learns HOW MUCH time should
          matter for that feature. Initialised near 0 (identity/no-op
          at init) so training starts close to "time barely matters"
          and learns to open the gate where useful.
   shift: small MLP, learns WHAT the additive adjustment should be.
   Adapted from temporal_gated.py's GatedTimeInjection (originally
   built for the pretrained model's [B, N, dim] token sequence) down
   to this from-scratch model's flat [B, n_genes] input.
"""

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Multi-frequency (sin/cos) embedding of the two raw time scalars
    (log1p(t_src), log1p(delta_t)), in the style of transformer
    positional encodings. Instead of handing the model two raw numbers
    and hoping it learns to separate short vs. long intervals on its
    own, this gives it a richer, multi-scale representation of time up
    front -- cheap to add, no architectural risk, since it's a fixed
    (non-learned) transform sitting in front of whatever conditioning
    mechanism (concatenation, FiLM, or gated injection) is already in
    use.

    Each of the 2 raw scalars is expanded into `2 * num_frequencies`
    values (sin and cos at `num_frequencies` exponentially-spaced
    scales), so output dim = 2 * 2 * num_frequencies.
    """
    def __init__(self, num_frequencies=8):
        super().__init__()
        self.num_frequencies = num_frequencies
        # exponentially spaced frequencies, matching standard
        # sinusoidal positional encoding practice
        freqs = torch.pow(2.0, torch.arange(num_frequencies).float())
        self.register_buffer("freqs", freqs)  # [num_frequencies]

    @property
    def out_dim(self):
        return 2 * 2 * self.num_frequencies  # 2 raw scalars x (sin+cos) x num_frequencies

    def forward(self, t_info):
        # t_info: [B, 2] -> [B, 2, 1] -> broadcast against freqs -> [B, 2, num_frequencies]
        scaled = t_info.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        emb = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # [B, 2, 2*num_frequencies]
        return emb.reshape(t_info.shape[0], -1)  # [B, 2*2*num_frequencies]


class MLPBaseline(nn.Module):
    def __init__(self, n_genes, n_proteins, hidden_dims=(512, 256, 128), dropout=0.2,
                 time_embed="raw", num_frequencies=8):
        super().__init__()
        self.time_embed = time_embed
        if time_embed == "sinusoidal":
            self.time_enc = SinusoidalTimeEmbedding(num_frequencies)
            t_dim = self.time_enc.out_dim
        else:
            self.time_enc = None
            t_dim = 2
        dims = [n_genes + t_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], n_proteins)

    def forward(self, x, t_info):
        t = self.time_enc(t_info) if self.time_enc is not None else t_info
        h = torch.cat([x, t], dim=-1)
        h = self.trunk(h)
        return torch.sigmoid(self.out(h))


class FiLMGenerator(nn.Module):
    """Maps t_info -> (gamma, beta) of size `feature_dim` each."""
    def __init__(self, t_dim, feature_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim * 2)
        )
        self.feature_dim = feature_dim

    def forward(self, t_info):
        out = self.net(t_info)
        gamma, beta = out[:, :self.feature_dim], out[:, self.feature_dim:]
        # gamma initialized around 1 (identity modulation) via +1 shift
        return 1.0 + gamma, beta


class FiLMPreMLP(nn.Module):
    """FiLM conditioning applied once, directly on the input gene vector."""
    def __init__(self, n_genes, n_proteins, hidden_dims=(512, 256, 128),
                 dropout=0.2, t_dim=2, time_embed="raw", num_frequencies=8):
        super().__init__()
        self.time_embed = time_embed
        if time_embed == "sinusoidal":
            self.time_enc = SinusoidalTimeEmbedding(num_frequencies)
            t_dim = self.time_enc.out_dim
        else:
            self.time_enc = None
        self.film_in = FiLMGenerator(t_dim, n_genes)
        dims = [n_genes] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], n_proteins)

    def forward(self, x, t_info):
        t = self.time_enc(t_info) if self.time_enc is not None else t_info
        gamma, beta = self.film_in(t)
        x_mod = gamma * x + beta
        h = self.trunk(x_mod)
        return torch.sigmoid(self.out(h))


class FiLMLayer(nn.Module):
    """One Linear -> FiLM -> ReLU -> Dropout block."""
    def __init__(self, in_dim, out_dim, t_dim=2, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.film = FiLMGenerator(t_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, t_info):
        h = self.linear(x)
        gamma, beta = self.film(t_info)
        h = gamma * h + beta
        h = self.act(h)
        return self.drop(h)


class FiLMPostMLP(nn.Module):
    """FiLM conditioning applied after every hidden layer (standard FiLM usage)."""
    def __init__(self, n_genes, n_proteins, hidden_dims=(512, 256, 128),
                 dropout=0.2, t_dim=2, time_embed="raw", num_frequencies=8):
        super().__init__()
        self.time_embed = time_embed
        if time_embed == "sinusoidal":
            self.time_enc = SinusoidalTimeEmbedding(num_frequencies)
            t_dim = self.time_enc.out_dim
        else:
            self.time_enc = None
        dims = [n_genes] + list(hidden_dims)
        self.layers = nn.ModuleList([
            FiLMLayer(dims[i], dims[i + 1], t_dim=t_dim, dropout=dropout)
            for i in range(len(dims) - 1)
        ])
        self.out = nn.Linear(dims[-1], n_proteins)

    def forward(self, x, t_info):
        t = self.time_enc(t_info) if self.time_enc is not None else t_info
        h = x
        for layer in self.layers:
            h = layer(h, t)
        return torch.sigmoid(self.out(h))


class GatedTimeInjection(nn.Module):
    """Additive time injection with a learned magnitude gate, adapted
    from temporal_gated.py's pretrained-model version down to flat
    [B, n_genes] RNA input. gate: sigmoid in [0,1] per-dimension,
    learns HOW MUCH time should matter for that feature. shift: small
    MLP, learns WHAT the additive adjustment should be. Gate is
    initialised near 0 (identity/no-op at init), same as the pretrained
    version, so training starts close to "time barely matters" and
    learns to open the gate where useful."""
    def __init__(self, n_genes, t_dim=2, hidden=64):
        super().__init__()
        self.shift = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_genes),
        )
        self.gate = nn.Sequential(
            nn.Linear(t_dim, n_genes),
            nn.Sigmoid(),
        )
        # init gate near 0 so it starts close to identity (matches
        # temporal_gated.py's GatedTimeInjection init exactly)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -2.0)  # sigmoid(-2) ~= 0.12

    def forward(self, x, t_info):
        shift = self.shift(t_info)  # [B, n_genes]
        gate = self.gate(t_info)    # [B, n_genes]
        return x + gate * shift


class GatedInjectionMLP(nn.Module):
    """Gated time injection applied ONCE to the raw input gene vector,
    same structural position as FiLMPreMLP (Section 4.4.1), for a
    direct, apples-to-apples architecture comparison against FiLM's
    multiplicative conditioning mechanism."""
    def __init__(self, n_genes, n_proteins, hidden_dims=(512, 256, 128),
                 dropout=0.2, t_dim=2, time_embed="raw", num_frequencies=8):
        super().__init__()
        self.time_embed = time_embed
        if time_embed == "sinusoidal":
            self.time_enc = SinusoidalTimeEmbedding(num_frequencies)
            t_dim = self.time_enc.out_dim
        else:
            self.time_enc = None
        self.gated_in = GatedTimeInjection(n_genes, t_dim=t_dim)
        dims = [n_genes] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], n_proteins)

    def forward(self, x, t_info):
        t = self.time_enc(t_info) if self.time_enc is not None else t_info
        x_mod = self.gated_in(x, t)
        h = self.trunk(x_mod)
        return torch.sigmoid(self.out(h))


class GatedLayer(nn.Module):
    """One Linear -> Gated time injection -> ReLU -> Dropout block,
    the gated-mechanism analog of FiLMLayer, for a GatedPostMLP
    architecture matching FiLMPostMLP's structure exactly."""
    def __init__(self, in_dim, out_dim, t_dim=2, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gated = GatedTimeInjection(out_dim, t_dim=t_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, t_info):
        h = self.linear(x)
        h = self.gated(h, t_info)
        h = self.act(h)
        return self.drop(h)


class GatedPostMLP(nn.Module):
    """Gated time injection applied after EVERY hidden layer (the
    gated-mechanism analog of FiLMPostMLP), for a direct comparison
    against GatedInjectionMLP (input-only) matching the existing
    FiLM-pre vs FiLM-post structural comparison exactly."""
    def __init__(self, n_genes, n_proteins, hidden_dims=(512, 256, 128),
                 dropout=0.2, t_dim=2, time_embed="raw", num_frequencies=8):
        super().__init__()
        self.time_embed = time_embed
        if time_embed == "sinusoidal":
            self.time_enc = SinusoidalTimeEmbedding(num_frequencies)
            t_dim = self.time_enc.out_dim
        else:
            self.time_enc = None
        dims = [n_genes] + list(hidden_dims)
        self.layers = nn.ModuleList([
            GatedLayer(dims[i], dims[i + 1], t_dim=t_dim, dropout=dropout)
            for i in range(len(dims) - 1)
        ])
        self.out = nn.Linear(dims[-1], n_proteins)

    def forward(self, x, t_info):
        t = self.time_enc(t_info) if self.time_enc is not None else t_info
        h = x
        for layer in self.layers:
            h = layer(h, t)
        return torch.sigmoid(self.out(h))


def build_model(model_name, n_genes, n_proteins, **kwargs):
    if model_name == "mlp":
        return MLPBaseline(n_genes, n_proteins, **kwargs)
    elif model_name == "film_pre":
        return FiLMPreMLP(n_genes, n_proteins, **kwargs)
    elif model_name == "film_post":
        return FiLMPostMLP(n_genes, n_proteins, **kwargs)
    elif model_name == "gated":
        return GatedInjectionMLP(n_genes, n_proteins, **kwargs)
    elif model_name == "gated_post":
        return GatedPostMLP(n_genes, n_proteins, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")