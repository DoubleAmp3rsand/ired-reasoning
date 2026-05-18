"""Mode-1 actor: one-shot regression from `z_q` to the Mode-2 optimum `z_final`.

§1.4 / §5.3 of gensis.md. The Mode-2 optimizer (`GaussianLatentDiffusion.sample`)
defines what "correct" means in latent space; the actor distills its converged
output into a single forward pass so deployment can default to one cheap pass and
escalate to the full IRED rollout only when the EBM flags low confidence (§5.4).

Architecture: a small transformer encoder over the K latent slots, with a
residual head. The head is zero-initialized so at step 0 the actor returns
`z_q` exactly — `z_q` is a vastly better starting point than random noise
(it already encodes the question), and the model only has to learn the
question → answer residual in latent space.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Fixed-shape latent regressor `(B, K, d_ae) → (B, K, d_ae)`."""

    def __init__(
        self,
        d_ae: int,
        k: int,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_ff_mult: int = 4,
    ):
        super().__init__()
        self.d_ae = d_ae
        self.k = k

        self.pos_emb = nn.Embedding(k, d_ae)
        layer = nn.TransformerEncoderLayer(
            d_model=d_ae,
            nhead=n_heads,
            dim_feedforward=dim_ff_mult * d_ae,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_ae)
        self.head = nn.Linear(d_ae, d_ae)

        # Identity-init: head outputs zero, so actor(z_q) == z_q at step 0.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        b, k, d = z_q.shape
        assert k == self.k and d == self.d_ae, (
            f"expected K={self.k}, d={self.d_ae}, got K={k}, d={d}"
        )
        pos = torch.arange(self.k, device=z_q.device)
        x = z_q + self.pos_emb(pos)[None]
        h = self.encoder(x)
        h = self.norm(h)
        return z_q + self.head(h)
