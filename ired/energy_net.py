"""Transformer-based energy network and DiffusionWrapper.

Inputs:
  z_q : (B, K, d) — frozen-encoded question latents (conditioning)
  z_t : (B, K, d) — noisy answer latents (the variable to denoise)
  t   : (B,)      — diffusion timestep

Architecture (matching gensis.md §3-§5 and §10.1):
  Tokens fed to self-attention:
    [time_token] + [z_q with type=0 + per-position emb] + [z_t with type=1 + per-position emb]
  Four transformer encoder layers, then a Linear head over the z_t slots.
  Scalar energy = ||head(h_zt)||² summed across the K*d dims, matching the
  squared-sum trick at models.py:211 in the original IRED MLP.

DiffusionWrapper.forward returns ∂E/∂z_t — the "noise prediction" for IRED-style
training and inference. `create_graph=True` is kept on during training so the MSE
on the gradient itself can backprop into the EBM's parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


# The fast (flash / mem-efficient) SDPA kernels do not implement double-backward,
# which we need because the IRED training loss MSEs against ∇_z E (so the outer
# loss.backward() differentiates *through* an inner autograd.grad). Forcing the
# math backend keeps double-backward available on both CPU and CUDA.
_DOUBLE_BWD_KERNELS = [SDPBackend.MATH]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:  # pad for odd dim
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class EnergyTransformer(nn.Module):
    """4-layer transformer EBM producing a scalar energy."""

    def __init__(
        self,
        d_model: int = 768,
        k: int = 32,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_ff_mult: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.out_shape = (k, d_model)

        # Type embedding: 0 = z_q, 1 = z_t
        self.type_emb = nn.Embedding(2, d_model)
        # Per-K-position embedding, shared between z_q and z_t blocks
        self.pos_emb = nn.Embedding(k, d_model)

        # Time embedding becomes a single prepended token
        self.time_proj = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff_mult * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # head: take z_t slots → project → square-sum to scalar
        self.head = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_q: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns scalar energy of shape (B, 1)."""
        b, k, d = z_q.shape
        assert k == self.k and d == self.d_model, (
            f"expected K={self.k}, d={self.d_model}, got K={k}, d={d}"
        )
        assert z_t.shape == z_q.shape, "z_q and z_t must have matching shape"

        pos = torch.arange(self.k, device=z_q.device)
        pe = self.pos_emb(pos)[None]                       # (1, K, d)

        type_q = self.type_emb.weight[0][None, None]       # (1, 1, d)
        type_t = self.type_emb.weight[1][None, None]       # (1, 1, d)

        z_q_in = z_q + pe + type_q                         # (B, K, d)
        z_t_in = z_t + pe + type_t                         # (B, K, d)

        t_tok = self.time_proj(t).unsqueeze(1)             # (B, 1, d)
        x = torch.cat([t_tok, z_q_in, z_t_in], dim=1)      # (B, 1+2K, d)

        with sdpa_kernel(_DOUBLE_BWD_KERNELS):
            h = self.encoder(x)
        h = self.norm(h)

        h_zt = h[:, 1 + self.k:]                           # (B, K, d) — z_t slots
        out = self.head(h_zt)                              # (B, K, d)

        # scalar energy via squared-sum (matches models.py:211 in IRED)
        energy = out.pow(2).flatten(1).sum(-1, keepdim=True)  # (B, 1)
        return energy


class DiffusionWrapper(nn.Module):
    """Exposes ∇_{z_t} E for IRED-style training and inference.

    `create_graph` defaults to `self.training` so that p_losses can backprop the
    denoising MSE through the gradient itself, while inference / inner-loop
    opt_step run without keeping the higher-order graph alive.
    """

    def __init__(self, ebm: EnergyTransformer):
        super().__init__()
        self.ebm = ebm
        self.out_shape = ebm.out_shape

    def forward(
        self,
        z_q: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        return_energy: bool = False,
        return_both: bool = False,
    ):
        with torch.enable_grad():
            z_in = z_t.detach().requires_grad_(True)
            energy = self.ebm(z_q, z_in, t)              # (B, 1)
            if return_energy:
                return energy
            create_graph = self.training
            grad = torch.autograd.grad(
                [energy.sum()], [z_in], create_graph=create_graph
            )[0]
        if return_both:
            return energy, grad
        return grad
