"""Gaussian latent diffusion with IRED inner-loop refinement and bad-step rejection.

This is the "thinking module" of the gensis.md proposal — adapted directly from
IRED's `GaussianDiffusion1D` (the matrix-addition / continuous variant) to live in
the latent space defined by the frozen T5 encoder + AttentionPool.

Key mechanisms (from gensis.md §10):
  - opt_step: inner-loop gradient descent on the energy with bad-step rejection
              (a step is rejected if the new energy is higher than the old one).
  - p_losses: denoising MSE + NCE energy contrast. The NCE term takes a heavily
              noised version of the clean target, refines it via 2 opt_step
              iterations to mine a "hard negative", then pushes the clean
              sample's energy below it at every timestep. This is what calibrates
              the absolute scale of E so bad-step rejection is meaningful.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# A decoder-aux callback receives a sub-batch of predicted clean latents
# x0_hat (B', K, d) and the matching list of gold-answer texts, and returns
# a scalar CE loss. Gradient flows back through the frozen decoder into the
# EBM via x0_hat.
DecoderLossFn = Callable[[torch.Tensor, list[str]], torch.Tensor]


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Sequence[int]) -> torch.Tensor:
    """Pull per-batch values out of a 1-D schedule and broadcast to `x_shape`."""
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    # IRED's formula assumes timesteps ~1000. For small T the upper end exceeds 1.0,
    # which makes alpha negative and sqrt(alpha_cumprod) NaN for late t. Clip to the
    # same upper bound as the cosine schedule so q_sample is always well-defined.
    scale = 1000.0 / timesteps
    return torch.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=torch.float64).clamp(max=0.999)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1.0 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 0.0, 0.999)


class GaussianLatentDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,                  # DiffusionWrapper around the EBM
        latent_shape: Sequence[int],       # (K, d_model)
        timesteps: int = 10,
        beta_schedule: str = "linear",
        opt_step_size: float = 1.0,
        loss_scale: float = 1.0,           # weight on NCE term
        continuous: bool = True,           # matches the matrix-addition variant
        supervise_energy_landscape: bool = True,
        objective: str = "pred_noise",
        # Clamping. IRED's defaults (x_start_clamp=2, envelope_sf=2) assume data
        # in roughly [-1, 1]. LayerNorm'd T5 latents have element-std ~1 so the
        # bulk lives in ~[-3, 3]; we relax x_start_clamp and disable the per-t
        # envelope clamp by default. Pass concrete floats to re-enable.
        x_start_clamp: float | None = 5.0,
        envelope_sf: float | None = None,
        # Optional decoder-CE auxiliary. When > 0 and a callback is registered
        # via set_decoder_loss_fn, p_losses also decodes x0_hat for samples
        # with t < decoder_aux_t_max and adds CE(decode(x0_hat), gold) at the
        # given weight. Only low-t samples are used because x0_hat is only a
        # reliable estimate of the clean latent near the end of the schedule.
        decoder_aux_weight: float = 0.0,
        decoder_aux_t_max: int = 2,
    ):
        super().__init__()
        self.model = model
        self.latent_shape = tuple(latent_shape)
        self.num_timesteps = timesteps
        self.objective = objective
        self.continuous = continuous
        self.supervise_energy_landscape = supervise_energy_landscape
        self.loss_scale = loss_scale
        self.x_start_clamp = x_start_clamp
        self.envelope_sf = envelope_sf
        self.decoder_aux_weight = float(decoder_aux_weight)
        self.decoder_aux_t_max = int(decoder_aux_t_max)
        self._decoder_loss_fn: DecoderLossFn | None = None

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(beta_schedule)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        if (alphas_cumprod <= 0).any():
            raise ValueError(
                f"beta schedule '{beta_schedule}' at T={timesteps} produced "
                f"non-positive alpha_cumprod={alphas_cumprod.tolist()} — "
                "q_sample would NaN. Choose a different schedule or T."
            )

        reg = lambda name, val: self.register_buffer(name, val.float())

        reg("betas", betas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        post_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        reg("posterior_variance", post_var)
        reg("posterior_log_variance_clipped", torch.log(post_var.clamp(min=1e-20)))
        reg(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        reg(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # uniform weighting for the noise-prediction objective
        reg("loss_weight", torch.ones(timesteps))

        # per-timestep step size for inner-loop optimization
        reg("opt_step_size_t", torch.full((timesteps,), float(opt_step_size)))

    def set_decoder_loss_fn(self, fn: DecoderLossFn | None) -> None:
        """Register a callback for the decoder-CE auxiliary. Must be set if
        decoder_aux_weight > 0 and the user calls forward with gold_texts."""
        self._decoder_loss_fn = fn

    # ------------------------------------------------------------------
    # forward / posterior helpers
    # ------------------------------------------------------------------
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, log_var

    def _clamp_envelope(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Clamp z into the per-t diffusion envelope. No-op if envelope_sf is None."""
        if self.envelope_sf is None:
            return z
        max_val = extract(self.sqrt_alphas_cumprod, t, z.shape) * float(self.envelope_sf)
        return torch.clamp(z, -max_val, max_val)

    # ------------------------------------------------------------------
    # inner-loop bad-step rejection — the heart of IRED
    # ------------------------------------------------------------------
    def opt_step(
        self,
        z_q: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        step: int = 5,
        sf: float = 1.0,
        detach: bool = True,
    ) -> torch.Tensor:
        with torch.enable_grad():
            for _ in range(step):
                energy, grad = self.model(z_q, z, t, return_both=True)
                step_size = extract(self.opt_step_size_t, t, grad.shape)
                z_new = z - step_size * grad * sf

                z_new = self._clamp_envelope(z_new, t)

                energy_new = self.model(z_q, z_new, t, return_energy=True)
                # energy is (B, 1); compare per-sample
                bad = (energy_new[:, 0] > energy[:, 0])

                # broadcast bad mask over the latent dims of z
                bad_b = bad.view(-1, *((1,) * (z.dim() - 1)))
                z_next = torch.where(bad_b, z, z_new)

                z = z_next.detach() if detach else z_next
        return z

    # ------------------------------------------------------------------
    # p_sample: one outer DDPM step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, z_q: torch.Tensor, z_t: torch.Tensor, t: int):
        b = z_t.shape[0]
        batched_t = torch.full((b,), t, device=z_t.device, dtype=torch.long)

        eps_hat = self.model(z_q, z_t, batched_t, return_energy=False)
        x_start = self.predict_start_from_noise(z_t, batched_t, eps_hat)
        if self.x_start_clamp is not None:
            bound = float(self.x_start_clamp)
            x_start = x_start.clamp(-bound, bound)

        mean, log_var = self.q_posterior(x_start, z_t, batched_t)
        noise = torch.randn_like(z_t) if t > 0 else torch.zeros_like(z_t)
        z_prev = mean + (0.5 * log_var).exp() * noise
        return z_prev.detach(), x_start.detach()

    @torch.no_grad()
    def sample(
        self,
        z_q: torch.Tensor,
        inner_steps: int = 5,
        show_tqdm: bool = False,
    ) -> torch.Tensor:
        """Full inference: outer T DDPM steps, each followed by `inner_steps` of IRED refinement."""
        b = z_q.size(0)
        device = z_q.device
        z = torch.randn((b, *self.latent_shape), device=device)

        iterator = reversed(range(self.num_timesteps))
        if show_tqdm:
            iterator = tqdm(iterator, total=self.num_timesteps, desc="sampling")

        for t in iterator:
            z, _ = self.p_sample(z_q, z, t)

            if inner_steps > 0:
                batched_t = torch.full((b,), t, device=device, dtype=torch.long)
                z = self.opt_step(z_q, z, batched_t, step=inner_steps, sf=1.0, detach=True)

            batched_t = torch.full((b,), t, device=device, dtype=torch.long)
            z = self._clamp_envelope(z, batched_t)

        return z

    # ------------------------------------------------------------------
    # training loss: denoising MSE + (optional) NCE energy-landscape supervision
    # ------------------------------------------------------------------
    def p_losses(
        self,
        z_q: torch.Tensor,
        z_a: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        gold_texts: list[str] | None = None,
    ):
        if noise is None:
            noise = torch.randn_like(z_a)
        z_t = self.q_sample(z_a, t, noise)

        # ε̂ = ∇_{z_t} E. With self.training=True, the gradient itself carries a
        # graph so the MSE can backprop into the EBM's weights.
        eps_hat = self.model(z_q, z_t, t, return_energy=False)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = z_a
        else:
            raise ValueError(self.objective)

        # mean over all non-batch dims, then per-t loss weighting
        loss = F.mse_loss(eps_hat, target, reduction="none")
        loss = loss.flatten(1).mean(-1, keepdim=True)         # (B, 1)
        loss = loss * extract(self.loss_weight, t, loss.shape) # (B, 1)
        loss_mse = loss

        # Scale-drift monitor: if ||eps_hat|| diverges from ||noise||, DDPM
        # math breaks even when mse(eps_hat, noise) looks small.
        with torch.no_grad():
            eps_norm   = eps_hat.flatten(1).norm(dim=-1).mean()
            noise_norm = target.flatten(1).norm(dim=-1).mean()
            scale_ratio = (eps_norm / noise_norm.clamp(min=1e-8)).item()

        if not self.supervise_energy_landscape:
            loss_total = loss_mse.mean()
            stats = {"mse": loss_mse.mean().item(), "eps_scale": scale_ratio}
            loss_total, stats = self._maybe_add_decoder_aux(
                loss_total, stats, z_t, eps_hat, t, gold_texts
            )
            return loss_total, stats

        # ----- NCE term -----
        noise2 = torch.randn_like(z_a)
        data_sample = self.q_sample(z_a, t, noise2)                       # real sample at t

        # heavy-noise the clean target, refine via 2 opt_steps → hard negative
        x_min_noise = self.q_sample(z_a, t, 3.0 * noise2)
        x_min_noise = self.opt_step(z_q, x_min_noise, t, step=2, sf=1.0, detach=True)

        # monitoring: distance from the noise-only mean of q(x_t | x_0)
        x_min = extract(self.sqrt_alphas_cumprod, t, z_a.shape) * z_a
        loss_opt = (x_min_noise - x_min).pow(2).mean()

        # un-scale to x_0 estimate, clamp, then re-noise with the original noise2
        # so the "fake" sample sits at the same noise level as data_sample
        x_min_noise = x_min_noise.detach()
        x_min_unscaled = self.predict_start_from_noise(
            x_min_noise, t, torch.zeros_like(x_min_noise)
        )
        if self.x_start_clamp is not None:
            bound = float(self.x_start_clamp)
            x_min_unscaled = torch.clamp(x_min_unscaled, -bound, bound)
        x_min_renoised = self.q_sample(x_min_unscaled, t, noise2)

        # NCE: clean should have lower energy than the mined fake at this t
        z_q_cat = torch.cat([z_q, z_q], dim=0)
        z_cat = torch.cat([data_sample, x_min_renoised], dim=0)
        t_cat = torch.cat([t, t], dim=0)
        energy = self.model(z_q_cat, z_cat, t_cat, return_energy=True)    # (2B, 1)
        e_real, e_fake = torch.chunk(energy, 2, dim=0)
        logits = torch.cat([-e_real, -e_fake], dim=-1)                     # softmax over [real, fake]
        ce_target = torch.zeros(e_real.size(0), device=e_real.device, dtype=torch.long)
        loss_nce = F.cross_entropy(logits, ce_target, reduction="none")[:, None]

        loss_total = (loss_mse + self.loss_scale * loss_nce).mean()
        stats = {
            "mse": loss_mse.mean().item(),
            "nce": loss_nce.mean().item(),
            "opt": loss_opt.item(),
            "e_real": e_real.mean().item(),
            "e_fake": e_fake.mean().item(),
            "eps_scale": scale_ratio,
        }
        loss_total, stats = self._maybe_add_decoder_aux(
            loss_total, stats, z_t, eps_hat, t, gold_texts
        )
        return loss_total, stats

    def _maybe_add_decoder_aux(
        self,
        loss_total: torch.Tensor,
        stats: dict,
        z_t: torch.Tensor,
        eps_hat: torch.Tensor,
        t: torch.Tensor,
        gold_texts: list[str] | None,
    ):
        """Mix in CE(decode(x0_hat), gold) for the sub-batch with t < t_max."""
        if self.decoder_aux_weight <= 0 or gold_texts is None:
            return loss_total, stats
        if self._decoder_loss_fn is None:
            raise RuntimeError(
                "decoder_aux_weight > 0 but no decoder loss fn registered; "
                "call set_decoder_loss_fn(ae.decode_loss)."
            )

        mask = t < self.decoder_aux_t_max  # (B,)
        if not bool(mask.any()):
            stats["dec"] = 0.0
            stats["dec_n"] = 0
            return loss_total, stats

        x0_hat = self.predict_start_from_noise(z_t, t, eps_hat)
        if self.x_start_clamp is not None:
            bound = float(self.x_start_clamp)
            x0_hat = x0_hat.clamp(-bound, bound)

        idx = mask.nonzero(as_tuple=True)[0]
        x0_sub = x0_hat[idx]
        texts_sub = [gold_texts[i] for i in idx.tolist()]
        loss_dec = self._decoder_loss_fn(x0_sub, texts_sub)

        stats["dec"] = float(loss_dec.detach().item())
        stats["dec_n"] = int(mask.sum().item())
        return loss_total + self.decoder_aux_weight * loss_dec, stats

    def forward(self, z_q: torch.Tensor, z_a: torch.Tensor, gold_texts: list[str] | None = None):
        b = z_q.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=z_q.device).long()
        return self.p_losses(z_q, z_a, t, gold_texts=gold_texts)
