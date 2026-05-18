"""Frozen T5 autoencoder with compression / reconstruction split.

Pipeline:

    text → E_frozen → AttentionPool (f_φ) → z ∈ R^(K × d_ae)
                                              │
                                              ▼
                                  [diffuse here, EBM]
                                              │
                                              ▼
                                     ReconstructionNet (f_ψ)
                                              │
                                              ▼
                                       (K × d_LM)
                                              │
                                              ▼
                                       D_frozen → text

Two pool implementations live here, selectable via `pool_type`:

- `"decoder"` (default): each block does separate self-attention then
  cross-attention, via `nn.TransformerDecoderLayer`. Distinct `W_k/W_v` per
  role. Empirically reaches loss ≈ 1.45 on GSM8K-full.
- `"resampler"`: Flamingo-style Perceiver Resampler — one MHA per block
  with `kv = concat([Z; E(w)])`. This is what LD4LG specifies. Includes the
  two fixes that our first attempt was missing: (a) shared LayerNorm for
  the KV path so latent and encoder positions are normalized identically
  before sharing `W_k/W_v`; (b) gated residuals (`tanh(α)` with α=0 init)
  so each block is bit-exact identity at step 0.

`ReconstructionNet` (f_ψ) is identity-initialized so it's bit-exact identity
at step 0 — gradient descent grows non-zero residuals as it becomes useful.

Milestone 1 in gensis.md is verifying that
    Decoder(Recon(Pool(Encoder(A)))) ≈ A
on the target domain. If that fails, no diffusion training can recover it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


POOL_TYPES = ("decoder", "resampler")


class PerceiverResamplerLayer(nn.Module):
    """One Flamingo-style Perceiver Resampler block (LD4LG-faithful).

    Update rule (Flamingo §3.1.1; LD4LG eq. on p.4):

        Z = Z + tanh(α_xattn) · MHA(q = LN_q(Z), kv = LN_kv([Z; E(w)]))
        Z = Z + tanh(α_ff)    · FF(LN_ff(Z))

    Two details that mattered (and were wrong in our first attempt; §11.6):

    1. **Shared LayerNorm for the KV path.** A single `LayerNorm` applied to
       the *concatenated* `[Z; E(w)]` means latents and encoder positions are
       normalized with the same gain/bias. The shared `W_k`, `W_v` projections
       then see distributionally-comparable inputs from both halves of KV.
       Separate LNs (our earlier bug) gave the two halves different stats,
       and the shared projection couldn't fit both.

    2. **Gated residuals with α init = 0.** At step 0, `tanh(0) = 0` so each
       block is bit-exact identity — no random transforms in the residual
       stream that could swamp the pool's input-conditional signal. The
       optimizer grows `α` as the block becomes useful.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dim_ff_mult: int = 4):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff_mult * d_model),
            nn.GELU(),
            nn.Linear(dim_ff_mult * d_model, d_model),
        )
        # Flamingo gated residuals: α starts at 0 → tanh(α) = 0 → identity at init.
        self.alpha_xattn = nn.Parameter(torch.zeros(1))
        self.alpha_ff    = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        z: torch.Tensor,                  # (B, K, d)
        enc_hidden: torch.Tensor,         # (B, L, d)
        enc_mask: torch.Tensor | None,    # (B, L) — 1 keep, 0 pad
    ) -> torch.Tensor:
        b, k, _ = z.shape

        # Single shared LN applied to the concatenation: same gain/bias for
        # all K+L positions. Equivalent to applying norm_kv separately to z
        # and enc_hidden since LayerNorm is per-position.
        kv_raw = torch.cat([z, enc_hidden], dim=1)               # (B, K+L, d)
        kv = self.norm_kv(kv_raw)

        if enc_mask is not None:
            latent_keep = torch.ones((b, k), dtype=enc_mask.dtype, device=enc_mask.device)
            kv_keep = torch.cat([latent_keep, enc_mask], dim=1)  # (B, K+L)
            key_padding_mask = ~kv_keep.bool()                   # True = mask
        else:
            key_padding_mask = None

        attn_out, _ = self.attn(
            self.norm_q(z), kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        z = z + torch.tanh(self.alpha_xattn) * attn_out
        z = z + torch.tanh(self.alpha_ff)    * self.ff(self.norm_ff(z))
        return z


class AttentionPool(nn.Module):
    """Compression network f_φ. Variable-length encoder → fixed K latents.

    Two attention patterns are supported via `pool_type`:

    - `"decoder"`: `nn.TransformerDecoderLayer` stack. Separate self-attn
      then cross-attn per block; distinct `W_k/W_v` per role.
    - `"resampler"`: Flamingo-style Perceiver Resampler stack. One combined
      MHA per block with `kv = [Z; E(w)]`, shared LayerNorm on the KV path,
      gated residuals (identity at init).

    Final linear projection `d_model → d_ae` matches LD4LG's dimensionality
    reduction. `nn.Identity` when `d_ae == d_model`.
    """

    def __init__(
        self,
        d_model: int,
        k: int,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_ff_mult: int = 4,
        d_ae: int | None = None,
        pool_type: str = "decoder",
    ):
        super().__init__()
        if pool_type not in POOL_TYPES:
            raise ValueError(f"pool_type must be one of {POOL_TYPES}, got {pool_type!r}")
        self.pool_type = pool_type
        self.k = k
        self.d_model = d_model
        self.d_ae = d_ae if d_ae is not None else d_model
        self.queries = nn.Parameter(torch.randn(k, d_model) * 0.02)

        if pool_type == "decoder":
            layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_ff_mult * d_model,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.attn = nn.TransformerDecoder(layer, num_layers=n_layers)
        else:  # "resampler"
            self.layers = nn.ModuleList([
                PerceiverResamplerLayer(d_model, n_heads, dim_ff_mult)
                for _ in range(n_layers)
            ])

        # LD4LG's "learnable linear projection" to d_ae.
        self.out_proj = (
            nn.Linear(d_model, self.d_ae) if self.d_ae != d_model else nn.Identity()
        )
        self.norm = nn.LayerNorm(self.d_ae)

    def forward(
        self,
        enc_hidden: torch.Tensor,
        enc_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b = enc_hidden.size(0)
        z = self.queries.unsqueeze(0).expand(b, -1, -1).contiguous()

        if self.pool_type == "decoder":
            mem_pad = ~enc_mask.bool() if enc_mask is not None else None
            z = self.attn(tgt=z, memory=enc_hidden, memory_key_padding_mask=mem_pad)
        else:  # "resampler"
            for layer in self.layers:
                z = layer(z, enc_hidden, enc_mask)

        z = self.out_proj(z)
        return self.norm(z)


class ReconstructionNet(nn.Module):
    """LD4LG reconstruction network f_ψ: (K, d_ae) → (K, d_LM).

    Decouples 'shape latents for the frozen decoder' from the pool's
    compression job. Self-attention transformer over the K latents, with an
    up-projection if `d_ae < d_LM`.

    **Identity-init.** Each transformer block's residual paths (attention
    out_proj and FF linear2) are zero-initialized so the block is exactly
    `x = x + 0 = x` at step 0. This keeps the input-conditional signal from
    the pool intact at random init — otherwise the recon's random noise
    swamps it and the frozen decoder falls back to unigram statistics. The
    optimizer then gradually grows non-zero residuals as training proceeds.

    **No final LayerNorm.** The frozen T5 decoder was trained on RMSNorm'd
    encoder hidden states; adding a third LayerNorm on top (pool's final LN
    + recon's internal LNs already happened) over-normalizes and squashes
    the magnitude information the decoder cross-attention needs.
    """

    def __init__(
        self,
        d_ae: int,
        d_LM: int,
        n_layers: int = 2,
        n_heads: int = 8,
        dim_ff_mult: int = 4,
    ):
        super().__init__()
        self.d_ae = d_ae
        self.d_LM = d_LM
        self.up_proj = nn.Linear(d_ae, d_LM) if d_ae != d_LM else nn.Identity()
        layer = nn.TransformerEncoderLayer(
            d_model=d_LM,
            nhead=n_heads,
            dim_feedforward=dim_ff_mult * d_LM,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)
        # No trailing LayerNorm — see class docstring.

        self._identity_init()

    def _identity_init(self) -> None:
        """Zero the residual paths of every transformer block so each block
        is bit-exactly identity at step 0. The optimizer learns them from
        zero, which avoids the recon's random noise drowning the pool's
        input-conditional signal."""
        for layer in self.layers.layers:
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(z)
        x = self.layers(x)
        return x


class FrozenT5Autoencoder(nn.Module):
    """T5 encoder + decoder, frozen; AttentionPool + ReconstructionNet, trainable.

    Public API:
      encode_to_latents(texts) -> (B, K, d_ae)              # diffusion-space
      decode_loss(z, target_texts) -> scalar CE loss        # for AE training
      decode(z) -> list[str]                                # greedy from latents

    Set `d_ae < d_model` to make the EBM diffuse in a smaller space; the
    reconstruction network up-projects back to d_model for the frozen decoder.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        k: int = 32,
        pool_layers: int = 2,
        pool_heads: int = 8,
        pool_type: str = "decoder",
        d_ae: int | None = None,
        recon_layers: int = 2,
        recon_heads: int = 8,
        use_recon: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.d_model = self.t5.config.d_model
        self.d_ae = d_ae if d_ae is not None else self.d_model
        self.k = k
        self.use_recon = use_recon
        self.pool_type = pool_type

        self.pool = AttentionPool(
            d_model=self.d_model,
            k=k,
            n_heads=pool_heads,
            n_layers=pool_layers,
            d_ae=self.d_ae,
            pool_type=pool_type,
        )
        if self.use_recon:
            self.recon: nn.Module = ReconstructionNet(
                d_ae=self.d_ae,
                d_LM=self.d_model,
                n_layers=recon_layers,
                n_heads=recon_heads,
            )
        else:
            # Diagnostic fallback: pool output goes straight to T5 decoder. Only
            # valid when d_ae == d_model (decoder needs d_LM-shaped input).
            if self.d_ae != self.d_model:
                raise ValueError(
                    f"use_recon=False requires d_ae == d_model "
                    f"(got d_ae={self.d_ae}, d_model={self.d_model}). "
                    "Without recon there's no up-projection back to d_model."
                )
            self.recon = nn.Identity()

        self.freeze_t5()

    # ------------------------------------------------------------------
    # freezing
    # ------------------------------------------------------------------
    def freeze_t5(self) -> None:
        for p in self.t5.parameters():
            p.requires_grad_(False)
        self.t5.eval()

    def train(self, mode: bool = True):
        # Pool + recon toggle to train mode; T5 stays in eval.
        super().train(mode)
        self.t5.eval()
        return self

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------
    def _tokenize(self, texts, device, max_length):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

    @torch.no_grad()
    def _t5_encode(self, texts, device, max_length):
        enc = self._tokenize(texts, device, max_length)
        out = self.t5.encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        return out.last_hidden_state, enc.attention_mask

    def encode_to_latents(self, texts, device, max_length: int = 128) -> torch.Tensor:
        """Encode text → (B, K, d_ae) via frozen encoder + trainable pool."""
        enc_hidden, enc_mask = self._t5_encode(texts, device, max_length=max_length)
        z = self.pool(enc_hidden, enc_mask)
        return z

    # ------------------------------------------------------------------
    # decoding: training loss + greedy generation
    # ------------------------------------------------------------------
    def _latents_to_decoder_input(self, z: torch.Tensor) -> torch.Tensor:
        """Apply f_ψ to map diffusion-space latents back to decoder-input shape."""
        return self.recon(z)

    def decode_loss(
        self,
        z: torch.Tensor,
        target_texts,
        device,
        max_length: int = 128,
    ) -> torch.Tensor:
        """T5 CE loss. `z` is in diffusion space (B, K, d_ae); recon shapes it
        to (B, K, d_model) before the frozen decoder reads it."""
        z_dec = self._latents_to_decoder_input(z)                   # (B, K, d_model)

        target_enc = self._tokenize(target_texts, device, max_length)
        labels = target_enc.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        enc_attn = torch.ones(z_dec.shape[:2], dtype=torch.long, device=z_dec.device)
        encoder_outputs = BaseModelOutput(last_hidden_state=z_dec)
        out = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attn,
            labels=labels,
        )
        return out.loss

    @torch.no_grad()
    def decode(self, z: torch.Tensor, max_length: int = 128) -> list[str]:
        """Greedy decoding from continuous latents using KV cache. `z` is in
        diffusion space (B, K, d_ae); recon expands to (B, K, d_model)."""
        self.t5.eval()
        z_dec = self._latents_to_decoder_input(z)
        b = z_dec.size(0)
        device = z_dec.device

        enc_attn = torch.ones(z_dec.shape[:2], dtype=torch.long, device=device)
        encoder_outputs = BaseModelOutput(last_hidden_state=z_dec)

        start_id = self.t5.config.decoder_start_token_id
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        decoder_ids = torch.full((b, 1), start_id, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        past = None

        for _ in range(max_length):
            step_input = decoder_ids if past is None else decoder_ids[:, -1:]
            out = self.t5(
                encoder_outputs=encoder_outputs,
                attention_mask=enc_attn,
                decoder_input_ids=step_input,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_tok = out.logits[:, -1].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
            decoder_ids = torch.cat([decoder_ids, next_tok[:, None]], dim=1)
            finished = finished | (next_tok == eos_id)
            if finished.all():
                break

        return self.tokenizer.batch_decode(decoder_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # checkpoint helpers
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        params = list(self.pool.parameters())
        if self.use_recon:
            params += list(self.recon.parameters())
        return params

    def state_dict_ae(self) -> dict:
        """Save format: {pool, recon, d_ae, use_recon, pool_type}. Both modules
        go together because they were trained jointly and only make sense as
        a pair. When use_recon=False, the recon entry is None."""
        return {
            "pool":      self.pool.state_dict(),
            "recon":     self.recon.state_dict() if self.use_recon else None,
            "d_ae":      self.d_ae,
            "use_recon": self.use_recon,
            "pool_type": self.pool_type,
        }

    def load_ae(self, state_dict: dict) -> None:
        saved_d_ae = state_dict.get("d_ae", self.d_ae)
        if saved_d_ae != self.d_ae:
            raise ValueError(
                f"checkpoint has d_ae={saved_d_ae} but this AE was built with "
                f"d_ae={self.d_ae}. Construct with matching d_ae and retry."
            )
        saved_use_recon = state_dict.get("use_recon", True)
        if saved_use_recon != self.use_recon:
            raise ValueError(
                f"checkpoint has use_recon={saved_use_recon} but this AE was "
                f"built with use_recon={self.use_recon}. Construct to match."
            )
        # Default to "decoder" for back-compat with pre-pool_type checkpoints.
        saved_pool_type = state_dict.get("pool_type", "decoder")
        if saved_pool_type != self.pool_type:
            raise ValueError(
                f"checkpoint has pool_type={saved_pool_type!r} but this AE was "
                f"built with pool_type={self.pool_type!r}. Construct to match."
            )
        self.pool.load_state_dict(state_dict["pool"])
        if self.use_recon:
            self.recon.load_state_dict(state_dict["recon"])
