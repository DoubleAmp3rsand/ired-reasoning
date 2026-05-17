"""Frozen T5 autoencoder with LD4LG-faithful compression / reconstruction split.

This is the Stable-Diffusion-style "VAE" in the gensis.md proposal:

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

Two design choices that follow LD4LG (Lovelace et al., NeurIPS 2023):

1. **Perceiver Resampler attention pattern.** Each pool layer does ONE MHA
   call with `q = Z, kv = [Z; E(w)]` instead of separate self-attention then
   cross-attention. Each head can dynamically split its attention budget
   between latent-latent mixing and encoder-extraction; see §7.6, §11.6.

2. **Explicit reconstruction network f_ψ.** Decouples the pool's compression
   job from the "shape latents for the frozen decoder" job. With `d_ae < d_LM`
   it also performs the up-projection that lets the EBM diffuse in a smaller
   space. Default `d_ae = d_model` (Step 1 of the rollout in §7.6).

Milestone 1 in gensis.md is verifying that
    Decoder(Recon(Pool(Encoder(A)))) ≈ A
on the target domain. If that fails, no diffusion training can recover it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class PerceiverResamplerLayer(nn.Module):
    """One block of the Perceiver Resampler (Flamingo §3.1.1; LD4LG eq.).

    Implements the LD4LG attention update:

        Z = Z + MHA(q = Z, kv = [Z; E(w)])
        Z = Z + FF(Z)

    The single combined MHA call (vs. separate self-attn + cross-attn) lets
    each head allocate attention dynamically between the latent slots and
    encoder positions, rather than forcing strict separation. See §11.6.
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

    def forward(
        self,
        z: torch.Tensor,                  # (B, K, d)
        enc_hidden: torch.Tensor,         # (B, L, d)
        enc_mask: torch.Tensor | None,    # (B, L) — 1 keep, 0 pad
    ) -> torch.Tensor:
        b, k, _ = z.shape

        z_n  = self.norm_q(z)
        kv   = torch.cat([z_n, self.norm_kv(enc_hidden)], dim=1)   # (B, K+L, d)

        if enc_mask is not None:
            latent_keep = torch.ones((b, k), dtype=enc_mask.dtype, device=enc_mask.device)
            kv_keep = torch.cat([latent_keep, enc_mask], dim=1)    # (B, K+L)
            key_padding_mask = ~kv_keep.bool()                     # True = mask
        else:
            key_padding_mask = None

        attn_out, _ = self.attn(
            z_n, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        z = z + attn_out
        z = z + self.ff(self.norm_ff(z))
        return z


class AttentionPool(nn.Module):
    """LD4LG-faithful compression network f_φ.

    Stack of `PerceiverResamplerLayer`s with K trainable latent queries.
    Outputs `(B, K, d_ae)` — the diffusion-space latent. The final linear
    projection (d_model → d_ae) matches LD4LG's "we reduce the dimensionality
    of the output to dimension d_ae with a learnable linear projection."

    When `d_ae == d_model` the projection is `nn.Identity` (no parameters).
    """

    def __init__(
        self,
        d_model: int,
        k: int,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_ff_mult: int = 4,
        d_ae: int | None = None,
    ):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.d_ae = d_ae if d_ae is not None else d_model
        self.queries = nn.Parameter(torch.randn(k, d_model) * 0.02)
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
        z = self.queries.unsqueeze(0).expand(b, -1, -1).contiguous()  # (B, K, d_model)
        for layer in self.layers:
            z = layer(z, enc_hidden, enc_mask)
        z = self.out_proj(z)                                          # (B, K, d_ae)
        return self.norm(z)


class ReconstructionNet(nn.Module):
    """LD4LG reconstruction network f_ψ: (K, d_ae) → (K, d_LM).

    Decouples 'shape latents for the frozen decoder' from the pool's
    compression job. Self-attention transformer over the K latents, with an
    up-projection if `d_ae < d_LM`. The pre-decoder LayerNorm is what the
    frozen T5 cross-attention will read.
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
        self.norm = nn.LayerNorm(d_LM)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(z)
        x = self.layers(x)
        return self.norm(x)


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
        d_ae: int | None = None,
        recon_layers: int = 2,
        recon_heads: int = 8,
    ):
        super().__init__()
        self.model_name = model_name
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.d_model = self.t5.config.d_model
        self.d_ae = d_ae if d_ae is not None else self.d_model
        self.k = k

        self.pool = AttentionPool(
            d_model=self.d_model,
            k=k,
            n_heads=pool_heads,
            n_layers=pool_layers,
            d_ae=self.d_ae,
        )
        self.recon = ReconstructionNet(
            d_ae=self.d_ae,
            d_LM=self.d_model,
            n_layers=recon_layers,
            n_heads=recon_heads,
        )

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
        return list(self.pool.parameters()) + list(self.recon.parameters())

    def state_dict_ae(self) -> dict:
        """Save format: {pool, recon, d_ae}. Both modules go together because
        they were trained jointly and only make sense as a pair."""
        return {
            "pool":  self.pool.state_dict(),
            "recon": self.recon.state_dict(),
            "d_ae":  self.d_ae,
        }

    def load_ae(self, state_dict: dict) -> None:
        saved_d_ae = state_dict.get("d_ae", self.d_ae)
        if saved_d_ae != self.d_ae:
            raise ValueError(
                f"checkpoint has d_ae={saved_d_ae} but this AE was built with "
                f"d_ae={self.d_ae}. Construct with matching d_ae and retry."
            )
        self.pool.load_state_dict(state_dict["pool"])
        self.recon.load_state_dict(state_dict["recon"])
