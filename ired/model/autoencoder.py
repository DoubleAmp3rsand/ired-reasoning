"""Frozen BART autoencoder with compression / reconstruction split.

BART (byte-level BPE tokenizer) is used instead of T5 (SentencePiece) because
T5's tokenizer normalizes `\\n`, `\\r`, and runs of spaces into a single space
during encoding — fatal for code reconstruction where whitespace is
load-bearing syntax. BART preserves every byte through encode/decode.

Pipeline
--------

The autoencoder compresses variable-length text into K fixed latent slots,
then reconstructs it through a frozen (or optionally fine-tuned) BART decoder.
The latent space between Pool and Recon is where the EBM diffuses.

Two-phase design, matching LD4LG's compression/reconstruction split:

    ┌── ENCODE (frozen BART encoder + trainable AttentionPool) ──┐
    │                                                            │
    │   text ──► BART Encoder (frozen) ──► AttentionPool (f_φ)   │
    │                                                │           │
    │                                                ▼           │
    │                                     z ∈ R^(K × d_ae)       │
    │                                          │                 │
    └──────────────────────────────────────────┼─────────────────┘
                                               │
                                    [EBM diffuses / denoises
                                     in this continuous space]
                                               │
    ┌── DECODE (trainable Recon + BART decoder) ────────────────┐
    │                                          │                │
    │                                          ▼                │
    │                                ReconstructionNet (f_ψ)    │
    │                                          │                │
    │                                          ▼                │
    │                                   (K × d_LM)              │
    │                                          │                │
    │                    ┌─────────────────────┤                │
    │                    ▼                     ▼                │
    │           BART Decoder           BART Decoder             │
    │           (frozen, default)      (fine-tuned)             │
    │                    │          train_decoder=True          │
    │                    ▼                         ▼            │
    │                  text                      text           │
    │                                                           │
    └───────────────────────────────────────────────────────────┘

One knob on the decode side:

- ``train_decoder`` — unfreeze the BART decoder (and tied LM head) and
  fine-tune it jointly with Pool + Recon. Encoder stays frozen. Weights
  saved under the checkpoint ``decoder`` key.

The decoder always generates from the latent alone — there is no copy/source
path. (A PointerGenerator copy mechanism was removed: at EBM generation time
there is no source sequence to copy from, so copying only inflated AE
reconstruction metrics during training without giving the EBM a usable
latent-only decoder. See docs/autoencoder.md.)

Three compression patterns, selectable via ``pool_type``
----------------------------------------------------------

- ``"decoder"`` (default) — each block does separate self-attention then
  cross-attention via ``nn.TransformerDecoderLayer``. Distinct ``W_k/W_v``
  per role. K learnable queries cross-attend the encoder. Empirically
  reaches loss ≈ 1.45 on GSM8K-full.
- ``"resampler"`` — Flamingo-style Perceiver Resampler. One combined MHA
  per block with ``kv = concat([Z; E(w)])``, matching LD4LG. Two fixes
  that matter: (a) shared LayerNorm on the KV path so latent and encoder
  positions are normalized identically before sharing ``W_k/W_v``;
  (b) gated residuals (``tanh(α)``, α=0 init) so each block is identity
  at step 0.
- ``"conv"`` — no learnable queries. A depthwise-separable 1-D conv scans
  the encoder sequence (one kernel per channel + a 1×1 pointwise mix),
  adaptive average pool collapses variable length L → fixed K slots, and
  a self-attention ``TransformerEncoder`` refines the K slots.

ReconstructionNet (f_ψ)
-----------------------

Identity-initialized (attention ``out_proj`` and FF ``linear2`` zeroed) so
the recon is bit-exact identity at step 0 — the pool's input-conditional
signal passes through unscathed. The optimizer grows non-zero residuals as
training proceeds. No trailing LayerNorm: the frozen decoder's cross-attn
needs the magnitude information in its key/value memory, and stacking a
third LN (pool's final LN + recon's internal LNs + trailing LN) would
over-normalize and squash that signal.

Milestone 1 gate (gensis.md §7)
-------------------------------

Verify that ``Decoder(Recon(Pool(Encoder(A)))) ≈ A`` on the target domain
before training the EBM. If round-trip reconstruction fails, no diffusion
training can recover it. The AE trains on OpenWebText only — never on the
EBM's code corpora — preserving the §3.2 anchoring property.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


POOL_TYPES = ("decoder", "resampler", "conv")


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

    Three pooling patterns are supported via `pool_type`:

    - `"decoder"`: `nn.TransformerDecoderLayer` stack. Separate self-attn
      then cross-attn per block; distinct `W_k/W_v` per role.
    - `"resampler"`: Flamingo-style Perceiver Resampler stack. One combined
      MHA per block with `kv = [Z; E(w)]`, shared LayerNorm on the KV path,
      gated residuals (identity at init).
    - `"conv"`: a **depthwise-separable 1-D conv** scans the encoder sequence
      (one length-`kernel_size` kernel per feature channel, then a 1×1
      pointwise mix), an adaptive average pool collapses the variable length
      L → fixed K slots, and a self-attention `TransformerEncoder` refines the
      K slots. Unlike the query-based pools, the K slots come from the conv +
      adaptive pool — there are no learnable `queries`, no cross-attention,
      and no trailing LayerNorm. This is the "1-D kernel that scans through the
      input tensor" variant.

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
        conv_kernel: int = 5,
    ):
        super().__init__()
        if pool_type not in POOL_TYPES:
            raise ValueError(f"pool_type must be one of {POOL_TYPES}, got {pool_type!r}")
        self.pool_type = pool_type
        self.k = k
        self.d_model = d_model
        self.d_ae = d_ae if d_ae is not None else d_model

        if pool_type == "conv":
            # Depthwise-separable temporal conv: one length-`conv_kernel` kernel
            # per channel (groups=d_model → weight (d_model, 1, k)) followed by a
            # 1×1 pointwise mix. The K slots come from the adaptive pool in
            # forward(), so there are no learnable queries and no trailing norm.
            self.conv = nn.Sequential(
                nn.Conv1d(
                    d_model, d_model, kernel_size=conv_kernel,
                    padding=conv_kernel // 2, groups=d_model,
                ),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_ff_mult * d_model,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.attn = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            # d_ae projection only; conv pool carries no final LayerNorm.
            self.out_proj = (
                nn.Linear(d_model, self.d_ae) if self.d_ae != d_model else nn.Identity()
            )
            return

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

        if self.pool_type == "conv":
            x = enc_hidden
            if enc_mask is not None:
                # Zero padded positions so they don't bleed into the conv /
                # adaptive pool (BART pads carry a real embedding otherwise).
                x = x * enc_mask.unsqueeze(-1).to(x.dtype)
            x = x.transpose(1, 2)                       # (B, d_model, L)
            x = self.conv(x)                            # (B, d_model, L)
            x = F.adaptive_avg_pool1d(x, self.k)        # (B, d_model, K)
            x = x.transpose(1, 2).contiguous()          # (B, K, d_model)
            x = self.attn(x)                            # self-attn over K slots
            return self.out_proj(x)

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

    **No final LayerNorm.** The frozen decoder's cross-attention needs the
    magnitude information in its key/value memory. Pool already applies a
    final LayerNorm and recon's internal norm_first blocks normalize
    per-token; stacking a third trailing LN over-normalizes and squashes
    that magnitude signal.
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


class FrozenBartAutoencoder(nn.Module):
    """BART autoencoder: encoder always frozen, AttentionPool + ReconstructionNet
    trainable, decoder optionally fine-tuned.

    BART is used over T5 because its byte-level BPE tokenizer preserves all
    whitespace (newlines, tabs, indentation) — required for code reconstruction.

    Public API:
      encode_to_latents(texts) -> (B, K, d_ae)              # diffusion-space
      decode_loss(z, target_texts) -> scalar loss           # CE
      decode(z, num_beams=1) -> list[str]                   # generate from latents

    Construction knobs:
      d_ae < d_model   make the EBM diffuse in a smaller space; recon up-projects
                       back to d_model for the decoder.
      pool_type        "decoder" | "resampler" | "conv" (see module docstring).
      train_decoder    fine-tune the BART decoder + LM head (encoder stays frozen).
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        k: int = 32,
        pool_layers: int = 2,
        pool_heads: int = 8,
        pool_type: str = "decoder",
        d_ae: int | None = None,
        recon_layers: int = 2,
        recon_heads: int = 8,
        train_decoder: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.d_model = self.bart.config.d_model
        self.d_ae = d_ae if d_ae is not None else self.d_model
        self.k = k
        self.pool_type = pool_type
        self.train_decoder = train_decoder

        self.pool = AttentionPool(
            d_model=self.d_model,
            k=k,
            n_heads=pool_heads,
            n_layers=pool_layers,
            d_ae=self.d_ae,
            pool_type=pool_type,
        )
        self.recon = ReconstructionNet(
            d_ae=self.d_ae,
            d_LM=self.d_model,
            n_layers=recon_layers,
            n_heads=recon_heads,
        )

        self.freeze_bart()

    # ------------------------------------------------------------------
    # freezing
    # ------------------------------------------------------------------
    def freeze_bart(self) -> None:
        for p in self.bart.parameters():
            p.requires_grad_(False)
        self.bart.eval()
        if self.train_decoder:
            # The conv/copy checkpoints fine-tune the decoder (and tied LM head)
            # jointly with the pool; keep it trainable + in train mode while the
            # encoder stays frozen.
            for p in self.bart.model.decoder.parameters():
                p.requires_grad_(True)
            self.bart.model.decoder.train()

    def train(self, mode: bool = True):
        # Pool + recon toggle to train mode; BART encoder stays in eval. The
        # decoder follows train mode only when it is being fine-tuned.
        super().train(mode)
        self.bart.eval()
        if self.train_decoder:
            self.bart.model.decoder.train(mode)
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
    def _bart_encode(self, texts, device, max_length):
        enc = self._tokenize(texts, device, max_length)
        out = self.bart.get_encoder()(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )
        return out.last_hidden_state, enc.attention_mask

    def encode_to_latents(self, texts, device, max_length: int = 128) -> torch.Tensor:
        """Encode text → (B, K, d_ae) via frozen encoder + trainable pool."""
        enc_hidden, enc_mask = self._bart_encode(texts, device, max_length=max_length)
        z = self.pool(enc_hidden, enc_mask)
        return z

    def encode_to_latents_from_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-tokenized variant of `encode_to_latents`. Same forward path as
        the text-based call but skips the main-thread tokenizer step — feed
        ids straight from `PreTokenizedDataset` + `collate_pretokenized`.
        Pool weights still receive gradients; the BART encoder stays frozen.
        """
        with torch.no_grad():
            out = self.bart.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        z = self.pool(out.last_hidden_state, attention_mask)
        return z

    # ------------------------------------------------------------------
    # decoding: training loss + generation (greedy or beam)
    # ------------------------------------------------------------------
    def _latents_to_decoder_input(self, z: torch.Tensor) -> torch.Tensor:
        """Apply f_ψ to map diffusion-space latents back to decoder-input shape."""
        return self.recon(z)

    def _decoder_states(
        self, z_dec: torch.Tensor, decoder_input_ids: torch.Tensor,
        last_only: bool = False,
    ):
        """Run the BART decoder over the recon latents and return
        (decoder hidden states, LM-head logits).

        `last_only` keeps just the final position's hidden state before the
        LM-head projection, so `lm_logits` is (B, 1, V) instead of (B, T, V).
        Autoregressive generation only needs the last step; projecting the
        whole prefix to vocab each step is the (B, T, V) memory blow-up that
        OOMs the unified-memory host. The decoder itself still runs over the
        full prefix (causal context is required); only the vocab projection
        is sliced."""
        enc_attn = torch.ones(z_dec.shape[:2], dtype=torch.long, device=z_dec.device)
        encoder_outputs = BaseModelOutput(last_hidden_state=z_dec)
        out = self.bart.model(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attn,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        hidden = out.last_hidden_state
        if last_only:
            hidden = hidden[:, -1:]
        lm_logits = self.bart.lm_head(hidden) + self.bart.final_logits_bias
        return hidden, lm_logits

    def decode_loss(
        self,
        z: torch.Tensor,
        target_texts,
        device,
        max_length: int = 128,
    ) -> torch.Tensor:
        """Reconstruction loss. `z` is in diffusion space (B, K, d_ae); recon
        shapes it to (B, K, d_model) before the decoder reads it. BART's plain
        teacher-forced cross-entropy — the decoder reconstructs from the latent
        alone."""
        z_dec = self._latents_to_decoder_input(z)                   # (B, K, d_model)

        target_enc = self._tokenize(target_texts, device, max_length)
        input_ids = target_enc.input_ids
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        enc_attn = torch.ones(z_dec.shape[:2], dtype=torch.long, device=z_dec.device)
        encoder_outputs = BaseModelOutput(last_hidden_state=z_dec)
        out = self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attn,
            labels=labels,
        )
        return out.loss

    def decode_loss_from_ids(
        self,
        z: torch.Tensor,
        label_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-tokenized variant of `decode_loss`. `label_ids` is the raw
        tokenizer output (pad positions masked out internally with -100)."""
        z_dec = self._latents_to_decoder_input(z)
        labels = label_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        enc_attn = torch.ones(z_dec.shape[:2], dtype=torch.long, device=z_dec.device)
        encoder_outputs = BaseModelOutput(last_hidden_state=z_dec)
        out = self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attn,
            labels=labels,
        )
        return out.loss

    @torch.no_grad()
    def decode_loss_per_example(
        self,
        z: torch.Tensor,
        target_texts,
        device,
        max_length: int = 128,
    ) -> torch.Tensor:
        """Per-example teacher-forced CE — returns shape (B,).

        Same forward pass as `decode_loss` but reduces independently per
        example, so per-example CE can be paired with per-example pass/fail
        for the exposure-bias-gap diagnostic (see docs/exposure_bias.md).
        Pad positions are excluded from each example's mean.
        """
        z_dec = self._latents_to_decoder_input(z)
        target_enc = self._tokenize(target_texts, device, max_length)
        input_ids = target_enc.input_ids
        labels = input_ids.clone()
        pad_mask = (labels == self.tokenizer.pad_token_id)
        labels[pad_mask] = -100

        enc_attn = torch.ones(z_dec.shape[:2], dtype=torch.long, device=z_dec.device)
        encoder_outputs = BaseModelOutput(last_hidden_state=z_dec)
        out = self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attn,
            labels=labels,
        )
        logits = out.logits  # (B, L, V); BART shifts labels internally
        B, L, V = logits.shape
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(B, L)
        valid = (~pad_mask).float()
        return (ce * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    @torch.no_grad()
    def decode(
        self,
        z: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 1,
    ) -> list[str]:
        """Decode from continuous latents. `z` is in diffusion space (B, K, d_ae);
        recon expands to (B, K, d_model). `num_beams=1` is greedy; >1 enables
        beam search, which usually lifts code pass-rate when per-token loss is
        in the 0.2–1.0 nat range (right token often #2/#3, not #1).

        The decoder generates from the latent alone (`bart.generate`) — the same
        path used at EBM sampling time."""
        self.bart.eval()
        z_dec = self._latents_to_decoder_input(z)

        enc_attn = torch.ones(z_dec.shape[:2], dtype=torch.long, device=z_dec.device)
        encoder_outputs = BaseModelOutput(last_hidden_state=z_dec)
        out_ids = self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attn,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=(num_beams > 1),
        )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # checkpoint helpers
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        params = list(self.pool.parameters()) + list(self.recon.parameters())
        if self.train_decoder:
            params += list(self.bart.model.decoder.parameters())
        return params

    def state_dict_ae(self) -> dict:
        """Save format: {pool, recon, d_ae, pool_type, [decoder]}.

        pool + recon are always saved together (trained jointly). The optional
        fine-tuned `decoder` weights are included only when `train_decoder` is
        active, so checkpoints stay back-compatible with the load path."""
        sd = {
            "pool":      self.pool.state_dict(),
            "recon":     self.recon.state_dict(),
            "d_ae":      self.d_ae,
            "pool_type": self.pool_type,
        }
        if self.train_decoder:
            sd["decoder"] = self.bart.model.decoder.state_dict()
        return sd

    def load_ae(self, state_dict: dict) -> None:
        saved_d_ae = state_dict.get("d_ae", self.d_ae)
        if saved_d_ae != self.d_ae:
            raise ValueError(
                f"checkpoint has d_ae={saved_d_ae} but this AE was built with "
                f"d_ae={self.d_ae}. Construct with matching d_ae and retry."
            )
        # Default to "decoder" for back-compat with pre-pool_type checkpoints.
        saved_pool_type = state_dict.get("pool_type", "decoder")
        if saved_pool_type != self.pool_type:
            raise ValueError(
                f"checkpoint has pool_type={saved_pool_type!r} but this AE was "
                f"built with pool_type={self.pool_type!r}. Construct to match."
            )
        self.pool.load_state_dict(state_dict["pool"])
        self.recon.load_state_dict(state_dict["recon"])
        # Fine-tuned decoder weights, if the checkpoint carried them. Loaded into
        # the BART decoder regardless of train_decoder so reconstruction matches.
        if "decoder" in state_dict:
            self.bart.model.decoder.load_state_dict(state_dict["decoder"])
