"""Frozen T5 autoencoder with a learned attention pool to K fixed latents.

This is the Stable-Diffusion-style "VAE" in the gensis.md proposal:
- T5 encoder + decoder stay frozen.
- A small AttentionPool (K trainable queries cross-attending to encoder output)
  compresses variable-length encoder hidden states into K fixed latents.
- For decoding, the K latents serve as encoder_hidden_states for the frozen T5
  decoder's cross-attention — the pool is responsible for producing outputs that
  the frozen decoder can read.

Milestone 1 in gensis.md is verifying that
    Decoder(Pool(Encoder(A))) ≈ A
on the target domain. If that fails, no diffusion training can recover it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class AttentionPool(nn.Module):
    """K trainable queries cross-attending to a variable-length sequence.

    Implemented as an `nn.TransformerDecoder` where the queries are the "target"
    and the encoder hidden states are the "memory". Self-attention among the K
    queries lets them share information; cross-attention pulls from the encoder.
    """

    def __init__(
        self,
        d_model: int,
        k: int,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_ff_mult: int = 4,
    ):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.queries = nn.Parameter(torch.randn(k, d_model) * 0.02)
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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor | None) -> torch.Tensor:
        b = enc_hidden.size(0)
        q = self.queries.unsqueeze(0).expand(b, -1, -1).contiguous()
        # nn.TransformerDecoder uses *True for masked positions*.
        mem_pad = None
        if enc_mask is not None:
            mem_pad = ~enc_mask.bool()
        out = self.attn(tgt=q, memory=enc_hidden, memory_key_padding_mask=mem_pad)
        return self.norm(out)


class FrozenT5Autoencoder(nn.Module):
    """T5 encoder + decoder, frozen; AttentionPool, trainable.

    Public API:
      encode_to_latents(texts) -> (B, K, d_model)
      decode_loss(z, target_texts) -> scalar CE loss (for pool training)
      decode(z) -> list[str] (greedy from continuous latents)
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        k: int = 32,
        pool_layers: int = 2,
        pool_heads: int = 8,
    ):
        super().__init__()
        self.model_name = model_name
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.d_model = self.t5.config.d_model
        self.k = k

        self.pool = AttentionPool(
            d_model=self.d_model,
            k=k,
            n_heads=pool_heads,
            n_layers=pool_layers,
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
        # Pool toggles to train mode; T5 stays in eval (no dropout, no BN updates).
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
        """Encode text → (B, K, d_model) via frozen encoder + trainable pool."""
        enc_hidden, enc_mask = self._t5_encode(texts, device, max_length=max_length)
        z = self.pool(enc_hidden, enc_mask)
        return z

    # ------------------------------------------------------------------
    # decoding: training loss + greedy generation
    # ------------------------------------------------------------------
    def decode_loss(self, z: torch.Tensor, target_texts, device, max_length: int = 128) -> torch.Tensor:
        """T5 CE loss with K latents `z` as encoder_hidden_states. Backprops into the pool."""
        target_enc = self._tokenize(target_texts, device, max_length)
        labels = target_enc.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        enc_attn = torch.ones(z.shape[:2], dtype=torch.long, device=z.device)
        encoder_outputs = BaseModelOutput(last_hidden_state=z)
        out = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attn,
            labels=labels,
        )
        return out.loss

    @torch.no_grad()
    def decode(self, z: torch.Tensor, max_length: int = 128) -> list[str]:
        """Greedy decoding from continuous latents using KV cache."""
        self.t5.eval()
        b = z.size(0)
        device = z.device

        enc_attn = torch.ones(z.shape[:2], dtype=torch.long, device=device)
        encoder_outputs = BaseModelOutput(last_hidden_state=z)

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
        return list(self.pool.parameters())

    def state_dict_pool(self):
        return self.pool.state_dict()

    def load_pool(self, state_dict):
        self.pool.load_state_dict(state_dict)
