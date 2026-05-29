# Autoencoder (AE) — Reference

The autoencoder is the **bottleneck** between raw text and the EBM's continuous
latent space. It compresses variable-length text into K fixed slots, then
reconstructs the original text from those slots through a frozen decoder.
Everything the EBM learns lives inside the latent space the AE defines.

---

## 1. Where it fits

```
   Q ──► BART Enc ──► Pool (f_φ) ──►  z_q ∈ R^(K×d_ae)
                                       │
                                       │   conditioning
                                       ▼
                z_T ~ N(0,I)  ──►  EBM(z_q, z, t)  ──►  z_0
                (Mode-2: T outer × N inner steps)
                                       │
                                       ▼
                z_0  ──►  Recon (f_ψ) ──► BART Dec ──►  A
```

The AE is trained first (Milestone 1 in `gensis.md` §7), then frozen while
the EBM is trained. The AE is deliberately **never trained on the EBM's code
corpora** — it trains on OpenWebText only, preserving the anchoring property
(`gensis.md` §3.2): the latent image of the gold answer `z_a = f_φ(Enc(A))`
is structurally fixed and cannot drift with EBM training.

---

## 2. Why BART over T5

T5 uses a SentencePiece tokenizer that **normalizes whitespace during
encoding**: `\n`, `\r`, and runs of spaces collapse to a single space. For
code reconstruction this is fatal — indentation, blank lines, and intra-line
spacing are load-bearing syntax. A single missing newline can break a Python
function.

BART uses a **byte-level BPE** tokenizer that preserves every byte through
`encode → decode`. Round-tripping a code snippet through BART's tokenizer is
lossless at the byte level, which is the minimum requirement for the AE's
reconstruction objective.

The model loaded is `facebook/bart-base` (`d_model = 768`).

---

## 3. Architecture

```
                           ┌─── ENCODE ───┐
text ──► BART Encoder ──► AttentionPool (f_φ) ──► z ∈ R^(K×d_ae)
       (frozen)         (trainable, 3 variants)         │
                                                        │
                                             [EBM operates here]
                                                        │
                           ┌─── DECODE ───┐             │
                                                        ▼
text ◄── BART Decoder ◄── ReconstructionNet (f_ψ) ◄─────┘
         (frozen or         (trainable, identity-init)
          fine-tuned)
              ▲
              │
     PointerGenerator (opt.)
     point-generator
```

| Component | Trainable | Role |
|-----------|-----------|------|
| BART Encoder | ❌ frozen | Tokenize & embed text → hidden states `(B, L, 768)` |
| **AttentionPool** (`f_φ`) | ✅ | Compress `(B, L, d_model)` → `(B, K, d_ae)` |
| **ReconstructionNet** (`f_ψ`) | ✅ | Shape `(B, K, d_ae)` → `(B, K, d_model)` for decoder cross-attn |
| BART Decoder | ❌ frozen (default) or ✅ `train_decoder=True` | Transduce latent slots → token sequence |
| **PointerGenerator** | ✅ if `use_copy=True` | Mix vocab distribution with copy-from-source distribution |

---

### 3.1 BART Encoder (frozen)

Standard BART encoder. Runs `bart.get_encoder()` with `torch.no_grad()`.
Returns `last_hidden_state` of shape `(B, L, d_model)` and an `attention_mask`.

The encoder is **always frozen** — even when `train_decoder=True`, only the
decoder side unfreezes.

---

### 3.2 AttentionPool (`f_φ`) — the compression network

Maps a variable-length encoder hidden sequence to K fixed latent slots.
Configurable via `pool_type`; all three variants share:

- `d_model` — input/output dimensionality (768 for bart-base)
- `K` — number of latent slots (default 32; 128 for code)
- `n_layers` — number of attention blocks (default 2)
- `n_heads` — attention heads (default 8)
- `d_ae` — output dimensionality (default `d_model`; reduce to shrink EBM space)
- Final linear projection `d_model → d_ae` (identity when equal)

#### "decoder" (default)

Each block: **self-attention over K queries**, then **cross-attention into
the encoder sequence**. Uses `nn.TransformerDecoderLayer` with `norm_first=True`.
K learnable queries (`nn.Parameter`) cross-attend into the encoder output.

Distinct `W_k/W_v` per attention role (self vs cross) — this separation is
what makes it train better than the combined-MHA "resampler" on smaller
datasets. Empirically reaches loss ≈ 1.45 on GSM8K-full.

#### "resampler"

Flamingo-style Perceiver Resampler, matching the LD4LG specification. One
combined MHA per block where `kv = concat([Z; E(w)])` — the learnable latents
and encoder positions share `W_k/W_v`.

Two fixes that were missing in the first attempt:

1. **Shared LayerNorm on the KV path.** A single `LayerNorm` applied to the
   concatenated `[Z; E(w)]` means latents and encoder positions are normalized
   with the same gain/bias. Separate LNs give the two halves different stats,
   and the shared projections can't fit both.

2. **Gated residuals with α=0 init.** `Z = Z + tanh(α) · MHA(...)`. At step 0
   `tanh(0) = 0` so each block is identity — no random transforms that could
   swamp the pool's input-conditional signal. The optimizer grows α as the
   block becomes useful.

#### "conv"

No learnable queries. A **depthwise-separable 1-D conv** scans the encoder
sequence (one length-`conv_kernel` kernel per channel via `groups=d_model`,
then a 1×1 pointwise mix), `adaptive_avg_pool1d` collapses L → K, and a
self-attention `TransformerEncoder` refines the K slots.

No cross-attention, no trailing LayerNorm — the K slots come directly from
the conv + pool, not from learnable queries attending the encoder.

---

### 3.3 ReconstructionNet (`f_ψ`) — the shaping network

A self-attention TransformerEncoder over the K latent slots, with an
up-projection `d_ae → d_model` when `d_ae < d_model`.

#### Identity initialization

Every transformer block's residual paths are zero-initialized:
```python
nn.init.zeros_(layer.self_attn.out_proj.weight)
nn.init.zeros_(layer.self_attn.out_proj.bias)
nn.init.zeros_(layer.linear2.weight)
nn.init.zeros_(layer.linear2.bias)
```
At step 0, `ReconstructionNet(z) == z` (after up-projection). Without this,
the recon's random noise swamps the pool's signal and the frozen decoder
falls back to unigram statistics. The optimizer grows non-zero residuals as
training proceeds.

#### No trailing LayerNorm

The pool already applies a final `LayerNorm`. Recon's internal blocks use
`norm_first=True`. Stacking a third trailing LN over-normalizes and squashes
the magnitude signal the frozen decoder's cross-attention was trained to read.

---

### 3.4 BART Decoder

Standard BART decoder. In the default (`train_decoder=False`) mode it is
frozen and runs in `eval()` mode. The decoder receives the recon's output
as cross-attention key/value (via `BaseModelOutput(last_hidden_state=z_dec)`)
with an all-ones attention mask (K slots, no padding).

When `train_decoder=True`, the decoder (and tied LM head) are unfrozen and
trained jointly with Pool + Recon. The encoder stays frozen.

---

### 3.5 PointerGenerator (`use_copy=True`)

A pointer-generator network (See et al., 2017) that controls the decoder's
output distribution:

```
attn_t  = softmax(q(h_t) · k(E_src)ᵀ / √d_attn)    over source positions
ctx_t   = attn_t · E_src                            copy context
p_gen   = σ(W · [ctx_t ; h_t ; x_t] + b) ∈ (0,1)    the gate
P(w)    = p_gen · P_vocab(w) + (1-p_gen) · Σ_{i: src_i=w} attn_{t,i}
```

At each decode step the model can either **generate** from BART's vocabulary
or **copy** from the source sequence. This matters for code reconstruction
where identifiers and literals must come back verbatim — copying makes that
explicit instead of forcing the softmax to memorize every rare token.

The gate's input is the `[ctx; dec_hidden; dec_input_emb]` concatenation
(3·d_model), matching the checkpoint's `gate.0` shape `(1, 3·768)`.

Active in `decode_loss*` (source = target text being reconstructed) and
`decode(..., src_texts=...)`. Latent-only `decode` (e.g. EBM sampling)
skips it.

---

### 3.6 `train_decoder=True`

Unfreezes the BART decoder (and tied LM head) for joint fine-tuning with
Pool + Recon. Used by the conv-pool checkpoints. The encoder remains frozen.
Decoder weights are saved under the checkpoint's `decoder` key.

---

## 4. Training (Milestone 1)

### Objective

Token-level cross-entropy reconstruction loss:

```
L = CE(BART_Dec(Recon(Pool(BART_Enc(A)))), A)
```

When `use_copy=True`, the loss is the negative log-likelihood under the
pointer-generator's copy-augmented distribution instead of plain CE.

### Dataset

**OpenWebText** — a broad natural-language corpus. The AE is deliberately
**never trained on MBPP or HumanEval**, preserving the §3.2 anchoring
property: the latent image of the gold answer is fixed and the EBM cannot
exploit AE-learned distribution bias.

### Metrics (per eval cycle)

| Metric | What it measures |
|--------|-----------------|
| `owt_recon_cer` | Char error rate on held-out OpenWebText. Smooth signal; 0 = exact byte-level reconstruction. |
| `mbpp_cer` | Same CER, on MBPP canonical solutions. Proxy for code recon before pass-rate moves. |
| `mbpp_pass` | Fraction of MBPP examples where the round-tripped solution passes its `test_list`. |
| `humaneval_cer` | Same CER, on HumanEval prompt+canonical. |
| `humaneval_pass` | Fraction of HumanEval examples where the decoded function passes `check(entry_point)`. |

CER gives a smooth signal early in training; pass-rate is the downstream-relevant
metric. A function can have low CER (few chars off) but still fail execution if
the drift hit an operator or identifier.

### Milestone 1 gate (gensis.md §7)

- MBPP test pass-rate ≥ 80%
- HumanEval pass-rate ≥ 80%
- OpenWebText byte-exact reconstruction ≥ 50%

If the AE clears OWT but fails both code bars, the OWT-trained latent space
doesn't transfer to Python — recovery options are (a) add a code slice to AE
pretraining, or (b) swap BART for CodeBERT/CodeT5.

---

## 5. API Reference

All methods live on `FrozenBartAutoencoder` in `ired/model/autoencoder.py`.

### 5.1 Construction

```python
ae = FrozenBartAutoencoder(
    model_name="facebook/bart-base",  # pretrained BART
    k=32,                              # number of latent slots
    pool_layers=2,                     # attention blocks in Pool
    pool_heads=8,                      # attention heads
    pool_type="decoder",               # "decoder" | "resampler" | "conv"
    d_ae=None,                         # latent dim (default d_model=768)
    recon_layers=2,                    # attention blocks in Recon
    recon_heads=8,
    use_copy=False,                    # add PointerGenerator
    train_decoder=False,               # fine-tune BART decoder
)
```

### 5.2 Encoding

```python
# Text → latents (with tokenization)
z = ae.encode_to_latents(texts, device, max_length=128)
# → (B, K, d_ae)

# Pre-tokenized variant (skips CPU tokenizer)
z = ae.encode_to_latents_from_ids(input_ids, attention_mask)
# → (B, K, d_ae)
```

Both methods keep the BART encoder frozen (`torch.no_grad()`). Pool weights
receive gradients in `encode_to_latents_from_ids` (the encoder output is
detached before the pool).

### 5.3 Decoding (loss)

```python
# Scalar reconstruction loss
loss = ae.decode_loss(z, target_texts, device, max_length=128)
# → scalar tensor

# Pre-tokenized variant
loss = ae.decode_loss_from_ids(z, label_ids)
# → scalar tensor

# Per-example loss (for exposure-bias diagnostic)
ce_per_example = ae.decode_loss_per_example(z, target_texts, device, max_length=128)
# → (B,) tensor
```

`decode_loss` runs:
1. `ReconstructionNet(z)` → `z_dec` of shape `(B, K, d_model)`
2. Tokenize target texts
3. If `use_copy`: compute copy-augmented log-probs via PointerGenerator, return NLL
4. Otherwise: feed `z_dec` as encoder output to `bart(...)` with teacher-forced labels

### 5.4 Decoding (generation)

```python
# Greedy
texts = ae.decode(z, max_length=128, num_beams=1)
# → list[str]

# Beam search
texts = ae.decode(z, max_length=128, num_beams=4)

# With copy (requires use_copy=True)
texts = ae.decode(z, max_length=128, src_texts=source_texts)
```

When `use_copy=True` and `src_texts` is supplied, runs a greedy copy-aware
loop. Otherwise delegates to `bart.generate()`.

### 5.5 Checkpoint I/O

```python
# Save
sd = ae.state_dict_ae()
# → {"pool": ..., "recon": ..., "d_ae": ..., "pool_type": ...,
#     "pointer_generator": ... (if use_copy), "decoder": ... (if train_decoder)}

# Load
ae.load_ae(sd)
# Validates d_ae, pool_type, and pointer_generator presence match the current AE config.
```

See §7 for the full checkpoint format.

---

## 6. Design Decisions

### 6.1 LD4LG compression/reconstruction split

The original single-pool design collapsed both roles into one module:

```
text → Enc → Pool → Dec → text
```

This plateaued at loss ≈ 1.45. The diagnosis: the pool was wearing two hats —
compressing information AND formatting it for the decoder — and the objectives
partially conflict.

The split gives each subnetwork a single job:

- **Pool (`f_φ`)**: compress variable-length encoder sequence → K latent slots.
  Wants information density (sparse, high-entropy, unique per input).
- **Recon (`f_ψ`)**: shape K slots into something the frozen decoder's
  cross-attention can read. Wants statistical similarity to natural BART
  encoder outputs (smoother, more redundant, on-manifold).

This is the standard separation-of-concerns win that LD4LG (after Flamingo's
Perceiver Resampler) validated empirically.

### 6.2 Identity initialization

Both `PerceiverResamplerLayer` (gated residuals, α=0) and `ReconstructionNet`
(zeroed `out_proj` + `linear2`) are identity at step 0. Two reasons:

1. **Preserve the pool's signal.** At random init, the recon's noise would
   swamp the pool's input-conditional output, and the frozen decoder would
   fall back to unigram statistics. Identity-init guarantees the decoder
   sees the pool's actual output from step 1.

2. **Let the optimizer choose what to learn.** Each block starts contributing
   zero — the optimizer only grows non-zero residuals when they reduce loss.
   Nothing is ever un-learned.

Validated: at random init, `decode_loss(with_recon) == decode_loss(without_recon)`
to 6 decimal places.

### 6.3 Pool type comparison

Empirical comparison on GSM8K:

| Pool type | Mechanism | GSM8K loss @ 5k steps | Best for |
|-----------|-----------|----------------------|----------|
| `decoder` | Separate self + cross attn, distinct W_k/W_v | ~0.38 | Smaller datasets, faster convergence |
| `resampler` | Combined MHA, shared W_k/W_v, gated residuals | ~2.18 | Large-data regime (LD4LG's original target) |
| `conv` | Depthwise conv + adaptive pool, no queries | — | Code (conv-pool checkpoints with copy) |

The "decoder" type's separated attention gives each role its own projection
space, which matters when data is limited. The "resampler" is architecturally
cleaner but needs more data to learn the shared projections.

### 6.4 `d_ae` — latent dimensionality

When `d_ae = d_model` (default), the projection is `nn.Identity()` and the
EBM diffuses in the full 768-dimensional space.

Setting `d_ae < d_model` (e.g. 128) shrinks the EBM's diffusion space by
the same factor — the single most impactful lever for Milestone 2 tractability.
The recon's up-projection `d_ae → d_model` handles the dimensionality gap.

### 6.5 PointerGenerator

For code reconstruction, identifiers and literals appear verbatim in the
source and must come back out exactly. The PointerGenerator makes this explicit:
instead of forcing the softmax to memorize every rare token, the model can
copy from the source via attention. See §3.5 for the formulation.

### 6.6 `train_decoder`

Fine-tuning the decoder alongside Pool + Recon lets the decode side adapt to
the latent distribution. This helps when the Pool/Recon output distribution
drifts away from natural BART encoder statistics (which frozen decoder
cross-attention expects). The conv-pool checkpoints use this.

---

## 7. Checkpoint Format

Saved via `state_dict_ae()`, loaded via `load_ae()`.

```python
{
    "pool":       state_dict,     # AttentionPool weights
    "recon":      state_dict,     # ReconstructionNet weights
    "d_ae":       int,            # latent dimensionality (for validation)
    "pool_type":  str,            # "decoder" | "resampler" | "conv" (for validation)
    # Optional:
    "pointer_generator":  state_dict,     # present iff use_copy=True
    "decoder":    state_dict,     # present iff train_decoder=True (BART decoder weights)
}
```

`load_ae()` validates that the checkpoint's `d_ae`, `pool_type`, and
`pointer_generator` presence match the current AE config. Mismatch raises `ValueError`
with a message telling you which constructor arg to fix.

Pre-`pool_type` checkpoints default to `"decoder"` for back-compat.
The fine-tuned decoder weights (`"decoder"` key) are loaded regardless of
`train_decoder` so reconstruction matches.

---

## 8. Trained parameter counts

| Component | Parameters (bart-base, K=32, d_ae=768, 2 layers) |
|-----------|---------------------------------------------------|
| AttentionPool (decoder) | ~9.5M |
| AttentionPool (resampler) | ~7.1M |
| AttentionPool (conv) | ~4.7M |
| ReconstructionNet | ~9.4M |
| PointerGenerator | ~1.2M |
| BART Decoder (train_decoder=True) | ~62M |
| **Total (default)** | **~18.9M trainable** |
| **Total (with recon)** | **~28.4M trainable** |

BART encoder is always frozen (~62M parameters, zero gradients).

---

## 9. Files

| File | Role |
|------|------|
| `ired/model/autoencoder.py` | Module definition — all classes |
| `ired/train_autoencoder.py` | Training script (Milestone 1) |
| `checkpoints/ae_*/` | Saved AE checkpoints |
| `gensis.md` §3.2, §7 | Architecture rationale, Milestone 1 gate |

---

## 10. Common issues

**Reconstruction loss won't go below ~10.**
The pool + recon at random init produce near-random decoder output. This is
expected at step 0. If loss stays above 5 after the first eval cycle,
check: (a) BART encoder is actually frozen, (b) gradient is flowing to pool
and recon parameters, (c) learning rate isn't too low (default `1e-4`).

**MBPP pass-rate is high but HumanEval pass-rate is near zero.**
HumanEval prompts are longer (signature + docstring + examples → up to 512
tokens). The AE may be losing information for long inputs. Increase
`max_length` or `K`.

**Checkpoint load fails with "d_ae mismatch".**
The checkpoint was saved with a different `d_ae`. Reconstruct the AE with
the matching `d_ae` value (printed in the error message).

**Checkpoint load fails with "pool_type mismatch".**
Pre-`pool_type` checkpoints default to `"decoder"`. If you're loading an
older checkpoint into a `"resampler"` or `"conv"` AE, reconstruct with
`pool_type="decoder"`.
