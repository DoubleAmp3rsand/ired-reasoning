# Implementation pitfalls (post-mortem)

> Split out of `gensis.md`. These are concrete bugs and infelicities that surfaced
> when building the §7 prototype against the IRED reference code. None of them
> invalidate the proposal — but every one is a place where a default that worked
> for IRED's small tabular tasks fails silently in the latent-LLM setting. Listed
> in one place so future implementers don't re-discover them.

## 1. Diffusion β schedule blows up at small T

`linear_beta_schedule` in the IRED reference (`denoising_diffusion_pytorch_1d.py`) is

```python
scale = 1000.0 / timesteps
betas = linspace(scale * 1e-4, scale * 0.02, timesteps)
```

This assumes `timesteps ~ 1000`. At our `T=10`:

```
betas = [0.01, 0.231, 0.452, 0.674, 0.895, 1.116, 1.338, 1.559, 1.781, 2.000]
alphas = 1 - betas        → negative for t ≥ 5
α̅ = cumprod(alphas)       → negative
sqrt(α̅)                   → NaN at t ∈ {5, 7, 9}
```

Any batch that samples one of those t values produces NaN loss; the first
`opt.step()` poisons the EBM permanently. With B=16 and uniform t, the
probability of a clean batch is `(0.7)^16 ≈ 0.3%`, so essentially every step
NaNs from step 1 — but it can take 50–350 logged steps to *notice* depending on
what `--log-every` is set to and whether the NaN propagates to a visible stat
first.

**Fix:** clip β to ≤ 0.999 (matching the cosine schedule's bound), and assert
`(alphas_cumprod > 0).all()` at construction so future bad (schedule, T) combos
fail loudly.

**Better fix:** for small T, use `cosine` directly. The clipped-linear schedule
at T=10 has β ≥ 0.999 for half its range, so the EBM only sees variance
interpolations near t=0; it never gets useful gradient at intermediate noise
levels.

## 2. EBM head magnitude inflates unboundedly without weight decay

IRED's reference setup uses an MLP energy net on small (~tens-of-dims) tabular
data with no weight decay. We inherited the no-weight-decay default. In the
latent-LLM setting the head is a `Linear(d_model=768, d_model=768)` and the
energy is the squared sum over `K · d_model = 24576` dims — so the absolute
energy scale is enormous and grows freely.

Observed in a 5000-step run:

| step | e_real | e_fake | contrast | nce loss |
|---:|---:|---:|---:|---:|
| 100 | 1,413 | 2,896 | 2.05× | 0.19 |
| 5000 | 131,601 | 134,133 | **1.02×** | 0.20 |

The NCE loss number is unchanged but the *relative* contrast collapsed from 2× to
2%. Cross-entropy is scale-equivariant in the logit difference, so the loss
minimum can be reached by either (a) learning real-vs-fake discrimination or
(b) growing both energies in proportion. Once the head weights start growing,
gradient descent prefers (b) — it's an easier optimization direction.

Downstream consequence: `opt_step` bad-step rejection becomes inert (a 2%
contrast is dominated by noise in `E(z_new)` vs `E(z)`), and ∇E grows with the
head, breaking DDPM's reverse process that expects ε̂ at noise-scale (std ≈ 1).

**Fix:** `--weight-decay 0.01` on `AdamW` keeps the head norm bounded. Should be
the default; the current `0.0` is a footgun.

**Diagnostic:** the `eps_scale = ||ε̂|| / ||noise||` stat added to per-step
logging surfaces this in one number — when head inflates, `eps_scale` drifts
upward in lockstep.

## 3. IRED's clamping bounds assume `[-1, 1]` data

IRED hardcodes `x_start_clamp = 2` and `envelope_sf = 2` (the per-t
`±sqrt(α̅_t)·sf` clamp inside `opt_step` and `p_sample_loop`). Both assume input
data normalized to `[-1, 1]` — true for IRED's tabular tasks, false for
LayerNorm'd T5 latents whose element std is ~1 and bulk lives in `[-3, 3]`.

Applied verbatim, these clamps crush noisy `z_t` to ~0 at large t (when
`sqrt(α̅_t)` is small), turning the reverse process into noise injection plus
clamping rather than denoising.

**Fix:** `x_start_clamp = 5.0` (loose enough to not bite typical T5 latents),
`envelope_sf = None` (disabled by default; pass an explicit float to re-enable
the IRED-style behavior). Both are CLI-configurable.

**Lesson:** any time you transplant IRED-style code to a new data domain, audit
every literal numeric constant against the new data's empirical range. The
original constants are a domain-specific tuning, not a general default.

## 4. ε̂ scale drift is invisible to `mse(ε̂, ε)`

The denoising loss `mse(ε̂, ε)` is dominated by direction agreement — a 2× scale
error contributes only `(2 - 1)^2 = 1` per-element to MSE, the same as a
perpendicular unit-noise direction. So `mse` can look small (~0.02) while
`||ε̂||` is drifting away from `||ε||`. DDPM's `predict_start_from_noise` and
`q_posterior` math assume ε̂ is at noise-scale; if it isn't, the reverse process
inflates or deflates `z` magnitude regardless of how good the *direction* is.

In our broken 5000-step run, the symptom was `std(z_sampled) = 2.75` vs
`std(z_a) = 1.00` after sampling — magnitude inflated almost 3× while
`mse(ε̂, ε)` was 0.017 at train time.

**Fix:** log `eps_scale = ||ε̂||_2 / ||ε||_2` per training step. It's free (one
no_grad norm) and is the earliest detector of the §2 head-inflation problem.

## 5. PyTorch fast SDPA kernels don't implement double-backward

IRED's training loss takes MSE against ε̂ = ∇<sub>z</sub>E, so the outer
`loss.backward()` differentiates *through* an inner `autograd.grad(...)`. This is
a second-order autograd path through every attention call inside the EBM.
PyTorch's default flash and mem-efficient SDPA kernels don't implement the
backward-of-backward; on CPU you get

```
RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward
is not implemented
```

and on GPU you get a similar (slightly less obvious) failure depending on
hardware.

**Fix:** force the math kernel inside the EBM's attention,

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel([SDPBackend.MATH]):
    h = self.encoder(x)
```

This costs throughput vs flash attention, but is the only kernel that supports
double-backward. A nicer long-term solution is to switch the EBM to a custom
transformer that uses an explicit attention implementation, but the kernel
switch is the smallest fix.

## 6. LD4LG-literal Perceiver Resampler did not work; reverted to separated attention

The naïve compression-network implementation uses `nn.TransformerLatentDecLayer`
(separate self-attention then cross-attention). That plateaued Milestone 1 at
loss ≈ 1.45 with input-conditional outputs — the model was clearly using the
latent code but couldn't push past a certain reconstruction fidelity. The
hypothesis was that LD4LG's combined-MHA Perceiver Resampler block,

```
Z = Z + MHA(q = Z, kv = [Z; E(w)])
Z = Z + FF(Z)
```

would let each head dynamically allocate attention budget between latents and
encoder, vs. the separated form's strict role assignment. I implemented it as a
custom `PerceiverResamplerLayer`.

**It made things dramatically worse.** Milestone 1 with the combined-MHA pool
plateaued at loss ≈ 7.6 across multiple LR settings (3e-4, 1e-4, and a 1000-step
warmup from 0 to 1e-4 — the loss curves were *identical*, ruling out
optimization). Predictions collapsed to a unigram-like mode: every input decoded
to the same `>>>>>>>>` token stream (`>>` being a common substring in GSM8K's
`<<...>>` calculation markers).

**The most plausible mechanism:** the combined form forces a single `W_k, W_v`
pair to project *both* the latent slots *and* the encoder hidden states. These
two roles want different projections — latents are LayerNorm'd identity-shared
queries; encoder hidden states are content-bearing input-conditional vectors. The
separated form has two distinct projection pairs (one for self-attn, one for
cross-attn) so each role specializes from the first SGD step. The combined form
has to either compromise or get stuck — and on the GSM8K-CoT task with frozen T5,
it gets stuck.

**Diagnostic that pinpointed it:** with the identity-initialized
ReconstructionNet producing bit-exact identity at step 0, recon could not be the
source of collapse at the initial eval. Yet the collapse was visible immediately.
By elimination, the pool was the problem. LR sweeps producing identical loss
curves confirmed it wasn't optimization.

**Fix:** reverted `AttentionPool` to `nn.TransformerLatentDecLayer`. Keeps the
LD4LG-style explicit `f_φ` / `f_ψ` separation and the `d_ae` projection, but uses
the separated-attention pool that has empirical evidence of working on this task.

**Lesson:** when a paper's formula doesn't translate to your setting, it doesn't
mean you implemented it wrong — sometimes the paper's design is calibrated for a
different (data, encoder, decoder, scale) regime and doesn't transfer. Empirical
falsification (matched loss curves across LR sweeps) is faster than mechanistic
debate. The "literal-fidelity vs. what-works" tradeoff was decided by data, not
by re-reading the paper.

---

## Quick reference: defaults to override when porting IRED to latents

| Default in IRED | Safe default for latents | Why |
|---|---|---|
| `linear` β schedule | `cosine` | linear at small T saturates or NaNs |
| `weight_decay = 0` | `weight_decay = 0.01` | bounds head magnitude, preserves NCE contrast |
| `x_start_clamp = 2.0` | `x_start_clamp = 5.0` | T5 latents have wider element range |
| `envelope_sf = 2.0` | `envelope_sf = None` | the per-t envelope crushes z at large t |
| default SDPA kernel | `SDPBackend.MATH` | double-backward not implemented in flash kernels |
| log `mse, nce` only | also log `eps_scale, e_real/e_fake ratio` | early detection of scale collapse |
