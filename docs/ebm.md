# Energy-Based Model (EBM) — Reference

The EBM is the **thinking module**. It learns a scalar energy `E(z_q, z_t, t)`
over the autoencoder's latent space, and IRED-style inference iteratively
descends that energy to turn a question latent `z_q` into an answer latent `z_0`.
The answer latent is then decoded back to text by the (frozen) autoencoder.

Everything here operates **inside the latent space the AE defines** — see
`docs/autoencoder.md`. The AE is trained first and frozen; only the EBM updates
during Milestone 2 (`gensis.md` §9, §10).

---

## 1. Where it fits

```
   Q ──► BART Enc ──► Pool (f_φ) ──►  z_q ∈ R^(K×d_ae)
                                       │
                                       │   conditioning
                                       ▼
                z_T ~ N(0,I)  ──►  EBM(z_q, z, t)  ──►  z_0
                (T outer DDPM steps × N inner opt_steps)
                                       │
                                       ▼
                z_0  ──►  Recon (f_ψ) ──► BART Dec ──►  A
```

The EBM never sees text. It conditions on the question latent `z_q` and denoises
a noisy answer latent `z_t` toward the clean answer latent
`z_a = f_φ(Enc(A))`. The AE's encoder/pool/recon/decoder are all frozen here.

---

## 2. Why an energy model (IRED)

A standard diffusion model predicts noise `ε̂ = model(z_t, t)` directly. IRED
instead parametrizes the noise prediction as the **gradient of a scalar energy**:

```
ε̂  =  ∇_{z_t} E(z_q, z_t, t)
```

Two consequences:

1. **A navigable landscape, not just a vector field.** Because `ε̂` is a true
   gradient, `E` is a scalar potential whose value is comparable across points.
   Inference can do gradient *descent* on `E` and **reject any step that raises
   the energy** (the `opt_step` bad-step rejection in §5) — the move that turns
   diffusion into iterative *reasoning*.

2. **The energy scale must be calibrated.** Bad-step rejection is only
   meaningful if a lower energy genuinely means a better latent. The MSE term
   alone fixes `∇E` but leaves the absolute scale of `E` free, so the NCE term
   (and the optional anchors in §4) exist to pin it down.

`gensis.md` §10 is the source for the diffusion / inner-loop design; the code
adapts IRED's continuous (matrix-addition) `GaussianDiffusion1D` into latent
space.

---

## 3. Architecture

```
            ┌──────────── DiffusionWrapper ────────────┐
            │   z_in = z_t.detach().requires_grad_()    │
   z_q, ───►│   E = EnergyTransformer(z_q, z_in, t)     │──► ε̂ = ∂E/∂z_t
   z_t, t   │   ε̂ = ∂E/∂z_in  (create_graph=training)   │   (and/or E)
            └───────────────────────────────────────────┘
                              ▲
                              │
            ┌──────── GaussianLatentDiffusion ─────────┐
            │  schedules, q_sample, opt_step,          │
            │  p_sample, sample, p_losses              │
            └───────────────────────────────────────────┘
```

| Component | Role |
|-----------|------|
| **EnergyTransformer** | Transformer that maps `(z_q, z_t, t)` → scalar energy `E` |
| **DiffusionWrapper** | Exposes `∇_{z_t} E` (and/or `E`) via autograd |
| **GaussianLatentDiffusion** | DDPM schedules, training losses, inner-loop refinement, sampling |

All three live in `ired/model/energy_net.py` (first two) and
`ired/model/diffusion.py` (the third).

---

### 3.1 EnergyTransformer

A 4-layer transformer encoder that produces one scalar per batch element.

**Token sequence** fed to self-attention:

```
[ time_token ]  +  [ z_q + pos_emb + type=0 ]  +  [ z_t + pos_emb + type=1 ]
   (1 token)            (K tokens)                     (K tokens)
                                                         → (B, 1 + 2K, d)
```

- `type_emb` (2 entries) distinguishes the conditioning slots (`z_q`, type 0)
  from the variable slots (`z_t`, type 1).
- `pos_emb` (K entries) is the per-slot position embedding, shared between the
  `z_q` and `z_t` blocks.
- The timestep `t` becomes a single prepended token via a sinusoidal embedding
  + 2-layer MLP.

**Scalar energy.** Four `TransformerEncoderLayer`s (`norm_first=True`, GELU),
a final LayerNorm, then a `Linear` head over the **`z_t` slots only**, and a
squared-sum reduction:

```
E = || head(h[z_t slots]) ||²   summed over all K·d dims     # (B, 1)
```

This matches the squared-sum energy trick in the original IRED MLP.

**Double-backward.** The training loss MSEs against `∇E`, so `loss.backward()`
must differentiate *through* an inner `autograd.grad`. The flash / mem-efficient
SDPA kernels don't implement double-backward, so the encoder forward is wrapped
in `sdpa_kernel([SDPBackend.MATH])` to force the math kernel (CPU and CUDA).

---

### 3.2 DiffusionWrapper

Thin module that turns the energy into a noise prediction:

```python
z_in = z_t.detach().requires_grad_(True)
E    = ebm(z_q, z_in, t)
ε̂    = autograd.grad(E.sum(), z_in, create_graph=self.training)[0]
```

- `create_graph` defaults to `self.training`: during training the gradient
  itself carries a graph so the denoising MSE can backprop into the EBM weights
  (the double-backward above); during inference / `opt_step` the higher-order
  graph is dropped for speed.
- Flags: `return_energy=True` returns `E` instead of `ε̂`; `return_both=True`
  returns `(E, ε̂)` in one pass (used by `opt_step`).

---

### 3.3 GaussianLatentDiffusion

Owns the DDPM schedules and all of training + sampling. Key construction knobs
(defaults in parentheses):

| Knob | Meaning |
|------|---------|
| `timesteps` (10) | number of outer DDPM steps `T` |
| `beta_schedule` ("linear") | `linear` or `cosine`; validated so `alpha_cumprod > 0` |
| `opt_step_size` (1.0) | inner-loop gradient-descent step size (per-`t` buffer) |
| `loss_scale` (1.0) | weight `λ` on the NCE term |
| `objective` ("pred_noise") | `pred_noise` (ε) or `pred_x0` |
| `x_start_clamp` (5.0) | clamp predicted `x_0` to ±this; `None` disables |
| `envelope_sf` (None) | if set, clamp `z` to `±√ᾱ_t · sf` per step |
| `decoder_aux_weight` / `rand_neg_weight` / `gen_neg_weight` | optional loss terms (§4) |

**Latent normalization.** AE-native latents are anisotropic, so the diffusion
noise schedule (designed for `N(0, I)`) is miscalibrated against them. A per-dim
shift/scale (`latent_mu`, `latent_sigma`) maps AE latents → `N(0, I)`-like space
before any diffusion math, and the inverse is applied to anything that leaves
the module (sampled outputs, `x0_hat` fed to a decode-grounded term). The stats
are computed once at the start of training (`set_latent_stats`) and stored as
**buffers**, so they ship with the checkpoint and `sample.py` picks them up
automatically. This is the LD4LG §4.1 / Stable-Diffusion-`0.18215` trick.

---

## 4. Training (Milestone 2)

Per step: encode question + answer with the frozen AE (`no_grad`), then
`loss, stats = diffusion(z_q, z_a, gold_texts=...)`, `loss.backward()`,
`opt.step()`. Only the EBM updates. Gradients are clipped to norm 1.0.

The total loss is assembled in `GaussianLatentDiffusion.p_losses`.

### 4.1 Always-on terms

| Term | Form | Purpose |
|------|------|---------|
| **MSE** | `‖ε̂ − ε‖²` (per-`t` weighted) | denoising regression — fixes `∇E` |
| **NCE** | `softmax`-CE pushing `E(z_a) < E(neg)` | calibrates the absolute energy scale |

The NCE **hard negative** is mined *geometrically*: heavy-noise the clean target
(`3·noise`), run 2 `opt_step` refinements, un-scale to an `x_0` estimate, clamp,
re-noise to the data noise level, and require the clean sample to have lower
energy at that `t`. A **scale-drift monitor** (`eps_scale = ‖ε̂‖/‖ε‖`) is logged
because DDPM math breaks if the predicted-noise norm drifts even when the MSE
looks small.

### 4.2 Optional anchors / grounding (off by default)

| Term | Flag | What it does |
|------|------|--------------|
| **rand-neg** | `--rand-neg-weight` | `softplus(E(z_a) − E(N(0,I)) + margin)`. Stops the EBM leaving random latents at the same energy as gold. Gate to low `t` (`--rand-neg-t-max`) — at high `t`, `q(z_t\|z_a)` and `N(0,I)` overlap and the anchor fights ε prediction. |
| **decoder-aux** | `--decoder-aux-weight` | `CE(decode(x0_hat), gold)` on low-`t` samples (`--decoder-aux-t-max`). Grounds the one-step denoising prediction in the decoder. |
| **gen-neg** | `--gen-neg-weight` | **Generator-grounded negative** (see §4.3). |

### 4.3 The generator-grounded negative

**The problem.** MSE and NCE shape the energy field using only the *geometry* of
the latent space — "good" = "close to `z_a`". Neither term knows what the
decoder / point-generator actually emits. So the energy minimum that `opt_step`
descends into can sit *off* the decoder-decodable manifold: energy keeps
dropping while decode quality doesn't. This is *"the energy manifold does not
know the shape of the generator network,"* and it shows up in the eval as a
large exposure-bias gap (§4.5).

**The fix.** Decode the NCE-mined latent's clean estimate through the
**inference (no-copy) generator** and, where that decode is wrong, treat the
latent as a validated negative:

```
ce  = decode_loss_per_example(refined_x0, gold, copy=False)   # detached gate
wrong = (ce > gen_neg_ce_thresh)
loss_gen = mean_over_wrong( softplus(E(z_a) − E(refined) + gen_neg_margin) )
```

- The decode/CE is a **detached gate only** — it decides *which* mined latents
  count as negatives. The learning signal comes solely from the energy margin,
  which reuses the energy the NCE term already computed. No gradient flows
  through the decoder or the inner loop, so the term is cheap.
- Gated to low `t` (`--gen-neg-t-max`, default 3) because the clean `x_0`
  estimate is only reliable at low noise, where a decode is meaningful.
- Stats: `gen` (loss), `gen_ce` (mean CE), `gen_wrong`, `gen_n`.

**No-copy is mandatory.** Both decode-grounded terms call `decode_loss*` with
`copy=False`. With the point-generator's copy path on, the source is the gold
answer itself, so the gate can drive `p_gen → 0` and copy gold verbatim — CE
collapses toward zero regardless of latent quality and the EBM gets no usable
signal. `copy=False` matches `ae.decode(z, src_texts=None)`, the path actually
run at EBM inference.

### 4.4 Dataset

Trains on **MBPP** (`--mbpp-config full` = 974 ex / `sanitized` = 427). Evaluates
on **MBPP test + HumanEval** (held out). HumanEval is never trained on. The AE
was never trained on either (`gensis.md` §3.2 anchoring), so the gold-answer
latent is a fixed target the EBM cannot exploit via AE distribution bias.

### 4.5 Metrics (per eval cycle, `eval_corpus`)

End-to-end: sample `z` from `z_q`, decode, execute the decoded code against each
example's tests.

| Metric | Meaning |
|--------|---------|
| `acc` | pass-rate with full `inner_steps` of refinement |
| `acc_inner0` | pass-rate with `inner_steps=0` — isolates the gain from `opt_step` |
| `ae_acc` | pass-rate of `decode(encode(answer))` — the Milestone-1 AE ceiling |
| `mse_z`, `corr_z` | L2 / cosine of sampled latent vs `z_a` |
| `std_za`, `std_zs` | element-std of gold vs sampled latents (scale drift) |
| `ce_pass` / `ce_fail` / `eb_gap` | teacher-forced CE on the sampled latent, split by free-running pass/fail. A large `eb_gap` (`ce_fail − ce_pass`) means latent quality tracks correctness; a small gap with both CEs low means the latent points the decoder at gold but free-running decode still flakes (AR leakage). |

`ae_acc < 0.5` on either corpus means the **AE is the bottleneck**, not the EBM.

---

## 5. Sampling / inference

```python
z_0 = diffusion.sample(z_q, inner_steps=5)   # AE-native in, AE-native out
```

The loop (`sample`): normalize `z_q`, start from `z ~ N(0, I)`, then for each of
the `T` timesteps run **one outer DDPM step** (`p_sample`) followed by
`inner_steps` of **IRED refinement** (`opt_step`), and finally denormalize.

### 5.1 `opt_step` — bad-step rejection (the heart of IRED)

```python
for _ in range(step):
    E, grad = model(z_q, z, t, return_both=True)
    z_new   = z − step_size · grad
    z_new   = clamp_envelope(z_new, t)
    E_new   = model(z_q, z_new, t, return_energy=True)
    z       = where(E_new > E, z, z_new)    # reject any step that raises energy
```

Per sample, a refinement step is **kept only if it lowers the energy**. Because
`E` is a calibrated scalar (§2), this is a meaningful accept/reject — it is what
distinguishes IRED's iterative reasoning from plain diffusion sampling.

### 5.2 `p_sample` — one DDPM step

Standard DDPM posterior: predict `ε̂`, recover `x_0` (clamped to
`±x_start_clamp`), compute the posterior mean/variance, and sample `z_{t-1}`.

---

## 6. API Reference

`EnergyTransformer`, `DiffusionWrapper` in `ired/model/energy_net.py`;
`GaussianLatentDiffusion` in `ired/model/diffusion.py`.

### 6.1 Construction

```python
ebm = EnergyTransformer(d_model=d_ae, k=K, n_layers=4, n_heads=8, dim_ff_mult=4)
wrapper = DiffusionWrapper(ebm)
diffusion = GaussianLatentDiffusion(
    model=wrapper,
    latent_shape=(K, d_ae),
    timesteps=10,
    beta_schedule="linear",
    opt_step_size=1.0,
    loss_scale=1.0,                 # NCE weight
    supervise_energy_landscape=True,  # turn NCE off via --no-nce
    x_start_clamp=5.0,
    decoder_aux_weight=0.0,
    rand_neg_weight=0.0,
    gen_neg_weight=0.0, gen_neg_margin=1.0, gen_neg_t_max=3, gen_neg_ce_thresh=0.5,
)
```

### 6.2 Latent stats (call once before training)

```python
diffusion.set_latent_stats(mu, sigma)   # 1-D tensors of length d_ae; stored as buffers
```

### 6.3 Decode-grounded callbacks

```python
diffusion.set_decoder_loss_fn(fn)   # fn(x0_hat, texts) -> scalar; needed if decoder_aux_weight>0
diffusion.set_gen_ce_fn(fn)         # fn(z_native, texts) -> (B',) no-copy CE; needed if gen_neg_weight>0
```

### 6.4 Forward / sample

```python
loss, stats = diffusion(z_q, z_a, gold_texts=...)   # training; gold only if a decode-grounded term is on
z_0 = diffusion.sample(z_q, inner_steps=5)          # inference (AE-native latents)
eps_hat = wrapper(z_q, z_t, t)                       # ∇_{z_t} E
energy  = wrapper(z_q, z_t, t, return_energy=True)   # scalar E
```

---

## 7. Design Decisions

### 7.1 Why MSE alone is not enough

MSE fixes the *direction* of `∇E` but not the *value* of `E`. Without the NCE
term the energy scale is free, so `opt_step`'s "reject if energy rose" test is
arbitrary. NCE pins the scale by forcing `E(real) < E(mined-negative)`.

### 7.2 Why the geometric hard negative isn't enough — the generator gap

The NCE negative is mined by geometry (noise + opt_step), so the energy field is
calibrated against latent *distance*, not against what the decoder produces. The
**gen-neg** term (§4.3) closes this gap by validating the mined negative with the
actual inference-time generator. Conceptually it ties the energy *minima* to the
decodable region; the existing terms only ensure the minima are *near* `z_a`.

### 7.3 No-copy decode grounding

The decode-grounded terms must use the same generator path as inference. The
point-generator's copy path copies from the gold source during a `decode_loss`,
which collapses CE and severs the signal to the EBM. Both terms therefore force
`copy=False` (§4.3).

### 7.4 Latent normalization as buffers

Storing `latent_mu` / `latent_sigma` as buffers (not recomputed at load time)
guarantees the EBM is sampled in exactly the space it was trained in, and makes
`sample.py` zero-config.

### 7.5 `d_ae` drives EBM cost

The EBM diffuses in `R^(K×d_ae)`. Shrinking `d_ae` (set on the AE) is the single
biggest lever on EBM tractability — see the parameter table in §9 and
`docs/autoencoder.md` §6.4.

---

## 8. Checkpoint Format

Saved each eval cycle to `ebm_step{N}.pt` and `ebm_latest.pt`:

```python
{
    "ebm":          state_dict,   # EnergyTransformer weights
    "opt":          state_dict,   # AdamW state (for --resume)
    "latent_mu":    tensor,       # (d_ae,) normalization shift
    "latent_sigma": tensor,       # (d_ae,) normalization scale
    "config":       vars(args),   # full training config
    "step":         int,
    "mbpp_pass":    float, "humaneval_pass": float,
    "mbpp_ae_pass": float, "humaneval_ae_pass": float,
    "mse_z_mbpp":   float, "mbpp_eb_gap": float, "humaneval_eb_gap": float,
}
```

`--resume` restores `ebm`, `opt`, `latent_mu/sigma`, and `step`, and skips the
one-shot latent-stats recompute. Both `--ae-ckpt` and `--resume` accept local
paths or `hf://<org>/<repo>[@<rev>]/<file>` Hub specs.

---

## 9. Trained parameter counts

The EBM is the **only** trained module (the entire AE is frozen). Counts for
`EnergyTransformer` (4 layers, 8 heads, ff×4):

| `d_ae` | `K` | Parameters |
|--------|-----|------------|
| 768 | 128 | ~30.2M |
| 768 | 256 | ~30.3M |
| 256 | 256 | ~3.4M |
| 128 | 256 | ~0.88M |

Param count is dominated by `d_ae²` (the transformer width), nearly independent
of `K`. Shrinking `d_ae` shrinks both the EBM and its diffusion space.

---

## 10. Files

| File | Role |
|------|------|
| `ired/model/energy_net.py` | `EnergyTransformer`, `DiffusionWrapper` |
| `ired/model/diffusion.py` | `GaussianLatentDiffusion` — schedules, losses, sampling |
| `ired/train_diffusion.py` | Training script (Milestone 2) + `eval_corpus` |
| `configs/ebm.yaml` | Reference training config |
| `ired/diagnose_ebm.py` | EBM diagnostics |
| `docs/autoencoder.md` | The latent space the EBM operates in |
| `gensis.md` §10 | Diffusion / inner-loop design rationale |

---

## 11. Common issues

**`mse_z` keeps dropping but `acc` is flat.** The energy minimum is off the
decodable manifold — the generator gap (§4.3). Turn on `--gen-neg-weight 1.0`
(and check `gen_wrong` is non-zero in the logs).

**`acc_inner0 ≈ acc`.** The inner-loop `opt_step` isn't adding anything — energy
isn't calibrated, so bad-step rejection is a no-op. Confirm NCE is on
(`--no-nce` not set) and `nce_scale > 0`; consider `--rand-neg-weight`.

**`ae_acc < 0.5`.** Milestone 1 (the AE) is the bottleneck, not the EBM. Fix the
AE first (`docs/autoencoder.md` §10).

**Loss is non-finite / `eps_scale` drifts far from 1.0.** The predicted-noise
norm has diverged from the true noise norm and DDPM math is breaking. Check
`x_start_clamp`, the beta schedule at this `T`, and that latent stats were
computed (sampling in an un-normalized space miscalibrates the schedule).

**`gen_neg_weight > 0` raises "no gen-CE fn registered".** Call
`diffusion.set_gen_ce_fn(...)` (the training script wires this automatically when
`--gen-neg-weight > 0`); the same applies to `set_decoder_loss_fn` for
`--decoder-aux-weight`.

**RuntimeError about double-backward / SDPA kernel.** The energy forward must use
the MATH SDPA backend so `∇E` is twice-differentiable. This is forced inside
`EnergyTransformer.forward`; don't wrap training in a competing `sdpa_kernel`.
```
