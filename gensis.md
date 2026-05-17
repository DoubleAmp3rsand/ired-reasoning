# IRED for LLM Reasoning: Latent Diffusion as a Thinking Module

A proposal for applying the IRED (Iterative Reasoning through Energy Diffusion) framework to language model reasoning, by treating "thinking" as a continuous latent diffusion process between a frozen encoder and a frozen decoder.

---

## 1. Motivation

Modern "thinking" LLMs (o1, R1, QwQ, etc.) implement reasoning as **autoregressive chain-of-thought**: each reasoning token is sampled left-to-right and cannot be revisited within a single forward pass. This couples three concerns into one process:

1. *Deciding* what to think.
2. *Refining* that thought.
3. *Verbalizing* it as tokens.

We want to separate these. IRED already gives us a principled separation for continuous problems: an energy landscape over candidate answers, with gradient-based iterative refinement and bad-step rejection. The question is whether the same machinery can drive **the thinking step itself** in a language model.

---

## 2. Recap: what IRED actually does

For the matrix-addition task (`dataset.py:327`, `models.py:164`, `diffusion_lib/denoising_diffusion_pytorch_1d.py:161`):

```
(inp, noisy_pred) → EBM → scalar E
   ε̂ = ∇_{noisy_pred} E       ← same shape as noisy_pred
   loss = MSE(ε̂, true ε) + NCE(E(clean) < E(noisy))
```

Two key mechanisms beyond standard diffusion:

- **Energy-supervised landscape** (`--supervise-energy-landscape True`) — the *scalar* `E` is trained to be lower at clean answers than at noisy ones, not just to produce the right gradient.
- **Inner-loop bad-step rejection** (`opt_step` at `diffusion_lib/denoising_diffusion_pytorch_1d.py:373`) — at inference, each refinement step is accepted only if `E` decreased.

Together these make `E` a **calibrated scalar proxy for answer quality**, which is exactly the property we want from a reasoning verifier.

---

## 3. The proposal

Replace surface-form reasoning with a three-stage pipeline:

```
            ┌────────────┐      ┌────────────────────┐      ┌────────────┐
   X  ────▶│  Encoder   │─────▶│  Latent Thinking   │────▶│  Decoder   │─────▶ Result
            │  (frozen)  │  z_q │   (IRED in latent  │  z_0 │  (frozen)  │   A
            └────────────┘      │       space)       │      └────────────┘
                                └────────────────────┘
```

- **Encoder** maps a question `Q` to a conditioning latent `z_q`.
- **Latent Thinking Model** is the IRED energy network. It denoises a candidate latent `z` toward a target latent `z*` conditioned on `z_q`, iteratively, with bad-step rejection.
- **Decoder** maps the final latent `z_0` directly to the answer `A`. No autoregressive conditioning on the latent — the latent *is* the answer's compressed representation.

Crucially, **encoder and decoder are frozen**. Only the thinking model is trained.

### 3.1 Solving the "no ground-truth `c*`" problem

Earlier discussion flagged that LLM reasoning has no canonical clean latent: many reasoning paths produce the same answer. This proposal sidesteps the problem entirely:

```
z_a*  =  Encoder(A)        ← deterministic, defined by the frozen encoder
```

The answer text gives a unique target latent. The diffusion model is trained to denoise toward `Encoder(A)` given `Encoder(Q)`. Standard IRED loss applies directly.

### 3.2 Why this is "real thinking" vs. Coconut-style latent CoT

Coconut and similar work still use the LLM as an *autoregressive token generator conditioned on continuous thoughts*. The thinking is fused with surface-form production. In the proposal above:

- **Thinking is isolated** in a continuous latent space with smooth gradients.
- **Surface form is isolated** in the frozen decoder.
- The two phases do not interfere — inner-loop compute can be spent freely on thinking without touching the decoder.

This is a more honest "reasoning model": the computation that decides *what* the answer is happens before any token is emitted, and uses gradient descent rather than token sampling.

---

## 4. Architectural analogy: Stable Diffusion for reasoning

This architecture is Stable Diffusion's recipe transplanted to text reasoning:

| Stable Diffusion | This proposal |
|---|---|
| VAE encoder (image → latent) | Text encoder (`Q`, `A` → latent) |
| UNet diffusion in latent space | IRED energy net in latent space |
| VAE decoder (latent → image) | Text decoder (latent → answer) |
| Conditioning: CLIP text embedding | Conditioning: `z_q = Encoder(Q)` |
| Forward process: Gaussian noise | Forward process: Gaussian noise |
| Reverse: noise prediction | Reverse: `ε̂ = ∇_z E` with bad-step rejection |

The elegance: factor the hard problem (modeling discrete text) into the **frozen autoencoder**, leaving only the **smooth, continuous reasoning problem** for the diffusion model. This is exactly the trick that made pixel-space diffusion tractable.

---

## 5. Training and inference

### 5.1 Training (encoder & decoder frozen)

```python
# Inputs
z_q  = Encoder(Q)                          # conditioning
z_a  = Encoder(A)                          # clean target

# Forward diffusion at random timestep t
t    = uniform(0, T)
eps  = randn_like(z_a)
z_t  = sqrt(alpha_bar_t) * z_a + sqrt(1 - alpha_bar_t) * eps

# Energy network (the IRED EBM, transformer over (z_q, z_t, t))
E_t        = EnergyNet(z_q, z_t, t)        # scalar
eps_hat    = grad(E_t.sum(), z_t,          # vector, same shape as z_t
                  create_graph=True)[0]

# Losses
L_denoise  = mse(eps_hat, eps)             # matches IRED p_losses
L_nce      = relu(margin + E_q_clean - E_q_noisy).mean()
L          = L_denoise + lam * L_nce

L.backward()                               # updates only EnergyNet
```

This is *exactly* the IRED training loop from `diffusion_lib/denoising_diffusion_pytorch_1d.py:552`, with the only change being that `inp` is now `z_q` and `opt_out` is now `z_t`.

### 5.2 Inference

```python
z_q = Encoder(Q)
z   = randn(latent_shape)
for t in reversed(range(T)):
    for _ in range(N_inner):
        E_before = EnergyNet(z_q, z, t)
        eps_hat  = grad(E_before.sum(), z, create_graph=False)[0]
        z_new    = z - step_size * eps_hat
        E_after  = EnergyNet(z_q, z_new, t)
        z = where(E_after < E_before, z_new, z)         # bad-step rejection
A = Decoder(z)
```

Test-time compute scales with `N_inner` and `T`. The decoder is invoked exactly once.

---

## 6. Prior work in this exact shape

This architecture has precedent for *generation*, not yet for energy-based reasoning:

- **LD4LG / "Latent Diffusion for Language Generation"** (Lovelace et al., NeurIPS 2023) — frozen BART encoder/decoder + Gaussian latent diffusion. Demonstrates the autoencoder-bottleneck approach works for text.
- **PLANNER** (Zhang et al., NeurIPS 2023) — frozen autoencoder + latent paragraph diffusion.
- **GENIE, DiffuSeq, SeqDiffuSeq** — sequence-to-sequence latent diffusion variants.
- **Coconut** (Hao et al., Meta, 2024) — continuous "thoughts" fed into AR LLM. Closest in spirit, but keeps AR decoding and skips diffusion.
- **Diffusion-of-Thought / DoT** — diffusion over CoT tokens directly, mostly discrete.
- **Process Reward Models** (Math-Shepherd, "Let's Verify Step by Step") — trained scalar scorers over partial reasoning. Used for reranking, not for gradient-based refinement.

**The unfilled gap:** none of these combine (a) latent diffusion for text with (b) energy-based formulation with (c) bad-step rejection. That's the contribution this proposal stakes out.

---

## 7. Hard problems and design choices

### 7.1 The encoder-decoder must be a real autoencoder

Load-bearing assumption: `Decoder(Encoder(A)) ≈ A` for the answers in your domain. Candidates:

| Choice | Latent shape | Pros | Cons |
|---|---|---|---|
| **T5 / BART (frozen)** | sequence of token embeddings | strong pretraining; works as autoencoder | latent length = answer length; unknown at inference |
| **Sentence-T5 / Instructor / E5** | single vector | easy to diffuse | bottlenecks long answers |
| **Custom perceiver-style autoencoder** | fixed `K` latents (e.g. 32–64) | controllable; LD4LG-validated | requires its own pretraining |
| **LLaDA / discrete diffusion decoder** | fixed `K` latents | competitive non-AR quality | extra complexity |

**Recommendation:** start with T5 + learned pooling to fixed `K=32` latents. Verify reconstruction on the target task before training the diffusion head. If `Decoder(Encoder(A))` doesn't reproduce `A`, the whole pipeline is dead — fix this first.

### 7.2 Non-autoregressive decoding is harder than it looks

Pure one-shot decoding from `z_0` loses the AR inductive bias responsible for most modern LM quality. Practical paths:

- **AR decoder conditioned on `z_0`** — pragmatic compromise; the latent carries the semantic content, the decoder just serializes. Still AR in form but the latent does the heavy lifting.
- **Mask-predict / parallel decoder** — truly non-AR; quality lags.
- **Discrete diffusion decoder** (LLaDA-style) — non-AR and increasingly competitive.

The strict reading of "truly think then decode" demands non-AR. The pragmatic reading allows AR-conditional decoding as long as the latent commits to the answer's semantic content before token emission begins.

### 7.3 Why NCE matters here (revisiting the JEPA question)

Yann LeCun's argument that contrastive terms can be dropped in JEPA-style models **does not transfer cleanly to this proposal**, for the same reason it doesn't transfer to IRED proper:

- JEPA's anti-collapse mechanisms (VICReg, BYOL asymmetry) calibrate *which directions in feature space carry signal*.
- They do **not** calibrate the *absolute scalar value* of an energy function.
- Bad-step rejection requires `E(z_a) < E(z_b)` to be a reliable proxy for "`z_a` is a better candidate than `z_b`." That's an absolute-value property, not a direction property.

So `--supervise-energy-landscape True` is **load-bearing** for this proposal, not optional.

### 7.4 Latent MSE alone may not punish the right errors

The training loss in §5.1 lives entirely in latent space: `MSE(ε̂, ε)` over the K·d latent entries, plus the NCE energy contrast. This is the LD4LG / Stable-Diffusion recipe and is what we use as the default. But it has a known weakness for short discriminative outputs like GSM8K's final-answer mode:

- The latent encoding of `"5"` and `"6"` may sit close in MSE distance while being maximally different in correctness.
- The MSE objective treats every dimension of the latent equally; it has no way to "spend more budget" on the dimensions the decoder reads most.

The decoder *does* know this distinction — its CE loss explicitly punishes producing the wrong token. So a natural augmentation is to mix in a small frozen-decoder CE term:

```
L = L_denoise + λ·L_nce + λ_dec · 1[t < t_max] · CE(Decoder(x0_hat), gold)
```

where `x0_hat = predict_start_from_noise(z_t, t, ε̂)` is the model's current estimate of the clean latent. Two design notes:

1. **Gate on low t.** `x0_hat` is only a reliable estimate of the clean latent near the end of the schedule. For early t the residual noise is large; backpropping decoder CE through a wildly-off `x0_hat` injects noise into the EBM. We restrict the auxiliary to `t < t_max` (default 2 of 10).
2. **Backprop, don't replace.** The decoder stays frozen — its params get zero gradient. But the CE loss depends on `x0_hat`, which depends on the EBM via `ε̂`. So gradient flows back to the EBM through the frozen decoder, treating the decoder as a fixed perceptual loss over latents.

Why keep this as an auxiliary rather than the primary objective: as discussed in §7.3, the inner-loop bad-step rejection assumes ∇E points toward the data manifold. The latent MSE is what calibrates that. A decoder-CE-only loss would optimize ∇E toward "whatever latent the decoder happens to decode well," which isn't the same set — and would break the bad-step contract at inference.

Default in this repo: `--decoder-aux-weight 0.0` (off, matches §5.1 exactly). Recommended for ablation on short-answer tasks: `--decoder-aux-weight 0.1 --decoder-aux-t-max 2`.

### 7.5 Inference cost

Each inner step is one forward+backward through the energy network. For an IRED schedule with `T=10` outer × `N=5` inner = 50 transformer passes through the energy net, versus ~200 AR tokens of CoT through the full LLM. FLOPs-comparable if the energy net is ~1/4 the size of the decoder.

The bet: a steeper accuracy-vs-compute curve than AR-CoT, justifying the cost.

---

## 8. Concrete first experiment

Realistic single-GPU prototype to validate the core thesis.

**Setup (as originally proposed):**
- **Task:** GSM8K (grade-school math). Short numeric answers, clean correctness signal.
- **Encoder/Decoder:** `flan-t5-large`, frozen. Custom learned attention pool compresses encoder output to `K=32` latents.
- **Energy net:** 4-layer transformer over `concat(z_q, z_t, time_embed)`. Scalar output via squared-sum of final layer (matching `models.py:211`).
- **Diffusion:** `T=10` timesteps, `continuous=True` (matrix-addition variant of `GaussianDiffusion1D`).
- **Inner loop:** 5 steps with bad-step rejection.
- **Loss:** denoising MSE + NCE with `λ=1`. Optional `--decoder-aux-weight 0.1 --decoder-aux-t-max 2` for a low-t frozen-decoder CE ablation (see §7.4).

**Practical defaults that survived first-run debugging (see §11):**
- **Model size:** `flan-t5-base` (d_model=768) for faster iteration. `large` is a drop-in.
- **Answer mode:** `full` (the CoT ending in `#### N`), not `final` (just the number). With `K=32, d=768` the latent has ~24k DoF; a final-only answer carries ~60 bits, leaving the data manifold ~5 orders of magnitude smaller than the latent volume. The EBM can find the rough direction but can't precisely localize, and head magnitude inflates trying to compensate. Full mode also gives reasoning structure to denoise toward. Eval extracts `#### N` from decoded text rather than byte-exact matching.
- **`max_a_length`:** `256` for full mode. Covers ~99% of GSM8K answers (p99=240, max=354). `128` (the previous default) truncates ~25% of the training set.
- **β schedule:** `cosine`, not `linear`. Linear at T=10 saturates immediately after clamping (β=0.999 for 5 of 10 steps), giving the EBM almost no useful t spread.
- **Weight decay:** `--weight-decay 0.01` on the EBM. **Load-bearing** — without it the head magnitude inflates ~100× during training, energies grow with it, and the relative NCE contrast collapses from ~2× to ~1.02× even though the NCE loss number looks fine.
- **Clamps:** `x_start_clamp=5.0`, `envelope_sf=None`. IRED's hardcoded `(2, 2)` assume `[-1, 1]` data; LayerNorm'd T5 latents live in ~`[-3, 3]`.
- **SDPA kernel:** force `SDPBackend.MATH` inside the EBM's attention — the fast kernels don't implement double-backward, which IRED's MSE-on-∇E requires.
- **Monitoring (per training step):** `mse, nce, e_real, e_fake, eps_scale = ||ε̂|| / ||noise||`. Target `eps_scale ≈ 1.0`. Drift above 1 = head inflation; below 1 = under-prediction; either breaks DDPM sampling regardless of MSE.
- **Monitoring (per eval):** `ae_recon_acc, ebm_acc(inner=N), ebm_acc(inner=0), mse_z, std_zs, corr_z`. `ae_recon_acc` separates Milestone-1 failures from EBM failures; `inner=0` vs `inner=N` isolates whether `opt_step` is helping or hurting; `std_zs/std_za` and `corr_z` localize whether the failure is magnitude inflation, direction misalignment, or both.

**Milestones:**
1. **Autoencoder check** — confirm `Decoder(Pool(Encoder(A)))` reproduces GSM8K answers. If accuracy < 95% in `final` mode (a strict floor on pool fidelity), redesign the pool/decoder before going further. For `full` mode the bar is final-answer extraction accuracy ≥ 90% — the CoT body needn't byte-match, only the `#### N` must come back. *(Final-mode benchmark on flan-t5-base + 2-layer pool + K=32: 98.8% on GSM8K-test.)*
2. **Diffusion training** — train energy net to convergence on GSM8K train set. *Watch `eps_scale` and the `e_real/e_fake` ratio over training; either drifting is the early warning sign §11.2 describes.*
3. **Test-time compute curve** — plot accuracy vs. `N_inner × T` and compare to AR-CoT at matched FLOPs.

**Decision rule:** if the diffusion curve is steeper than AR-CoT at matched compute, the thesis is validated. If it plateaus below AR-CoT, that's an informative negative result about smoothness of reasoning in latent space.

---

## 9. Honest assessment

**Promising aspects:**
- Solves the "no `c*`" problem elegantly via the frozen encoder.
- Cleanly separates thinking from surface-form generation — a sharper definition of "reasoning model" than current practice.
- Test-time compute scaling via inner-loop steps is a natural fit for the o1-era frontier.
- Architectural precedent (Stable Diffusion, LD4LG) suggests the frozen-autoencoder approach is viable.

**Risks:**
- Autoencoder quality bottleneck — most teams who tried latent-diffusion-for-text reported "the decoder is the limiting factor."
- RL-on-AR-CoT (DeepSeek-R1, o1) is currently the empirical winner. Any diffusion-based reasoning system has to justify *not* using RL.
- Reasoning may not be smooth in latent space. If correct reasoning traces are isolated discrete modes rather than connected regions, diffusion will struggle.

**Why try it anyway:** the architecture is clean, the prior work is encouraging, and even a clear negative result tells us something concrete about the geometry of reasoning. Two-week prototype is feasible on a single GPU.

---

## 10. Code reference

Code in this repo that maps directly to the proposal. Adapting to the LLM setting requires only swapping `(inp, opt_out)` for `(Encoder(Q), Encoder(A))` and replacing the MLP `EBM` with a transformer.

### 10.1 The energy network — `EBM` (models.py)

A 4-layer MLP with FiLM-style time conditioning. The scalar energy is produced by squaring and summing the last layer's output.

```python
class EBM(nn.Module):
    def __init__(self, inp_dim, out_dim, is_ebm: bool = True):
        super(EBM, self).__init__()
        h = 512

        fourier_dim, time_dim = 128, 128

        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.fc1 = nn.Linear(inp_dim + out_dim, h)
        self.is_ebm = is_ebm

        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, out_dim if is_ebm else out_dim)

        self.t_map_fc2 = nn.Linear(time_dim, 2 * h)
        self.t_map_fc3 = nn.Linear(time_dim, 2 * h)

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.is_ebm = is_ebm

    def forward(self, *args):
        if self.is_ebm:
            x, t = args
        else:
            x, y, t = args
            x = torch.cat([x, y], dim=-1)

        t_emb = self.time_mlp(t)

        fc2_gain, fc2_bias = torch.chunk(self.t_map_fc2(t_emb), 2, dim=-1)
        fc3_gain, fc3_bias = torch.chunk(self.t_map_fc3(t_emb), 2, dim=-1)

        h = swish(self.fc1(x))
        h = swish(self.fc2(h) * (fc2_gain + 1) + fc2_bias)
        h = swish(self.fc3(h) * (fc3_gain + 1) + fc3_bias)

        if self.is_ebm:
            output = self.fc4(h).pow(2).sum(dim=-1)[..., None]   # scalar energy E = ||fc4(h)||²
        else:
            output = self.fc4(h)

        return output
```

### 10.2 Scalar → vector — `DiffusionWrapper` (models.py)

Takes a scalar energy and returns its gradient w.r.t. `opt_out`. The `create_graph=True` flag keeps the autograd graph alive so the outer training loss can backprop through this gradient operation.

```python
class DiffusionWrapper(nn.Module):
    def __init__(self, ebm):
        super(DiffusionWrapper, self).__init__()
        self.ebm = ebm
        self.inp_dim = ebm.inp_dim
        self.out_dim = ebm.out_dim

        if hasattr(self.ebm, 'is_ebm'):
            assert self.ebm.is_ebm, 'DiffusionWrapper only works for EBMs'

    def forward(self, inp, opt_out, t, return_energy=False, return_both=False):
        opt_out.requires_grad_(True)
        opt_variable = torch.cat([inp, opt_out], dim=-1)

        energy = self.ebm(opt_variable, t)

        if return_energy:
            return energy

        opt_grad = torch.autograd.grad([energy.sum()], [opt_out], create_graph=True)[0]

        if return_both:
            return energy, opt_grad
        else:
            return opt_grad
```

### 10.3 Inner loop with bad-step rejection — `opt_step` (denoising_diffusion_pytorch_1d.py)

For each step inside the inner loop: take a gradient step on the energy, clamp to the diffusion envelope, then **reject** if the new energy is higher than the old one. This is the mechanism that requires `E` to be a *calibrated* scalar — which is why the NCE term in `p_losses` is load-bearing.

```python
def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0, detach=True):
    with torch.enable_grad():
        for i in range(step):
            energy, grad = self.model(inp, img, t, return_both=True)
            img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf

            if mask is not None:
                img_new = img_new * (1 - mask) + mask * data_cond

            if self.continuous:
                sf = 2.0
            else:
                sf = 1.0

            max_val = extract(self.sqrt_alphas_cumprod, t, img_new.shape)[0, 0] * sf
            img_new = torch.clamp(img_new, -max_val, max_val)

            energy_new = self.model(inp, img_new, t, return_energy=True)
            if len(energy_new.shape) == 2:
                bad_step = (energy_new > energy)[:, 0]
            elif len(energy_new.shape) == 1:
                bad_step = (energy_new > energy)
            else:
                raise ValueError('Bad shape!!!')

            img_new[bad_step] = img[bad_step]    # bad-step rejection

            if eval:
                img = img_new.detach()
            else:
                img = img_new

    return img
```

### 10.4 Inference schedule — `p_sample_loop` (denoising_diffusion_pytorch_1d.py)

Outer loop over diffusion timesteps `T..0`; at each step a standard denoising step followed by the inner `opt_step` refinement. The continuous-vs-discrete `sf` scaling lives here.

```python
@torch.no_grad()
def p_sample_loop(self, batch_size, shape, inp, cond, mask, return_traj=False):
    device = self.betas.device

    if hasattr(self.model, 'randn'):
        img = self.model.randn(batch_size, shape, inp, device)
    else:
        img = torch.randn((batch_size, *shape), device=device)

    x_start = None

    if self.show_inference_tqdm:
        iterator = tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps)
    else:
        iterator = reversed(range(0, self.num_timesteps))

    preds = []

    for t in iterator:
        self_cond = x_start if self.self_condition else None
        batched_times = torch.full((img.shape[0],), t, device=inp.device, dtype=torch.long)

        cond_val = None
        if mask is not None:
            cond_val = self.q_sample(x_start=inp, t=batched_times, noise=torch.zeros_like(inp))
            img = img * (1 - mask) + cond_val * mask

        img, x_start = self.p_sample(inp, img, t, self_cond, scale=False, with_noise=self.baseline)

        if mask is not None:
            img = img * (1 - mask) + cond_val * mask

        if self.sudoku:
            step = 20
        else:
            step = 5

        if self.use_innerloop_opt:
            img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)
            img = img.detach()

        if self.continuous:
            sf = 2.0
        elif self.shortest_path:
            sf = 0.1
        else:
            sf = 1.0

        max_val = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape)[0, 0] * sf
        img = torch.clamp(img, -max_val, max_val)

        img_unscaled = self.predict_start_from_noise(img, batched_times, torch.zeros_like(img))
        preds.append(img_unscaled)
```

### 10.5 Combined training loss — `p_losses` (denoising_diffusion_pytorch_1d.py)

The denoising MSE term plus the NCE energy contrast. For continuous tasks (matrix addition, inverse, lowrank) the contrast samples are produced by running `opt_step` on a heavily-noised version of the clean target — a "hard negative" mined from the current energy landscape.

```python
def p_losses(self, inp, x_start, mask, t, noise=None):
    b, *c = x_start.shape
    noise = default(noise, lambda: torch.randn_like(x_start))

    # noise sample
    x = self.q_sample(x_start=x_start, t=t, noise=noise)

    if mask is not None:
        x_cond = self.q_sample(x_start=inp, t=t, noise=torch.zeros_like(noise))
        x = x * (1 - mask) + mask * x_cond

    # predict noise via gradient of energy
    model_out = self.model(inp, x, t)

    if self.objective == 'pred_noise':
        target = noise
    elif self.objective == 'pred_x0':
        target = x_start
    elif self.objective == 'pred_v':
        v = self.predict_v(x_start, t, noise)
        target = v
    else:
        raise ValueError(f'unknown objective {self.objective}')

    if mask is not None:
        model_out = model_out * (1 - mask) + mask * target

    loss = F.mse_loss(model_out, target, reduction='none')
    loss = reduce(loss, 'b ... -> b (...)', 'mean')
    loss = loss * extract(self.loss_weight, t, loss.shape)
    loss_mse = loss

    if self.supervise_energy_landscape:
        noise = torch.randn_like(x_start)
        data_sample = self.q_sample(x_start=x_start, t=t, noise=noise)

        if mask is not None:
            data_cond = self.q_sample(x_start=x_start, t=t, noise=torch.zeros_like(noise))
            data_sample = data_sample * (1 - mask) + mask * data_cond

        # heavy-noise negative, then refine via inner-loop opt -> "hard" negative
        xmin_noise = self.q_sample(x_start=x_start, t=t, noise=3.0 * noise)
        if mask is None:
            data_cond = None

        # (continuous-task branch shown; sudoku / connectivity / shortest_path have their own variants)
        xmin_noise = self.opt_step(inp, xmin_noise, t, mask, data_cond, step=2, sf=1.0)
        xmin = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        loss_opt = torch.pow(xmin_noise - xmin, 2).mean()

        xmin_noise = xmin_noise.detach()
        xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
        xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)
        loss_scale = 0.5

        xmin_noise = self.q_sample(x_start=xmin_noise_rescale, t=t, noise=noise)
        if mask is not None:
            xmin_noise = xmin_noise * (1 - mask) + mask * data_cond

        # NCE: real (clean) sample should have lower energy than the mined negative
        inp_concat = torch.cat([inp, inp], dim=0)
        x_concat   = torch.cat([data_sample, xmin_noise], dim=0)
        t_concat   = torch.cat([t, t], dim=0)
        energy = self.model(inp_concat, x_concat, t_concat, return_energy=True)

        energy_real, energy_fake = torch.chunk(energy, 2, 0)
        energy_stack = torch.cat([energy_real, energy_fake], dim=-1)
        target = torch.zeros(energy_real.size(0)).to(energy_stack.device)
        loss_energy = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')[:, None]

        loss = loss_mse + loss_scale * loss_energy
        return loss.mean(), (loss_mse.mean(), loss_energy.mean(), loss_opt.mean())
    else:
        loss = loss_mse
        return loss.mean(), (loss_mse.mean(), -1, -1)
```

---

## 11. Implementation pitfalls (post-mortem)

Five concrete bugs and infelicities surfaced when building the §8 prototype against the IRED reference code. None of them invalidate the proposal — but every one is a place where a default that worked for IRED's small tabular tasks fails silently in the latent-LLM setting. Listing them in one place so future implementers don't re-discover them.

### 11.1 Diffusion β schedule blows up at small T

`linear_beta_schedule` in `denoising_diffusion_pytorch_1d.py` is

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

Any batch that samples one of those t values produces NaN loss; the first `opt.step()` poisons the EBM permanently. With B=16 and uniform t, the probability of a clean batch is `(0.7)^16 ≈ 0.3%`, so essentially every step NaNs from step 1 — but it can take 50–350 logged steps to *notice* depending on what `--log-every` is set to and whether the NaN propagates to a visible stat first.

**Fix:** clip β to ≤ 0.999 (matching the cosine schedule's bound), and assert `(alphas_cumprod > 0).all()` at construction so future bad (schedule, T) combos fail loudly.

**Better fix:** for small T, use `cosine` directly. The clipped-linear schedule at T=10 has β ≥ 0.999 for half its range, so the EBM only sees variance interpolations near t=0; it never gets useful gradient at intermediate noise levels.

### 11.2 EBM head magnitude inflates unboundedly without weight decay

IRED's reference setup uses an MLP energy net on small (~tens-of-dims) tabular data with no weight decay. We inherited the no-weight-decay default. In the latent-LLM setting the head is a `Linear(d_model=768, d_model=768)` and the energy is the squared sum over `K · d_model = 24576` dims — so the absolute energy scale is enormous and grows freely.

Observed in a 5000-step run:

| step | e_real | e_fake | contrast | nce loss |
|---:|---:|---:|---:|---:|
| 100 | 1,413 | 2,896 | 2.05× | 0.19 |
| 5000 | 131,601 | 134,133 | **1.02×** | 0.20 |

The NCE loss number is unchanged but the *relative* contrast collapsed from 2× to 2%. Cross-entropy is scale-equivariant in the logit difference, so the loss minimum can be reached by either (a) learning real-vs-fake discrimination or (b) growing both energies in proportion. Once the head weights start growing, gradient descent prefers (b) — it's an easier optimization direction.

Downstream consequence: `opt_step` bad-step rejection becomes inert (a 2% contrast is dominated by noise in `E(z_new)` vs `E(z)`), and ∇E grows with the head, breaking DDPM's reverse process that expects ε̂ at noise-scale (std ≈ 1).

**Fix:** `--weight-decay 0.01` on `AdamW` keeps the head norm bounded. Should be the default; the current `0.0` is a footgun.

**Diagnostic:** the `eps_scale = ||ε̂|| / ||noise||` stat added to per-step logging surfaces this in one number — when head inflates, `eps_scale` drifts upward in lockstep.

### 11.3 IRED's clamping bounds assume `[-1, 1]` data

IRED hardcodes `x_start_clamp = 2` and `envelope_sf = 2` (the per-t `±sqrt(α̅_t)·sf` clamp inside `opt_step` and `p_sample_loop`). Both assume input data normalized to `[-1, 1]` — true for IRED's tabular tasks, false for LayerNorm'd T5 latents whose element std is ~1 and bulk lives in `[-3, 3]`.

Applied verbatim, these clamps crush noisy `z_t` to ~0 at large t (when `sqrt(α̅_t)` is small), turning the reverse process into noise injection plus clamping rather than denoising.

**Fix:** `x_start_clamp = 5.0` (loose enough to not bite typical T5 latents), `envelope_sf = None` (disabled by default; pass an explicit float to re-enable the IRED-style behavior). Both are CLI-configurable.

**Lesson:** any time you transplant IRED-style code to a new data domain, audit every literal numeric constant against the new data's empirical range. The original constants are a domain-specific tuning, not a general defaults.

### 11.4 ε̂ scale drift is invisible to `mse(ε̂, ε)`

The denoising loss `mse(ε̂, ε)` is dominated by direction agreement — a 2× scale error contributes only `(2 - 1)^2 = 1` per-element to MSE, the same as a perpendicular unit-noise direction. So `mse` can look small (~0.02) while `||ε̂||` is drifting away from `||ε||`. DDPM's `predict_start_from_noise` and `q_posterior` math assume ε̂ is at noise-scale; if it isn't, the reverse process inflates or deflates `z` magnitude regardless of how good the *direction* is.

In our broken 5000-step run, the symptom was `std(z_sampled) = 2.75` vs `std(z_a) = 1.00` after sampling — magnitude inflated almost 3× while `mse(ε̂, ε)` was 0.017 at train time.

**Fix:** log `eps_scale = ||ε̂||_2 / ||ε||_2` per training step. It's free (one no_grad norm) and is the earliest detector of the §11.2 head-inflation problem.

### 11.5 PyTorch fast SDPA kernels don't implement double-backward

IRED's training loss takes MSE against ε̂ = ∇<sub>z</sub>E, so the outer `loss.backward()` differentiates *through* an inner `autograd.grad(...).` This is a second-order autograd path through every attention call inside the EBM. PyTorch's default flash and mem-efficient SDPA kernels don't implement the backward-of-backward; on CPU you get

```
RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward
is not implemented
```

and on GPU you get a similar (slightly less obvious) failure depending on hardware.

**Fix:** force the math kernel inside the EBM's attention,

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel([SDPBackend.MATH]):
    h = self.encoder(x)
```

This costs throughput vs flash attention, but is the only kernel that supports double-backward. A nicer long-term solution is to switch the EBM to a custom transformer that uses an explicit attention implementation, but the kernel switch is the smallest fix.

---

### Quick reference: defaults to override when porting IRED to latents

| Default in IRED | Safe default for latents | Why |
|---|---|---|
| `linear` β schedule | `cosine` | linear at small T saturates or NaNs |
| `weight_decay = 0` | `weight_decay = 0.01` | bounds head magnitude, preserves NCE contrast |
| `x_start_clamp = 2.0` | `x_start_clamp = 5.0` | T5 latents have wider element range |
| `envelope_sf = 2.0` | `envelope_sf = None` | the per-t envelope crushes z at large t |
| default SDPA kernel | `SDPBackend.MATH` | double-backward not implemented in flash kernels |
| log `mse, nce` only | also log `eps_scale, e_real/e_fake ratio` | early detection of scale collapse |
