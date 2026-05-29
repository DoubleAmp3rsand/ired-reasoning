# Reasoning Without Language — Latent Energy Diffusion for LLM Reasoning

Chain-of-thought is a *representation* of thinking, not thinking itself. When a
modern language model "reasons" step by step, it is not deliberating — it is
maximizing the posterior probability of the next token under a distribution fit
to human-written text. The visible reasoning trace is a fluent rendering of a
search that is really happening in the network's activations; the tokens are the
*shadow* of the computation, not the computation. RL-trained reasoners (o1, R1)
make this explicit: the reward shapes which token prefixes the policy prefers,
and the prefix is chosen because it raises the likelihood of a good final answer
— not because each token is a true cognitive step.

This proposal takes that observation seriously and moves the reasoning *off* the
token substrate. It treats thinking as a continuous optimization in a latent
space between a frozen encoder and a frozen decoder: an **energy-based diffusion
model** descends a learned cost until a latent answer-representation converges,
and the surface form is rendered exactly once, after the thinking is done.
Reasoning is gradient descent on an explicit cost, not token sampling.

It builds on two existing pieces of work and one framework:

1. **IRED** — *Iterative Reasoning through Energy Diffusion* (Du et al., ICML
   2024). Provides the optimizer: an energy-based diffusion model whose scalar
   energy is calibrated to be a quality proxy, with inner-loop bad-step
   rejection.
2. **LD4LG** — *Latent Diffusion for Language Generation* (Lovelace et al.,
   NeurIPS 2023). Provides the frozen-autoencoder bottleneck: a pretrained
   text encoder/decoder pair with diffusion operating purely in their latent
   space.
3. **APTAMI** — LeCun's *A Path Towards Autonomous Machine Intelligence*. Provides
   the Mode-1 / Mode-2 framework this system is a concrete instantiation of
   (§4).

Companion docs: `docs/autoencoder.md` (the frozen-AE bottleneck), `docs/ebm.md`
(the energy net and its training), and `docs/implementation_pitfalls.md` (the
post-mortem of bugs that bite when porting IRED to latents).

---

## 1. The thesis: reasoning is not language

### 1.1 Chain-of-thought is search in token space

The cleanest way to read chain-of-thought is as **a search for a token prefix
whose conditioning yields a better final-answer distribution**. Each emitted token
is a move, the growing prefix is the state, and the (implicit) objective is
roughly the likelihood of the gold answer given the prefix. Seen this way,
step-by-step "reasoning" is not deliberation — it is a search over token
sequences, carried out by sampling and re-rolling against an objective the model
never names.

This token-space search has three properties this proposal trades away. It is
worth being precise about which are forced by the *medium* and which are merely
properties of the *dominant architecture* — a single autoregressive transformer —
because that distinction decides how much work moving off tokens is really doing.

1. **Discrete, non-differentiable search (medium-forced).** Tokens are discrete,
   so there is no useful gradient of the objective with respect to a move.
   Refinement cannot *be* gradient descent; it has to be sampling, re-rolling, or
   discrete search (best-of-N, beam, MCTS, RL). No token-space architecture
   escapes this — and it is the load-bearing reason to consider a continuous space
   at all.
2. **The predictor and the proposer are the same network (architectural, not
   medium-forced).** In the standard single-transformer reasoner, "predict the
   consequence of a step" and "propose the next step" are one autoregressive
   rollout, with no clean separation between them. But that is a design choice, not
   a property of tokens — AlphaZero runs a discrete search with a *separate* value
   model and policy. You can keep them distinct in token space.
3. **The objective is implicit (architectural, not medium-forced).** In vanilla
   CoT the only quality signal is baked into the policy by pretraining/RL, so you
   cannot point at it, score an off-policy candidate, or use it as a verifier. But
   Process Reward Models (§6) are precisely explicit scalar scores over token
   sequences — evaluable off-policy, usable as verifiers. Implicitness is a
   training-objective choice, not a medium constraint.

So two of the three are patchable without leaving token space. The honest case for
a continuous substrate rests on property 1: **only in a continuous space can the
refinement itself be gradient descent on an explicit cost**, turning the search
into optimization. The other two then come for free rather than as bolt-ons — a
cost network structurally distinct from the renderer, and an explicit scalar score
that doubles as a verifier. Whether continuous optimization actually beats
well-engineered discrete search is an empirical question — the one §7 is built to
test; the claim here is only that the substrate makes the optimization formulation
*available*, which token space does not.

### 1.2 Current "latent reasoning" is latent-guided prompting, not this

Recent latent-reasoning work — Coconut, LaDiR, Soft Thinking — runs computation
in continuous space but then hands the result to an autoregressive LLM that does
the actual reasoning during decoding. On inspection these are **latent-guided
prompting**: the latent is rich conditioning, and the decoder inherits whatever
cognitive capacity the underlying LLM has. They fall short on two counts: (a)
inference is generative sampling, not optimization against a cost, and (b) the
decoder is doing the reasoning, not transduction.

The system this proposal builds instead rests on three structural commitments:

1. **Inference is optimization against an explicit cost**, not sampling from a
   learned distribution over "what reasoning looks like."
2. **The cost is structurally anchored** so it resists being gamed — the proposal
   has to supply something that plays that anchor role for a language reasoner.
3. **The decoder transduces, it does not reason** — a hard boundary between
   cognition and surface form, with the full cognitive burden in the latent
   optimization.

(These three commitments are where the proposal lines up with LeCun's
Mode-1 / Mode-2 framework, which §4 develops in full.)

### 1.3 Why energy-based diffusion is the right substrate

A diffusion model trained by score matching learns `∇log p(x) = −∇E(x)` for an
implicit energy `E(x) = −log p(x)`; reverse diffusion *is* iterative gradient
descent on that energy. A trained diffusion model therefore already defines a
cost whose gradient field is anchored in its training data and whose minima sit
on the data manifold.

IRED makes this explicit. The energy `E` is no longer implicit — it is trained
to be a **calibrated scalar proxy for answer quality** and exploited at inference
via bad-step rejection. The mechanism, stripped to essentials:

```
(z_q, z_t) → EnergyNet → scalar E
   ε̂ = ∇_{z_t} E              ← same shape as z_t, used as the "noise prediction"
   loss = MSE(ε̂, true ε)      ← denoising objective (direction of the field)
        + NCE(E(clean) < E(noisy))  ← calibrates the *absolute* scalar
```

Two mechanisms beyond standard diffusion do the work:

- **Energy-supervised landscape** — `E` is trained to be lower at clean answers
  than at noisy ones, not merely to produce the right gradient.
- **Inner-loop bad-step rejection** — at inference, each refinement step is
  accepted only if `E` decreased.

Together they make `E` a calibrated quality scalar — exactly the property a
reasoning verifier needs. This is, to our knowledge, the cleanest existing
instantiation of reasoning-as-optimization in continuous space; it has not been
scaled to language. This proposal scales it: keep IRED's energy-diffusion core as
the optimizer, isolate it between a frozen encoder and a frozen decoder so that no
reasoning leaks into surface-form generation, and use the encoder of the answer
text as the structural anchor for the cost.

---

## 2. The method

Replace surface-form reasoning with a three-stage pipeline:

```
            ┌────────────┐      ┌────────────────────┐      ┌────────────┐
   X  ────▶│  LatentEnc  │────▶│  Latent Thinking   │────▶│  LatentDec  │─────▶ Result
            │  (frozen)  │  z_q │   (IRED in latent  │  z_0 │  (frozen)  │   A
            └────────────┘      │       space)       │      └────────────┘
                                └────────────────────┘
```

- **LatentEnc** maps a question `Q` to a conditioning latent `z_q`.
- **Latent Thinking Model** is the IRED energy network. It denoises a candidate
  latent `z` toward a target latent `z*` conditioned on `z_q`, iteratively, with
  bad-step rejection.
- **LatentDec** maps the final latent `z_0` directly to the answer `A`. No
  autoregressive conditioning — the latent *is* the answer's compressed
  representation.

Crucially, the **underlying pretrained encoder and decoder are frozen** (the
pipeline also contains two small *learned* autoencoder modules — the compressor
`f_φ` and the reconstructor `f_ψ` — that adapt the frozen backbone's
representations to a diffusion-friendly latent space; see §3.1 for the full
diagram). Only the thinking model (EBM) and these lightweight AE adapters are
trained.

### 2.1 Solving the "no ground-truth `c*`" problem

LLM reasoning has no canonical clean latent: many reasoning paths produce the
same answer. The frozen encoder sidesteps this entirely:

```
z_a*  =  LatentEnc(A)        ← deterministic, defined by the frozen encoder
```

The answer text gives a unique target latent. The diffusion model is trained to
denoise toward `LatentEnc(A)` given `LatentEnc(Q)`. Standard IRED loss applies
directly. The framing here largely follows how other diffusion language
model perform training.

### 2.2 Why this is reasoning-as-optimization, not latent-guided prompting

The three commitments of §1.2 are realized as follows (the first two hold by
construction; the third is an architectural invariant whose practical fidelity
depends on autoencoder quality — see §8):

- **Inference is optimization** — reverse diffusion with bad-step rejection is
  explicit gradient descent on a learned energy, not sampling from a generative
  model.
- **The cost is anchored** — the target latent `LatentEnc(A)` is fixed by the
  frozen encoder applied to ground-truth answers. The NCE term that calibrates
  `E` is anchored to this same fixed point. There is no learnable target the
  optimizer can drift toward.
- **The decoder is architected to transduce** — it sees a single latent and
  produces tokens; inner-loop compute can be spent freely on thinking without
  touching the decoder. The architecture enforces this boundary (the decoder
  fires exactly once, on the final latent), but whether the decoder *in practice*
  transduces rather than implicitly problem-solves depends on how much
  information the latent carries — an autoencoder-fidelity question tested in
  §7 Milestone 1.

The computation that decides *what* the answer is happens entirely before any
token is emitted, and it happens via gradient descent rather than token sampling.

### 2.3 The cost-anchoring simplification

A fuller treatment of commitment (2) would use a **composite anchored cost**
combining structurally-different signals — world-model coherence, decoded-output
stability under latent perturbation, capacity bottleneck, target-encoder
distance, external verifier — so that gaming the cost requires fooling several
independent anchors at once. That is the broader research direction, and the
natural place to extend this prototype once the core pipeline works.

For the first experiment we deliberately collapse that composite cost to its
single most load-bearing term: the **target-encoder distance**, realized as
IRED's denoising MSE plus NCE against `LatentEnc(A)`. Two small steps toward the
composite form are already implemented and described in §5: a low-`t`
frozen-decoder CE auxiliary, and a generator-grounded negative that pins the
energy minima to latents the decoder can actually render. The retrieval/KB
conditioning of §9 adds a structurally-distinct conditioning source — but
conditioning is not an anchor (it shapes what the energy network sees, not what
low energy *means*). The next proper anchor is the external verifier; anything
richer is deferred until the single-anchor version is shown to train.

---

## 3. Architecture

### 3.1 Stable Diffusion's recipe, transplanted to reasoning

| Stable Diffusion | This proposal |
|---|---|
| VAE encoder (image → latent) | Text encoder (`Q`, `A` → latent) |
| UNet diffusion in latent space | IRED energy net in latent space |
| VAE decoder (latent → image) | Text decoder (latent → answer) |
| Conditioning: CLIP text embedding | Conditioning: `z_q = LatentEnc(Q)` |
| Forward process: Gaussian noise | Forward process: Gaussian noise |
| Reverse: noise prediction | Reverse: `ε̂ = ∇_z E` with bad-step rejection |

The elegance: factor the hard problem (modeling discrete text) into the **frozen
autoencoder**, leaving only the **smooth, continuous reasoning problem** for the
diffusion model — the trick that made pixel-space diffusion tractable.

The full pipeline is five modules — two frozen (the T5/BART encoder and decoder),
two learned for the autoencoder stage (the Compressor `f_φ` and the Reconstructor
`f_ψ`), and one learned for the reasoning stage (the EBM):

```
   Q ──▶ Encode ──▶ f_φ (Compressor) ──▶  z_q ∈ R^(K×d_ae)
                                          │  conditioning
                                          ▼
                   z_T ~ N(0, I)  ──▶  EBM(z_q, z, t)  ──▶  z_0
                   (reasoning loop: T outer × N_inner steps, bad-step rejection on E)
                                          │
                                          ▼
                   z_0  ──▶  f_ψ (Reconstructor) ──▶ Decode ──▶  A
```

### 3.2 The autoencoder: compression / reconstruction split

LD4LG splits the autoencoder into two networks with one job each (full detail in
`docs/autoencoder.md`):

```
text → E_frozen → Pool (f_φ) → z ∈ R^(K×d_ae) → ReconstructionNet (f_ψ) → R^(K×d_LM) → D_frozen → text
                                     │
                                     ▼
                            [diffuse here, EBM]
```

1. **Compression** (`f_φ`, the pool) maps a variable-length encoder hidden
   sequence to `K` latent slots that pack answer-relevant content densely.
2. **Reconstruction** (`f_ψ`) shapes those slots into something the frozen
   decoder's cross-attention can read — the decoder was trained on encoder
   outputs with specific statistics, and arbitrary pool outputs aren't guaranteed
   to match.

These objectives pull in different directions (density vs. on-manifold
smoothness), so separating them lets each module specialize. The pool ends with a
learnable `d_LM → d_ae` projection — the dimensionality knob: once the AE has
converged, dropping `d_ae` shrinks the EBM's diffusion space by the same factor
and is the most impactful lever for reasoning-stage tractability. The
`ReconstructionNet` is identity-initialized (attention `out_proj` and FF
`linear2` zero-init'd) so the AE is bit-exact identity at step 0 and grows
residuals only as they help.

### 3.3 The energy network

A transformer EBM over `(z_q, z_t, t)` produces a scalar energy (`ired/model/energy_net.py`;
full reference in `docs/ebm.md`). Token layout:

```
[ time_token(1) | z_q + pos + type0 (K) | z_t + pos + type1 (K) ]      # (B, 1+2K, d)
```

- `z_q` (the question) and `z_t` (the noisy answer latent) each get a per-slot
  positional embedding and a learned type embedding.
- Four `norm_first` transformer encoder layers, then a `Linear` head over the
  **z_t slots only**.
- **Scalar energy** = `||head(h_zt)||²` summed across the `K·d` dims (the
  squared-sum trick from IRED).

`DiffusionWrapper` exposes `ε̂ = ∇_{z_t} E` via `autograd.grad`, with
`create_graph=True` during training so the denoising MSE can backprop *through*
the gradient into the EBM's weights. Note `z_q` receives **no gradient** — it is
pure conditioning, attended over but never optimized. (This is the property §9's
retrieval extension reuses for a third conditioning block.)

Because training MSEs against `∇_z E`, the outer `loss.backward()` is a
second-order autograd path through every attention call. PyTorch's flash /
mem-efficient SDPA kernels do not implement double-backward, so the EBM forces
`SDPBackend.MATH` (see `docs/implementation_pitfalls.md` §5).

### 3.4 The diffusion wrapper: schedules, inner loop, normalization

`GaussianLatentDiffusion` owns the DDPM schedules and the train/inference loops.
Salient choices, all hard-won (see `docs/implementation_pitfalls.md`):

- **`cosine` β schedule**, not linear — linear at small `T` saturates or NaNs.
- **Latent normalization as buffers.** AE-native latents are anisotropic, so the
  noise schedule (designed for `N(0, I)`) is miscalibrated against them. A
  per-dim `latent_mu`/`latent_sigma` maps AE latents → `N(0, I)`-like space and
  the inverse is applied to anything that leaves the module. Computed once at the
  start of training, stored as buffers so they ship with the checkpoint — the
  LD4LG / Stable-Diffusion `0.18215` trick.
- **`opt_step` — bad-step rejection (the heart of IRED).** Each inner refinement
  step is kept only if it lowers the energy:

  ```python
  for _ in range(step):
      E, grad = model(z_q, z, t, return_both=True)
      z_new   = z − step_size · grad
      E_new   = model(z_q, z_new, t, return_energy=True)
      z       = where(E_new > E, z, z_new)    # reject any step that raises energy
  ```

  Because `E` is calibrated (§1.3), this accept/reject is meaningful — it is what
  distinguishes IRED's iterative reasoning from plain diffusion sampling.

- **Optional stochastic inner loop.** The deterministic step is greedy and can
  stall at the first energy bump; an annealed-Langevin / SGLD variant
  (`opt_noise_scale > 0`) lets the chain explore instead. Off by default. The
  trade-off and the energy-is-a-proxy caveat are in §5.4; config in
  `docs/ebm.md` §5.1.1.

### 3.5 Training and inference loops

**Training** (encoder & decoder frozen — only the EBM updates):

```python
z_q  = LatentEnc(Q)                          # conditioning
z_a  = LatentEnc(A)                          # clean target
t    = uniform(0, T);  eps = randn_like(z_a)
z_t  = sqrt(alpha_bar_t)*z_a + sqrt(1-alpha_bar_t)*eps

E_t      = EnergyNet(z_q, z_t, t)            # scalar
eps_hat  = grad(E_t.sum(), z_t, create_graph=True)

L_denoise = mse(eps_hat, eps)
L_nce     = relu(margin + E(clean) − E(hard_negative)).mean()   # see §5
L         = L_denoise + λ·L_nce
L.backward()                                 # updates only EnergyNet
```

**Inference** (`T` outer DDPM steps, each followed by `N_inner` refinements;
decoder invoked exactly once):

```python
z_q = LatentEnc(Q);  z = randn(latent_shape)
for t in reversed(range(T)):
    z = p_sample(z_q, z, t)                  # one DDPM step
    for _ in range(N_inner):                 # IRED refinement
        E_before = EnergyNet(z_q, z, t)
        eps_hat  = grad(E_before.sum(), z)
        z_new    = z − step_size · eps_hat
        E_after  = EnergyNet(z_q, z_new, t)
        z = where(E_after < E_before, z_new, z)   # bad-step rejection
A = LatentDec(z)
```

Test-time compute scales with `N_inner × T`; the decoder fires once.

---

## 4. Mode-1 / Mode-2, and distilling the actor

### 4.1 The framework

LeCun's APTAMI separates a fast reactive policy from a slow deliberative planner:

**Mode-1 — Reactive agent:** `Enc(X) → S → Actor(S) → A`. An action directly from
the perceived state, no trajectory search.

**Mode-2 — Planning agent:** `Enc(X) → S → Pred(S, A) → S' → ... → optimal path`.
Unroll a world model under an action proposal and search for the action that
minimizes a cost over the rollout. Mode-2 is **constraint satisfaction, not
sampling**: an action is chosen because it minimizes an explicit cost, not
because it has high likelihood.

**Combining the two:** `argmin_A D(Actor(S), A) s.t. Cost(Pred(S, A)) minimized` —
the Mode-1 actor is distilled to imitate Mode-2's optimum, so deployment can fall
back to a single forward pass.

This proposal instantiates the *cost-and-optimizer* half of Mode-2 in latent
space: the EBM is the cost, reverse diffusion with bad-step rejection is the
constrained search, and `z_0` is the chosen "action" (the answer latent). It does
not implement the full Mode-2 world-model loop `Pred(S, A) → S'` — the EBM scores
candidates directly rather than predicting the consequence of an action and
unrolling a trajectory. The mapping to APTAMI is therefore partial: the
cost/optimizer split is faithfully carried over, but without environment dynamics
the "planning" is single-step optimization rather than rollout search.
Token-CoT, by contrast, is Mode-2 forced through the discrete substrate of §1.1.

### 4.2 The energy trajectory is a distillation target

The inner loop produces not just a denoising vector but a full **energy
trajectory** — a sequence `z_T → z_{T−1} → ... → z_0` that each step provably
decreases `E` (bad-step rejection guarantees it). That trajectory is a
supervision signal: once the Mode-2 EBM is trained, a separate **Mode-1 actor**
`Actor_θ(z_q) → z*` can be distilled to reproduce its converged output in a single
forward pass.

```python
# Encoder, decoder, EBM all frozen; only Actor_θ updates
for Q in dataloader:
    z_q = LatentEnc(Q)
    with torch.no_grad():
        z_final = p_sample_loop(EnergyNet, z_q, T=T, N_inner=N_inner)  # teacher rollout
    z_hat = Actor_θ(z_q)
    L_act = mse(z_hat, z_final)          # fixed-shape → fixed-shape regression
    L_act.backward()
```

Notes:

- **No backprop through the teacher.** `p_sample_loop` runs under `no_grad`;
  `z_final` is a fixed regression target. This is what makes the actor cheap to
  train despite the teacher's second-order autograd.
- **Replay buffer.** Caching `(z_q, z_final)` pairs amortizes the teacher rollout
  to one per `Q` for the whole actor run.
- **EBM-weighted loss (optional).** Weight each pair by `exp(−E(z_q, z_final, 0))`
  so the actor imitates the teacher's *confident* minima and ignores ones where
  Mode-2 itself didn't converge well. Cheap — the EBM is already on the GPU.

Why this target is cleaner than AR-CoT distillation: distilling token-CoT into a
non-CoT model means compressing a variable-length sequence into a shorter one — an
autoregressive-to-autoregressive problem with no closed-form "optimum to
imitate." Here the source is a fixed-shape `z_q` and the target is a fixed-shape
`z_final`; the actor learns a fixed-input → fixed-output regression, among the
best-behaved learning problems available.

### 4.3 Adaptive Mode-1 / Mode-2 inference

At deployment, default to the fast actor and escalate to the slow optimizer only
when the EBM says the actor's output is poor — the EBM doubles as a confidence
scorer, so no separate verifier is needed:

```python
z_q   = LatentEnc(Q)
z_hat = Actor_θ(z_q)                       # one forward pass
E_hat = EnergyNet(z_q, z_hat, t=0)         # calibrated quality scalar

if E_hat < threshold:                      # actor confident
    z_final = z_hat
else:                                      # hard case → fall back to Mode-2
    z_final = p_sample_loop(EnergyNet, z_q, T=T, N_inner=N_inner)
    # optional: add (z_q, z_final) to the actor's replay buffer
A = LatentDec(z_final)
```

`threshold` is calibrated on held-out data so that, say, 80% of queries take the
Mode-1 path. Average inference cost becomes a tunable knob between one forward
pass and the full `T × N_inner` rollout — and unlike a bolt-on confidence head,
the gate is *the same scalar* that defined correctness during Mode-2 training, so
there is no second calibration problem.

This stage is the natural follow-up once the test-time-compute curve (§7,
Milestone 3) is in hand; the §1.2 commitments earn their keep at deployment time
— the Mode-2 path is the cost-anchored optimizer that *defines* correctness in
latent space, the Mode-1 path is the fast approximation that inherits its
guarantees, with the EBM still available as a fallback verifier.

---

## 5. Design decisions

### 5.1 Why NCE is load-bearing (the JEPA question)

LeCun's argument that contrastive terms can be dropped in JEPA-style models does
**not** transfer here, for the same reason it doesn't transfer to IRED:

- JEPA's anti-collapse mechanisms (VICReg, BYOL asymmetry) calibrate *which
  directions in feature space carry signal*.
- They do **not** calibrate the *absolute scalar value* of an energy function.
- Bad-step rejection requires `E(z_a) < E(z_b)` to be a reliable proxy for "`z_a`
  is a better candidate than `z_b`" — an absolute-value property, not a direction
  property.

So energy-landscape supervision is load-bearing, not optional. The NCE term mines
a **hard negative** from the current energy landscape: heavy-noise the clean
target, run two `opt_step` refinements to find a low-energy (confusing) point,
then push `E(clean) < E(that point)`. The mined negative is detached — it shapes
*where* the contrast sits, contributes no gradient itself. (This is why the inner
loop is reused in training even though bad-step rejection is, on its face, an
inference trick: it is the negative *sampler*, drawing negatives from the same
optimizer inference will run.)

*Caveat: moving-target instability.* Because the hard negative is mined from the
current energy landscape and the landscape is updated every training step, the
negative and the energy function co-evolve — the same joint-optimization dynamic
that can destabilize GAN training. In practice the negative is detached (no
gradient flows through it), so the coupling is one-directional (the EBM chases a
moving negative) rather than fully adversarial. Empirically this has been stable
in IRED, but it warrants monitoring: if the NCE margin saturates or the
`e_real/e_fake` ratio oscillates, the negative-mining schedule may need to lag
behind the EBM update rate.

### 5.2 Latent MSE alone may not punish the right errors

The default loss lives entirely in latent space — `MSE(ε̂, ε)` over the `K·d`
entries plus the NCE contrast. This is the LD4LG recipe, but it has a known
weakness when decoded outputs have near-isomorphic forms whose correctness flips
on a few tokens:

- Two MBPP solutions differing only by `<` vs `<=` sit *very* close in MSE
  distance while being maximally different in correctness.
- MSE treats every latent dimension equally; it cannot "spend more budget" on the
  operator-bearing tokens.

The decoder *does* know this distinction — its CE explicitly punishes the wrong
token. So an optional low-`t` frozen-decoder CE auxiliary mixes in:

```
L = L_denoise + λ·L_nce + λ_dec · 1[t < t_max] · CE(LatentDec(x0_hat), gold)
```

where `x0_hat = predict_start_from_noise(z_t, t, ε̂)`. Two design notes:

1. **Gate on low `t`.** `x0_hat` is only a reliable clean estimate near the end
   of the schedule; at high `t` it is too noisy and backpropping CE through it
   injects noise into the EBM. Restrict to `t < t_max` (default 2 of 10).
2. **Backprop, don't replace.** The decoder stays frozen (zero gradient), but CE
   depends on `x0_hat`, which depends on the EBM via `ε̂` — so gradient flows back
   to the EBM *through* the frozen decoder, treating it as a fixed perceptual loss
   over latents.

Kept as an auxiliary, not the primary objective: a decoder-CE-only loss would
push `∇E` toward "whatever latent the decoder happens to decode well," which is
not the same set the latent MSE calibrates, and would break the bad-step contract
at inference. Off by default; recommended for ablation on short-answer tasks.

### 5.3 The generator-grounded negative

The geometric NCE negative (§5.1) is mined purely by geometry — the energy field
is shaped without any knowledge of what the decoder/point-generator actually
emits. *The energy manifold does not know the shape of the generator.* A low-energy
minimum can therefore sit off the decodable manifold (it manifests as the eval's
exposure-bias gap: energy looks good, the decode is wrong).

The **generator-grounded negative** closes that gap. It decodes the mined
latent's clean estimate through the *inference (no-copy)* generator path, and only
when that decode is wrong (gold-CE above a threshold) treats the latent as a
validated negative and pushes `E(real) + margin < E(refined)`. The decode/CE is a
**detached gate** — the learning signal is the energy margin, so no gradient flows
through the decoder or the inner loop (cheap; reuses the sample NCE already
mined). This is a second concrete anchor in the §2.3 composite-cost program: it
forces the energy minima to coincide with latents the generator can actually
render. Full detail in `docs/ebm.md` §4.3.

### 5.4 The inner loop is greedy — an optional stochastic search

Bad-step rejection (§1.3, §3.4) is **monotone greedy descent**: a refinement step
is kept only if it lowers `E`. That is what makes the accept/reject meaningful, but
it has two costs. First, for a *deterministic* gradient step an energy *increase*
almost always means the step overshot the local minimum — yet the rejection leaves
`z` unchanged, so the next inner iteration recomputes the identical gradient at the
identical point and stalls, wasting the rest of the inner budget. Second, strict
monotone descent cannot cross a small energy ridge to reach a deeper basin; it
freezes at the first bump.

The cheap fix for overshoot is step-size backtracking, but the more general option
is to make the inner loop **stochastic** — an annealed Langevin / SGLD step
(`opt_noise_scale > 0`):

```
z ← z − α·∇E + σ·ξ,   σ = opt_noise_scale · √(2α) · decay      (ξ ~ N(0, I))
```

`decay` falls linearly to 0 on the last inner step, so the chain explores early and
settles into a minimum at the end; `opt_noise_scale ≈ 1` targets the Boltzmann
density `p ∝ exp(−E)`, smaller is greedier. The `opt_reject` flag then chooses the
regime:

- **reject = true + noise** — *stochastic greedy*: fresh noise each iteration
  breaks the deterministic stall, but only downhill moves stick (still cannot cross
  a ridge).
- **reject = false + noise** — *true Langevin*: uphill moves are accepted, so the
  chain can climb over a small bump into a deeper basin.

This is **inference-only** — training negative-mining (§5.1) stays deterministic,
since the mined negative only needs to be hard and is detached anyway. The default
is the deterministic IRED behavior; the stochastic path is opt-in. Config and the
exact update live in `docs/ebm.md` §5.1.1.

**The caveat that decides how to use it — and what it does *not* concede.** The
whole architecture rests on `E` being a calibrated quality scalar (§1.3): bad-step
rejection, the §4.3 gate, and the §4.2 distillation weighting all read the *same*
scalar, so they stand or fall together on that one property — which is not assumed
but *measured*, as the `E`–pass/fail correlation in §7's Milestone 2. Stochastic
search does not weaken that claim; it stress-tests it. `E` is best calibrated where
it was trained — near the data manifold and the deterministic trajectory the
NCE/gen-neg negatives are mined along. Pushing harder with Langevin can drive `z`
into regions `E` never saw, where a lower energy is *extrapolation*, not a better
answer (the off-manifold minima §5.3 attacks). So the risk is **search-induced
distribution shift**, not "`E` is only a direction": on the deterministic path `E`
is the quality judge the design claims; aggressive exploration is simply where its
calibration is most likely to run out.

That is why, *in the stochastic regime and only there*, it is worth cross-checking
the candidates the search visits against the external verifier or the no-copy
decoder CE — not as a confession that `E` cannot judge quality, but as (a) the
ground-truth signal `E` is calibrated against in the first place, and (b) a guard
for exactly the extrapolation regime aggressive search creates. For code that
verifier is cheap and available at inference, so best-of-K-by-verifier is a genuine
option — but it is a domain luxury, not the general claim. The general architecture
has only `E` at deployment; if Milestone 2 finds the `E`–correctness correlation is
low, the honest read (§8) is that the calibration bet failed — not that `E` was
only ever a search direction.

---

## 6. Prior work in this exact shape

This architecture has precedent for *generation*, not yet for energy-based
reasoning:

- **LD4LG** (Lovelace et al., 2023) — frozen BART encoder/decoder + Gaussian
  latent diffusion. Demonstrates the autoencoder-bottleneck approach works for
  text.
- **PLANNER** (Zhang et al., 2023) — frozen autoencoder + latent paragraph
  diffusion.
- **GENIE, DiffuSeq, SeqDiffuSeq** — sequence-to-sequence latent diffusion.
- **Coconut** (Hao et al., 2024) — continuous "thoughts" fed into an AR LLM.
  Closest in spirit, but keeps AR decoding and skips diffusion.
- **Diffusion-of-Thought** — diffusion over CoT tokens directly, mostly discrete.
- **Process Reward Models** (Math-Shepherd, "Let's Verify Step by Step") — trained
  scalar scorers over partial reasoning. Used for reranking, not gradient-based
  refinement.

Two closely related concurrent works converge on the same intersection from
different directions, but each holds only one half of the transplant — either the
energy-based optimizer (IRED's lineage) or the frozen-AE bottleneck (LD4LG's),
never both, and never aimed at reasoning:

- **Energy-Based Transformers** (Gladstone, Du et al., 2025) — energy minimization
  via gradient descent for text reasoning. The closest existing work in spirit: it
  has the energy-based optimizer, but operates directly in the model's embedding
  space rather than in a frozen autoencoder bottleneck. It holds the IRED-side
  ingredient, not the LD4LG one.
- **STAR-LDM** (Lovelace et al., 2025) — latent diffusion planning for language
  from the same LD4LG authors. Holds the frozen-AE bottleneck, but uses standard
  diffusion with no energy-based optimizer.

**The contribution is the synthesis, and the bet it rests on** — not a checklist of
mechanisms. Concretely:

1. **The transplant.** Put IRED's energy-diffusion optimizer — its calibrated
   scalar energy *and* its inner-loop bad-step rejection, adopted as a single unit —
   inside a frozen text autoencoder's latent space, and aim it at *reasoning* rather
   than generation. The concurrent works above each sit on one side of this; none
   make the join.
2. **The bet.** That the latent energy, calibrated against decoder reconstruction,
   tracks *reasoning correctness* well enough to gate refinement (§3.4) and rank
   candidates (§4.3). This is the falsifiable claim, and §8 treats it as one.

Bad-step rejection is a mechanism inherited wholesale from IRED, not a
differentiator — "competitor X lacks it" merely means X did not build on IRED. The
novelty is the combination and the calibration bet.

---

## 7. Concrete first experiment

A realistic single-GPU prototype to validate the core thesis.

**Why code, not math or token-CoT.** GSM8K answers are 1–3 numeric tokens (~13
bits) — too little structure for "reasoning *in* latent space" to be testable.
ProofWriter/BBH give richer structure but the correctness-bearing part is a
one-token verdict, and the proof chains are themselves token-CoT traces — which
puts the prototype back in latent-encoded-token-CoT territory and defeats the
decoder-transduces commitment. **Code generation cleans up both:** the target is a
structured artifact (a Python function, 50–200 tokens of identifiers, operators,
control flow), so the latent manifold has volume and the EBM has room to refine;
the decoder genuinely transduces (Python's grammar is rigid, the decoder isn't
problem-solving); and correctness is verifiable by an **external** signal
(`assert` + subprocess), independent of `E` — the strongest possible test of the
calibrated-scalar claim.

**Setup:**
- **Task:** MBPP (974 examples, full config) — NL description, canonical Python
  solution, ≥3 assert tests. Primary EBM-training corpus. **HumanEval** (164) is
  held-out eval only.
- **AE pretraining corpus:** OpenWebText. The AE is deliberately **never** trained
  on MBPP/HumanEval, to keep the latent space free of EBM-training-distribution
  leakage and preserve the §2.2 anchoring property.
- **LatentEnc/LatentDec:** `flan-t5-base` (d_model=768), frozen. Learned
  `AttentionPool` (K=128–256) + `ReconstructionNet`.
- **Energy net:** 4-layer transformer over `(z_q, z_t, t)`; scalar via squared-sum
  head.
- **Diffusion:** `T=10`, `cosine` schedule, `continuous=True`. Inner loop: 5 steps
  with bad-step rejection.
- **Loss:** denoising MSE + NCE (`λ=1`). Optional low-`t` decoder-CE (§5.2) and
  generator-grounded negative (§5.3).
- **Verifier:** decoded code executed in a subprocess with a 5s timeout. *Not
  sandboxed* — fine for offline eval on MBPP/HumanEval, never point it at
  adversarial input.

Practical defaults that survived first-run debugging (the *why* for each lives in
`docs/implementation_pitfalls.md`): `cosine` schedule; `weight_decay=0.01`
(load-bearing — bounds head magnitude, preserves NCE contrast); `x_start_clamp=5.0`,
`envelope_sf=None`; `SDPBackend.MATH`; per-step monitoring of
`eps_scale = ||ε̂||/||noise||` (target ≈ 1.0) and the `e_real/e_fake` ratio.

**Milestones:**

1. **Autoencoder check** — `LatentDec(Recon(Pool(LatentEnc(code))))` round-trips on
   the eval corpora *even though the AE saw only OpenWebText*. Bars: MBPP test
   pass-rate ≥ 80%, HumanEval ≥ 80%, OWT byte-exact recon ≥ 50%.
   These are aspirational — an 80% HumanEval pass-rate through a frozen
   general-purpose T5 encoder/decoder trained only on OpenWebText is aggressive;
   a more plausible initial bar is 30–50% on both code corpora, with the
   critical signal being that round-tripping preserves enough structure for the
   EBM to refine (i.e., the `inner=N − inner=0` gap in Milestone 2 is positive
   despite imperfect reconstruction). If OWT clears but both code bars fail
   badly, the OWT latent space doesn't transfer to Python — recover by adding a
   code slice to AE pretraining or swapping to CodeT5.
2. **Diffusion training** — train the EBM on MBPP. Watch `eps_scale` and the
   `e_real/e_fake` ratio (either drifting is the head-inflation warning sign).
   Report `pass(inner=N) − pass(inner=0)` on both corpora: positive deltas mean
   iterative refinement is doing real work. Also report Pearson correlation
   between `E(z_q, z, 0)` and observed pass/fail — the calibrated-scalar claim,
   with code execution as the cleanest external signal.
3. **Test-time compute curve** — plot pass-rate vs. `N_inner × T` and compare to
   AR-CoT at matched FLOPs.

**Decision rule:** if the diffusion curve is steeper than AR-CoT at matched
compute *and* the `inner=N` vs `inner=0` gap is positive and grows with N, the
thesis is validated. If accuracy plateaus below AR-CoT, or the two are
indistinguishable, that is an informative negative about the smoothness of
reasoning in this latent space. A separate signal: high `E`–pass-rate correlation
(>0.5) means the calibrated-scalar claim holds even when accuracy is modest —
which is what justifies using `E` as the §4.3 gate.

---

## 8. Honest assessment

**Promising:**
- Solves the "no `c*`" problem elegantly via the frozen encoder.
- Cleanly separates thinking from surface-form generation — the actuator-style
  boundary that distinguishes true Mode-2 from latent-guided prompting.
- Test-time compute scaling via inner-loop steps fits the o1-era frontier, and is
  governed by an explicit cost rather than a sampling temperature.
- Architectural precedent (Stable Diffusion, LD4LG) suggests the frozen-AE
  approach is viable.

**Risks:**
- Autoencoder quality bottleneck — most teams who tried latent-diffusion-for-text
  reported the decoder as the limiting factor. It also caps world knowledge: the
  EBM searches the space the AE defines, it does not store facts (§9 attacks this).
- RL-on-AR-CoT (R1, o1) is the current empirical winner. Any diffusion-based
  reasoner has to justify *not* using RL.
- Reasoning may not be smooth in latent space. If correct reasoning traces are
  isolated discrete modes rather than connected regions, diffusion will struggle —
  and the descent degenerates from "reasoning" into "retrieve the nearest
  memorized answer-latent." The `inner=N − inner=0` gap and a trajectory-decode
  diagnostic are the tests that distinguish the two.
- The decoder-as-transducer assumption only holds for constrained output formats.
  This prototype targets code; the approach is structurally restricted to domains
  (code, formal logic, structured planning) where rendering doesn't itself require
  reasoning. Open-ended natural language is out of scope — a real limitation, not
  a temporary one.
- Collapsing the composite cost (§2.3) to a single anchor is deliberate. If the
  single anchor proves gameable, additional anchors must be added and the
  bad-step contract re-checked under each.

**Why try it anyway.** The architecture is clean, the prior work is encouraging,
and even a clear negative tells us something concrete about the geometry of
reasoning in continuous space. Brains, by every available indication, do not
reason in language — language regions are largely inactive during reasoning, and
verbalization happens at output. If artificial reasoning is ever to reach the
fluid cognition biological systems exhibit, the substrate likely has to be
representational rather than lexical. The current latent-reasoning literature
gestures at this without implementing it; this prototype is the smallest concrete
step toward the system the framework actually demands.

---

## 9. Extension: retrieval / knowledge-base conditioning (external knowledge)

**Status: deferred design, not in the §7 first experiment.** This extends the
architecture with external knowledge and is the answer to the knowledge ceiling
in §8: the EBM's world knowledge is otherwise bounded by whatever the
small frozen `LatentEnc`/`LatentDec` can encode and render. Scaling the EBM's
weights does not add knowledge — it learns to *search* the latent space the AE
defines, not to store facts. The fix is to **decouple knowledge from reasoning**:
keep facts in an external store the energy net *reads*, and let the EBM weights
carry only the reasoning. This is the RETRO / kNN-LM / KBLaM split (retrieve into
the architecture, don't memorize in the weights) transplanted onto the Mode-2
optimizer.

### 9.1 The conditioning block

The energy net already treats `z_q` as pure conditioning — attended over, but it
receives **no gradient** (§3.3). Retrieved knowledge `z_k` gets the identical
treatment, as a *third* conditioning block. The semantic is exactly right:
**knowledge conditions the reasoning; it is not the thing being reasoned toward.**

```
today:        [ t(1) | z_q+type0 (K) | z_t+type1 (K) ]                    # (B, 1+2K,   d)
with z_k:     [ t(1) | z_q+type0 (K) | z_t+type1 (K) | z_k+type2 (M) ]    # (B, 1+2K+M, d)
```

Three additive edits, all backward-compatible when `z_k = None`:

1. **`type_emb` 2 → 3** — add type 2 for the knowledge block.
2. **Append `z_k` last, fix the head slice.** The head reads `h[:, 1+K:]` today
   ("everything after `z_q`"); with a trailing `z_k` that slice would swallow it.
   Change to the explicit z_t window `h[:, 1+K : 1+2K]`. This is the only
   correctness-bearing change; everything else is additive.
3. **No within-memory positional embedding.** Retrieved passages are a *set*, so
   the permutation-invariant choice (just `z_k + type2`) is also principled. Add a
   coarse per-doc id embedding only if boundaries turn out to matter.

The gradient path is unchanged: `ε̂ = ∇_{z_t} E(z_q, z_t, z_k, t)` still
differentiates only w.r.t. `z_t`. Bad-step rejection, NCE, and the denoising MSE
are untouched; the energy surface is simply conditioned on more context.

### 9.2 Retrieve → encode → compress

```
Q ──► retriever (frozen; BM25 or dense) ──► top-m passages
   passages ──► LatentEnc (frozen, same encoder as z_q) ──► hidden states
   hidden states ──► memory resampler (Perceiver cross-attn) ──► z_k ∈ R^(M×d_ae)
```

- **Compress to a fixed `M` budget** (~32–64); `m` passages × `K` slots each is too
  many tokens (mean-pool-per-passage is the cheap baseline).
- **Precompute and cache `z_k` per example**, exactly as the verifier fixtures
  already live on the dataset record — retrieval over a frozen corpus is
  deterministic, so it runs once, with no change to the hot training loop.
- **Normalize `z_k` into the diffusion `N(0, I)` space** like `z_q` (reuse
  `latent_mu`/`latent_sigma`; add a separate `mem_*` buffer if stats diverge).

### 9.3 Train/inference parity and the contamination guard

Unlike the gold-answer auxiliaries (§5.2, §5.3), retrieval is keyed on **`Q`, not
`A`**, so `z_k` is present at *both* train and inference — a legitimate input with
no leakage. The one rule: **hold the gold answers out of the retrieval corpus**
(and dedup near-duplicates), or the retriever returns the answer and the EBM
learns nothing — the standard RAG contamination guard.

### 9.4 Where it touches the code

| File | Change |
|---|---|
| `ired/model/energy_net.py` | `type_emb` 2→3; optional `z_k` in `EnergyTransformer.forward` and `DiffusionWrapper.forward`; head slice → `h[:, 1+K : 1+2K]` |
| `ired/model/diffusion.py` | thread `z_k` through `opt_step` / `p_sample` / `p_losses`; normalize `z_k`; `sample(z_q, z_k=...)` |
| `ired/data.py` | retrieval + per-example `z_k` caching |
| `ired/train_diffusion.py`, `configs/ebm.yaml` | `--n-mem`, retrieval corpus path, top-`m`, on/off flag |

### 9.5 Design decisions and risks

- **Checkpoint migration.** Growing `type_emb` 2→3 means existing EBM checkpoints
  load with `strict=False`: copy the two existing rows, zero-init row 2 (type-2
  begins as a no-op offset). Must be handled or `load_state_dict` fails.
- **Fair baseline.** When comparing against AR-CoT (§7, Milestone 3), feed the AR
  baseline the *same retrieved passages in-context* — otherwise it is a rigged
  comparison.
- **Does it help *here*?** MBPP/HumanEval are more reasoning than knowledge-lookup,
  so generic document retrieval may add little. The corpus likely to help is
  **retrieved exemplars** (similar solved problems — retrieval-as-few-shot) or
  library/API docs. Choose the corpus for signal, not convenience.
- **Attention cost.** Sequence length goes `1+2K → 1+2K+M`; with `K=256`, `+64` is
  negligible beside the double-backward MATH-kernel cost (`docs/implementation_pitfalls.md` §5).

### 9.6 Mapping back to the three commitments

- *Inference is optimization, not sampling.* Unchanged — `z_k` only conditions the
  energy; the inner loop is still gradient descent with bad-step rejection.
- *The cost is structurally anchored.* **Unchanged but better-informed** — `z_k`
  gives the energy network richer context for distinguishing good from bad
  latents. This is conditioning, not an independent cost anchor: it changes the
  input to `E`, not what the scalar is constrained to correlate with. The next
  proper anchor (external verifier, decoded-output stability under perturbation)
  is deferred.
- *The decoder transduces, it does not reason.* Unchanged — `z_k` enters the EBM,
  never the decoder; `f_ψ → Decode` still fires exactly once on `z_0`.

### 9.7 Suggested rollout

1. **Plumbing with mock `z_k`.** Wire `z_k` end-to-end (`energy_net` → `diffusion`
   → trainer) with random/zero memory slots. Verify `z_k=None` is byte-identical
   to the current model, `z_k` provided trains, and checkpoint migration works. No
   retrieval yet.
2. **Real retrieval.** Add the retriever + resampler + caching in `ired/data.py`,
   pick the exemplar/doc corpus per §9.5, and run the augmented-vs-baseline
   comparison with the matched-context AR baseline.
