# LLM Reasoning via IRED — Inspired by *A Path Towards Autonomous Machine Intelligence*

This proposal is a concrete instantiation of Yann LeCun's Mode-1 / Mode-2 framework from *A Path Towards Autonomous Machine Intelligence* (APTAMI), applied to language reasoning. It combines two existing pieces of work:

1. **IRED** — *Iterative Reasoning through Energy Diffusion* (Du et al., ICML 2024). Provides the Mode-2 optimizer: an energy-based diffusion model whose scalar energy is calibrated to be a quality proxy, with inner-loop bad-step rejection.
2. **LD4LG** — *Latent Diffusion for Language Generation* (Lovelace et al., NeurIPS 2023). Provides the frozen-autoencoder bottleneck: a pretrained text encoder/decoder pair with diffusion operating purely in their latent space.

The result is a system that forgoes the generative nature of a traditional language model and treats "thinking" as a continuous latent diffusion process between a frozen encoder and a frozen decoder. Reasoning is gradient descent on a learned energy, not token sampling; surface form is rendered exactly once, after the latent has converged.

---

## 1. Motivation

### 1.1 Mode-1 and Mode-2 in *A Path Towards Autonomous Machine Intelligence*

LeCun's framework separates a fast reactive policy from a slow deliberative planner:

**Mode-1 — Reactive agent (inference):**
```
Enc(X) → S[0] → Actor(S[0]) → A[0] → return
```
Mode-1 produces an action directly from the perceived state with no trajectory search.

**Mode-2 — Planning agent (inference):**
```
Enc(X) → S[0] → Pred(S[0], A[0]) → S[1] → ... → optimal path → return A[0]
                              ↳ A[0] → ...
```
Mode-2 unrolls a world model `Pred(S, A)` under an action proposal `A` and searches for the `A[0]` that minimizes a cost `Cost(S)` over the rollout. Crucially, Mode-2 is **constraint satisfaction**, not sampling: an action is chosen because it minimizes an explicit cost, not because it has high likelihood under some learned generative distribution.

**Combining Mode-1 and Mode-2:**
```
argmin_A  D(Actor(S), A)   s.t.   Cost(Pred(S, A)) is minimized
```
The Mode-1 actor is distilled to imitate Mode-2's optimum, so deployment can fall back to a single forward pass.


### 1.1a Translating the framework to language modeling

The cleanest mapping from LeCun's Mode-2 onto current LLM practice is this: **chain-of-thought (CoT) is Mode-2 carried out in token space**. When an LLM emits intermediate reasoning tokens before its final answer, it is not really "thinking out loud" — it is searching for a token prefix whose conditioning yields a better final-answer distribution. Each emitted token plays the role of an action `a[t]`, the resulting prefix is the new state `s[t+1]`, and the implicit cost being minimized is roughly the negative log-likelihood of the gold answer given the prefix. RL-trained reasoners (o1, R1) make this almost literal: the reward shapes which prefixes the policy prefers, which is the cost-function role in LeCun's diagram.

Token-space Mode-2 works, but it inherits three properties of the substrate that are accidents of the medium rather than features of reasoning:

1. **The search space is discrete and combinatorial.** There is no useful gradient with respect to an action; refinement happens via sampling and rerolling.
2. **The world model and the actor are the same network**, both rolled out autoregressively. There is no clean separation between *predicting the consequence of a step* and *proposing the next step* — the Pred / Actor split that LeCun's diagram relies on collapses.
3. **The cost is implicit.** Whatever quality signal exists is baked into the policy by pretraining and RL. You cannot point at it, evaluate it on an off-policy candidate, or use it as a verifier.

The question this proposal asks — and that motivates everything that follows — is whether the same search can be carried out in a continuous latent space, where each of those three properties flips: gradient descent replaces sampling, the energy net is structurally distinct from the renderer, and the cost becomes an explicit scalar that doubles as a verifier. The rest of §1 argues that energy-based diffusion is precisely the construction that makes this flip possible.

### 1.2 Latent reasoning today is *latent-guided prompting*, not Mode-2

Recent "latent reasoning" work — Coconut, LaDiR, Soft Thinking — runs computation in continuous space but then hands the result to an autoregressive LLM that does the actual reasoning during decoding. On close inspection these systems are **latent-guided prompting**: the latent serves as rich conditioning, and the decoder inherits whatever cognitive capacity the underlying LLM has. They are not faithful to LeCun's formulation because (a) inference is generative sampling, not optimization against a cost, and (b) the decoder is doing reasoning, not transduction.

A faithful Mode-2 system in latent space needs three structural commitments:
1. **Inference is optimization against an explicit cost**, not sampling from a learned distribution over "what reasoning looks like."
2. **The cost signal is structurally anchored** so it resists being gamed (LeCun's hand-specified Intrinsic Cost serves this role for embodied agents; a language reasoner has to substitute something).
3. **The decoder transduces, it does not reason** — the analogue of LeCun's actuator boundary. The full cognitive burden lives in the latent optimization.

### 1.3 Why energy-based diffusion is the right substrate

A diffusion model trained by score matching learns `∇log p(x) = −∇E(x)` for an implicit energy `E(x) = −log p(x)`; reverse diffusion *is* iterative gradient descent on that energy. So a trained diffusion model already defines a cost function whose gradient field is anchored in its training data and whose minima sit on the data manifold.

IRED (Du et al., ICML 2024) makes this explicit. The energy `E` is no longer implicit — it is learned to be a *calibrated scalar proxy for answer quality* via an NCE term (`E(clean) < E(noisy)`) and exploited at inference via bad-step rejection (each refinement step is accepted only if `E` decreased). This is, to our knowledge, the cleanest existing instantiation of LeCun-style Mode-2 in continuous space. It has not been scaled to language.

This proposal scales it: keep IRED's energy-diffusion core as the Mode-2 optimizer, isolate it between a frozen encoder and a frozen decoder so that no reasoning leaks into surface-form generation, and use the encoder of the answer text — `LatentEnc(A)` — as the structural anchor for the cost. The result satisfies, by construction, the three commitments of §1.2.

### 1.4 Closing the loop: training a Mode-1 actor by distillation

Because the inner loop of the energy diffusion model produces not just a denoising vector but a full **energy trajectory** — a sequence of refinements `z_T → z_{T−1} → ... → z_0` that each provably decrease `E` (bad-step rejection guarantees this) — that trajectory is a supervision signal in its own right. I can faithfully implement a separate model `Actor(X) → z*` by distilling the slow-acting energy diffusion model into a single forward pass.

Concretely, once the diffusion model is trained, we collect pairs `(z_q, z_final)` where `z_q = LatentEnc(Q)` and `z_final = p_sample_loop(z_q)` is the latent the IRED outer-and-inner loop converges to. A small feed-forward (or single-pass denoising) network is then trained to regress `z_final` directly from `z_q`:

```
argmin_θ   E_Q [ || Actor_θ(z_q) − z_final ||² ]
         where z_final = argmin_z E(z_q, z, t=0)  (obtained by the trained Mode-2 optimizer)
```

This realizes LeCun's combined Mode-1/Mode-2 objective from §1.1: the fast actor is trained to imitate the slow optimum, and deployment can fall back to a single forward pass once the imitation is faithful enough.

Two reasons this matters here, beyond the generic LeCun framing:

- **Deployment economics.** Mode-2 inference costs `T × N_inner` energy-network passes per query (50 by default; see §7.6). Mode-1 inference costs one. After distillation the system defaults to Mode-1 and only invokes Mode-2 on hard inputs — and we get a natural difficulty signal for free, since the trained EBM doubles as a confidence scorer: `E(z_q, Actor(z_q), 0)` is exactly the calibrated quality scalar §2 calls out. High energy → escalate to Mode-2, re-run the inner loop, and (optionally) add the resulting `(z_q, z_final)` pair to the actor's training buffer.
- **A structurally clean distillation target that AR-CoT lacks.** Distilling token-CoT into a non-CoT model means compressing a variable-length token sequence into a shorter one — fundamentally an autoregressive-to-autoregressive problem with no closed-form notion of "the optimum to imitate." Here the source is a fixed-shape latent `z_q` and the target is a fixed-shape latent `z_final`. The actor only has to learn a fixed-input-fixed-output regression, which is among the best-behaved learning problems available.

Out of scope for the §9 first experiment, which validates the Mode-2 optimizer in isolation. Mode-1 distillation is the natural follow-up once Milestone 3 (the test-time compute curve) is in hand, and is also where the §1.2 commitments earn their keep at deployment time: the Mode-2 path is the cost-anchored optimizer that *defines* what "correct" means in latent space; the Mode-1 path is the fast approximation that inherits its guarantees, with the EBM still available as a fallback verifier.

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
   X  ────▶│  LatentEnc  │────▶│  Latent Thinking   │────▶│  LatentDec  │─────▶ Result
            │  (frozen)  │  z_q │   (IRED in latent  │  z_0 │  (frozen)  │   A
            └────────────┘      │       space)       │      └────────────┘
                                └────────────────────┘
```

- **LatentEnc** maps a question `Q` to a conditioning latent `z_q`.
- **Latent Thinking Model** is the IRED energy network. It denoises a candidate latent `z` toward a target latent `z*` conditioned on `z_q`, iteratively, with bad-step rejection.
- **LatentDec** maps the final latent `z_0` directly to the answer `A`. No autoregressive conditioning on the latent — the latent *is* the answer's compressed representation.

Crucially, **encoder and decoder are frozen**. Only the thinking model is trained.

### 3.1 Solving the "no ground-truth `c*`" problem

Earlier discussion flagged that LLM reasoning has no canonical clean latent: many reasoning paths produce the same answer. This proposal sidesteps the problem entirely:

```
z_a*  =  LatentEnc(A)        ← deterministic, defined by the frozen encoder
```

The answer text gives a unique target latent. The diffusion model is trained to denoise toward `LatentEnc(A)` given `LatentEnc(Q)`. Standard IRED loss applies directly.

### 3.2 Why this is true Mode-2 vs. latent-guided prompting

Coconut, LaDiR, and similar approaches still use the LLM as an *autoregressive token generator conditioned on continuous thoughts*. Reasoning is fused with surface-form production: the latent conditions an AR decoder that does substantial cognitive work during generation. By the three criteria in §1.2, these systems fail (1) — inference is sampling, not optimization — and fail (3) — the decoder reasons.

The proposal above satisfies all three by construction:

- **Inference is optimization** (commitment 1): reverse diffusion with bad-step rejection is explicit gradient descent on a learned energy, not sampling from a generative model.
- **The cost is anchored** (commitment 2): the target latent `LatentEnc(A)` is fixed by the frozen encoder applied to ground-truth answers. The NCE term that calibrates `E` is anchored to this same fixed point. There is no learnable target the optimizer can drift toward.
- **The decoder transduces** (commitment 3): it sees a single latent and produces tokens. Inner-loop compute can be spent freely on thinking without touching the decoder; the boundary between cognition and surface form is sharp, in the same sense LeCun's actuator boundary is sharp.

The computation that decides *what* the answer is happens entirely before any token is emitted, and it happens via gradient descent rather than token sampling. This is the structural property that distinguishes the proposal from current latent-CoT work, regardless of how it ranks empirically.

### 3.3 The cost-anchoring simplification

A fuller treatment of commitment (2) would use a **composite anchored cost** combining multiple structurally-different signals — e.g. world-model coherence, decoded-output stability under latent perturbation, capacity bottleneck, target-encoder distance, and external verifier where available — so that gaming the cost requires fooling several independent anchors simultaneously. That program is what the broader research direction calls for, and it is the natural place to extend this prototype once the core pipeline is working.

For this first experiment we deliberately collapse that composite cost to its single most-load-bearing term: the target-encoder distance, realized as IRED's denoising MSE plus NCE against `LatentEnc(A)`. The §7.4 frozen-decoder CE auxiliary is a small step toward the composite form — it adds a second anchor (the decoder's CE against the gold tokens) and demonstrates that additional terms can be folded in without disturbing the bad-step-rejection contract. Anything richer is deferred until the single-anchor version is shown to train.

---

## 4. Architectural analogy: Stable Diffusion for reasoning

This architecture is Stable Diffusion's recipe transplanted to text reasoning:

| Stable Diffusion | This proposal |
|---|---|
| VAE encoder (image → latent) | Text encoder (`Q`, `A` → latent) |
| UNet diffusion in latent space | IRED energy net in latent space |
| VAE decoder (latent → image) | Text decoder (latent → answer) |
| Conditioning: CLIP text embedding | Conditioning: `z_q = LatentEnc(Q)` |
| Forward process: Gaussian noise | Forward process: Gaussian noise |
| Reverse: noise prediction | Reverse: `ε̂ = ∇_z E` with bad-step rejection |

The elegance: factor the hard problem (modeling discrete text) into the **frozen autoencoder**, leaving only the **smooth, continuous reasoning problem** for the diffusion model. This is exactly the trick that made pixel-space diffusion tractable.

---

## 5. Training and inference

### 5.1 Training (encoder & decoder frozen)

```python
# Inputs
z_q  = LatentEnc(Q)                          # conditioning
z_a  = LatentEnc(A)                          # clean target

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
z_q = LatentEnc(Q)
z   = randn(latent_shape)
for t in reversed(range(T)):
    for _ in range(N_inner):
        E_before = EnergyNet(z_q, z, t)
        eps_hat  = grad(E_before.sum(), z, create_graph=False)[0]
        z_new    = z - step_size * eps_hat
        E_after  = EnergyNet(z_q, z_new, t)
        z = where(E_after < E_before, z_new, z)         # bad-step rejection
A = LatentDec(z)
```

Test-time compute scales with `N_inner` and `T`. The decoder is invoked exactly once.

### 5.3 Training the Mode-1 actor (post-hoc distillation)

After the Mode-2 EBM has converged, train a one-shot actor `Actor_θ(z_q) → z*` against the converged optimizer's outputs. The encoder, decoder, and EBM are all frozen at this stage; only `Actor_θ` is updated.

```python
# All three of these are frozen during actor training
LatentEnc, LatentDec, EnergyNet = load_pretrained(...)

for Q, A in dataloader:                        # A is unused at train time;
    z_q = LatentEnc(Q)                           # only z_q drives the actor

    # Teacher rollout: run the trained Mode-2 optimizer to convergence
    with torch.no_grad():
        z_final = p_sample_loop(EnergyNet, z_q,        # exactly §5.2
                                T=T, N_inner=N_inner)

    # Student: one-shot regression onto the teacher's converged latent
    z_hat   = Actor_theta(z_q)
    L_act   = mse(z_hat, z_final)              # fixed-shape → fixed-shape
    L_act.backward()
```

Notes:

- **No backprop through the teacher.** `p_sample_loop` is run under `no_grad`; `z_final` is a fixed regression target. This is what makes the actor cheap to train despite the teacher's second-order autograd.
- **Replay buffer.** Caching `(z_q, z_final)` pairs across epochs lets the teacher rollout amortize — one Mode-2 rollout per `Q` for the whole actor training run.
- **Optional EBM-weighted loss.** Weight each training pair by `exp(−E(z_q, z_final, 0))` so the actor preferentially imitates the teacher's *confident* converged latents and ignores ones where Mode-2 itself didn't reach a good minimum. Cheap because the EBM is already on the GPU.

### 5.4 Adaptive Mode-1 / Mode-2 inference

At deployment, default to the fast actor and only escalate to the slow optimizer when the EBM says the actor's output is poor. The EBM acts as a built-in confidence scorer — no separate verifier needed.

```python
z_q     = LatentEnc(Q)
z_hat   = Actor_theta(z_q)                     # one forward pass
E_hat   = EnergyNet(z_q, z_hat, t=0)           # calibrated quality scalar (§2)

if E_hat < threshold:                          # actor is confident
    z_final = z_hat
else:                                          # hard case → fall back to Mode-2
    z_final = p_sample_loop(EnergyNet, z_q,
                            T=T, N_inner=N_inner)
    # optional: add (z_q, z_final) to the actor's replay buffer for later

A = LatentDec(z_final)
```

The `threshold` is calibrated on a held-out set so that, say, 80% of queries take the Mode-1 path. Average inference cost becomes a tunable knob between one forward pass and the full `T × N_inner` rollout — and unlike a simple confidence head, the gate is *the same scalar* that defines correctness during Mode-2 training, so there's no second calibration problem.

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

### 7.1 Why NCE matters here (revisiting the JEPA question)

Yann LeCun's argument that contrastive terms can be dropped in JEPA-style models **does not transfer cleanly to this proposal**, for the same reason it doesn't transfer to IRED proper:

- JEPA's anti-collapse mechanisms (VICReg, BYOL asymmetry) calibrate *which directions in feature space carry signal*.
- They do **not** calibrate the *absolute scalar value* of an energy function.
- Bad-step rejection requires `E(z_a) < E(z_b)` to be a reliable proxy for "`z_a` is a better candidate than `z_b`." That's an absolute-value property, not a direction property.

So `--supervise-energy-landscape True` is **load-bearing** for this proposal, not optional.

### 7.2 Latent MSE alone may not punish the right errors

The training loss in §5.1 lives entirely in latent space: `MSE(ε̂, ε)` over the K·d latent entries, plus the NCE energy contrast. This is the LD4LG / Stable-Diffusion recipe and is what we use as the default. But it has a known weakness any time decoded outputs have multiple near-isomorphic forms whose correctness flips on a small set of tokens:

- Two MBPP solutions that differ only by `<` vs `<=` in a comparison, or by a `+1` boundary tweak in a slice, sit *very* close in MSE distance — a handful of token-positions in an otherwise-identical function — while being maximally different in correctness (one passes the assertions, the other doesn't).
- The MSE objective treats every dimension of the latent equally; it has no way to "spend more budget" on the dimensions the decoder reads as the operator-bearing tokens.

With the code corpora used in §9 (MBPP solutions ~50–200 tokens, HumanEval canonical bodies ~30–150 tokens) the answer manifold has enough volume that random noise rarely lands on a near-miss program by accident — so the auxiliary CE term is *less* load-bearing than it would be for short numeric answers. It still helps in the operator-flip regime where small surface edits flip test pass/fail. We keep it as an optional knob (`--decoder-aux-weight 0.1 --decoder-aux-t-max 2`) but don't ship it on by default.

The decoder *does* know this distinction — its CE loss explicitly punishes producing the wrong token. So a natural augmentation is to mix in a small frozen-decoder CE term:

```
L = L_denoise + λ·L_nce + λ_dec · 1[t < t_max] · CE(LatentDec(x0_hat), gold)
```

where `x0_hat = predict_start_from_noise(z_t, t, ε̂)` is the model's current estimate of the clean latent. Two design notes:

1. **Gate on low t.** `x0_hat` is only a reliable estimate of the clean latent near the end of the schedule. For early t the residual noise is large; backpropping decoder CE through a wildly-off `x0_hat` injects noise into the EBM. I restrict the auxiliary to `t < t_max` (default 2 of 10).
2. **Backprop, don't replace.** The decoder stays frozen — its params get zero gradient. But the CE loss depends on `x0_hat`, which depends on the EBM via `ε̂`. So gradient flows back to the EBM through the frozen decoder, treating the decoder as a fixed perceptual loss over latents.

Why keep this as an auxiliary rather than the primary objective: as discussed in §7.3, the inner-loop bad-step rejection assumes ∇E points toward the data manifold. The latent MSE is what calibrates that. A decoder-CE-only loss would optimize ∇E toward "whatever latent the decoder happens to decode well," which isn't the same set — and would break the bad-step contract at inference.

Default in this repo: `--decoder-aux-weight 0.0` (off, matches §5.1 exactly). Recommended for ablation on short-answer tasks: `--decoder-aux-weight 0.1 --decoder-aux-t-max 2`.

### 7.3 LD4LG-faithful compression / reconstruction split

The original draft of this proposal collapsed LD4LG's two-network design into a single "pool" module:

```
text → E_frozen → Pool → z → D_frozen → text       (original)
```

Empirically that plateaued Milestone 1 at loss ≈ 1.45 (token p ≈ 0.23). The diagnosis: the pool was wearing two hats simultaneously and the objectives partially conflict. Switching to LD4LG's split gives each subnetwork a single job:

```
text → E_frozen → Pool (f_φ) → z ∈ R^(K×d_ae) → ReconstructionNet (f_ψ) → R^(K×d_LM) → D_frozen → text
                                       │
                                       ▼
                              [diffuse here, EBM]
```

**Two distinct jobs**:

1. **Compression** (`f_φ` = the pool). Maps a variable-length encoder hidden sequence to K latent slots that pack answer-relevant content densely.
2. **LatentDec compatibility** (`f_ψ` = the reconstruction network). Shapes those K slots into something the frozen T5 cross-attention can actually read. T5's decoder was trained on encoder outputs with specific statistical properties; arbitrary pool outputs aren't guaranteed to match.

These objectives pull in different directions: (1) wants information density (sparse, high-entropy, unique-per-input); (2) wants statistical similarity to natural T5 encoder hidden states (smoother, more redundant, on-manifold). Separating them lets each module specialize. This is the standard "separation of concerns" win that LD4LG (after Flamingo's Perceiver Resampler) validated empirically.

**Pool attention pattern**: separated self-attn + cross-attn (`nn.TransformerLatentDecLayer`), not LD4LG's combined-MHA Perceiver Resampler. I tried the combined form first because the paper specifies it that way; it plateaued at loss ~7.6 with collapsed outputs across LR sweeps. See §12.6 for the post-mortem. The separated form has distinct `W_k/W_v` per attention role (self-attn vs. cross-attn) and is what actually trains on this task.

The pool ends with a learnable linear projection `d_LM → d_ae` (no-op when `d_ae = d_LM`). This is the dimensionality knob — when the AE has converged, dropping `d_ae` shrinks the EBM's diffusion space by the same factor and is the most impactful lever for Milestone 2 tractability.

**Recon initialization**: each transformer block in `ReconstructionNet` is identity-initialized (attention `out_proj` and FF `linear2` zero-init'd) so the recon is bit-exact identity at step 0. Without this, the recon's random transforms scramble the pool's input-conditional signal and the frozen decoder falls back to unigram statistics. Validated: at random init, `decode_loss(with_recon) == decode_loss(without_recon)` to 6 decimal places. From step 1 the optimizer grows non-zero residuals as they become useful.

I also dropped the trailing `LayerNorm` from the recon — three sequential LayerNorms in series (pool's final LN + recon's internal LNs + a trailing LN) over-normalized the latents and squashed the magnitude signal T5's decoder cross-attention was trained to read.

**Two-step rollout**:

I attempt these as separate steps so we can attribute which fix is doing the work.

- **Step 1 (default).** `d_ae = d_model = 768`. Adds the recon net and the combined-MHA pool without changing latent dimensionality. This isolates the "dual-role pool was the bottleneck" hypothesis. Diagnostic: does Milestone 1's final-answer extraction accuracy clear the 90% bar that the single-pool design plateaued well below?
- **Step 2.** Drop `d_ae` to ~128 once Step 1 is verified. Shrinks the EBM diffusion space by ~6× — the structural fix for "the EBM has too many DoF to denoise into" that §7.4 motivated for short answers. Step 2 changes the EBM input shape, so M2 must be retrained.

**Trainable parameter cost**: Step 1 adds ~9M (the 2-layer ReconstructionNet at `d=768`), bringing the default AE from 18.9M → 28.4M trainable. Step 2 is parameter-neutral relative to Step 1 (the dimensionality projection is a single linear layer).

**Risk**: the bootstrap cost. With f_ψ at random init, the very first decode is *worse* than the single-pool version was at random init (one more random module in the path). At step 0 of Milestone 1 we observed `decode_loss ≈ 13.0` vs the pool-only random-init baseline of ~10.4. The model has to learn through this initial deficit. If you see eval loss still above 10 after a few hundred steps, that's pathological — but if it drops below 5 within the first eval cycle, you're on track.

---
## 8. Wrap it Up

Pulling §3–§7 together, the complete pipeline is five modules: two frozen (the T5 encoder and decoder), two learned for the autoencoder stage (the Compressor `f_φ` and the Reconstructor `f_ψ`), and one learned for the reasoning stage (the EBM).

```
   Q ──▶ T5_Encode ──▶ f_φ (Compressor) ──▶  z_q ∈ R^(K×d_ae)
                                              │
                                              │   conditioning
                                              ▼
                       z_T ~ N(0, I)  ──▶  EBM(z_q, z, t)  ──▶  z_0
                       (Mode-2: T outer × N_inner steps, bad-step rejection on E)
                                              │
                                              ▼
                       z_0  ──▶  f_ψ (Reconstructor) ──▶ T5_Decode ──▶  A
```

(At training time, `z_a = f_φ(T5_Encode(A))` is the clean target the EBM denoises toward; at inference time only `z_q` is supplied and `z_0` is produced by the Mode-2 loop of §5.2.)

**Training stages, in order:**

1. **Autoencoder.** Train `f_φ` and `f_ψ` end-to-end against the frozen T5 reconstruction objective (`CE(T5_Decode(f_ψ(f_φ(T5_Encode(A)))), A)`). T5 stays frozen. Milestone 1 in §9 gates progression.
2. **EBM (Mode-2 optimizer).** Freeze `f_φ`, `f_ψ`, and T5. Train the EBM with IRED's `L_denoise + λ·L_nce` (§5.1), with optional low-t frozen-decoder CE auxiliary (§7.4). Milestone 2 + 3 in §9 validate the thesis.
3. **Mode-1 actor (optional, post-hoc).** Freeze everything from stages 1–2. Distill the trained Mode-2 optimizer into a one-shot `Actor_θ(z_q) → z*` per §5.3, and deploy with the adaptive Mode-1/Mode-2 gate of §5.4.

**Mapping back to §1.2's three commitments:**

- *Inference is optimization, not sampling.* The EBM stage runs reverse diffusion with bad-step rejection on a calibrated scalar `E` — gradient descent, not a generative model over reasoning traces.
- *The cost is structurally anchored.* `z_a = f_φ(T5_Encode(A))` is fixed by the frozen encoder and a frozen-once compressor; nothing the EBM can learn drifts the target.
- *The decoder transduces, it does not reason.* `f_ψ → T5_Decode` is invoked exactly once on `z_0`. All cognitive work happens in the latent loop, behind the actuator boundary.

The next section is the smallest experiment that exercises all three stages on a real reasoning task and tests whether test-time compute in latent space actually buys accuracy.

---

## 9. Concrete first experiment

Realistic single-GPU prototype to validate the core thesis.

**Task choice — why code instead of math or token-CoT.** An earlier draft targeted GSM8K. The grade-school-math choice ran into a structural problem: GSM8K answers are 1–3 numeric tokens. The effective rank of the answer manifold is ~log₂(10⁴) ≈ 13 bits, so even with K=128 latents × d_model dimensions the diffusion model is fighting for crumbs of structured signal. The "reasoning *in* latent space" thesis (each inner step is a thinking step that refines a noisy answer-latent) is hard to test rigorously when there is so little structure for the latent to encode.

A second draft targeted ProofWriter + BBH logical-deduction puzzles. That gave richer per-answer structure (50–150 token proof chains) but introduced a different problem: the *correctness-bearing portion* of the target is a 3-way verdict (True/False/Unknown), so the latent's structural budget is spent encoding boilerplate proof prose while the bit that determines accuracy is a one-token verdict the EBM can lose to MSE-noise. Worse, the proof chain is itself a token-CoT trace — so the EBM was being supervised to denoise toward the latent image of English-language reasoning, putting the prototype back in "latent-encoded token-CoT" territory and partially defeating §1.2's commitment that the decoder transduces rather than reasons.

Code generation cleans up both issues at once. The target is a **structured artifact** (a Python function), not a writeup of how to solve the problem. The artifact has rich per-token variability — 50–200 tokens of identifiers, operators, control flow — so the latent manifold has volume and the EBM has room to refine. The decoder's job is genuinely transduction: Python's grammar is rigid enough that the frozen T5 decoder is not doing problem-solving when it emits tokens, just rendering. And correctness is verifiable by an **external** signal (`assert` statements + `python -c`), independent of the EBM's own `E` — which gives us the strongest possible test of §2's calibrated-quality-scalar claim: does the trained `E(z_q, z, 0)` correlate with whether the decoded code passes its tests?

**Setup:**
- **Task:** MBPP — Mostly Basic Python Problems (974 examples, full config; 374 train / 90 val / 500 test / 10 prompt). Each example has a natural-language description, a canonical Python solution, and ≥3 assert tests. Primary EBM-training corpus. **HumanEval** (164 problems) is held-out eval only; never seen during training. See `ired/data.py` for the loaders.
- **AE pretraining corpus:** OpenWebText (broad natural language). The AE is deliberately **never** trained on MBPP or HumanEval — keeping the latent space free of EBM-training-distribution leakage preserves the §3.2 anchoring property. The known risk this raises is documented in Milestone 1 below: an OWT-only AE may not round-trip Python well, since OWT is text-heavy. If Milestone 1 fails, the recovery is to either (a) add a code slice to AE pretraining (e.g., a sample from The Stack), or (b) swap T5 for CodeT5.
- **LatentEnc/LatentDec:** `flan-t5-base` (d_model=768), frozen. Learned `AttentionPool` (K=128 latents) + `ReconstructionNet`. See §7.3.
- **Energy net:** 4-layer transformer over `concat(z_q, z_t, time_embed)`. Scalar energy via squared-sum of final layer.
- **Diffusion:** `T=10` timesteps, `continuous=True`.
- **Inner loop:** 5 steps with bad-step rejection.
- **Loss:** denoising MSE + NCE with `λ=1`. Optional `--decoder-aux-weight 0.1 --decoder-aux-t-max 2` for a low-t frozen-decoder CE ablation (see §7.2).
- **Verifier:** decoded code is written to a temp file and executed in a subprocess with a 5-second timeout. Pass = subprocess exits 0; fail = non-zero return code, timeout, or unparseable output. The verifier is *not* sandboxed; it runs benchmark code with the same privileges as the parent process — fine for offline eval on MBPP/HumanEval, never point it at adversarial input.

**Practical defaults that survived first-run debugging (see §12):**
- **Model size:** `flan-t5-base` (d_model=768) for faster iteration. `large` is a drop-in. CodeT5-base is the natural swap if the OWT-trained T5 fails Milestone 1 on code recon.
- **Autoencoder architecture:** LD4LG-faithful — `AttentionPool` (Perceiver Resampler with shared LN + gated residuals OR the `decoder`-style separated self+cross attention) + explicit `ReconstructionNet` (f_ψ). See §7.3 for the rationale and §12.6 for what goes wrong with naïve combined-MHA. Default `d_ae = d_model`; set lower once Milestone 1 is converged to shrink the EBM's diffusion space.
- **Pool/recon capacity:** `K=128, pool_layers=2–4, recon_layers=2`. K=128 gives ~2:1 compression on MBPP's typical ~100-token solution; K=32 (the GSM8K-era default) is undersized for code-shaped outputs.
- **Answer target:** the full canonical solution code (MBPP `code` field — full `def`-block) or the full executable function `prompt + canonical_solution` (HumanEval). Both corpora share the "answer is a complete executable Python function" convention so the EBM sees uniform target shape. Per-example test fixtures are stored on the dataset record (`_test_script`) and used by `verify_code` at eval time. See `ired/data.py`.
- **`max_q_length`:** `512` — HumanEval prompts (signature + long docstrings + I/O examples) can be long; MBPP descriptions are short but we keep the cap consistent.
- **`max_a_length`:** `384` — MBPP solutions and HumanEval canonical bodies fit comfortably; the small bump from the ProofWriter-era `256` covers HumanEval's longer functions.
- **β schedule:** `cosine`, not `linear`. Linear at T=10 saturates immediately after clamping (β=0.999 for 5 of 10 steps), giving the EBM almost no useful t spread.
- **Weight decay:** `--weight-decay 0.01` on the EBM. **Load-bearing** — without it the head magnitude inflates ~100× during training and the relative NCE contrast collapses (§12.2).
- **Clamps:** `x_start_clamp=5.0`, `envelope_sf=None`. IRED's hardcoded `(2, 2)` assume `[-1, 1]` data; LayerNorm'd T5 latents live in ~`[-3, 3]`.
- **SDPA kernel:** force `SDPBackend.MATH` inside the EBM's attention — fast kernels don't implement double-backward.
- **Monitoring (per training step):** `mse, nce, e_real, e_fake, eps_scale = ||ε̂|| / ||noise||`. Target `eps_scale ≈ 1.0`.
- **Monitoring (per eval):** `mbpp_pass(inner=N), mbpp_pass(inner=0), mbpp_ae_pass, humaneval_pass(inner=N), humaneval_pass(inner=0), humaneval_ae_pass, mse_z, std_zs, corr_z`. Per-corpus AE pass-rate separates Milestone-1 failures from EBM failures *per corpus* — important since HumanEval has longer prompts than MBPP and the AE may handle one better than the other.

**Milestones:**
1. **Autoencoder check** — `LatentDec(Recon(Pool(LatentEnc(code))))` round-trips on the EBM-eval corpora *even though the AE was only trained on OpenWebText*. Bars:
   - MBPP test pass-rate ≥ 80% (decode the canonical, run against `test_list`)
   - HumanEval pass-rate ≥ 80% (decode prompt+canonical, run against `check`)
   - OpenWebText held-out byte-exact recon ≥ 50% (sanity check that the AE works at all)

   The code bars are lower than the ProofWriter-era 90% because Python tokens (especially identifiers and operators) are higher-entropy than English prose, and a single mis-rendered operator can flake a whole test. If the AE clears OWT but fails both code bars by a large margin, the OWT-trained latent space doesn't transfer to Python — the recovery is to either (a) add a small code slice to AE pretraining (a sample from The Stack), or (b) swap T5 for CodeT5-base, both of which break the strict "no EBM-corpus exposure" rule slightly but preserve the more important property that the AE was never trained on MBPP/HumanEval specifically.
2. **Diffusion training** — train the energy net on MBPP. *Watch `eps_scale` and the `e_real/e_fake` ratio over training; either drifting is the §12.2 warning sign. Report `mbpp_pass(inner=N) - mbpp_pass(inner=0)` and the HumanEval equivalent: positive deltas mean iterative refinement is doing real work.* Additionally report Pearson correlation between `E(z_q, z, 0)` on the model's own samples and observed pass/fail — this is the §2 calibrated-scalar claim, and code execution gives us the cleanest external signal to test it against.
3. **Test-time compute curve** — plot MBPP pass-rate and HumanEval pass-rate vs. `N_inner × T` and compare to AR-CoT at matched FLOPs (e.g., flan-t5-base CoT-prompted to write the function).

**Decision rule:** if the diffusion curve is steeper than AR-CoT at matched compute, *and* the `inner=N` vs `inner=0` gap is positive and grows with N, the thesis is validated — iterative energy descent is performing genuine refinement on code generation. If accuracy plateaus below AR-CoT, or if `inner=N` and `inner=0` are indistinguishable, that's an informative negative about smoothness of reasoning in this latent space. A separate informative signal: if `E`–pass-rate correlation is high (>0.5), the calibrated-scalar claim holds even when accuracy is modest — which is what justifies using `E` as the gate in §5.4's adaptive inference.

---

## 10. Honest assessment

**Promising aspects:**
- Solves the "no `c*`" problem elegantly via the frozen encoder — `LatentEnc(A)` is the structural anchor that §1.2's commitment (2) demands.
- Cleanly separates thinking from surface-form generation — the actuator-style boundary that distinguishes true Mode-2 from latent-guided prompting.
- Test-time compute scaling via inner-loop steps is a natural fit for the o1-era frontier, and is governed by an explicit cost rather than a sampling temperature.
- Architectural precedent (Stable Diffusion, LD4LG) suggests the frozen-autoencoder approach is viable.

**Risks:**
- Autoencoder quality bottleneck — most teams who tried latent-diffusion-for-text reported "the decoder is the limiting factor."
- RL-on-AR-CoT (DeepSeek-R1, o1) is currently the empirical winner. Any diffusion-based reasoning system has to justify *not* using RL.
- Reasoning may not be smooth in latent space. If correct reasoning traces are isolated discrete modes rather than connected regions, diffusion will struggle.
- The decoder-as-transducer assumption only holds for constrained output formats. This prototype targets code generation; the approach is structurally restricted to domains (code, formal logic, structured planning) where linguistic rendering doesn't itself require reasoning. Open-ended natural language is outside scope — not a temporary limitation but a real one.
- Collapsing the composite anchored cost (§3.3) to a single target-encoder term is a deliberate simplification. If the single anchor turns out to be gameable in the LLM-latent setting, additional anchors (world-model coherence, counterfactual stability, capacity bottleneck) have to be added — and the bad-step-rejection contract has to be re-checked under each addition.

**Why try it anyway.** The architecture is clean, the prior work is encouraging, and even a clear negative result tells us something concrete about the geometry of reasoning in continuous space. Brains, by every available indication, do not reason in language — language regions are largely inactive during reasoning, verbalization happens at output. If artificial reasoning systems are ever to operate at the level of fluid cognition that biological systems exhibit, the substrate likely has to be representational rather than lexical. The current "latent reasoning" literature gestures at this without implementing it; this prototype is the smallest concrete step toward the system the framework actually demands. Feasible on a single GPU in roughly two weeks.

---

## 11. Code reference


The following code block serve as anchor for the codes in the project, this is pulled directly from the IRED github repo `https://github.com/yilundu/ired_code_release`

### 11.1 The energy network — `EBM` (models.py)

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

### 11.2 Scalar → vector — `DiffusionWrapper` (models.py)

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

### 11.3 Inner loop with bad-step rejection — `opt_step` (denoising_diffusion_pytorch_1d.py)

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

### 11.4 Inference schedule — `p_sample_loop` (denoising_diffusion_pytorch_1d.py)

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

### 11.5 Combined training loss — `p_losses` (denoising_diffusion_pytorch_1d.py)

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

## 12. Implementation pitfalls (post-mortem)

Five concrete bugs and infelicities surfaced when building the §9 prototype against the IRED reference code. None of them invalidate the proposal — but every one is a place where a default that worked for IRED's small tabular tasks fails silently in the latent-LLM setting. Listing them in one place so future implementers don't re-discover them.

### 12.1 Diffusion β schedule blows up at small T

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

### 12.2 EBM head magnitude inflates unboundedly without weight decay

IRED's reference setup uses an MLP energy net on small (~tens-of-dims) tabular data with no weight decay. I inherited the no-weight-decay default. In the latent-LLM setting the head is a `Linear(d_model=768, d_model=768)` and the energy is the squared sum over `K · d_model = 24576` dims — so the absolute energy scale is enormous and grows freely.

Observed in a 5000-step run:

| step | e_real | e_fake | contrast | nce loss |
|---:|---:|---:|---:|---:|
| 100 | 1,413 | 2,896 | 2.05× | 0.19 |
| 5000 | 131,601 | 134,133 | **1.02×** | 0.20 |

The NCE loss number is unchanged but the *relative* contrast collapsed from 2× to 2%. Cross-entropy is scale-equivariant in the logit difference, so the loss minimum can be reached by either (a) learning real-vs-fake discrimination or (b) growing both energies in proportion. Once the head weights start growing, gradient descent prefers (b) — it's an easier optimization direction.

Downstream consequence: `opt_step` bad-step rejection becomes inert (a 2% contrast is dominated by noise in `E(z_new)` vs `E(z)`), and ∇E grows with the head, breaking DDPM's reverse process that expects ε̂ at noise-scale (std ≈ 1).

**Fix:** `--weight-decay 0.01` on `AdamW` keeps the head norm bounded. Should be the default; the current `0.0` is a footgun.

**Diagnostic:** the `eps_scale = ||ε̂|| / ||noise||` stat added to per-step logging surfaces this in one number — when head inflates, `eps_scale` drifts upward in lockstep.

### 12.3 IRED's clamping bounds assume `[-1, 1]` data

IRED hardcodes `x_start_clamp = 2` and `envelope_sf = 2` (the per-t `±sqrt(α̅_t)·sf` clamp inside `opt_step` and `p_sample_loop`). Both assume input data normalized to `[-1, 1]` — true for IRED's tabular tasks, false for LayerNorm'd T5 latents whose element std is ~1 and bulk lives in `[-3, 3]`.

Applied verbatim, these clamps crush noisy `z_t` to ~0 at large t (when `sqrt(α̅_t)` is small), turning the reverse process into noise injection plus clamping rather than denoising.

**Fix:** `x_start_clamp = 5.0` (loose enough to not bite typical T5 latents), `envelope_sf = None` (disabled by default; pass an explicit float to re-enable the IRED-style behavior). Both are CLI-configurable.

**Lesson:** any time you transplant IRED-style code to a new data domain, audit every literal numeric constant against the new data's empirical range. The original constants are a domain-specific tuning, not a general defaults.

### 12.4 ε̂ scale drift is invisible to `mse(ε̂, ε)`

The denoising loss `mse(ε̂, ε)` is dominated by direction agreement — a 2× scale error contributes only `(2 - 1)^2 = 1` per-element to MSE, the same as a perpendicular unit-noise direction. So `mse` can look small (~0.02) while `||ε̂||` is drifting away from `||ε||`. DDPM's `predict_start_from_noise` and `q_posterior` math assume ε̂ is at noise-scale; if it isn't, the reverse process inflates or deflates `z` magnitude regardless of how good the *direction* is.

In our broken 5000-step run, the symptom was `std(z_sampled) = 2.75` vs `std(z_a) = 1.00` after sampling — magnitude inflated almost 3× while `mse(ε̂, ε)` was 0.017 at train time.

**Fix:** log `eps_scale = ||ε̂||_2 / ||ε||_2` per training step. It's free (one no_grad norm) and is the earliest detector of the §12.2 head-inflation problem.

### 12.5 PyTorch fast SDPA kernels don't implement double-backward

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

### 12.6 LD4LG-literal Perceiver Resampler did not work; reverted to separated attention

The naïve compression-network implementation uses `nn.TransformerLatentDecLayer` (separate self-attention then cross-attention). That plateaued Milestone 1 at loss ≈ 1.45 with input-conditional outputs — the model was clearly using the latent code but couldn't push past a certain reconstruction fidelity. The hypothesis was that LD4LG's combined-MHA Perceiver Resampler block,

```
Z = Z + MHA(q = Z, kv = [Z; E(w)])
Z = Z + FF(Z)
```

would let each head dynamically allocate attention budget between latents and encoder, vs. the separated form's strict role assignment. I implemented it as a custom `PerceiverResamplerLayer`.

**It made things dramatically worse.** Milestone 1 with the combined-MHA pool plateaued at loss ≈ 7.6 across multiple LR settings (3e-4, 1e-4, and a 1000-step warmup from 0 to 1e-4 — the loss curves were *identical*, ruling out optimization). Predictions collapsed to a unigram-like mode: every input decoded to the same `>>>>>>>>` token stream (`>>` being a common substring in GSM8K's `<<...>>` calculation markers).

**The most plausible mechanism:** the combined form forces a single `W_k, W_v` pair to project *both* the latent slots *and* the encoder hidden states. These two roles want different projections — latents are LayerNorm'd identity-shared queries; encoder hidden states are content-bearing input-conditional vectors. The separated form has two distinct projection pairs (one for self-attn, one for cross-attn) so each role specializes from the first SGD step. The combined form has to either compromise or get stuck — and on the GSM8K-CoT task with frozen T5, it gets stuck.

**Diagnostic that pinpointed it:** with the identity-initialized ReconstructionNet (§7.5) producing bit-exact identity at step 0, recon could not be the source of collapse at the initial eval. Yet the collapse was visible immediately. By elimination, the pool was the problem. LR sweeps producing identical loss curves confirmed it wasn't optimization.

**Fix:** reverted `AttentionPool` to `nn.TransformerLatentDecLayer`. Keeps the LD4LG-style explicit `f_φ` / `f_ψ` separation and the `d_ae` projection, but uses the separated-attention pool that has empirical evidence of working on this task.

**Lesson:** when a paper's formula doesn't translate to your setting, it doesn't mean you implemented it wrong — sometimes the paper's design is calibrated for a different (data, encoder, decoder, scale) regime and doesn't transfer. Empirical falsification (matched loss curves across LR sweeps) is faster than mechanistic debate. The "literal-fidelity vs. what-works" tradeoff was decided by data, not by re-reading the paper.

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
