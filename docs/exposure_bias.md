# Exposure-bias gap — diagnosing AR decoder leakage

A short diagnostic to answer: **how much of the answer is the autoregressive decoder making up vs. just reading off the latent?**

This matters because of gensis.md §1.2 commitment (3): *"the decoder transduces, it does not reason."* If the frozen T5 decoder is making real generative choices at inference (which AR decoders historically do — exposure bias is the textbook example), then the latent is no longer the sole locus of cognition and the proposal's Mode-2 story has a leak.

The gap is a single per-eval-cycle number that tells us whether that leak is empirically real in our current setup. It costs one extra forward pass per eval batch and lives next to the existing pass-rate metrics in `eval_corpus` (`ired/train_diffusion.py`).

---

## What we measure

Two scores per example, both conditioned on the same EBM-sampled latent `z_0`:

**(a) Teacher-forced CE on canonical.** Feed the decoder `z_0` via cross-attention AND the gold answer tokens as previous-step inputs. At each position `t`, the decoder predicts the next token given `(z_0, gold_{<t})`. Score is the average `−log P(gold_t | gold_{<t}, z_0)`. Implemented as `FrozenT5Autoencoder.decode_loss_per_example(z_0, gold, ...)` → shape `(B,)`.

**(b) Free-running decode + verifier.** Feed the decoder `z_0` only, let it generate left-to-right from its own outputs (greedy, the `generate(...)` call), then execute the produced code against the example's test fixture. Pass/fail. Implemented as `verify_code(ae.decode(z_0), example)`.

We collect both per example, then bucket CE by the pass/fail outcome:

- `ce_pass` = mean teacher-forced CE on examples that the free-running decoder got right
- `ce_fail` = mean teacher-forced CE on examples that the free-running decoder got wrong
- `eb_gap` = `ce_fail − ce_pass`

All three are returned by `eval_corpus(...)` and printed each eval cycle in `train_diffusion.py`.

---

## Why these two scores can differ

In (a) the decoder sees the gold prefix at every step. It only ever predicts one token given perfect context. The latent only has to "point in roughly the right direction" — anything ambiguous gets resolved by the gold prefix.

In (b) the decoder sees its own prefix. A wrong variable name at position 4 forces position 17 to be consistent with the wrong name; the whole function flakes. The latent has to specify enough that the decoder doesn't drift.

The difference between (a) and (b) is the **autoregressive contribution** to the final output — work the decoder does beyond what the latent specified.

---

## Interpretation

Read `ce_pass`, `ce_fail`, `eb_gap`, and the absolute pass-rate together:

| ce_pass | ce_fail | gap | pass-rate | Diagnosis |
|---|---|---|---|---|
| low | high | large positive | mixed | **Healthy.** CE tracks pass/fail. Latent quality varies per example, AR isn't smuggling cognition. Free-running decode reflects latent quality faithfully. |
| low | low | ≈ 0 | low | **AR is leaking cognition.** Latent uniformly points decoder toward gold (CE is low everywhere), yet the free-running decoder flakes anyway. The drift between teacher-forced and free-running is doing real work — that's cognition happening in the AR step. Architectural swap (non-AR decoder) is justified. |
| low | low | ≈ 0 | high | **Healthy and saturated.** Both metrics agree the model is good. No leak to chase; just keep training. |
| high | high | small | low | **Latent is broken.** Even with the gold prefix the decoder is confused — the latent isn't conditioning it usefully. This is an EBM problem, not an AR problem. Fix the EBM (or the AE before it). |
| high | low | negative | mixed | **Suspicious — investigate.** This shouldn't happen for a well-trained pair. Likely a bug in the eval (e.g., verifier accepting on technicalities) or an AE that's so lossy the gold tokens are themselves out-of-distribution for the decoder. |

The case that says "go rebuild the decoder" is specifically **row 2**: `ce_pass ≈ ce_fail` AND both low AND pass-rate low. Any other pattern points elsewhere.

---

## Why this matters before swapping decoders

Replacing the frozen T5 decoder with a non-AR head (as discussed in conversation) is a meaningful commitment:

- A new decoder to train from scratch (or from a non-causal init).
- Fixed-length output budget (`<pad>` fills the tail) — wasteful and weakens variable-length generation.
- Known quality limits for non-AR text generation, especially on tasks requiring cross-position consistency (variable naming in code).

We should make that commitment if and only if the AR concern is empirically real. The gap measurement is the empirical evidence. Doing it once before rebuilding the decoder is the cheapest way to make the call.

---

## Reading the metric over training

Useful trajectories to watch:

- **`eb_gap` over training steps.** Early in training, both CEs may be high and the gap noisy (small `n_pass`). As pass-rate climbs, the gap should grow positive — pass/fail starts to discriminate. A persistently-near-zero gap as pass-rate stagnates around 30–50% is the canonical AR-leakage signature.
- **`ce_pass` over training steps.** Should decrease monotonically. If it plateaus high, the EBM is not learning to produce latents the decoder reads well.
- **`(pass-rate, ce_pass)` trajectory.** Plotting one against the other across checkpoints traces whether the latent and the verifier agree on what "good" looks like.

The simplest single-number health check is **`eb_gap / (ce_pass + 0.1)`** — gap normalized by the absolute CE level. Larger = more discrimination per unit of remaining loss = less AR leakage suspicion.

---

## What's wired up

- `FrozenT5Autoencoder.decode_loss_per_example(z, target_texts, device, max_length)` → `(B,)` tensor of per-example teacher-forced CE. Same forward pass as `decode_loss`; reduces independently per example, ignoring pad positions.
- `eval_corpus(...)` in `ired/train_diffusion.py` now collects `(ce_per_example, ok)` tuples and returns `ce_pass`, `ce_fail`, `eb_gap`, `n_pass`, `n_fail` alongside the existing keys. The eval log prints them as a `[exposure ...]` line per corpus.
- EBM checkpoints (`ebm_step{N}.pt` and `ebm_latest.pt`) store `mbpp_eb_gap` and `humaneval_eb_gap` so the trajectory can be reconstructed later from `checkpoints/ebm_*/`.

## What's deliberately not wired up

- **`sample.py` / `sample_adaptive.py`.** Those scripts measure pass-rate only; exposure-bias diagnosis belongs in the training loop where the same `z_0` is decoded both ways every eval cycle.
- **`train_actor.py`.** The same diagnostic applies to the Mode-1 actor's output — pair `decode_loss_per_example(actor(z_q), gold)` with `verify_code(decode(actor(z_q)))`. Adding it is a 4-line follow-up; left out of this pass because Milestone 2 (this measurement's primary use case) gates Milestone 3 actor training.
- **AE-only exposure-bias.** The AE eval in `train_autoencoder.py` already round-trips `z_a = encode(answer)` directly. The exposure-bias question doesn't apply there in the same way — the latent is the encoder's own output, so the AR concern is about decoder behavior given a well-formed latent, which is already captured by `mbpp_pass` / `humaneval_pass` vs. token-level CE that the eval prints.

---

## Decision rule for the architectural swap

Run the EBM for one full Milestone 2 schedule with the current AR decoder, watching `eb_gap` over training:

- If `eb_gap` grows positive and tracks `pass-rate` improvements → leave the AR decoder, keep going.
- If `eb_gap` stays near zero across a wide pass-rate range and pass-rate plateaus below the AR-CoT baseline → the AR-decoder swap is justified, and `ce_pass` becomes the target metric for the non-AR replacement (lower is better, since the non-AR decoder eliminates the gap by construction).
- If `ce_pass` itself plateaus high → fix the EBM / AE first; the AR question is downstream of that.
