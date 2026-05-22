"""Milestone 2 (gensis.md §9): train the IRED energy network in latent space.

Pipeline per step:
  with no_grad:
      z_q = ae.encode_to_latents(question)
      z_a = ae.encode_to_latents(answer)
  loss, stats = diffusion(z_q, z_a)
  loss.backward()
  opt.step()

Only the EBM updates. The autoencoder (frozen BART + previously trained pool) is
held fixed. Every `eval_every` steps we sample end-to-end and report per-corpus
pass-rates on MBPP test and HumanEval separately (correctness via subprocess
execution of the decoded code against per-example test fixtures).
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ired.autoencoder import FrozenBartAutoencoder
from ired.data import (
    HumanEvalDataset,
    MBPPDataset,
    PreTokenizedDataset,
    make_collate_pretokenized,
    verify_code_batch,
)
from ired.diffusion import GaussianLatentDiffusion
from ired.energy_net import DiffusionWrapper, EnergyTransformer
from ired.tui import make_reporter


def build_parser():
    p = argparse.ArgumentParser(description="Train IRED energy network in latent space")
    # autoencoder
    p.add_argument("--model", default="facebook/bart-base")
    p.add_argument("--ae-ckpt", required=True, help="AE checkpoint from train_autoencoder")
    p.add_argument("--k", type=int, default=128, help="Must match the AE checkpoint's k.")
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
    p.add_argument("--pool-type", choices=["decoder", "resampler"], default="decoder",
                   help="Must match the AE checkpoint's pool_type.")
    p.add_argument("--d-ae", type=int, default=-1,
                   help="diffusion-space latent dim. Must match the AE checkpoint.")
    p.add_argument("--recon-layers", type=int, default=2)
    p.add_argument("--recon-heads", type=int, default=8)
    # ebm
    p.add_argument("--ebm-layers", type=int, default=4)
    p.add_argument("--ebm-heads", type=int, default=8)
    p.add_argument("--ebm-ff-mult", type=int, default=4)
    # diffusion
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--inner-steps", type=int, default=5)
    p.add_argument("--beta-schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--opt-step-size", type=float, default=1.0)
    p.add_argument("--continuous", action="store_true", default=True)
    p.add_argument("--no-nce", action="store_true",
                   help="disable energy-landscape supervision (NCE)")
    p.add_argument("--nce-scale", type=float, default=1.0,
                   help="λ in genesis (NCE term weight)")
    p.add_argument("--x-start-clamp", type=float, default=5.0,
                   help="clamp predicted x_0 to ±this (set <=0 to disable)")
    p.add_argument("--envelope-sf", type=float, default=-1.0,
                   help="if >0, clamp z to ±sqrt(alpha_bar_t)*sf (IRED uses 2.0 on [-1,1] data)")
    p.add_argument("--decoder-aux-weight", type=float, default=0.0,
                   help="weight on the optional decoder-CE auxiliary loss "
                        "(0 disables; recommended start: 0.1)")
    p.add_argument("--decoder-aux-t-max", type=int, default=2,
                   help="only apply decoder-CE for samples with t < this. "
                        "Low t is where x0_hat ≈ z_a is reliable.")
    p.add_argument("--rand-neg-weight", type=float, default=0.0,
                   help="weight on the random-Gaussian negative margin loss. "
                        "Forces E(real) + margin < E(N(0,I)) so the EBM cannot "
                        "leave random latents at the same energy as gold. "
                        "0 disables; recommended start: 1.0.")
    p.add_argument("--rand-neg-margin", type=float, default=1.0,
                   help="margin in softplus(e_real - e_rand + margin). Larger = "
                        "stronger push for a real energy gap.")
    p.add_argument("--rand-neg-t-max", type=int, default=1_000_000,
                   help="only apply the rand-neg anchor for samples with "
                        "t < this. At high t, q(z_t|z_a) and N(0,I) overlap, "
                        "so the anchor there fights eps prediction. With T=10, "
                        "try 3 (low-noise only).")
    # data & opt
    p.add_argument("--train-dataset", choices=["mbpp"], default="mbpp",
                   help="Corpus the EBM trains on. Only MBPP is wired in by default; "
                        "HumanEval is reserved for held-out eval. To mix in other code "
                        "corpora, build a MixedDataset before calling main().")
    p.add_argument("--mbpp-config", choices=["full", "sanitized"], default="full",
                   help="MBPP HF config. 'full' = 974 examples (374 train / 90 val / "
                        "500 test). 'sanitized' = 427 cleaned examples; use if the full "
                        "set has annotation noise.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="accumulate gradients over this many micro-batches before "
                        "each optimizer step. Effective batch = batch_size × this. "
                        "Use to reduce per-step gradient variance and improve "
                        "timestep coverage when batch_size is memory-capped.")
    p.add_argument("--latent-stats-samples", type=int, default=1024,
                   help="number of training answers to encode for computing the "
                        "per-dim shift+scale that maps AE latents → N(0, I). "
                        "Without normalization the diffusion noise schedule is "
                        "miscalibrated for anisotropic AE latents (LD4LG §4.1, "
                        "Stable Diffusion's '0.18215' constant). Stats are stored "
                        "as buffers in the diffusion module and saved in the "
                        "checkpoint, so sample.py picks them up automatically.")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-dir", default="checkpoints/ebm")
    p.add_argument("--resume", default=None,
                   help="path to an EBM checkpoint to resume from. Restores ebm "
                        "weights, optimizer state, latent stats, and step "
                        "counter; skips the one-shot latent-stats recompute.")
    p.add_argument("--max-q-length", type=int, default=512,
                   help="HumanEval prompts (signature + long docstrings) can be long.")
    p.add_argument("--max-a-length", type=int, default=384,
                   help="Covers MBPP solutions (~50–200 tokens) and HumanEval functions.")
    p.add_argument("--eval-n-examples", type=int, default=80,
                   help="per-corpus examples to score each eval cycle.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--tui", action="store_true",
                   help="render training output as a Rich+plotext TUI "
                        "(graphs + scrolling log). Off by default so log "
                        "redirection / notebooks keep plain prints.")
    return p


@torch.no_grad()
def eval_corpus(ae, diffusion, dataset, dataset_name, device,
                max_q_length, max_a_length, inner_steps,
                n_examples=80, batch_size=16):
    """End-to-end EBM eval on a single code corpus.

    Reports:
      - acc           : pass-rate at inner_steps (execute decoded code vs tests)
      - acc_inner0    : pass-rate with inner_steps=0 (isolates opt_step)
      - ae_acc        : pass-rate of decode(encode(answer)) — Milestone 1 ceiling
      - mse_z, std_za, std_zs, corr_z : latent diagnostics

    `dataset_name` is "mbpp" or "humaneval"; it only affects logging — the
    correctness criterion comes from each example's `_test_script` via
    `verify_code`.
    """
    diffusion.eval()
    correct = correct0 = ae_correct = total = 0
    mse_z_sum = mse_z0_sum = 0.0
    sq_za_sum = sq_zs_sum = 0.0
    n_elems = 0
    cos_sum = 0.0
    # Exposure-bias-gap diagnostic: collect (teacher-forced CE on z_sampled,
    # free-running pass/fail) pairs per example so we can split CE by outcome.
    # See docs/exposure_bias.md for what these numbers mean.
    ce_pass_sum = ce_fail_sum = 0.0
    n_pass = n_fail = 0
    samples = []
    n = min(n_examples, len(dataset))
    for start in range(0, n, batch_size):
        batch_idx = list(range(start, min(start + batch_size, n)))
        batch_examples = [dataset[i] for i in batch_idx]
        questions = [b["question"] for b in batch_examples]
        answers = [b["answer"] for b in batch_examples]

        z_q = ae.encode_to_latents(questions, device, max_length=max_q_length)
        z_a = ae.encode_to_latents(answers, device, max_length=max_a_length)
        b = z_q.size(0)

        z_sampled  = diffusion.sample(z_q, inner_steps=inner_steps)
        z_sampled0 = diffusion.sample(z_q, inner_steps=0) if inner_steps > 0 else z_sampled

        preds_ebm  = ae.decode(z_sampled,  max_length=max_a_length)
        preds_ebm0 = ae.decode(z_sampled0, max_length=max_a_length) if inner_steps > 0 else preds_ebm
        preds_ae   = ae.decode(z_a, max_length=max_a_length)

        mse_z_sum  += (z_sampled  - z_a).pow(2).mean().item() * b
        mse_z0_sum += (z_sampled0 - z_a).pow(2).mean().item() * b
        sq_za_sum  += z_a.pow(2).sum().item()
        sq_zs_sum  += z_sampled.pow(2).sum().item()
        n_elems    += z_a.numel()

        za_flat = z_a.reshape(b, -1)
        zs_flat = z_sampled.reshape(b, -1)
        cos_sum += torch.nn.functional.cosine_similarity(za_flat, zs_flat, dim=-1).sum().item()

        # Teacher-forced CE on z_sampled — the other half of the exposure-bias
        # pair. Free-running pass/fail comes from verify_code below; CE is per-
        # example so we can bucket by outcome.
        ce_pe = ae.decode_loss_per_example(
            z_sampled, answers, device, max_length=max_a_length,
        ).detach().cpu().tolist()

        # Parallel subprocess verify for all three prediction streams.
        ok_ebm  = verify_code_batch(preds_ebm,  batch_examples)
        ok_ebm0 = verify_code_batch(preds_ebm0, batch_examples)
        ok_ae_l = verify_code_batch(preds_ae,   batch_examples)
        for j, (p_ebm, p_ebm0, p_ae, gold, q, ok, ok0, ok_ae) in enumerate(
                zip(preds_ebm, preds_ebm0, preds_ae, answers, questions,
                    ok_ebm, ok_ebm0, ok_ae_l)):
            total += 1
            if ok:    correct    += 1
            if ok0:   correct0   += 1
            if ok_ae: ae_correct += 1
            if ok:
                ce_pass_sum += ce_pe[j]
                n_pass += 1
            else:
                ce_fail_sum += ce_pe[j]
                n_fail += 1
            if len(samples) < 3:
                samples.append((q[:80], gold, p_ae, p_ebm, p_ebm0))
    diffusion.train()
    n_t = max(total, 1)
    ce_pass = (ce_pass_sum / n_pass) if n_pass else float("nan")
    ce_fail = (ce_fail_sum / n_fail) if n_fail else float("nan")
    # Exposure-bias gap: how much higher is teacher-forced CE on examples that
    # the free-running decoder got wrong, vs ones it got right? Large gap →
    # CE tracks pass/fail → latent quality varies per example → AR is doing
    # its job. Small gap with both CEs low → latent uniformly points decoder
    # toward gold, but free-running decode flakes → AR is leaking cognition.
    # See docs/exposure_bias.md.
    eb_gap = (ce_fail - ce_pass) if (n_pass and n_fail) else float("nan")
    return {
        "acc": correct / n_t,
        "acc_inner0": correct0 / n_t,
        "ae_acc": ae_correct / n_t,
        "mse_z": mse_z_sum / n_t,
        "mse_z_inner0": mse_z0_sum / n_t,
        "std_za": (sq_za_sum / max(n_elems, 1)) ** 0.5,
        "std_zs": (sq_zs_sum / max(n_elems, 1)) ** 0.5,
        "corr_z": cos_sum / n_t,
        "ce_pass": ce_pass,
        "ce_fail": ce_fail,
        "eb_gap": eb_gap,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "samples": samples,
        "n": total,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    with make_reporter(
        use_tui=args.tui,
        eval_series=(
            "mbpp_pass", "mbpp_pass_inner0", "mbpp_ae_pass",
            "humaneval_pass", "humaneval_pass_inner0", "humaneval_ae_pass",
        ),
        title="train_diffusion",
    ) as r:
        # autoencoder
        r.log(f"loading autoencoder: {args.model}")
        d_ae = args.d_ae if args.d_ae > 0 else None
        ae = FrozenBartAutoencoder(
            model_name=args.model,
            k=args.k,
            pool_layers=args.pool_layers,
            pool_heads=args.pool_heads,
            pool_type=args.pool_type,
            d_ae=d_ae,
            recon_layers=args.recon_layers,
            recon_heads=args.recon_heads,
        ).to(args.device)
        ckpt = torch.load(args.ae_ckpt, map_location=args.device)
        if "ae" in ckpt:
            ae.load_ae(ckpt["ae"])
        else:
            raise RuntimeError(
                f"checkpoint {args.ae_ckpt} predates the LD4LG split (no 'ae' key). "
                "Retrain the autoencoder with the new architecture before training the EBM."
            )
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        mbpp_at_ckpt = ckpt.get("mbpp_pass", "unknown")
        he_at_ckpt = ckpt.get("humaneval_pass", "unknown")
        r.log(
            f"autoencoder loaded (mbpp_pass@ckpt={mbpp_at_ckpt} humaneval_pass@ckpt={he_at_ckpt})  "
            f"d_model={ae.d_model}  d_ae={ae.d_ae}"
        )

        # ebm — diffuses in d_ae space
        d_diff = ae.d_ae
        ebm = EnergyTransformer(
            d_model=d_diff,
            k=args.k,
            n_layers=args.ebm_layers,
            n_heads=args.ebm_heads,
            dim_ff_mult=args.ebm_ff_mult,
        ).to(args.device)
        wrapper = DiffusionWrapper(ebm).to(args.device)

        diffusion = GaussianLatentDiffusion(
            model=wrapper,
            latent_shape=(args.k, d_diff),
            timesteps=args.timesteps,
            beta_schedule=args.beta_schedule,
            opt_step_size=args.opt_step_size,
            loss_scale=args.nce_scale,
            continuous=args.continuous,
            supervise_energy_landscape=not args.no_nce,
            x_start_clamp=args.x_start_clamp if args.x_start_clamp > 0 else None,
            envelope_sf=args.envelope_sf if args.envelope_sf > 0 else None,
            decoder_aux_weight=args.decoder_aux_weight,
            decoder_aux_t_max=args.decoder_aux_t_max,
            rand_neg_weight=args.rand_neg_weight,
            rand_neg_margin=args.rand_neg_margin,
            rand_neg_t_max=args.rand_neg_t_max,
        ).to(args.device)
        diffusion.train()

        if args.decoder_aux_weight > 0:
            def _decoder_loss_fn(x0_hat, texts):
                return ae.decode_loss(x0_hat, texts, args.device, max_length=args.max_a_length)
            diffusion.set_decoder_loss_fn(_decoder_loss_fn)
            r.log(f"decoder-aux enabled: weight={args.decoder_aux_weight} t_max={args.decoder_aux_t_max}")

        if args.rand_neg_weight > 0:
            r.log(
                f"rand-neg anchor enabled: weight={args.rand_neg_weight} "
                f"margin={args.rand_neg_margin} t_max={args.rand_neg_t_max}"
            )

        n_params = sum(p.numel() for p in ebm.parameters())
        r.log(f"ebm params: {n_params:,}  (diffuses in d_ae={d_diff}, layers={args.ebm_layers})")

        # data — train on MBPP; eval on MBPP test + HumanEval (held-out).
        r.log("loading datasets...")
        train_ds = MBPPDataset(
            split="train", config=args.mbpp_config, seed=args.seed,
        )
        r.log(f"train dataset: MBPP ({len(train_ds)} examples, config={args.mbpp_config})")

        mbpp_test_ds = MBPPDataset(
            split="test", config=args.mbpp_config,
            max_samples=args.eval_n_examples, seed=args.seed,
        )
        he_test_ds = HumanEvalDataset(max_samples=args.eval_n_examples, seed=args.seed)
        r.log(f"eval datasets: MBPP test ({len(mbpp_test_ds)}), HumanEval ({len(he_test_ds)})")

        # Pre-tokenize question + answer once so the train hot loop never
        # touches the (single-threaded) BART tokenizer.
        r.log("pre-tokenizing train corpus (question + answer)...")
        train_ds = PreTokenizedDataset(
            train_ds, ae.tokenizer,
            max_q_length=args.max_q_length,
            max_a_length=args.max_a_length,
            fields=("question", "answer"),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=make_collate_pretokenized(ae.tokenizer.pad_token_id),
            num_workers=args.num_workers,
            pin_memory=(args.device.startswith("cuda")),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(4 if args.num_workers > 0 else None),
            drop_last=True,
        )

        opt = AdamW(ebm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        start_step = 0
        if args.resume is not None:
            r.log(f"resuming from {args.resume}")
            rckpt = torch.load(args.resume, map_location=args.device)
            ebm.load_state_dict(rckpt["ebm"])
            diffusion.set_latent_stats(
                rckpt["latent_mu"].to(args.device),
                rckpt["latent_sigma"].to(args.device),
            )
            if "opt" in rckpt:
                opt.load_state_dict(rckpt["opt"])
            else:
                r.log("  [!] resume checkpoint has no optimizer state — restarting AdamW moments from zero")
            start_step = int(rckpt.get("step", 0))
            r.log(
                f"  resumed at step {start_step}  "
                f"(mbpp_pass@ckpt={rckpt.get('mbpp_pass', 'n/a')} "
                f"humaneval_pass@ckpt={rckpt.get('humaneval_pass', 'n/a')})"
            )
        else:
            # Compute per-dim latent statistics from a chunk of training answers
            # and install them as the diffusion module's shift/scale buffers. This
            # normalizes z_a (and z_q) to ~N(0, I) so the noise schedule is well
            # calibrated. Done once before training; saved with the EBM checkpoint.
            r.log(
                f"computing latent stats from up to {args.latent_stats_samples} train answers "
                "(answers AE-encoded, then per-dim mean/std across batch × K slots)..."
            )
            stats_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=make_collate_pretokenized(ae.tokenizer.pad_token_id),
                num_workers=0,
                drop_last=False,
            )
            zs: list[torch.Tensor] = []
            seen = 0
            with torch.no_grad():
                for batch in stats_loader:
                    a_ids = batch["answer_input_ids"].to(args.device, non_blocking=True)
                    a_mask = batch["answer_attention_mask"].to(args.device, non_blocking=True)
                    z_a = ae.encode_to_latents_from_ids(a_ids, a_mask)
                    zs.append(z_a.float().cpu())
                    seen += z_a.size(0)
                    if seen >= args.latent_stats_samples:
                        break
            z_stack = torch.cat(zs, dim=0)  # (N, K, d_ae)
            latent_mu = z_stack.mean(dim=(0, 1))  # (d_ae,)
            latent_sigma = z_stack.std(dim=(0, 1))  # (d_ae,)
            diffusion.set_latent_stats(latent_mu.to(args.device), latent_sigma.to(args.device))
            r.log(
                f"  latent stats: N={z_stack.size(0)} K={z_stack.size(1)} d_ae={z_stack.size(2)}  "
                f"|mu|_mean={latent_mu.abs().mean().item():.3f}  "
                f"sigma_mean={latent_sigma.mean().item():.3f}  "
                f"sigma_range=[{latent_sigma.min().item():.3f}, {latent_sigma.max().item():.3f}]"
            )
            del zs, z_stack

        step = start_step
        train_iter = iter(train_loader)
        t0 = time.time()
        accum = max(args.grad_accum_steps, 1)
        while step < args.steps:
            opt.zero_grad()
            loss_sum = 0.0
            stats_accum: dict[str, float] = {}
            for _ in range(accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                with torch.no_grad():
                    q_ids = batch["question_input_ids"].to(args.device, non_blocking=True)
                    q_mask = batch["question_attention_mask"].to(args.device, non_blocking=True)
                    a_ids = batch["answer_input_ids"].to(args.device, non_blocking=True)
                    a_mask = batch["answer_attention_mask"].to(args.device, non_blocking=True)
                    z_q = ae.encode_to_latents_from_ids(q_ids, q_mask)
                    z_a = ae.encode_to_latents_from_ids(a_ids, a_mask)

                loss, stats = diffusion(
                    z_q, z_a,
                    gold_texts=batch["answer"] if args.decoder_aux_weight > 0 else None,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"step {step}: non-finite loss {loss.item()} (stats={stats}). "
                        "Stopping before opt.step() corrupts the EBM weights."
                    )
                (loss / accum).backward()
                loss_sum += loss.item()
                for k, v in stats.items():
                    stats_accum[k] = stats_accum.get(k, 0.0) + float(v)
            torch.nn.utils.clip_grad_norm_(ebm.parameters(), 1.0)
            opt.step()

            step += 1
            avg_loss = loss_sum / accum
            avg_stats = {k: v / accum for k, v in stats_accum.items()}
            cur_lr = opt.param_groups[0]["lr"]
            r.train_point(step, loss=avg_loss, lr=cur_lr)
            r.set_status(f"step {step}/{args.steps} | loss {avg_loss:.4f} | lr {cur_lr:.2e}")
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                extras = " | ".join(f"{k} {v:.4f}" for k, v in avg_stats.items())
                r.log(f"step {step:6d} | loss {avg_loss:.4f} | {extras} | {elapsed:.1f}s")

            if step % args.eval_every == 0 or step == args.steps:
                ev_mbpp = eval_corpus(
                    ae, diffusion, mbpp_test_ds, "mbpp", args.device,
                    args.max_q_length, args.max_a_length, args.inner_steps,
                    n_examples=args.eval_n_examples, batch_size=args.batch_size,
                )
                ev_he = eval_corpus(
                    ae, diffusion, he_test_ds, "humaneval", args.device,
                    args.max_q_length, args.max_a_length, args.inner_steps,
                    n_examples=args.eval_n_examples, batch_size=args.batch_size,
                )
                r.eval_point(
                    step,
                    mbpp_pass=ev_mbpp["acc"],
                    mbpp_pass_inner0=ev_mbpp["acc_inner0"],
                    mbpp_ae_pass=ev_mbpp["ae_acc"],
                    humaneval_pass=ev_he["acc"],
                    humaneval_pass_inner0=ev_he["acc_inner0"],
                    humaneval_ae_pass=ev_he["ae_acc"],
                )
                r.log(
                    f"  [eval mbpp n={ev_mbpp['n']}] "
                    f"mbpp_pass(inner={args.inner_steps})={ev_mbpp['acc']:.3f}  "
                    f"mbpp_pass(inner=0)={ev_mbpp['acc_inner0']:.3f}  "
                    f"mbpp_ae_pass={ev_mbpp['ae_acc']:.3f}"
                )
                r.log(
                    f"  [eval humaneval n={ev_he['n']}] "
                    f"humaneval_pass(inner={args.inner_steps})={ev_he['acc']:.3f}  "
                    f"humaneval_pass(inner=0)={ev_he['acc_inner0']:.3f}  "
                    f"humaneval_ae_pass={ev_he['ae_acc']:.3f}"
                )
                r.log(
                    f"  [latent mbpp] mse_z={ev_mbpp['mse_z']:.3f}  mse_z(inner=0)={ev_mbpp['mse_z_inner0']:.3f}  "
                    f"std_za={ev_mbpp['std_za']:.3f}  std_zs={ev_mbpp['std_zs']:.3f}  corr_z={ev_mbpp['corr_z']:+.3f}"
                )
                r.log(
                    f"  [exposure mbpp] ce_pass={ev_mbpp['ce_pass']:.3f} (n={ev_mbpp['n_pass']})  "
                    f"ce_fail={ev_mbpp['ce_fail']:.3f} (n={ev_mbpp['n_fail']})  "
                    f"gap={ev_mbpp['eb_gap']:+.3f}"
                )
                r.log(
                    f"  [exposure humaneval] ce_pass={ev_he['ce_pass']:.3f} (n={ev_he['n_pass']})  "
                    f"ce_fail={ev_he['ce_fail']:.3f} (n={ev_he['n_fail']})  "
                    f"gap={ev_he['eb_gap']:+.3f}"
                )
                if ev_mbpp["ae_acc"] < 0.5 or ev_he["ae_acc"] < 0.5:
                    r.log("  [!] ae_pass < 0.5 on at least one corpus — Milestone 1 (AE) is the bottleneck.")
                def _snip(s, n=120):
                    s = s.replace("\n", " ⏎ ")
                    return s if len(s) <= n else s[:n] + "…"
                for name, ev in [("mbpp", ev_mbpp), ("humaneval", ev_he)]:
                    for q, gold, p_ae, p_ebm, p_ebm0 in ev["samples"][:2]:
                        r.log(f"    [{name}] Q: {_snip(q)}")
                        r.log(f"      gold       : {_snip(gold)}")
                        r.log(f"      ae_recon   : {_snip(p_ae)}")
                        r.log(f"      ebm(inner={args.inner_steps:>2}): {_snip(p_ebm)}")
                        r.log(f"      ebm(inner= 0): {_snip(p_ebm0)}")
                ck = {
                    "ebm": ebm.state_dict(),
                    "opt": opt.state_dict(),
                    "latent_mu": diffusion.latent_mu.detach().cpu(),
                    "latent_sigma": diffusion.latent_sigma.detach().cpu(),
                    "config": vars(args),
                    "step": step,
                    "mbpp_pass": ev_mbpp["acc"],
                    "humaneval_pass": ev_he["acc"],
                    "mbpp_ae_pass": ev_mbpp["ae_acc"],
                    "humaneval_ae_pass": ev_he["ae_acc"],
                    "mse_z_mbpp": ev_mbpp["mse_z"],
                    "mbpp_eb_gap": ev_mbpp["eb_gap"],
                    "humaneval_eb_gap": ev_he["eb_gap"],
                }
                torch.save(ck, os.path.join(args.save_dir, f"ebm_step{step}.pt"))
                torch.save(ck, os.path.join(args.save_dir, "ebm_latest.pt"))


if __name__ == "__main__":
    main()
