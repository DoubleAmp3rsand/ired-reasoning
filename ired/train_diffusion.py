"""Milestone 2 (gensis.md §7): train the IRED energy network in latent space.

Pipeline per step:
  with no_grad:
      z_q = ae.encode_to_latents(question)
      z_a = ae.encode_to_latents(answer)
  loss, stats = diffusion(z_q, z_a, gold_texts=...)   # gold passed only if a
                                                      # decode-grounded term is on
  loss.backward()
  opt.step()

Only the EBM updates. The autoencoder (frozen BART + previously trained pool) is
held fixed. Every `eval_every` steps we sample end-to-end and report per-corpus
exact-match rates on ZebraLogic puzzles (puzzle-level grid comparison).

Data protocol (gensis.md §7 — "reasoning, not generator-fitting")
-----------------------------------------------------------------
`WildEval/ZebraLogic` ships only a 1000-puzzle *test* split, so the EBM trains on
a **different** generator: `ClueZebraGridDataset` synthesizes unlimited fresh
`(prose clues, unique solution)` pairs (`ired/puzzle_gen.py`), eval-disjoint by
construction. Two evals keep us honest:
  - eval-1 (primary): held-out WildEval prose — generator→WildEval *transfer*,
    so a model that merely fit the training generator cannot score.
  - eval-2: generator puzzles at *larger* sizes than training — length
    generalization.
On both, the headline discriminator is the **inner-gap** `acc − acc_inner0`:
iterative energy descent beating single-shot is the evidence of reasoning rather
than disguised lookup.

Loss terms (see GaussianLatentDiffusion.p_losses)
-------------------------------------------------
Always on:
  - MSE        : denoising regression, ε̂ = ∇_{z_t}E ≈ noise.
  - NCE        : E(z_a) < E(mined hard-negative), where the negative is mined
                 geometrically (heavy-noise + 2 opt_steps).

Optional (off by default; enable per the flags below):
  - rand-neg   (--rand-neg-weight)   : E(real) + margin < E(N(0,I)).
  - decoder-aux(--decoder-aux-weight): CE(decode(x0_hat), gold) on low-t samples.
  - gen-neg    (--gen-neg-weight)    : generator-grounded negative.
"""

from __future__ import annotations

import argparse
import os
import time

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ired.model.autoencoder import FrozenBartAutoencoder
from ired.data import (
    ClueZebraGridDataset,
    ZebraLogicDataset,
    PreTokenizedDataset,
    make_collate_pretokenized,
    zebra_match_batch,
)
from ired.model.diffusion import GaussianLatentDiffusion
from ired.model.energy_net import DiffusionWrapper, EnergyTransformer
from ired.tui import make_reporter


def _resolve_ckpt_path(spec: str) -> str:
    """Resolve a checkpoint spec to a local file path.

    Accepts a local path (returned unchanged) or
    'hf://<org>/<repo>[@<revision>]/<filename>', which is downloaded via
    huggingface_hub.hf_hub_download and the cached path returned.
    """
    if not spec.startswith("hf://"):
        return spec
    from huggingface_hub import hf_hub_download
    parts = spec[len("hf://"):].split("/", 2)
    if len(parts) < 3:
        raise ValueError(
            f"hf:// spec must be 'hf://<org>/<repo>[@<revision>]/<filename>', got: {spec}"
        )
    org, repo_or_rev, filename = parts
    revision = None
    if "@" in repo_or_rev:
        repo_or_rev, revision = repo_or_rev.split("@", 1)
    return hf_hub_download(repo_id=f"{org}/{repo_or_rev}", filename=filename, revision=revision)


def build_parser():
    p = argparse.ArgumentParser(description="Train IRED energy network in latent space")
    p.add_argument("--config", default=None,
                   help="path to a YAML config file. CLI flags override it.")
    # autoencoder
    p.add_argument("--model", default="facebook/bart-base")
    p.add_argument("--ae-ckpt",
                   default="hf://roy-W/ired-reasoning/checkpoints/ae_zebra_conv/ae_latest.pt",
                   help="AE checkpoint from train_autoencoder.")
    p.add_argument("--k", type=int, default=128, help="Must match the AE checkpoint's k.")
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
    p.add_argument("--pool-type", choices=["decoder", "resampler", "conv"], default="decoder",
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
    p.add_argument("--beta-schedule", choices=["cosine"], default="cosine")
    p.add_argument("--opt-step-size", type=float, default=1.0)
    p.add_argument("--opt-noise-scale", type=float, default=0.0)
    p.add_argument("--opt-reject", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--continuous", action="store_true", default=True)
    p.add_argument("--no-nce", action="store_true",
                   help="disable energy-landscape supervision (NCE)")
    p.add_argument("--nce-scale", type=float, default=1.0)
    p.add_argument("--x-start-clamp", type=float, default=5.0)
    p.add_argument("--envelope-sf", type=float, default=-1.0)
    p.add_argument("--decoder-aux-weight", type=float, default=0.0)
    p.add_argument("--decoder-aux-t-max", type=int, default=2)
    p.add_argument("--rand-neg-weight", type=float, default=0.0)
    p.add_argument("--rand-neg-margin", type=float, default=1.0)
    p.add_argument("--rand-neg-t-max", type=int, default=1_000_000)
    p.add_argument("--gen-neg-weight", type=float, default=0.0)
    p.add_argument("--gen-neg-margin", type=float, default=1.0)
    p.add_argument("--gen-neg-t-max", type=int, default=3)
    p.add_argument("--gen-neg-ce-thresh", type=float, default=0.5)
    # data & opt
    # Train corpus: synthetic clue puzzles (different generator than the eval).
    p.add_argument("--clue-samples", type=int, default=20000,
                   help="synthetic clue puzzles to materialize for training.")
    p.add_argument("--clue-min-size", type=int, default=2,
                   help="smallest training puzzle (houses).")
    p.add_argument("--clue-max-size", type=int, default=4,
                   help="largest training puzzle (houses); eval-2 tests above this.")
    p.add_argument("--clue-min-attrs", type=int, default=3)
    p.add_argument("--clue-max-attrs", type=int, default=5)
    p.add_argument("--clue-min-level", type=int, default=5,
                   help="generator difficulty floor (WildEval-aligned relation set).")
    p.add_argument("--clue-max-level", type=int, default=8)
    p.add_argument("--clue-minimize-seconds", type=float, default=2.0,
                   help="per-puzzle cap for clue-count minimization.")
    # eval-1 (primary): held-out WildEval prose — generator→WildEval transfer.
    p.add_argument("--zebra-min-size", type=int, default=2,
                   help="smallest WildEval puzzle (houses) to eval on.")
    p.add_argument("--zebra-max-size", type=int, default=6,
                   help="largest WildEval puzzle (houses) to eval on.")
    # eval-2: generator puzzles ABOVE the training size band — length generalization.
    p.add_argument("--gen-eval-min-size", type=int, default=5)
    p.add_argument("--gen-eval-max-size", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--latent-stats-samples", type=int, default=1024,
                   help="number of training answers to encode for latent stats.")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-dir", default="checkpoints/ebm")
    p.add_argument("--resume", default=None,
                   help="EBM checkpoint to resume from.")
    p.add_argument("--push-to-hub", default=None,
                   help="if set (e.g. 'user/repo'), upload ebm_latest.pt to HF Hub.")
    p.add_argument("--hub-private", action="store_true")
    p.add_argument("--max-q-length", type=int, default=512)
    p.add_argument("--max-a-length", type=int, default=384)
    p.add_argument("--eval-n-examples", type=int, default=80,
                   help="per-corpus examples to score each eval cycle.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--tui", action="store_true",
                   help="render training output as a Rich+plotext TUI.")
    return p


@torch.no_grad()
def eval_corpus(ae, diffusion, dataset, device,
                max_q_length, max_a_length, inner_steps,
                n_examples=80, batch_size=16):
    """End-to-end EBM eval on ZebraLogic.

    Reports:
      - acc           : exact-match rate at inner_steps (zebra_match vs gold grid)
      - acc_inner0    : exact-match rate with inner_steps=0 (isolates opt_step)
      - ae_acc        : exact-match rate of decode(encode(answer)) — Milestone 1 ceiling
      - mse_z, std_za, std_zs, corr_z : latent diagnostics
    """
    diffusion.eval()
    correct = correct0 = ae_correct = total = 0
    mse_z_sum = mse_z0_sum = 0.0
    sq_za_sum = sq_zs_sum = 0.0
    n_elems = 0
    cos_sum = 0.0
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

        ok_ebm  = zebra_match_batch(preds_ebm,  batch_examples)
        ok_ebm0 = zebra_match_batch(preds_ebm0, batch_examples)
        ok_ae_l = zebra_match_batch(preds_ae,   batch_examples)
        for j, (p_ebm, p_ebm0, p_ae, gold, q, ok, ok0, ok_ae) in enumerate(
                zip(preds_ebm, preds_ebm0, preds_ae, answers, questions,
                    ok_ebm, ok_ebm0, ok_ae_l)):
            total += 1
            if ok:    correct    += 1
            if ok0:   correct0   += 1
            if ok_ae: ae_correct += 1
            if len(samples) < 3:
                samples.append((q[:80], gold, p_ae, p_ebm, p_ebm0))
    diffusion.train()
    n_t = max(total, 1)
    return {
        "acc": correct / n_t,
        "acc_inner0": correct0 / n_t,
        "ae_acc": ae_correct / n_t,
        "mse_z": mse_z_sum / n_t,
        "mse_z_inner0": mse_z0_sum / n_t,
        "std_za": (sq_za_sum / max(n_elems, 1)) ** 0.5,
        "std_zs": (sq_zs_sum / max(n_elems, 1)) ** 0.5,
        "corr_z": cos_sum / n_t,
        "samples": samples,
        "n": total,
    }


def main(argv=None):
    parser = build_parser()
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config is not None:
        if yaml is None:
            parser.error("--config requires PyYAML (`pip install pyyaml`)")
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            parser.error(f"config file {pre_args.config!r} must be a YAML mapping")
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    with make_reporter(
        use_tui=args.tui,
        eval_series=(
            "zebra_exact", "zebra_exact_inner0", "zebra_ae_exact",
            "gen_exact", "gen_exact_inner0",
        ),
        title="train_diffusion",
    ) as r:
        # autoencoder
        r.log(f"loading autoencoder: {args.model}")
        ae_ckpt_path = _resolve_ckpt_path(args.ae_ckpt)
        if ae_ckpt_path != args.ae_ckpt:
            r.log(f"  pulled AE checkpoint from HF Hub → {ae_ckpt_path}")
        ckpt = torch.load(ae_ckpt_path, map_location=args.device)

        # Architecture must match the checkpoint exactly.
        ck_cfg = ckpt.get("config", {})
        for field, attr in (("k", "k"), ("pool_type", "pool_type"),
                            ("d_ae", "d_ae")):
            if field in ck_cfg and getattr(args, attr) != ck_cfg[field]:
                r.log(f"  [arch] overriding --{attr.replace('_', '-')} "
                      f"{getattr(args, attr)} → {ck_cfg[field]} (from checkpoint config)")
                setattr(args, attr, ck_cfg[field])

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
        zebra_at_ckpt = ckpt.get("zebra_exact", "unknown")
        r.log(
            f"autoencoder loaded (zebra_exact@ckpt={zebra_at_ckpt})  "
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
            opt_noise_scale=args.opt_noise_scale,
            opt_reject=args.opt_reject,
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
            gen_neg_weight=args.gen_neg_weight,
            gen_neg_margin=args.gen_neg_margin,
            gen_neg_t_max=args.gen_neg_t_max,
            gen_neg_ce_thresh=args.gen_neg_ce_thresh,
        ).to(args.device)
        diffusion.train()

        if args.decoder_aux_weight > 0:
            def _decoder_loss_fn(x0_hat, texts):
                return ae.decode_loss(
                    x0_hat, texts, args.device,
                    max_length=args.max_a_length,
                )
            diffusion.set_decoder_loss_fn(_decoder_loss_fn)
            r.log(f"decoder-aux enabled: weight={args.decoder_aux_weight} t_max={args.decoder_aux_t_max}")

        if args.rand_neg_weight > 0:
            r.log(
                f"rand-neg anchor enabled: weight={args.rand_neg_weight} "
                f"margin={args.rand_neg_margin} t_max={args.rand_neg_t_max}"
            )

        if args.gen_neg_weight > 0:
            def _gen_ce_fn(z_native, texts):
                return ae.decode_loss_per_example(
                    z_native, texts, args.device,
                    max_length=args.max_a_length,
                )
            diffusion.set_gen_ce_fn(_gen_ce_fn)
            r.log(
                f"generator-grounded negative enabled: weight={args.gen_neg_weight} "
                f"margin={args.gen_neg_margin} t_max={args.gen_neg_t_max} "
                f"ce_thresh={args.gen_neg_ce_thresh}"
            )

        n_params = sum(p.numel() for p in ebm.parameters())
        r.log(f"ebm params: {n_params:,}  (diffuses in d_ae={d_diff}, layers={args.ebm_layers})")

        # data — train on the synthetic clue generator; eval on a DIFFERENT
        # generator (WildEval, held-out prose) plus larger synthetic puzzles.
        r.log("loading datasets...")
        r.log(f"generating {args.clue_samples} synthetic clue puzzles "
              f"(sizes {args.clue_min_size}-{args.clue_max_size}, "
              f"levels {args.clue_min_level}-{args.clue_max_level})...")
        train_ds = ClueZebraGridDataset(
            max_samples=args.clue_samples,
            min_size=args.clue_min_size, max_size=args.clue_max_size,
            min_attrs=args.clue_min_attrs, max_attrs=args.clue_max_attrs,
            min_level=args.clue_min_level, max_level=args.clue_max_level,
            max_seconds_for_minimizing=args.clue_minimize_seconds,
            seed=args.seed,
        )
        r.log(f"train dataset: ClueZebraGrid ({len(train_ds)} puzzles)")

        # eval-1 (primary): held-out WildEval prose — the transfer test.
        zebra_test_ds = ZebraLogicDataset(
            split="test", max_samples=args.eval_n_examples, seed=args.seed,
            min_size=args.zebra_min_size, max_size=args.zebra_max_size,
        )
        r.log(f"eval-1 (transfer): WildEval/ZebraLogic test ({len(zebra_test_ds)} puzzles)")

        # eval-2: generator puzzles above the training size band.
        gen_eval_ds = ClueZebraGridDataset(
            max_samples=args.eval_n_examples,
            min_size=args.gen_eval_min_size, max_size=args.gen_eval_max_size,
            min_attrs=args.clue_min_attrs, max_attrs=args.clue_max_attrs,
            min_level=args.clue_min_level, max_level=args.clue_max_level,
            max_seconds_for_minimizing=args.clue_minimize_seconds,
            seed=args.seed + 1,
        )
        r.log(f"eval-2 (length-gen): ClueZebraGrid sizes "
              f"{args.gen_eval_min_size}-{args.gen_eval_max_size} ({len(gen_eval_ds)} puzzles)")

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
            resume_path = _resolve_ckpt_path(args.resume)
            r.log(f"resuming from {args.resume}")
            if resume_path != args.resume:
                r.log(f"  pulled resume checkpoint from HF Hub → {resume_path}")
            rckpt = torch.load(resume_path, map_location=args.device)
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
                f"(zebra_exact@ckpt={rckpt.get('zebra_exact', 'n/a')})"
            )
        else:
            # Compute per-dim latent statistics from a chunk of training answers.
            r.log(
                f"computing latent stats from up to {args.latent_stats_samples} train answers ..."
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
            z_stack = torch.cat(zs, dim=0)
            latent_mu = z_stack.mean(dim=(0, 1))
            latent_sigma = z_stack.std(dim=(0, 1))
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

                need_gold = args.decoder_aux_weight > 0 or args.gen_neg_weight > 0
                loss, stats = diffusion(
                    z_q, z_a,
                    gold_texts=batch["answer"] if need_gold else None,
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
                ev_zebra = eval_corpus(
                    ae, diffusion, zebra_test_ds, args.device,
                    args.max_q_length, args.max_a_length, args.inner_steps,
                    n_examples=args.eval_n_examples, batch_size=args.batch_size,
                )
                ev_gen = eval_corpus(
                    ae, diffusion, gen_eval_ds, args.device,
                    args.max_q_length, args.max_a_length, args.inner_steps,
                    n_examples=args.eval_n_examples, batch_size=args.batch_size,
                )
                z_gap = ev_zebra["acc"] - ev_zebra["acc_inner0"]
                g_gap = ev_gen["acc"] - ev_gen["acc_inner0"]
                r.eval_point(
                    step,
                    zebra_exact=ev_zebra["acc"],
                    zebra_exact_inner0=ev_zebra["acc_inner0"],
                    zebra_ae_exact=ev_zebra["ae_acc"],
                    gen_exact=ev_gen["acc"],
                    gen_exact_inner0=ev_gen["acc_inner0"],
                )
                r.log(
                    f"  [eval-1 transfer/WildEval n={ev_zebra['n']}] "
                    f"exact(inner={args.inner_steps})={ev_zebra['acc']:.3f}  "
                    f"exact(inner=0)={ev_zebra['acc_inner0']:.3f}  "
                    f"inner-gap={z_gap:+.3f}  ae_exact={ev_zebra['ae_acc']:.3f}"
                )
                r.log(
                    f"  [eval-2 length-gen n={ev_gen['n']}] "
                    f"exact(inner={args.inner_steps})={ev_gen['acc']:.3f}  "
                    f"exact(inner=0)={ev_gen['acc_inner0']:.3f}  "
                    f"inner-gap={g_gap:+.3f}  ae_exact={ev_gen['ae_acc']:.3f}"
                )
                r.log(
                    f"  [latent] mse_z={ev_zebra['mse_z']:.3f}  "
                    f"mse_z(inner=0)={ev_zebra['mse_z_inner0']:.3f}  "
                    f"std_za={ev_zebra['std_za']:.3f}  std_zs={ev_zebra['std_zs']:.3f}  "
                    f"corr_z={ev_zebra['corr_z']:+.3f}"
                )
                if ev_zebra["ae_acc"] < 0.5:
                    r.log("  [!] ae_exact < 0.5 — Milestone 1 (AE) is the bottleneck.")
                if z_gap <= 0 and ev_zebra["acc"] > 0:
                    r.log("  [!] inner-gap ≤ 0 on WildEval — optimization not beating "
                          "single-shot (gensis §7: lookup, not reasoning).")

                def _snip(s, n=120):
                    s = s.replace("\n", " ⏎ ")
                    return s if len(s) <= n else s[:n] + "…"
                for q, gold, p_ae, p_ebm, p_ebm0 in ev_zebra["samples"][:2]:
                    r.log(f"    [zebra] Q: {_snip(q)}")
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
                    "zebra_exact": ev_zebra["acc"],
                    "zebra_exact_inner0": ev_zebra["acc_inner0"],
                    "zebra_ae_exact": ev_zebra["ae_acc"],
                    "gen_exact": ev_gen["acc"],
                    "gen_exact_inner0": ev_gen["acc_inner0"],
                    "mse_z": ev_zebra["mse_z"],
                }
                torch.save(ck, os.path.join(args.save_dir, f"ebm_step{step}.pt"))
                torch.save(ck, os.path.join(args.save_dir, "ebm_latest.pt"))

        if args.push_to_hub:
            from huggingface_hub import HfApi, create_repo
            latest = os.path.join(args.save_dir, "ebm_latest.pt")
            r.log(f"pushing {latest} → https://huggingface.co/{args.push_to_hub}")
            create_repo(args.push_to_hub, private=args.hub_private, exist_ok=True)
            HfApi().upload_file(
                path_or_fileobj=latest,
                path_in_repo="ebm_latest.pt",
                repo_id=args.push_to_hub,
            )
            r.log(f"upload complete: https://huggingface.co/{args.push_to_hub}/blob/main/ebm_latest.pt")


if __name__ == "__main__":
    main()
