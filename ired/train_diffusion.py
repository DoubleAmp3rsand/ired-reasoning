"""Milestone 2 (gensis.md §8): train the IRED energy network in latent space.

Pipeline per step:
  with no_grad:
      z_q = ae.encode_to_latents(question)
      z_a = ae.encode_to_latents(answer)
  loss, stats = diffusion(z_q, z_a)
  loss.backward()
  opt.step()

Only the EBM updates. The autoencoder (frozen T5 + previously trained pool) is
held fixed. Every `eval_every` steps we run inference end-to-end and report
exact-match accuracy on a few test batches.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ired.autoencoder import FrozenT5Autoencoder
from ired.data import GSM8KDataset, collate, extract_final_answer
from ired.diffusion import GaussianLatentDiffusion
from ired.energy_net import DiffusionWrapper, EnergyTransformer


def build_parser():
    p = argparse.ArgumentParser(description="Train IRED energy network in latent space")
    # autoencoder
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--ae-ckpt", required=True, help="AE checkpoint from train_autoencoder")
    p.add_argument("--k", type=int, default=32)
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
    # data & opt
    p.add_argument("--answer-mode", choices=["final", "full"], default="full",
                   help="Must match the AE checkpoint's training mode.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-dir", default="checkpoints/ebm")
    p.add_argument("--max-q-length", type=int, default=256)
    p.add_argument("--max-a-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    return p


@torch.no_grad()
def eval_accuracy(ae, diffusion, loader, device, max_q_length, max_a_length, inner_steps, n_batches=5):
    """Joint diagnostic on the same batches:
      - ebm_acc          : end-to-end accuracy (sample → decode) at inner_steps
      - ebm_acc_inner0   : same but with inner_steps=0 (isolates opt_step)
      - ae_acc           : decode(encode(answer)) — Milestone 1 ceiling
      - mse_z, mse_z0    : MSE(z_sampled, z_a) at inner_steps and 0
      - std_za, std_zs   : per-element std of z_a vs z_sampled (collapse / blow-up?)
      - corr_z           : cosine similarity between z_sampled and z_a (flattened)

    Baselines:
      - randn z_sampled gives mse_z ≈ 2·std(z_a)² and corr_z ≈ 0.
      - A perfect denoiser gives mse_z ≈ 0, corr_z ≈ 1, std_zs ≈ std_za.
    """
    diffusion.eval()
    correct = correct0 = ae_correct = total = 0
    mse_z_sum = mse_z0_sum = 0.0
    sq_za_sum = sq_zs_sum = 0.0
    n_elems = 0
    cos_sum = 0.0
    samples = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        z_q = ae.encode_to_latents(batch["question"], device, max_length=max_q_length)
        z_a = ae.encode_to_latents(batch["answer"], device, max_length=max_a_length)
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

        for p_ebm, p_ebm0, p_ae, gold, q in zip(preds_ebm, preds_ebm0, preds_ae, batch["answer"], batch["question"]):
            total += 1
            # Score on the extracted final number so 'full' mode predictions
            # (variable-length CoT ending in "#### N") are compared on N, not
            # byte-exact CoT. For 'final' mode this is a no-op — both sides
            # are already just the number.
            g = extract_final_answer(gold)
            if extract_final_answer(p_ebm)  == g: correct    += 1
            if extract_final_answer(p_ebm0) == g: correct0   += 1
            if extract_final_answer(p_ae)   == g: ae_correct += 1
            if len(samples) < 5:
                samples.append((q[:80], gold, p_ae, p_ebm, p_ebm0))
    diffusion.train()
    n = max(total, 1)
    return {
        "acc": correct / n,
        "acc_inner0": correct0 / n,
        "ae_acc": ae_correct / n,
        "mse_z": mse_z_sum / n,
        "mse_z_inner0": mse_z0_sum / n,
        "std_za": (sq_za_sum / max(n_elems, 1)) ** 0.5,
        "std_zs": (sq_zs_sum / max(n_elems, 1)) ** 0.5,
        "corr_z": cos_sum / n,
        "samples": samples,
        "n": total,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # autoencoder
    print(f"loading autoencoder: {args.model}")
    d_ae = args.d_ae if args.d_ae > 0 else None
    ae = FrozenT5Autoencoder(
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
    # Support both new format (ckpt["ae"]) and legacy (ckpt["pool"]).
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
    print(
        f"autoencoder loaded (eval acc at ckpt: {ckpt.get('eval_acc', 'unknown')})  "
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
    ).to(args.device)
    diffusion.train()

    if args.decoder_aux_weight > 0:
        # Bind the AE's frozen-decoder CE as the auxiliary loss. Gradients flow
        # through the decoder (its weights stay frozen) back into the EBM via
        # x0_hat = predict_start_from_noise(z_t, t, eps_hat).
        def _decoder_loss_fn(x0_hat, texts):
            return ae.decode_loss(x0_hat, texts, args.device, max_length=args.max_a_length)
        diffusion.set_decoder_loss_fn(_decoder_loss_fn)
        print(f"decoder-aux enabled: weight={args.decoder_aux_weight} t_max={args.decoder_aux_t_max}")

    n_params = sum(p.numel() for p in ebm.parameters())
    print(f"ebm params: {n_params:,}  (diffuses in d_ae={d_diff}, layers={args.ebm_layers})")

    # data
    train_ds = GSM8KDataset("train", answer_mode=args.answer_mode)
    test_ds = GSM8KDataset("test", answer_mode=args.answer_mode)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers,
    )

    opt = AdamW(ebm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    train_iter = iter(train_loader)
    t0 = time.time()
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with torch.no_grad():
            z_q = ae.encode_to_latents(batch["question"], args.device, max_length=args.max_q_length)
            z_a = ae.encode_to_latents(batch["answer"], args.device, max_length=args.max_a_length)

        loss, stats = diffusion(
            z_q, z_a,
            gold_texts=batch["answer"] if args.decoder_aux_weight > 0 else None,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"step {step}: non-finite loss {loss.item()} (stats={stats}). "
                "Stopping before opt.step() corrupts the EBM weights."
            )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ebm.parameters(), 1.0)
        opt.step()

        step += 1
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            extras = " | ".join(f"{k} {v:.4f}" for k, v in stats.items())
            print(f"step {step:6d} | loss {loss.item():.4f} | {extras} | {elapsed:.1f}s")

        if step % args.eval_every == 0 or step == args.steps:
            ev = eval_accuracy(
                ae, diffusion, test_loader, args.device,
                args.max_q_length, args.max_a_length, args.inner_steps, n_batches=5,
            )
            print(
                f"  [eval n={ev['n']}] "
                f"ebm_acc(inner={args.inner_steps})={ev['acc']:.3f}  "
                f"ebm_acc(inner=0)={ev['acc_inner0']:.3f}  "
                f"ae_recon_acc={ev['ae_acc']:.3f}"
            )
            print(
                f"  [latent] mse_z={ev['mse_z']:.3f}  mse_z(inner=0)={ev['mse_z_inner0']:.3f}  "
                f"std_za={ev['std_za']:.3f}  std_zs={ev['std_zs']:.3f}  corr_z={ev['corr_z']:+.3f}"
            )
            print(
                f"  [baseline] randn vs z_a expected mse ≈ {2*ev['std_za']**2:.3f}, corr_z ≈ 0"
            )
            if ev["ae_acc"] < 0.5:
                print("  [!] ae_recon_acc < 0.5 — Milestone 1 (pool) is the bottleneck.")
            def _snip(s, n=120):
                s = s.replace("\n", " ⏎ ")
                return s if len(s) <= n else s[:n] + "…"
            for q, gold, p_ae, p_ebm, p_ebm0 in ev["samples"][:3]:
                print(f"    Q: {q}")
                print(f"      gold final  : {extract_final_answer(gold)!r}")
                print(f"      ae_recon    : {extract_final_answer(p_ae)!r:>10}  | text: {_snip(p_ae)}")
                print(f"      ebm(inner={args.inner_steps:>2}): {extract_final_answer(p_ebm)!r:>10}  | text: {_snip(p_ebm)}")
                print(f"      ebm(inner= 0): {extract_final_answer(p_ebm0)!r:>10}  | text: {_snip(p_ebm0)}")
            ck = {
                "ebm": ebm.state_dict(),
                "config": vars(args),
                "step": step,
                "eval_acc": ev["acc"],
                "ae_recon_acc": ev["ae_acc"],
                "mse_z": ev["mse_z"],
            }
            torch.save(ck, os.path.join(args.save_dir, f"ebm_step{step}.pt"))
            torch.save(ck, os.path.join(args.save_dir, "ebm_latest.pt"))


if __name__ == "__main__":
    main()
