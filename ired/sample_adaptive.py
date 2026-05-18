"""Adaptive Mode-1 / Mode-2 inference (gensis.md §5.4).

Default to the fast actor; escalate to the slow IRED rollout when the EBM
energy on the actor's output exceeds `threshold`. The gate scalar is exactly
the calibrated quality scalar that defines correctness during Mode-2 training
(§2), so no second calibration problem.

Outputs a per-sample breakdown:

  inner   acc    n   ebm_passes   actor_frac   time_s
  -----   ---   --   ----------   ----------   ------
  ...

`actor_frac` is the fraction of test queries that took the cheap Mode-1 path
at the given threshold — the inference-economics knob from §1.4 / §5.4.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from ired.actor import Actor
from ired.autoencoder import FrozenT5Autoencoder
from ired.data import GSM8KDataset, collate, extract_final_answer
from ired.diffusion import GaussianLatentDiffusion
from ired.energy_net import DiffusionWrapper, EnergyTransformer


def build_parser():
    p = argparse.ArgumentParser(description="Adaptive Mode-1 / Mode-2 inference")
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--ae-ckpt", required=True)
    p.add_argument("--ebm-ckpt", required=True)
    p.add_argument("--actor-ckpt", required=True)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
    p.add_argument("--pool-type", choices=["decoder", "resampler"], default="decoder")
    p.add_argument("--d-ae", type=int, default=-1)
    p.add_argument("--recon-layers", type=int, default=2)
    p.add_argument("--recon-heads", type=int, default=8)
    p.add_argument("--ebm-layers", type=int, default=4)
    p.add_argument("--ebm-heads", type=int, default=8)
    p.add_argument("--ebm-ff-mult", type=int, default=4)
    p.add_argument("--actor-layers", type=int, default=4)
    p.add_argument("--actor-heads", type=int, default=8)
    p.add_argument("--actor-ff-mult", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--inner-steps", type=int, default=5)
    p.add_argument("--x-start-clamp", type=float, default=5.0)
    p.add_argument("--envelope-sf", type=float, default=-1.0)
    p.add_argument("--answer-mode", choices=["final", "full"], default="full")
    p.add_argument("--threshold", type=float, default=None,
                   help="Energy threshold. If unset, calibrate to keep --actor-target "
                        "of queries on the Mode-1 path.")
    p.add_argument("--actor-target", type=float, default=0.8,
                   help="Fraction of queries to route through the actor when "
                        "calibrating threshold automatically (default 80%%, §5.4).")
    p.add_argument("--calib-batches", type=int, default=5,
                   help="Number of test batches used to calibrate the threshold.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--max-q-length", type=int, default=256)
    p.add_argument("--max-a-length", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    return p


@torch.no_grad()
def calibrate_threshold(
    ae, actor, ebm, loader, device, max_q_length, target, n_batches,
):
    """Pick `threshold` so ~`target` fraction of E(z_q, actor(z_q), 0) values
    fall below it. That's the gate value at which `target`% of queries take the
    Mode-1 path on average."""
    energies: list[float] = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        z_q = ae.encode_to_latents(batch["question"], device, max_length=max_q_length)
        z_hat = actor(z_q)
        t0 = torch.zeros(z_q.size(0), device=device, dtype=torch.long)
        e = ebm(z_q, z_hat, t0).squeeze(-1)
        energies.extend(e.tolist())
    if not energies:
        return float("inf")
    energies.sort()
    idx = max(0, min(len(energies) - 1, int(round(target * len(energies))) - 1))
    return energies[idx]


@torch.no_grad()
def adaptive_sample(
    ae, actor, ebm, diffusion, loader, device,
    max_q_length, max_a_length, inner_steps, threshold, n_batches,
):
    a_correct = total = 0
    actor_pass = 0
    ebm_actor_passes = 0          # only the actor was invoked
    ebm_full_passes = 0           # actor + full Mode-2 (escalated)
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        z_q = ae.encode_to_latents(batch["question"], device, max_length=max_q_length)
        b = z_q.size(0)

        z_hat = actor(z_q)
        t0 = torch.zeros(b, device=device, dtype=torch.long)
        e_hat = ebm(z_q, z_hat, t0).squeeze(-1)
        take_actor = e_hat < threshold                                   # (B,)
        actor_pass += int(take_actor.sum().item())

        z_final = z_hat.clone()
        if (~take_actor).any():
            idx_t = (~take_actor).nonzero(as_tuple=True)[0]
            z_q_hard = z_q.index_select(0, idx_t)
            z_full = diffusion.sample(z_q_hard, inner_steps=inner_steps)
            z_final[idx_t] = z_full
            ebm_full_passes += int(idx_t.numel())
        ebm_actor_passes += b

        preds = ae.decode(z_final, max_length=max_a_length)
        for p, gold in zip(preds, batch["answer"]):
            total += 1
            if extract_final_answer(p) == extract_final_answer(gold):
                a_correct += 1

    acc = a_correct / max(total, 1)
    actor_frac = actor_pass / max(total, 1)
    # 1 actor pass per query + (T * (1 + inner_steps)) per escalation
    teacher_cost = (diffusion.num_timesteps * (1 + inner_steps)) if diffusion is not None else 0
    avg_passes = 1.0 + (1.0 - actor_frac) * teacher_cost
    return {
        "acc": acc,
        "actor_frac": actor_frac,
        "avg_ebm_passes": avg_passes,
        "n": total,
        "n_escalated": total - actor_pass,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    d_ae_arg = args.d_ae if args.d_ae > 0 else None

    ae = FrozenT5Autoencoder(
        model_name=args.model,
        k=args.k,
        pool_layers=args.pool_layers,
        pool_heads=args.pool_heads,
        pool_type=args.pool_type,
        d_ae=d_ae_arg,
        recon_layers=args.recon_layers,
        recon_heads=args.recon_heads,
    ).to(args.device)
    ae_ckpt = torch.load(args.ae_ckpt, map_location=args.device)
    if "ae" not in ae_ckpt:
        raise RuntimeError(f"{args.ae_ckpt} predates the LD4LG split (no 'ae' key).")
    ae.load_ae(ae_ckpt["ae"])
    ae.eval()
    d_diff = ae.d_ae

    ebm = EnergyTransformer(
        d_model=d_diff,
        k=args.k,
        n_layers=args.ebm_layers,
        n_heads=args.ebm_heads,
        dim_ff_mult=args.ebm_ff_mult,
    ).to(args.device)
    ebm.load_state_dict(torch.load(args.ebm_ckpt, map_location=args.device)["ebm"])
    ebm.eval()

    actor = Actor(
        d_ae=d_diff,
        k=args.k,
        n_layers=args.actor_layers,
        n_heads=args.actor_heads,
        dim_ff_mult=args.actor_ff_mult,
    ).to(args.device)
    actor.load_state_dict(torch.load(args.actor_ckpt, map_location=args.device)["actor"])
    actor.eval()

    wrapper = DiffusionWrapper(ebm).to(args.device)
    diffusion = GaussianLatentDiffusion(
        model=wrapper,
        latent_shape=(args.k, d_diff),
        timesteps=args.timesteps,
        x_start_clamp=args.x_start_clamp if args.x_start_clamp > 0 else None,
        envelope_sf=args.envelope_sf if args.envelope_sf > 0 else None,
    ).to(args.device)
    diffusion.eval()

    test_ds = GSM8KDataset("test", answer_mode=args.answer_mode)
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers,
    )

    if args.threshold is None:
        threshold = calibrate_threshold(
            ae, actor, ebm, loader, args.device,
            args.max_q_length, args.actor_target, args.calib_batches,
        )
        print(f"calibrated threshold = {threshold:.4f}  (target actor frac {args.actor_target:.0%})")
    else:
        threshold = args.threshold
        print(f"using threshold = {threshold:.4f}")

    t0 = time.time()
    out = adaptive_sample(
        ae, actor, ebm, diffusion, loader, args.device,
        args.max_q_length, args.max_a_length, args.inner_steps,
        threshold, args.n_batches,
    )
    dt = time.time() - t0
    print(
        f"adaptive: acc={out['acc']:.3f}  n={out['n']}  "
        f"actor_frac={out['actor_frac']:.2%}  "
        f"escalated={out['n_escalated']}/{out['n']}  "
        f"avg_ebm_passes={out['avg_ebm_passes']:.2f}  "
        f"time={dt:.1f}s"
    )
    full_cost = 1 + args.timesteps * (1 + args.inner_steps)
    actor_only = 1
    print(
        f"  reference: actor-only={actor_only} passes/query, "
        f"full Mode-2={full_cost} passes/query, "
        f"adaptive saves {(1 - out['avg_ebm_passes'] / full_cost):.0%} vs full"
    )


if __name__ == "__main__":
    main()
