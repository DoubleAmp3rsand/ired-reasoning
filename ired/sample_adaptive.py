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

from ired.actor import Actor
from ired.autoencoder import FrozenBartAutoencoder
from ired.data import (
    HumanEvalDataset,
    MBPPDataset,
    verify_code,
)
from ired.diffusion import GaussianLatentDiffusion
from ired.energy_net import DiffusionWrapper, EnergyTransformer


def build_parser():
    p = argparse.ArgumentParser(description="Adaptive Mode-1 / Mode-2 inference")
    p.add_argument("--model", default="facebook/bart-base")
    p.add_argument("--ae-ckpt", required=True)
    p.add_argument("--ebm-ckpt", required=True)
    p.add_argument("--actor-ckpt", required=True)
    p.add_argument("--k", type=int, default=128)
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
    p.add_argument("--eval-dataset", choices=["mbpp", "humaneval"], default="mbpp")
    p.add_argument("--mbpp-config", choices=["full", "sanitized"], default="full")
    p.add_argument("--threshold", type=float, default=None,
                   help="Energy threshold. If unset, calibrate to keep --actor-target "
                        "of queries on the Mode-1 path.")
    p.add_argument("--actor-target", type=float, default=0.8,
                   help="Fraction of queries to route through the actor when "
                        "calibrating threshold automatically (default 80%%, §5.4).")
    p.add_argument("--calib-examples", type=int, default=80,
                   help="Number of test examples used to calibrate the threshold.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-examples", type=int, default=320)
    p.add_argument("--max-q-length", type=int, default=512)
    p.add_argument("--max-a-length", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def _iter_batches(dataset, n_examples, batch_size):
    n = min(n_examples, len(dataset))
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        yield idx, [dataset[i] for i in idx]


def _score(example, pred) -> bool:
    return verify_code(pred, example)


@torch.no_grad()
def calibrate_threshold(
    ae, actor, ebm, dataset, device, max_q_length, target, n_examples, batch_size,
):
    """Pick `threshold` so ~`target` fraction of E(z_q, actor(z_q), 0) values
    fall below it."""
    energies: list[float] = []
    for _, batch in _iter_batches(dataset, n_examples, batch_size):
        questions = [b["question"] for b in batch]
        z_q = ae.encode_to_latents(questions, device, max_length=max_q_length)
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
    ae, actor, ebm, diffusion, dataset, dataset_name, device,
    max_q_length, max_a_length, inner_steps, threshold, n_examples, batch_size,
):
    a_correct = total = 0
    actor_pass = 0
    for batch_idx, batch in _iter_batches(dataset, n_examples, batch_size):
        questions = [b["question"] for b in batch]
        z_q = ae.encode_to_latents(questions, device, max_length=max_q_length)
        b = z_q.size(0)

        z_hat = actor(z_q)
        t0 = torch.zeros(b, device=device, dtype=torch.long)
        e_hat = ebm(z_q, z_hat, t0).squeeze(-1)
        take_actor = e_hat < threshold
        actor_pass += int(take_actor.sum().item())

        z_final = z_hat.clone()
        if (~take_actor).any():
            idx_t = (~take_actor).nonzero(as_tuple=True)[0]
            z_q_hard = z_q.index_select(0, idx_t)
            z_full = diffusion.sample(z_q_hard, inner_steps=inner_steps)
            z_final[idx_t] = z_full

        preds = ae.decode(z_final, max_length=max_a_length)
        for ex, pred in zip(batch, preds):
            total += 1
            if _score(ex, pred):
                a_correct += 1

    acc = a_correct / max(total, 1)
    actor_frac = actor_pass / max(total, 1)
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

    ae = FrozenBartAutoencoder(
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
    ebm_ckpt = torch.load(args.ebm_ckpt, map_location=args.device)
    ebm.load_state_dict(ebm_ckpt["ebm"])
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
    if "latent_mu" in ebm_ckpt and "latent_sigma" in ebm_ckpt:
        diffusion.set_latent_stats(
            ebm_ckpt["latent_mu"].to(args.device),
            ebm_ckpt["latent_sigma"].to(args.device),
        )

    if args.eval_dataset == "mbpp":
        test_ds = MBPPDataset(
            split="test", config=args.mbpp_config,
            max_samples=args.n_examples, seed=args.seed,
        )
    else:
        test_ds = HumanEvalDataset(max_samples=args.n_examples, seed=args.seed)
    print(f"eval dataset: {args.eval_dataset} ({len(test_ds)} examples)")

    if args.threshold is None:
        threshold = calibrate_threshold(
            ae, actor, ebm, test_ds, args.device,
            args.max_q_length, args.actor_target, args.calib_examples, args.batch_size,
        )
        print(f"calibrated threshold = {threshold:.4f}  (target actor frac {args.actor_target:.0%})")
    else:
        threshold = args.threshold
        print(f"using threshold = {threshold:.4f}")

    t0 = time.time()
    out = adaptive_sample(
        ae, actor, ebm, diffusion, test_ds, args.eval_dataset, args.device,
        args.max_q_length, args.max_a_length, args.inner_steps,
        threshold, args.n_examples, args.batch_size,
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
