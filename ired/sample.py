"""Milestone 3 (gensis.md §9): test-time compute curve.

Sweeps `inner_steps` ∈ {0, 1, 2, 5, 10} and reports pass-rate on the chosen
eval corpus (MBPP or HumanEval). The "decision rule" in gensis: a steeper
pass-rate-vs-compute curve than AR-CoT at matched FLOPs validates the thesis;
a plateau below is an informative negative about smoothness of reasoning in
latent space.
"""
from __future__ import annotations

import argparse
import time

import torch

from ired.model.autoencoder import FrozenBartAutoencoder
from ired.data import (
    HumanEvalDataset,
    MBPPDataset,
    verify_code,
)
from ired.model.diffusion import GaussianLatentDiffusion
from ired.model.energy_net import DiffusionWrapper, EnergyTransformer


def build_parser():
    p = argparse.ArgumentParser(description="Sample and measure test-time compute curve")
    p.add_argument("--model", default="facebook/bart-base")
    p.add_argument("--ae-ckpt", required=True)
    p.add_argument("--ebm-ckpt", required=True)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
    p.add_argument("--pool-type", choices=["decoder", "resampler"], default="decoder",
                   help="Must match AE and EBM checkpoints.")
    p.add_argument("--d-ae", type=int, default=-1,
                   help="diffusion-space latent dim. Must match AE and EBM checkpoints.")
    p.add_argument("--recon-layers", type=int, default=2)
    p.add_argument("--recon-heads", type=int, default=8)
    p.add_argument("--ebm-layers", type=int, default=4)
    p.add_argument("--ebm-heads", type=int, default=8)
    p.add_argument("--ebm-ff-mult", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--inner-steps", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--x-start-clamp", type=float, default=5.0)
    p.add_argument("--envelope-sf", type=float, default=-1.0)
    p.add_argument("--eval-dataset", choices=["mbpp", "humaneval"], default="mbpp",
                   help="Test corpus to sweep over.")
    p.add_argument("--mbpp-config", choices=["full", "sanitized"], default="full")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-examples", type=int, default=320,
                   help="total test examples to score per inner_steps setting.")
    p.add_argument("--max-q-length", type=int, default=512)
    p.add_argument("--max-a-length", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


@torch.no_grad()
def sample_accuracy(ae, diffusion, dataset, dataset_name, device,
                    max_q_length, max_a_length, inner_steps, n_examples, batch_size):
    correct = 0
    total = 0
    n = min(n_examples, len(dataset))
    for start in range(0, n, batch_size):
        batch_idx = list(range(start, min(start + batch_size, n)))
        batch = [dataset[i] for i in batch_idx]
        questions = [b["question"] for b in batch]

        z_q = ae.encode_to_latents(questions, device, max_length=max_q_length)
        z = diffusion.sample(z_q, inner_steps=inner_steps)
        preds = ae.decode(z, max_length=max_a_length)
        for ex, pred in zip(batch, preds):
            total += 1
            if verify_code(pred, ex):
                correct += 1
    return correct / max(total, 1), total


def main(argv=None):
    args = build_parser().parse_args(argv)

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
    ae_ckpt = torch.load(args.ae_ckpt, map_location=args.device)
    if "ae" in ae_ckpt:
        ae.load_ae(ae_ckpt["ae"])
    else:
        raise RuntimeError(
            f"checkpoint {args.ae_ckpt} predates the LD4LG split (no 'ae' key)."
        )
    ae.eval()

    ebm = EnergyTransformer(
        d_model=ae.d_ae,
        k=args.k,
        n_layers=args.ebm_layers,
        n_heads=args.ebm_heads,
        dim_ff_mult=args.ebm_ff_mult,
    ).to(args.device)
    ebm_ckpt = torch.load(args.ebm_ckpt, map_location=args.device)
    ebm.load_state_dict(ebm_ckpt["ebm"])
    ebm.eval()

    wrapper = DiffusionWrapper(ebm).to(args.device)
    diffusion = GaussianLatentDiffusion(
        model=wrapper,
        latent_shape=(args.k, ae.d_ae),
        timesteps=args.timesteps,
        x_start_clamp=args.x_start_clamp if args.x_start_clamp > 0 else None,
        envelope_sf=args.envelope_sf if args.envelope_sf > 0 else None,
    ).to(args.device)
    diffusion.eval()

    # Restore the per-dim latent normalization (LD4LG shift+scale). Missing
    # stats = old checkpoint trained before normalization existed; fall back
    # to identity and warn — sampling will be miscalibrated.
    if "latent_mu" in ebm_ckpt and "latent_sigma" in ebm_ckpt:
        diffusion.set_latent_stats(
            ebm_ckpt["latent_mu"].to(args.device),
            ebm_ckpt["latent_sigma"].to(args.device),
        )
        print(
            f"loaded latent stats: |mu|_mean={ebm_ckpt['latent_mu'].abs().mean().item():.3f}  "
            f"sigma_mean={ebm_ckpt['latent_sigma'].mean().item():.3f}"
        )
    else:
        print(
            "WARNING: ebm checkpoint has no latent_mu/sigma — sampling will use "
            "identity normalization, which is likely miscalibrated."
        )

    if args.eval_dataset == "mbpp":
        test_ds = MBPPDataset(
            split="test", config=args.mbpp_config,
            max_samples=args.n_examples, seed=args.seed,
        )
    else:
        test_ds = HumanEvalDataset(max_samples=args.n_examples, seed=args.seed)
    print(f"eval dataset: {args.eval_dataset} ({len(test_ds)} examples)")

    print(f"{'inner':>6}  {'acc':>6}  {'n':>5}  {'ebm_passes':>11}  {'time_s':>7}")
    for n_inner in args.inner_steps:
        t0 = time.time()
        acc, total = sample_accuracy(
            ae, diffusion, test_ds, args.eval_dataset, args.device,
            args.max_q_length, args.max_a_length, n_inner,
            args.n_examples, args.batch_size,
        )
        dt = time.time() - t0
        ebm_passes = args.timesteps * (1 + n_inner)
        print(f"{n_inner:>6}  {acc:>.3f}  {total:>5}  {ebm_passes:>11}  {dt:>7.1f}")


if __name__ == "__main__":
    main()
