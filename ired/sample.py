"""Milestone 3 (gensis.md §8): test-time compute curve.

Sweeps `inner_steps` ∈ {0, 1, 2, 5, 10} and reports exact-match accuracy. The
"decision rule" in gensis: a steeper accuracy-vs-compute curve than AR-CoT at
matched FLOPs validates the thesis; a plateau below is an informative negative
about smoothness of reasoning in latent space.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from ired.autoencoder import FrozenT5Autoencoder
from ired.data import GSM8KDataset, collate, extract_final_answer
from ired.diffusion import GaussianLatentDiffusion
from ired.energy_net import DiffusionWrapper, EnergyTransformer


def build_parser():
    p = argparse.ArgumentParser(description="Sample and measure test-time compute curve")
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--ae-ckpt", required=True)
    p.add_argument("--ebm-ckpt", required=True)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
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
    p.add_argument("--answer-mode", choices=["final", "full"], default="full",
                   help="Must match the AE+EBM training mode.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--max-q-length", type=int, default=256)
    p.add_argument("--max-a-length", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    return p


@torch.no_grad()
def sample_accuracy(ae, diffusion, loader, device, max_q_length, max_a_length, inner_steps, n_batches):
    correct = 0
    total = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        z_q = ae.encode_to_latents(batch["question"], device, max_length=max_q_length)
        z = diffusion.sample(z_q, inner_steps=inner_steps)
        preds = ae.decode(z, max_length=max_a_length)
        for p, a in zip(preds, batch["answer"]):
            total += 1
            if extract_final_answer(p) == extract_final_answer(a):
                correct += 1
    return correct / max(total, 1), total


def main(argv=None):
    args = build_parser().parse_args(argv)

    d_ae = args.d_ae if args.d_ae > 0 else None
    ae = FrozenT5Autoencoder(
        model_name=args.model,
        k=args.k,
        pool_layers=args.pool_layers,
        pool_heads=args.pool_heads,
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
    ebm.load_state_dict(torch.load(args.ebm_ckpt, map_location=args.device)["ebm"])
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

    test_ds = GSM8KDataset("test", answer_mode=args.answer_mode)
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers,
    )

    print(f"{'inner':>6}  {'acc':>6}  {'n':>5}  {'ebm_passes':>11}  {'time_s':>7}")
    for n_inner in args.inner_steps:
        t0 = time.time()
        acc, total = sample_accuracy(
            ae, diffusion, loader, args.device,
            args.max_q_length, args.max_a_length, n_inner, args.n_batches,
        )
        dt = time.time() - t0
        # Per sample: T outer DDPM passes (1 fwd+bwd each) + T * n_inner inner passes
        ebm_passes = args.timesteps * (1 + n_inner)
        print(f"{n_inner:>6}  {acc:>.3f}  {total:>5}  {ebm_passes:>11}  {dt:>7.1f}")


if __name__ == "__main__":
    main()
