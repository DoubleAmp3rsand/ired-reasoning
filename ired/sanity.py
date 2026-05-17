"""Quick shape / forward-pass sanity check. Runs entirely on CPU with random
inputs — no GSM8K download needed. Good for verifying the wiring before
launching a real training run.
"""
from __future__ import annotations

import argparse

import torch

from ired.autoencoder import FrozenT5Autoencoder
from ired.diffusion import GaussianLatentDiffusion
from ired.energy_net import DiffusionWrapper, EnergyTransformer


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--d-ae", type=int, default=-1,
                   help="-1 = d_model. Smaller = test the LD4LG dim-reduction path.")
    p.add_argument("--ebm-layers", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--inner-steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    device = args.device
    d_ae = args.d_ae if args.d_ae > 0 else None

    print(f"[1/4] loading autoencoder ({args.model}, d_ae={'d_model' if d_ae is None else d_ae}) ...")
    ae = FrozenT5Autoencoder(model_name=args.model, k=args.k, d_ae=d_ae).to(device)
    ae.eval()
    print(f"     d_model={ae.d_model}  d_ae={ae.d_ae}")

    print("[2/4] encoding sample texts ...")
    qs = ["What is 2 + 3?", "Jane has 7 apples and eats 2. How many remain?"][: args.batch_size]
    ans = ["5", "5"][: args.batch_size]
    z_q = ae.encode_to_latents(qs, device, max_length=64)
    z_a = ae.encode_to_latents(ans, device, max_length=16)
    print(f"     z_q.shape={tuple(z_q.shape)}  z_a.shape={tuple(z_a.shape)}")
    assert z_q.shape == (len(qs), args.k, ae.d_ae)
    assert z_a.shape == (len(ans), args.k, ae.d_ae)

    print("[3/4] building diffusion + EBM ...")
    ebm = EnergyTransformer(
        d_model=ae.d_ae, k=args.k, n_layers=args.ebm_layers,
    ).to(device)
    wrapper = DiffusionWrapper(ebm).to(device)
    diffusion = GaussianLatentDiffusion(
        model=wrapper,
        latent_shape=(args.k, ae.d_ae),
        timesteps=args.timesteps,
    ).to(device)
    diffusion.train()

    print("     forward (p_losses) ...")
    loss, stats = diffusion(z_q.detach(), z_a.detach())
    loss.backward()
    print(f"     loss={loss.item():.4f}  stats={stats}")
    assert torch.isfinite(loss), "loss is not finite"

    print("[4/4] sampling end-to-end ...")
    diffusion.eval()
    with torch.no_grad():
        z_pred = diffusion.sample(z_q, inner_steps=args.inner_steps)
    print(f"     z_pred.shape={tuple(z_pred.shape)}")
    preds = ae.decode(z_pred, max_length=16)
    print(f"     decoded (untrained EBM, expected to be garbage): {preds}")

    print("ok — all shapes and forward/backward passes succeed.")


if __name__ == "__main__":
    main()
