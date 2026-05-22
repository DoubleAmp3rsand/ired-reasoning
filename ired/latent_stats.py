"""Quick diagnostic: dump statistics of AE latents on MBPP train answers.

If z.std() is far from 1.0, the diffusion noise schedule is miscalibrated:
the EBM trains on noise added to non-unit-variance latents but inference
initializes from N(0, 1), so the inference distribution never matches anything
the EBM saw during training. Fix by applying a constant scale factor to z
before feeding the diffusion process.

Usage:
    uv run python -m ired.latent_stats --ae-ckpt checkpoints/ae/ae_latest.pt
"""
from __future__ import annotations

import argparse

import torch

from ired.autoencoder import FrozenBartAutoencoder
from ired.data import MBPPDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae-ckpt", required=True)
    p.add_argument("--model", default="facebook/bart-base")
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
    p.add_argument("--pool-type", choices=["decoder", "resampler"], default="decoder")
    p.add_argument("--d-ae", type=int, default=-1)
    p.add_argument("--recon-layers", type=int, default=4)
    p.add_argument("--recon-heads", type=int, default=8)
    p.add_argument("--max-a-length", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=256,
                   help="MBPP train answers to encode for the stats.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    d_ae = args.d_ae if args.d_ae > 0 else None
    print(f"loading AE from {args.ae_ckpt}")
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
    ae.load_ae(ckpt["ae"])
    ae.eval()
    print(f"  d_model={ae.d_model}  d_ae={ae.d_ae}  k={ae.k}")

    print(f"loading {args.n_samples} MBPP train answers")
    ds = MBPPDataset(split="train", max_samples=args.n_samples, seed=0)
    answers = [ds[i]["answer"] for i in range(len(ds))]
    print(f"  got {len(answers)} examples")

    with torch.no_grad():
        z = ae.encode_to_latents(answers, args.device, max_length=args.max_a_length)
    print(f"\nlatent tensor shape: {tuple(z.shape)}  (B, K, d_ae)")

    z_flat = z.flatten().float().cpu()
    print("\n=== global stats (across B*K*d_ae elements) ===")
    print(f"  mean        : {z_flat.mean().item():+.4f}")
    print(f"  std         : {z_flat.std().item():.4f}")
    print(f"  min         : {z_flat.min().item():+.4f}")
    print(f"  max         : {z_flat.max().item():+.4f}")
    print(f"  abs median  : {z_flat.abs().median().item():.4f}")

    per_dim_std = z.float().std(dim=(0, 1)).cpu()  # (d_ae,)
    print("\n=== per-dim std (across B and K) ===")
    print(f"  mean of stds: {per_dim_std.mean().item():.4f}")
    print(f"  std of stds : {per_dim_std.std().item():.4f}")
    print(f"  min std     : {per_dim_std.min().item():.4f}")
    print(f"  max std     : {per_dim_std.max().item():.4f}")

    per_slot_std = z.float().std(dim=(0, 2)).cpu()  # (K,)
    print("\n=== per-slot std (across B and d_ae) ===")
    print(f"  mean        : {per_slot_std.mean().item():.4f}")
    print(f"  min         : {per_slot_std.min().item():.4f}")
    print(f"  max         : {per_slot_std.max().item():.4f}")

    suggested_scale = 1.0 / max(z_flat.std().item(), 1e-6)
    print(f"\n=== interpretation ===")
    if 0.5 <= z_flat.std().item() <= 2.0:
        print(f"  std={z_flat.std().item():.3f} is near 1.0 — noise schedule is well calibrated.")
        print("  the eval gibberish is NOT a latent-scale issue.")
    else:
        print(f"  std={z_flat.std().item():.3f} is far from 1.0 — noise schedule MISCALIBRATED.")
        print(f"  suggested scale factor to apply to z before diffusion: {suggested_scale:.4f}")
        print("  (multiply z_q and z_a by this; divide by it before decoding.)")


if __name__ == "__main__":
    main()
