"""One-off diagnostic: is the EBM's energy minimum at the right place?

For 3 MBPP test examples, prints:
  1. AE round-trip on gold answer  → does decoder work at all?
  2. EBM sample output              → what's the user seeing?
  3. ||z_pred - z_a|| / ||z_a||     → did sampler land near gold?
  4. E(gold) vs E(pred)             → where is the energy minimum?

Reads --ae-ckpt / --ebm-ckpt; mirrors sample.py construction so arch matches.
"""
from __future__ import annotations

import argparse

import torch

from ired.autoencoder import FrozenBartAutoencoder
from ired.data import MBPPDataset
from ired.diffusion import GaussianLatentDiffusion
from ired.energy_net import DiffusionWrapper, EnergyTransformer


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/bart-base")
    p.add_argument("--ae-ckpt", required=True)
    p.add_argument("--ebm-ckpt", required=True)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
    p.add_argument("--pool-type", choices=["decoder", "resampler"], default="decoder")
    p.add_argument("--recon-layers", type=int, default=4)
    p.add_argument("--recon-heads", type=int, default=8)
    p.add_argument("--ebm-layers", type=int, default=6)
    p.add_argument("--ebm-heads", type=int, default=8)
    p.add_argument("--ebm-ff-mult", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--x-start-clamp", type=float, default=3.0)
    p.add_argument("--max-q-length", type=int, default=512)
    p.add_argument("--max-a-length", type=int, default=512)
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    ae = FrozenBartAutoencoder(
        model_name=args.model,
        k=args.k,
        pool_layers=args.pool_layers,
        pool_heads=args.pool_heads,
        pool_type=args.pool_type,
        recon_layers=args.recon_layers,
        recon_heads=args.recon_heads,
    ).to(args.device)
    ae_ckpt = torch.load(args.ae_ckpt, map_location=args.device)
    ae.load_ae(ae_ckpt["ae"])
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
    diff = GaussianLatentDiffusion(
        model=wrapper,
        latent_shape=(args.k, ae.d_ae),
        timesteps=args.timesteps,
        x_start_clamp=args.x_start_clamp if args.x_start_clamp > 0 else None,
    ).to(args.device)
    diff.eval()
    if "latent_mu" in ebm_ckpt and "latent_sigma" in ebm_ckpt:
        diff.set_latent_stats(
            ebm_ckpt["latent_mu"].to(args.device),
            ebm_ckpt["latent_sigma"].to(args.device),
        )
        print(
            f"loaded latent stats: |mu|_mean={ebm_ckpt['latent_mu'].abs().mean().item():.4f}  "
            f"sigma_mean={ebm_ckpt['latent_sigma'].mean().item():.4f}"
        )
    else:
        print("WARNING: no latent stats in ebm ckpt — using identity")

    ds = MBPPDataset(split="test", config="full", max_samples=64, seed=0)
    n = min(args.n_examples, len(ds))
    print(f"\nrunning {n} examples with inner_steps={args.inner_steps}\n")

    t0 = torch.zeros(1, dtype=torch.long, device=args.device)
    for i in range(n):
        ex = ds[i]
        q = ex["question"]
        a = ex["answer"]

        with torch.no_grad():
            z_q = ae.encode_to_latents([q], args.device, max_length=args.max_q_length)
            z_a = ae.encode_to_latents([a], args.device, max_length=args.max_a_length)

            # 1. AE round-trip on the gold answer
            ae_recon = ae.decode(z_a, max_length=args.max_a_length)[0]

            # 2. Run the sampler
            z_pred = diff.sample(z_q, inner_steps=args.inner_steps)
            ebm_pred = ae.decode(z_pred, max_length=args.max_a_length)[0]

            # 3. Distance in native space
            num = (z_pred - z_a).norm().item()
            den = z_a.norm().item()
            rel_dist = num / max(den, 1e-9)

            # 4. Energy at gold vs at prediction (in normalized space)
            zq_n = diff._normalize(z_q)
            za_n = diff._normalize(z_a)
            zp_n = diff._normalize(z_pred)
            e_gold = ebm(zq_n, za_n, t0).item()
            e_pred = ebm(zq_n, zp_n, t0).item()

            # Also: energy at random Gaussian (sanity for "high" energy region)
            z_rand_n = torch.randn_like(za_n)
            e_rand = ebm(zq_n, z_rand_n, t0).item()

        print(f"=== example {i} ===")
        print(f"  Q     : {q[:100]!r}")
        print(f"  gold  : {a[:100]!r}")
        print(f"  ae_rec: {ae_recon[:100]!r}")
        print(f"  ebm   : {ebm_pred[:100]!r}")
        print(f"  ||z_pred - z_a|| / ||z_a||  = {rel_dist:.4f}")
        print(f"  E(gold)  = {e_gold:>14,.1f}")
        print(f"  E(pred)  = {e_pred:>14,.1f}")
        print(f"  E(rand)  = {e_rand:>14,.1f}    [random N(0,I) reference]")
        if e_pred < e_gold:
            print(f"  ** E(pred) < E(gold): energy minimum is OFF the gold latent **")
        else:
            print(f"  E(pred) > E(gold): gold is lower-energy (sampler didn't find it)")
        print()


if __name__ == "__main__":
    main()
