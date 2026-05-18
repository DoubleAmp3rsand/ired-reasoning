"""Distillation training for the Mode-1 actor (gensis.md §5.3).

Pipeline per step:

  with no_grad:
      z_q     = ae.encode_to_latents(question)            # conditioning
      z_final = diffusion.sample(z_q, inner_steps=N)      # Mode-2 teacher

  z_hat = actor(z_q)
  loss  = w * mse(z_hat, z_final)                         # optional EBM weighting
  loss.backward()
  opt.step()

Everything except the actor is frozen: the T5 encoder/decoder, the AttentionPool
+ ReconstructionNet (loaded from the AE checkpoint), and the EBM (loaded from
the EBM checkpoint). `p_sample_loop` is run under `no_grad`; `z_final` is a
plain regression target. That's what makes the actor cheap to train despite the
teacher's second-order autograd.

Two optional knobs from §5.3:

  --replay-buffer-size N   cache (question → z_final) so the slow teacher
                           rollout amortizes across epochs (one rollout per Q
                           for the whole training run).
  --ebm-weight             weight each pair by exp(-E(z_q, z_final, 0)) so the
                           actor preferentially imitates the teacher's confident
                           converged latents. Cheap — the EBM is already on GPU.

Eval (§5.4 prerequisite): every `eval_every` steps we measure
  - actor_acc       : one-shot accuracy (decode(actor(z_q)))
  - ebm_acc         : full Mode-2 accuracy on the same batches (reference)
  - mse_z, corr_z   : actor output vs teacher's z_final on the eval set
"""
from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ired.actor import Actor
from ired.autoencoder import FrozenT5Autoencoder
from ired.data import GSM8KDataset, collate, extract_final_answer
from ired.diffusion import GaussianLatentDiffusion
from ired.energy_net import DiffusionWrapper, EnergyTransformer


def build_parser():
    p = argparse.ArgumentParser(description="Distill Mode-2 IRED into a Mode-1 actor")
    # autoencoder (must match the AE + EBM checkpoints exactly)
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--ae-ckpt", required=True)
    p.add_argument("--ebm-ckpt", required=True)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
    p.add_argument("--pool-type", choices=["decoder", "resampler"], default="decoder")
    p.add_argument("--d-ae", type=int, default=-1)
    p.add_argument("--recon-layers", type=int, default=2)
    p.add_argument("--recon-heads", type=int, default=8)
    # ebm (architecture must match the ebm ckpt)
    p.add_argument("--ebm-layers", type=int, default=4)
    p.add_argument("--ebm-heads", type=int, default=8)
    p.add_argument("--ebm-ff-mult", type=int, default=4)
    # diffusion / teacher rollout
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--inner-steps", type=int, default=5,
                   help="N_inner for the teacher rollout (Mode-2).")
    p.add_argument("--x-start-clamp", type=float, default=5.0)
    p.add_argument("--envelope-sf", type=float, default=-1.0)
    # actor
    p.add_argument("--actor-layers", type=int, default=4)
    p.add_argument("--actor-heads", type=int, default=8)
    p.add_argument("--actor-ff-mult", type=int, default=4)
    p.add_argument("--ebm-weight", action="store_true",
                   help="weight each pair by exp(-E(z_q, z_final, 0)).")
    p.add_argument("--ebm-weight-tau", type=float, default=1.0,
                   help="temperature on the EBM weighting (w ∝ exp(-E/tau)).")
    p.add_argument("--replay-buffer-size", type=int, default=0,
                   help=">0 enables an in-memory cache keyed by question text. "
                        "On hit, skip the teacher rollout.")
    # data & opt
    p.add_argument("--answer-mode", choices=["final", "full"], default="full")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-dir", default="checkpoints/actor")
    p.add_argument("--max-q-length", type=int, default=256)
    p.add_argument("--max-a-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    return p


class ReplayBuffer:
    """LRU cache keyed by question text → z_final tensor (on CPU)."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def __len__(self):
        return len(self.cache)

    def get(self, key: str) -> torch.Tensor | None:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: torch.Tensor) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value.detach().cpu()
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


@torch.no_grad()
def teacher_rollout(
    ae: FrozenT5Autoencoder,
    diffusion: GaussianLatentDiffusion,
    questions: list[str],
    device: str,
    max_q_length: int,
    inner_steps: int,
    buffer: ReplayBuffer | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (z_q, z_final) for the batch. Uses `buffer` when populated.

    Misses on the buffer trigger a single fresh teacher rollout for the missing
    sub-batch; hits are reused from cache. Both halves are stitched back into
    the original batch order so the caller never has to think about it.
    """
    z_q = ae.encode_to_latents(questions, device, max_length=max_q_length)

    if buffer is None:
        z_final = diffusion.sample(z_q, inner_steps=inner_steps)
        return z_q, z_final

    cached: list[torch.Tensor | None] = [buffer.get(q) for q in questions]
    miss_idx = [i for i, c in enumerate(cached) if c is None]

    z_final = torch.empty_like(z_q)
    for i, c in enumerate(cached):
        if c is not None:
            z_final[i] = c.to(device)

    if miss_idx:
        idx_t = torch.tensor(miss_idx, device=device, dtype=torch.long)
        z_q_miss = z_q.index_select(0, idx_t)
        z_final_miss = diffusion.sample(z_q_miss, inner_steps=inner_steps)
        for j, i in enumerate(miss_idx):
            z_final[i] = z_final_miss[j]
            buffer.put(questions[i], z_final_miss[j])

    return z_q, z_final


@torch.no_grad()
def eval_actor(
    ae: FrozenT5Autoencoder,
    diffusion: GaussianLatentDiffusion,
    ebm: EnergyTransformer,
    actor: Actor,
    loader,
    device: str,
    max_q_length: int,
    max_a_length: int,
    inner_steps: int,
    n_batches: int = 5,
) -> dict:
    """Joint eval comparing actor (Mode-1) and IRED (Mode-2) on the same batches.

      actor_acc   : decode(actor(z_q))
      ebm_acc     : decode(p_sample_loop(z_q, inner_steps))   — reference upper bound
      mse_z       : MSE between actor output and the teacher's z_final
      corr_z      : cosine sim between actor output and z_final
      e_actor     : mean E(z_q, actor(z_q), 0) — the §5.4 confidence scalar
      e_teacher   : mean E(z_q, z_final, 0) — what the gate compares against
    """
    actor.eval()
    diffusion.eval()
    a_correct = e_correct = total = 0
    mse_sum = 0.0
    cos_sum = 0.0
    e_actor_sum = e_teacher_sum = 0.0
    samples = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        z_q = ae.encode_to_latents(batch["question"], device, max_length=max_q_length)
        b = z_q.size(0)

        z_final = diffusion.sample(z_q, inner_steps=inner_steps)
        z_hat = actor(z_q)

        preds_actor = ae.decode(z_hat, max_length=max_a_length)
        preds_ebm = ae.decode(z_final, max_length=max_a_length)

        mse_sum += (z_hat - z_final).pow(2).mean().item() * b
        cos_sum += torch.nn.functional.cosine_similarity(
            z_hat.reshape(b, -1), z_final.reshape(b, -1), dim=-1
        ).sum().item()

        t0 = torch.zeros(b, device=device, dtype=torch.long)
        e_actor_sum += ebm(z_q, z_hat, t0).sum().item()
        e_teacher_sum += ebm(z_q, z_final, t0).sum().item()

        for p_a, p_e, gold, q in zip(
            preds_actor, preds_ebm, batch["answer"], batch["question"]
        ):
            total += 1
            g = extract_final_answer(gold)
            if extract_final_answer(p_a) == g: a_correct += 1
            if extract_final_answer(p_e) == g: e_correct += 1
            if len(samples) < 3:
                samples.append((q[:80], gold, p_a, p_e))
    actor.train()
    diffusion.train()
    n = max(total, 1)
    return {
        "actor_acc": a_correct / n,
        "ebm_acc":   e_correct / n,
        "mse_z":     mse_sum / n,
        "corr_z":    cos_sum / n,
        "e_actor":   e_actor_sum / n,
        "e_teacher": e_teacher_sum / n,
        "samples":   samples,
        "n":         total,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- autoencoder (frozen) ----
    print(f"loading autoencoder: {args.model}")
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
    for p in ae.parameters():
        p.requires_grad_(False)
    d_diff = ae.d_ae

    # ---- ebm (frozen) ----
    print(f"loading ebm: {args.ebm_ckpt}")
    ebm = EnergyTransformer(
        d_model=d_diff,
        k=args.k,
        n_layers=args.ebm_layers,
        n_heads=args.ebm_heads,
        dim_ff_mult=args.ebm_ff_mult,
    ).to(args.device)
    ebm.load_state_dict(torch.load(args.ebm_ckpt, map_location=args.device)["ebm"])
    ebm.eval()
    for p in ebm.parameters():
        p.requires_grad_(False)

    wrapper = DiffusionWrapper(ebm).to(args.device)
    diffusion = GaussianLatentDiffusion(
        model=wrapper,
        latent_shape=(args.k, d_diff),
        timesteps=args.timesteps,
        x_start_clamp=args.x_start_clamp if args.x_start_clamp > 0 else None,
        envelope_sf=args.envelope_sf if args.envelope_sf > 0 else None,
    ).to(args.device)
    diffusion.eval()

    # ---- actor (trainable) ----
    actor = Actor(
        d_ae=d_diff,
        k=args.k,
        n_layers=args.actor_layers,
        n_heads=args.actor_heads,
        dim_ff_mult=args.actor_ff_mult,
    ).to(args.device)
    actor.train()
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"actor params: {n_params:,}  (d_ae={d_diff}, K={args.k}, layers={args.actor_layers})")

    # ---- data ----
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

    opt = AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buffer = ReplayBuffer(args.replay_buffer_size) if args.replay_buffer_size > 0 else None
    if buffer is not None:
        print(f"replay buffer enabled (max_size={args.replay_buffer_size})")

    step = 0
    train_iter = iter(train_loader)
    t0 = time.time()
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # --- teacher rollout (no grad) ---
        z_q, z_final = teacher_rollout(
            ae, diffusion, batch["question"], args.device,
            args.max_q_length, args.inner_steps, buffer,
        )

        # --- student one-shot regression ---
        z_hat = actor(z_q)
        per_sample = (z_hat - z_final).pow(2).flatten(1).mean(dim=-1)   # (B,)

        if args.ebm_weight:
            with torch.no_grad():
                t0_b = torch.zeros(z_q.size(0), device=args.device, dtype=torch.long)
                e_teacher = ebm(z_q, z_final, t0_b).squeeze(-1)         # (B,)
                # subtract per-batch min for numerical stability — pure rescaling
                w = torch.softmax(-e_teacher / args.ebm_weight_tau, dim=0)
                w = w * w.numel()                                       # mean weight = 1
            loss = (w * per_sample).mean()
        else:
            loss = per_sample.mean()

        if not torch.isfinite(loss):
            raise RuntimeError(f"step {step}: non-finite loss {loss.item()}")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()

        step += 1
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            extra = ""
            if buffer is not None:
                extra = f" | buf {len(buffer)}"
            print(
                f"step {step:6d} | loss {loss.item():.4f} | "
                f"mse {per_sample.mean().item():.4f}{extra} | {elapsed:.1f}s"
            )

        if step % args.eval_every == 0 or step == args.steps:
            ev = eval_actor(
                ae, diffusion, ebm, actor, test_loader, args.device,
                args.max_q_length, args.max_a_length, args.inner_steps, n_batches=5,
            )
            print(
                f"  [eval n={ev['n']}] actor_acc={ev['actor_acc']:.3f}  "
                f"ebm_acc(inner={args.inner_steps})={ev['ebm_acc']:.3f}  "
                f"mse_z={ev['mse_z']:.3f}  corr_z={ev['corr_z']:+.3f}"
            )
            print(
                f"  [energy] e_actor={ev['e_actor']:.2f}  e_teacher={ev['e_teacher']:.2f}  "
                f"gap={ev['e_actor'] - ev['e_teacher']:+.2f}  "
                f"(gap≈0 means actor is at the teacher's minimum)"
            )

            def _snip(s, n=120):
                s = s.replace("\n", " ⏎ ")
                return s if len(s) <= n else s[:n] + "…"
            for q, gold, p_a, p_e in ev["samples"]:
                print(f"    Q: {q}")
                print(f"      gold        : {extract_final_answer(gold)!r}")
                print(f"      actor       : {extract_final_answer(p_a)!r:>10}  | {_snip(p_a)}")
                print(f"      ebm(inner={args.inner_steps:>2}): {extract_final_answer(p_e)!r:>10}  | {_snip(p_e)}")

            ck = {
                "actor": actor.state_dict(),
                "config": vars(args),
                "step": step,
                "actor_acc": ev["actor_acc"],
                "ebm_acc": ev["ebm_acc"],
                "mse_z": ev["mse_z"],
            }
            torch.save(ck, os.path.join(args.save_dir, f"actor_step{step}.pt"))
            torch.save(ck, os.path.join(args.save_dir, "actor_latest.pt"))


if __name__ == "__main__":
    main()
