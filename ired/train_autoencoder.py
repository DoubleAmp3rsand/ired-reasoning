"""Milestone 1 (gensis.md §8): train the AttentionPool so the frozen T5 decoder
can reconstruct GSM8K answers from K latents.

The whole T5 model stays frozen. Only the pool's parameters update. We measure
reconstruction accuracy on the test split — if it isn't ≥95% in final-answer
mode, the autoencoder is the bottleneck and there is no point training the EBM
on top of it.
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


def build_parser():
    p = argparse.ArgumentParser(description="Train AttentionPool autoencoder")
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
    p.add_argument("--answer-mode", choices=["final", "full"], default="full",
                   help="'full' uses the GSM8K chain-of-thought; 'final' uses only "
                        "the numeric answer. 'full' is recommended — see gensis.md §8.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-dir", default="checkpoints/ae")
    p.add_argument("--max-a-length", type=int, default=256,
                   help="Covers ~99% of GSM8K full-mode answers (p99=240, max=354).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    return p


@torch.no_grad()
def eval_reconstruction(ae, loader, device, max_a_length, n_batches=10):
    """Two reconstruction metrics:
      - exact_acc: byte-exact reproduction of the gold answer text. Strict;
                   essentially 0 unless `--answer-mode final`.
      - final_acc: extract_final_answer(pred) == extract_final_answer(gold).
                   This is the gensis §8 bar for `full` mode (≥ 90%).
    Also returns sample reconstructions so we can see *where* it's failing.
    """
    ae.eval()
    exact = 0
    final = 0
    total = 0
    losses = []
    samples = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        z = ae.encode_to_latents(batch["answer"], device, max_length=max_a_length)
        loss = ae.decode_loss(z, batch["answer"], device, max_length=max_a_length)
        losses.append(loss.item())
        preds = ae.decode(z, max_length=max_a_length)
        for p, a in zip(preds, batch["answer"]):
            total += 1
            if p.strip() == a.strip():
                exact += 1
            if extract_final_answer(p) == extract_final_answer(a):
                final += 1
            if len(samples) < 3:
                samples.append((a, p))
    ae.train()
    n = max(total, 1)
    return {
        "exact_acc": exact / n,
        "final_acc": final / n,
        "loss": sum(losses) / max(len(losses), 1),
        "samples": samples,
        "n": total,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"loading autoencoder: {args.model} (k={args.k}, pool_layers={args.pool_layers})")
    ae = FrozenT5Autoencoder(
        model_name=args.model,
        k=args.k,
        pool_layers=args.pool_layers,
        pool_heads=args.pool_heads,
    ).to(args.device)
    ae.train()

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

    pool_params = ae.trainable_parameters()
    n_params = sum(p.numel() for p in pool_params)
    print(f"trainable pool params: {n_params:,}")
    opt = AdamW(pool_params, lr=args.lr)

    step = 0
    train_iter = iter(train_loader)
    t0 = time.time()
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Reconstruction objective: Decoder(Pool(Encoder(A))) ≈ A
        z = ae.encode_to_latents(batch["answer"], args.device, max_length=args.max_a_length)
        loss = ae.decode_loss(z, batch["answer"], args.device, max_length=args.max_a_length)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pool_params, 1.0)
        opt.step()

        step += 1
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d} | loss {loss.item():.4f} | {elapsed:.1f}s")

        if step % args.eval_every == 0 or step == args.steps:
            ev = eval_reconstruction(
                ae, test_loader, args.device, args.max_a_length, n_batches=10,
            )
            print(
                f"  [eval n={ev['n']}] exact_acc={ev['exact_acc']:.3f}  "
                f"final_acc={ev['final_acc']:.3f}  loss={ev['loss']:.4f}"
            )
            def _snip(s, n=120):
                s = s.replace("\n", " ⏎ ")
                return s if len(s) <= n else s[:n] + "…"
            for gold, pred in ev["samples"]:
                print(f"    gold final: {extract_final_answer(gold)!r}  pred final: {extract_final_answer(pred)!r}")
                print(f"      gold: {_snip(gold)}")
                print(f"      pred: {_snip(pred)}")
            ckpt = {
                "pool": ae.state_dict_pool(),
                "config": vars(args),
                "step": step,
                "eval_acc": ev["final_acc"],
                "exact_acc": ev["exact_acc"],
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"pool_step{step}.pt"))
            torch.save(ckpt, os.path.join(args.save_dir, "pool_latest.pt"))


if __name__ == "__main__":
    main()
