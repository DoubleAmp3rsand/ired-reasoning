"""Milestone 1 (gensis.md §8): train the AttentionPool so the frozen T5 decoder
can reconstruct GSM8K answers from K latents.

The whole T5 model stays frozen. Only the pool's parameters update. We measure
reconstruction accuracy on the test split — if it isn't ≥95% in final-answer
mode, the autoencoder is the bottleneck and there is no point training the EBM
on top of it.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ired.autoencoder import FrozenT5Autoencoder
from ired.data import (
    GSM8KDataset,
    MixedDataset,
    WikiTextDataset,
    collate,
    extract_final_answer,
)


def build_parser():
    p = argparse.ArgumentParser(description="Train AttentionPool autoencoder")
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--k", type=int, default=128,
                   help="number of latent slots the pool produces. Compression "
                        "ratio is L_in/K; at K=32 (old default) GSM8K-full was "
                        "~7.5:1 and the frozen T5-base decoder couldn't reliably "
                        "recover. K=128 gives ~2:1 at p99 length 240. Raises "
                        "diffusion-tensor size 4×, so you may need to drop "
                        "--batch-size or raise --grad-accum-steps to fit memory.")
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
    p.add_argument("--pool-type", choices=["decoder", "resampler"], default="decoder",
                   help="'decoder' = nn.TransformerDecoderLayer (separated self+cross attn). "
                        "'resampler' = Flamingo-style combined-MHA Perceiver Resampler "
                        "with shared LN + gated residuals (LD4LG-faithful, identity at init).")
    p.add_argument("--d-ae", type=int, default=-1,
                   help="diffusion-space latent dimension (LD4LG d_ae). "
                        "-1 (default) = d_model (no down-projection); "
                        "set <d_model to shrink the EBM's diffusion space.")
    p.add_argument("--recon-layers", type=int, default=2,
                   help="depth of the ReconstructionNet (f_ψ)")
    p.add_argument("--recon-heads", type=int, default=8)
    p.add_argument("--answer-mode", choices=["final", "full"], default="full",
                   help="'full' uses the GSM8K chain-of-thought; 'final' uses only "
                        "the numeric answer. 'full' is recommended — see gensis.md §8. "
                        "Only affects GSM8K train (when --train-dataset != openwebtext) "
                        "and the GSM8K test loader used for the final_acc eval.")
    p.add_argument("--train-dataset", choices=["wikitext", "gsm8k", "mix"],
                   default="wikitext",
                   help="Corpus the AE reconstructs on. 'wikitext' (default) avoids "
                        "shaping the latent space with the same examples the diffusion "
                        "model later sees on GSM8K. 'mix' samples 80/20 wikitext/GSM8K. "
                        "Test eval is always GSM8K (final_acc is the gate).")
    p.add_argument("--wikitext-samples", type=int, default=50_000,
                   help="number of WikiText lines to materialize (filtered by length).")
    p.add_argument("--wikitext-min-chars", type=int, default=200,
                   help="drop lines shorter than this — wikitext has many empty lines "
                        "and ` = Title = ` headers we don't want.")
    p.add_argument("--wikitext-max-chars", type=int, default=4_000,
                   help="hard char truncation before tokenization. The tokenizer also "
                        "truncates to --max-a-length tokens, which is the real bound.")
    p.add_argument("--mix-wikitext-weight", type=float, default=0.8,
                   help="when --train-dataset=mix, fraction of samples drawn from wikitext.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="base LR after warmup. 3e-4 (old default) was too hot for "
                        "the LD4LG-faithful pool + recon stack; collapsed to unigram.")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="bounds AE param magnitudes, reduces local-minima stickiness.")
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="linear LR warmup from 0 to --lr over this many steps.")
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="cosine-decay floor as a fraction of --lr (default 0.1 → "
                        "LR ends at 0.1·--lr). Set to 0 for decay all the way to zero.")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="accumulate gradients over this many micro-batches before "
                        "each optimizer step. Effective batch = batch_size × this. "
                        "Use to reduce per-step gradient variance without OOM.")
    p.add_argument("--steps", type=int, default=2000,
                   help="number of optimizer steps (not micro-batches).")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-dir", default="checkpoints/ae")
    p.add_argument("--resume", default=None,
                   help="path to a checkpoint (e.g. checkpoints/ae/ae_latest.pt) to "
                        "resume from. Restores AE weights, optimizer/scheduler state, "
                        "and step counter; training continues until --steps total.")
    p.add_argument("--max-a-length", type=int, default=256,
                   help="Covers ~99% of GSM8K full-mode answers (p99=240, max=354).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    return p


def build_param_groups(ae, weight_decay: float):
    """Split trainable params into decay / no-decay groups.

    No-decay: 1-D tensors (LayerNorm gains, biases, the Flamingo gated `alpha_*`
    scalars) plus the learnable `queries`. WD on these is harmful — it pulls
    `alpha` away from the values the optimizer is trying to grow, shrinks LN
    gains, and decays the pool's query embeddings toward zero.
    """
    decay, no_decay = [], []
    seen = set()
    for name, p in ae.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_query = name.endswith("pool.queries")
        is_alpha = "alpha_" in name
        is_bias = name.endswith(".bias")
        is_1d = p.ndim <= 1
        if is_query or is_alpha or is_bias or is_1d:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


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

    d_ae = args.d_ae if args.d_ae > 0 else None
    print(
        f"loading autoencoder: {args.model} "
        f"(k={args.k}, pool_layers={args.pool_layers}, recon_layers={args.recon_layers}, "
        f"d_ae={'d_model' if d_ae is None else d_ae})"
    )
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
    ae.train()
    print(f"d_model={ae.d_model}  d_ae={ae.d_ae}")

    # Test loader is always GSM8K: final_acc on GSM8K answers is the gensis §8 gate.
    test_ds = GSM8KDataset("test", answer_mode=args.answer_mode)

    if args.train_dataset == "gsm8k":
        train_ds = GSM8KDataset("train", answer_mode=args.answer_mode)
        print(f"train dataset: GSM8K-{args.answer_mode} ({len(train_ds)} examples)")
    elif args.train_dataset == "wikitext":
        print(
            f"train dataset: WikiText-103 (materializing up to "
            f"{args.wikitext_samples} lines with "
            f"{args.wikitext_min_chars}≤len≤{args.wikitext_max_chars})"
        )
        train_ds = WikiTextDataset(
            max_samples=args.wikitext_samples,
            min_chars=args.wikitext_min_chars,
            max_chars=args.wikitext_max_chars,
            seed=args.seed,
        )
        print(f"  materialized {len(train_ds)} lines")
    elif args.train_dataset == "mix":
        print(
            f"train dataset: MIX ({args.mix_wikitext_weight:.2f} wikitext / "
            f"{1 - args.mix_wikitext_weight:.2f} GSM8K-{args.answer_mode})"
        )
        wiki = WikiTextDataset(
            max_samples=args.wikitext_samples,
            min_chars=args.wikitext_min_chars,
            max_chars=args.wikitext_max_chars,
            seed=args.seed,
        )
        gsm = GSM8KDataset("train", answer_mode=args.answer_mode)
        train_ds = MixedDataset(
            [wiki, gsm],
            [args.mix_wikitext_weight, 1.0 - args.mix_wikitext_weight],
            length=max(len(wiki), len(gsm)),
        )
        print(f"  wikitext={len(wiki)}  GSM8K={len(gsm)}  effective epoch={len(train_ds)}")
    else:
        raise ValueError(args.train_dataset)

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
    print(f"trainable AE params (pool + recon): {n_params:,}")
    param_groups = build_param_groups(ae, args.weight_decay)
    n_decay = sum(p.numel() for p in param_groups[0]["params"])
    n_nodecay = sum(p.numel() for p in param_groups[1]["params"])
    opt = AdamW(param_groups, lr=args.lr)

    # Warmup → cosine decay. Factor: 0→1 linear over warmup_steps, then cosine
    # from 1 down to min_lr_ratio across the remaining steps.
    def _lr_lambda(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        decay_steps = max(args.steps - args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
    sched = LambdaLR(opt, _lr_lambda)
    print(
        f"optimizer: AdamW(lr={args.lr}, wd={args.weight_decay} on "
        f"{n_decay:,} params; 0 on {n_nodecay:,})  "
        f"warmup={args.warmup_steps}  min_lr_ratio={args.min_lr_ratio}  "
        f"grad_accum={args.grad_accum_steps}"
    )

    step = 0
    if args.resume is not None:
        print(f"resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device)
        ae.load_ae(ckpt["ae"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        else:
            print("  (no optimizer state in checkpoint — starting optimizer fresh)")
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        step = ckpt.get("step", 0)
        print(f"  resumed at step {step}")
        if step >= args.steps:
            print(
                f"  warning: --steps={args.steps} <= resumed step={step}; "
                "nothing to do. Increase --steps to continue training."
            )
    train_iter = iter(train_loader)
    t0 = time.time()
    accum = max(args.grad_accum_steps, 1)
    while step < args.steps:
        # Reconstruction objective: Decoder(Pool(Encoder(A))) ≈ A
        opt.zero_grad()
        loss_sum = 0.0
        for _ in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            z = ae.encode_to_latents(batch["answer"], args.device, max_length=args.max_a_length)
            loss = ae.decode_loss(z, batch["answer"], args.device, max_length=args.max_a_length)
            (loss / accum).backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(pool_params, 1.0)
        opt.step()
        sched.step()

        step += 1
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            cur_lr = opt.param_groups[0]["lr"]
            avg_loss = loss_sum / accum
            print(f"step {step:6d} | loss {avg_loss:.4f} | lr {cur_lr:.2e} | {elapsed:.1f}s")

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
                "ae": ae.state_dict_ae(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "config": vars(args),
                "step": step,
                "eval_acc": ev["final_acc"],
                "exact_acc": ev["exact_acc"],
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"ae_step{step}.pt"))
            torch.save(ckpt, os.path.join(args.save_dir, "ae_latest.pt"))


if __name__ == "__main__":
    main()
