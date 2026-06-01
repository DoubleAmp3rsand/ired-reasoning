"""Milestone 1 (gensis.md §7): train the AttentionPool + ReconstructionNet so
the frozen BART decoder can round-trip text through K latent slots.

The whole BART model stays frozen. Only the pool's + recon's parameters update.

Task substrate: **ZebraLogic** — natural-language logic-grid puzzles. The AE
trains on **OpenWebText + synthetic logic-grid solutions** (random grids for
format exposure, mixed at `--zebra-mix-ratio`) and is evaluated on held-out
**ZebraLogic** puzzles by puzzle-level exact match. ZebraLogic is *never* used
for training, so the latent space is not shaped by the eval distribution
(gensis §2.2 anchoring).

Metrics per eval cycle:
  - owt_recon_cer      CER (torchmetrics) of decoded OpenWebText vs gold.
  - zebra_recon_cer    same CER, on ZebraLogic gold grids. Smooth early signal.
  - zebra_exact        fraction of puzzles whose round-trip matches gold
                       assignment exactly — the correctness metric.
CER gives a smooth signal; exact match stays the downstream-relevant one.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

try:
    from huggingface_hub import HfApi, create_repo
    _has_hf_hub = True
except ImportError:
    HfApi = None  # type: ignore
    create_repo = None  # type: ignore
    _has_hf_hub = False

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate

from ired.model.autoencoder import FrozenBartAutoencoder
from ired.data import (
    MixedDataset,
    OpenWebTextDataset,
    SyntheticZebraGridDataset,
    ZebraLogicDataset,
    ZebraLogicVerifier,
    collate,
)
from ired.tui import make_reporter


def build_parser():
    p = argparse.ArgumentParser(description="Train AttentionPool autoencoder on OpenWebText")
    p.add_argument("--model", default="facebook/bart-base")
    p.add_argument("--k", type=int, default=128,
                   help="number of latent slots the pool produces.")
    p.add_argument("--pool-layers", type=int, default=2)
    p.add_argument("--pool-heads", type=int, default=8)
    p.add_argument("--pool-type", choices=["decoder", "resampler", "conv"], default="decoder",
                   help="'decoder' = nn.TransformerDecoderLayer (separated self+cross attn). "
                        "'resampler' = Flamingo-style combined-MHA Perceiver Resampler. "
                        "'conv' = depthwise-separable 1-D conv scan + adaptive pool.")
    p.add_argument("--unfreeze-decoder", action="store_true",
                   help="fine-tune the BART decoder (and tied LM head) jointly with the "
                        "pool/recon. The encoder stays frozen.")
    p.add_argument("--decoder-lr", type=float, default=1e-5,
                   help="separate (usually smaller) LR for the unfrozen decoder params.")
    p.add_argument("--d-ae", type=int, default=-1,
                   help="diffusion-space latent dimension (LD4LG d_ae). "
                        "-1 (default) = d_model (no down-projection).")
    p.add_argument("--recon-layers", type=int, default=2,
                   help="depth of the ReconstructionNet (f_ψ)")
    p.add_argument("--recon-heads", type=int, default=8)
    p.add_argument("--train-dataset", choices=["openwebtext"], default="openwebtext",
                   help="AE training corpus (OpenWebText).")
    p.add_argument("--owt-samples", type=int, default=100_000,
                   help="number of OpenWebText documents to materialize.")
    p.add_argument("--owt-min-chars", type=int, default=200,
                   help="drop documents shorter than this.")
    p.add_argument("--owt-max-chars", type=int, default=4_000,
                   help="hard char truncation before tokenization.")
    p.add_argument("--owt-eval-samples", type=int, default=256,
                   help="size of the held-out OpenWebText eval slice.")
    p.add_argument("--zebra-mix-ratio", type=float, default=0.5,
                   help="fraction of AE training examples drawn from synthetic "
                        "ZebraLogic-style grids (SyntheticZebraGridDataset); the "
                        "rest from OpenWebText. The synthetic grids are random "
                        "assignments — disjoint from the WildEval/ZebraLogic eval "
                        "puzzles, so anchoring holds. 0 = pure OWT.")
    p.add_argument("--zebra-samples", type=int, default=50_000,
                   help="number of synthetic grids to materialize.")
    p.add_argument("--mixed-length", type=int, default=200_000,
                   help="virtual length of the OWT+grids mixed dataset.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="cosine-decay floor as a fraction of --lr.")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-dir", default="checkpoints/ae")
    p.add_argument("--resume", default=None,
                   help="path to a checkpoint to resume from.")
    p.add_argument("--max-a-length", type=int, default=384,
                   help="tokenizer truncation cap for the round-tripped text.")
    p.add_argument("--eval-n-examples", type=int, default=64,
                   help="per-dataset examples to score each eval cycle.")
    p.add_argument("--zebra-min-size", type=int, default=2,
                   help="smallest puzzle (houses) to keep in eval.")
    p.add_argument("--zebra-max-size", type=int, default=6,
                   help="largest puzzle (houses) to keep in eval.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--tui", action="store_true",
                   help="render training output as a Rich+plotext TUI.")
    p.add_argument("--config", default=None,
                   help="path to a YAML config file. CLI flags override it.")
    p.add_argument("--push-to-hub", action="store_true",
                   help="push the final AE checkpoint to HuggingFace Hub.")
    p.add_argument("--hub-repo", default=None,
                   help="HuggingFace Hub repo id, e.g. 'username/ae-conv-pg'.")
    p.add_argument("--hub-token", default=None,
                   help="HF API token. Defaults to the HF_TOKEN env var.")
    p.add_argument("--hub-private", action="store_true",
                   help="create the HF Hub repo as private.")
    return p


def build_param_groups(ae, weight_decay: float, decoder_lr: float | None = None):
    """Split trainable params into decay / no-decay groups.

    No-decay: 1-D tensors (LayerNorm gains, biases, the Flamingo gated `alpha_*`
    scalars) plus the learnable `queries`.
    """
    decay, no_decay = [], []
    dec_decay, dec_no_decay = [], []
    seen = set()
    for name, p in ae.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_query = name.endswith("pool.queries")
        is_alpha = "alpha_" in name
        is_bias = name.endswith(".bias")
        is_1d = p.ndim <= 1
        no_wd = is_query or is_alpha or is_bias or is_1d
        is_decoder = name.startswith("bart.")
        if is_decoder:
            (dec_no_decay if no_wd else dec_decay).append(p)
        else:
            (no_decay if no_wd else decay).append(p)
    groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if decoder_lr is not None and (dec_decay or dec_no_decay):
        groups += [
            {"params": dec_decay,    "weight_decay": weight_decay, "lr": decoder_lr},
            {"params": dec_no_decay, "weight_decay": 0.0,          "lr": decoder_lr},
        ]
    return groups


@torch.no_grad()
def eval_reconstruction(ae, dataset, mode, device, max_a_length,
                        n_examples=64, batch_size=16, num_beams=1,
                        verify_fn=None):
    """AE reconstruction eval.

    `mode` selects which scores are returned:
      - "byte"  : CER only. Returned as `cer` (lower=better). `pass_rate=None`.
      - "zebra" : both CER and puzzle-level exact-match rate against the gold
                  ZebraLogic grid, via `verify_fn`, in the `pass_rate` field.
    Always round-trips the "answer" field — the artifact the EBM is supervised
    to denoise toward.
    """
    assert mode in ("byte", "zebra")
    ae.eval()
    losses: list[float] = []
    samples: list[tuple[str, str]] = []
    all_preds: list[str] = []
    all_golds: list[str] = []
    correct = 0
    total = 0
    n = min(n_examples, len(dataset))
    for start in range(0, n, batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, n))]
        gold = [b["answer"] for b in batch]
        z = ae.encode_to_latents(gold, device, max_length=max_a_length)
        loss = ae.decode_loss(z, gold, device, max_length=max_a_length)
        losses.append(loss.item())
        preds = ae.decode(z, max_length=max_a_length, num_beams=num_beams)
        if mode == "zebra":
            results = verify_fn(preds, batch)
        else:
            results = [False] * len(preds)
        for pred, g, ok in zip(preds, gold, results):
            all_preds.append(pred.strip())
            all_golds.append(g.strip())
            if mode == "zebra":
                if ok is not None:
                    total += 1
                    if ok:
                        correct += 1
            if len(samples) < 3:
                samples.append((g, pred))
    cer = CharErrorRate()(all_preds, all_golds).item()
    pass_rate = (correct / max(total, 1)) if mode == "zebra" else None
    ae.train()
    return {
        "cer": cer,
        "pass_rate": pass_rate,
        "loss": sum(losses) / max(len(losses), 1),
        "samples": samples,
        "n": len(all_golds),
    }


def _push_checkpoint_to_hub(
    ae: FrozenBartAutoencoder,
    repo_id: str,
    save_dir: str,
    step: int,
    token: str | None = None,
    private: bool = False,
    log_fn=print,
) -> None:
    """Push the final AE checkpoint + config to HuggingFace Hub."""
    import json

    token = token or os.environ.get("HF_TOKEN")
    api = HfApi()

    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
    except Exception as e:
        log_fn(f"  warning: create_repo failed ({e}); continuing with upload")

    ae_config = {
        "model_name": ae.model_name,
        "k": ae.k,
        "pool_layers": ae.pool.attn.num_layers if hasattr(ae.pool, "attn") and hasattr(ae.pool.attn, "num_layers") else None,
        "pool_heads": None,
        "pool_type": ae.pool_type,
        "d_ae": ae.d_ae,
        "recon_layers": len(ae.recon.layers.layers),
        "recon_heads": None,
        "train_decoder": ae.train_decoder,
    }
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(ae_config, f, indent=2)

    ckpt_path = os.path.join(save_dir, "ae_checkpoint.pt")
    torch.save({"ae": ae.state_dict_ae(), "config": ae_config, "step": step}, ckpt_path)

    card = (
        "---\n"
        f"library_name: ired\n"
        f"tags:\n"
        f"- autoencoder\n"
        f"- bart\n"
        f"- latent-diffusion\n"
        "---\n\n"
        f"# IRED Autoencoder — `{repo_id}`\n\n"
        f"Frozen BART autoencoder with compression/reconstruction split.\n\n"
        f"- **Pool type:** `{ae.pool_type}`\n"
        f"- **Latent slots (K):** {ae.k}\n"
        f"- **d_ae:** {ae.d_ae}\n"
        f"- **Decoder fine-tuned:** {'yes' if ae.train_decoder else 'no'}\n"
        f"- **Base model:** `{ae.model_name}`\n"
        f"- **Training steps:** {step}\n\n"
        "## Usage\n\n"
        "```python\n"
        "from ired.model.autoencoder import FrozenBartAutoencoder\n"
        "import torch\n\n"
        "ae = FrozenBartAutoencoder(\n"
        f"    model_name='{ae.model_name}',\n"
        f"    k={ae.k},\n"
        f"    pool_type='{ae.pool_type}',\n"
        f"    d_ae={ae.d_ae},\n"
        f"    train_decoder={ae.train_decoder},\n"
        ")\n"
        "ckpt = torch.load('ae_checkpoint.pt', map_location='cpu')\n"
        "ae.load_ae(ckpt['ae'])\n"
        "```\n"
    )
    card_path = os.path.join(save_dir, "model_card.md")
    with open(card_path, "w") as f:
        f.write(card)

    for path, path_in_repo in [
        (ckpt_path, "ae_checkpoint.pt"),
        (config_path, "config.json"),
        (card_path, "README.md"),
    ]:
        log_fn(f"  uploading {path_in_repo} …")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            token=token,
        )

    log_fn(f"  done — https://huggingface.co/{repo_id}")


def main(argv=None):
    parser = build_parser()
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config is not None:
        if yaml is None:
            parser.error("--config requires PyYAML (`pip install pyyaml`)")
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            parser.error(f"config file {pre_args.config!r} must be a YAML mapping")
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    eval_series = ("owt_recon_cer", "zebra_recon_cer", "zebra_exact")

    with make_reporter(
        use_tui=args.tui,
        eval_series=eval_series,
        title="train_autoencoder",
    ) as r:
        d_ae = args.d_ae if args.d_ae > 0 else None
        r.log(
            f"loading autoencoder: {args.model} "
            f"(k={args.k}, pool_layers={args.pool_layers}, recon_layers={args.recon_layers}, "
            f"d_ae={'d_model' if d_ae is None else d_ae})"
        )
        ae = FrozenBartAutoencoder(
            model_name=args.model,
            k=args.k,
            pool_layers=args.pool_layers,
            pool_heads=args.pool_heads,
            pool_type=args.pool_type,
            d_ae=d_ae,
            recon_layers=args.recon_layers,
            recon_heads=args.recon_heads,
            train_decoder=args.unfreeze_decoder,
        ).to(args.device)
        ae.train()
        r.log(f"d_model={ae.d_model}  d_ae={ae.d_ae}")

        # Training corpus: OpenWebText.
        r.log(
            f"train dataset: OpenWebText (materializing up to {args.owt_samples} docs, "
            f"{args.owt_min_chars}≤len≤{args.owt_max_chars})"
        )
        train_ds = OpenWebTextDataset(
            max_samples=args.owt_samples,
            min_chars=args.owt_min_chars,
            max_chars=args.owt_max_chars,
            seed=args.seed,
        )
        r.log(f"  materialized {len(train_ds)} OWT docs")

        # Optionally mix in synthetic ZebraLogic grids for format exposure.
        if args.zebra_mix_ratio > 0.0:
            r.log(
                f"  + synthetic ZebraLogic grids (materializing {args.zebra_samples}, "
                f"sizes {args.zebra_min_size}–{args.zebra_max_size}, "
                f"mix ratio={args.zebra_mix_ratio:.2f})"
            )
            grid_ds = SyntheticZebraGridDataset(
                max_samples=args.zebra_samples,
                min_size=args.zebra_min_size,
                max_size=args.zebra_max_size,
                seed=args.seed,
            )
            r.log(f"  materialized {len(grid_ds)} synthetic grids")
            train_ds = MixedDataset(
                datasets=[train_ds, grid_ds],
                weights=[1.0 - args.zebra_mix_ratio, args.zebra_mix_ratio],
                length=args.mixed_length,
            )
            r.log(
                f"  mixed train dataset: virtual length {len(train_ds)} "
                f"({(1.0 - args.zebra_mix_ratio):.0%} OWT + {args.zebra_mix_ratio:.0%} grids)"
            )

        # Held-out OWT slice (different seed → effectively disjoint).
        r.log(f"  building OWT eval slice ({args.owt_eval_samples} docs, seed={args.seed + 1})")
        owt_eval_ds = OpenWebTextDataset(
            max_samples=args.owt_eval_samples,
            min_chars=args.owt_min_chars,
            max_chars=args.owt_max_chars,
            seed=args.seed + 1,
        )

        # Held-out ZebraLogic eval.
        r.log("  loading ZebraLogic (held-out NL logic-grid eval)")
        zebra_eval_ds = ZebraLogicDataset(
            split="test", max_samples=args.eval_n_examples,
            min_size=args.zebra_min_size, max_size=args.zebra_max_size,
            seed=args.seed,
        )
        r.log(f"    {len(zebra_eval_ds)} puzzles")
        zebra_verifier = ZebraLogicVerifier()

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=(args.device.startswith("cuda")),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(4 if args.num_workers > 0 else None),
            drop_last=True,
        )

        pool_params = ae.trainable_parameters()
        n_params = sum(p.numel() for p in pool_params)
        r.log(f"trainable AE params (pool + recon): {n_params:,}")
        param_groups = build_param_groups(
            ae, args.weight_decay,
            decoder_lr=args.decoder_lr if args.unfreeze_decoder else None,
        )
        n_decay = sum(p.numel() for p in param_groups[0]["params"])
        n_nodecay = sum(p.numel() for p in param_groups[1]["params"])
        opt = AdamW(param_groups, lr=args.lr)

        # Warmup → cosine decay.
        def _lr_lambda(step: int) -> float:
            if args.warmup_steps > 0 and step < args.warmup_steps:
                return float(step + 1) / float(args.warmup_steps)
            decay_steps = max(args.steps - args.warmup_steps, 1)
            progress = (step - args.warmup_steps) / decay_steps
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        sched = LambdaLR(opt, _lr_lambda)
        r.log(
            f"optimizer: AdamW(lr={args.lr}, wd={args.weight_decay} on "
            f"{n_decay:,} params; 0 on {n_nodecay:,})  "
            f"warmup={args.warmup_steps}  min_lr_ratio={args.min_lr_ratio}  "
            f"grad_accum={args.grad_accum_steps}"
        )

        step = 0
        if args.resume is not None:
            r.log(f"resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=args.device)
            ae.load_ae(ckpt["ae"])
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            else:
                r.log("  (no optimizer state in checkpoint — starting optimizer fresh)")
            if "scheduler" in ckpt:
                sched.load_state_dict(ckpt["scheduler"])
            step = ckpt.get("step", 0)
            r.log(f"  resumed at step {step}")
            if step >= args.steps:
                r.log(
                    f"  warning: --steps={args.steps} <= resumed step={step}; "
                    "nothing to do. Increase --steps to continue training."
                )

        train_iter = iter(train_loader)
        t0 = time.time()
        accum = max(args.grad_accum_steps, 1)
        while step < args.steps:
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
            cur_lr = opt.param_groups[0]["lr"]
            avg_loss = loss_sum / accum
            r.train_point(step, loss=avg_loss, lr=cur_lr)
            r.set_status(f"step {step}/{args.steps} | loss {avg_loss:.4f} | lr {cur_lr:.2e}")
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                r.log(f"step {step:6d} | loss {avg_loss:.4f} | lr {cur_lr:.2e} | {elapsed:.1f}s")

            if step % args.eval_every == 0 or step == args.steps:
                ev_owt = eval_reconstruction(
                    ae, owt_eval_ds, "byte", args.device, args.max_a_length,
                    n_examples=args.eval_n_examples, batch_size=args.batch_size,
                    num_beams=1,
                )
                ev_zebra = eval_reconstruction(
                    ae, zebra_eval_ds, "zebra", args.device, args.max_a_length,
                    n_examples=args.eval_n_examples, batch_size=args.batch_size,
                    num_beams=1, verify_fn=zebra_verifier.verify_batch,
                )
                r.eval_point(step, **{
                    "owt_recon_cer": ev_owt["cer"],
                    "zebra_recon_cer": ev_zebra["cer"],
                    "zebra_exact": ev_zebra["pass_rate"],
                })
                r.log(
                    f"  [eval] owt_recon_cer={ev_owt['cer']:.3f} (n={ev_owt['n']})  "
                    f"zebra_recon_cer={ev_zebra['cer']:.3f} "
                    f"zebra_exact={ev_zebra['pass_rate']:.3f} (n={ev_zebra['n']})"
                )
                r.log(f"    losses: owt={ev_owt['loss']:.3f}  zebra={ev_zebra['loss']:.3f}")

                def _snip(s, n=180):
                    s = re.sub(r"[ \t]*\n[ \t]*", " ⏎ ", s)
                    s = re.sub(r"[ \t]{2,}", " ", s)
                    return s if len(s) <= n else s[:n] + "…"

                for name, ev in [("owt", ev_owt), ("zebra", ev_zebra)]:
                    if not ev["samples"]:
                        continue
                    gold, pred = ev["samples"][0]
                    r.log(f"    [{name}] gold: {_snip(gold)}")
                    r.log(f"    [{name}] pred: {_snip(pred)}")
                    if name == "zebra":
                        r.log(f"    [{name}] gold(raw): {gold[:240]!r}")
                        r.log(f"    [{name}] pred(raw): {pred[:240]!r}")

                ckpt_metrics = {
                    "owt_recon_cer": ev_owt["cer"],
                    "zebra_recon_cer": ev_zebra["cer"],
                    "zebra_exact": ev_zebra["pass_rate"],
                }
                ckpt = {
                    "ae": ae.state_dict_ae(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "config": vars(args),
                    "step": step,
                    **ckpt_metrics,
                }
                torch.save(ckpt, os.path.join(args.save_dir, f"ae_step{step}.pt"))
                torch.save(ckpt, os.path.join(args.save_dir, "ae_latest.pt"))

        # ── push final checkpoint to HuggingFace Hub ───────────────────
        if args.push_to_hub:
            if not _has_hf_hub:
                r.log("ERROR: --push-to-hub requires huggingface_hub")
            elif args.hub_repo is None:
                r.log("ERROR: --push-to-hub requires --hub-repo (e.g. 'username/ae-conv-pg')")
            else:
                r.log(f"pushing final checkpoint to HF Hub: {args.hub_repo}")
                _push_checkpoint_to_hub(
                    ae=ae,
                    repo_id=args.hub_repo,
                    save_dir=args.save_dir,
                    step=step,
                    token=args.hub_token,
                    private=args.hub_private,
                    log_fn=r.log,
                )


if __name__ == "__main__":
    main()
