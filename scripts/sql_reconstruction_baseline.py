"""Vanilla AE reconstruction baseline on SQL (Spider) — whitespace-robust target.

Companion to `scripts/bart_reconstruction_baseline.py`. The code baseline showed
the dominant round-trip failure is whitespace/newline collapse — fatal for Python
(whitespace is syntax), but SQL is whitespace- and case-insensitive, so the same
AE mangling need not break correctness. This script tests whether an off-the-shelf
frozen AE round-trips SQL well enough to clear a Milestone-1-style reconstruction
bar *without* a custom autoencoder.

We round-trip each Spider `query` (the SQL answer) straight through the raw
seq2seq model (encoder→decoder.generate, full cross-attention, NO pool/recon).

Metrics:
  - cer         CharErrorRate of raw reconstruction vs gold (0 = exact).
  - exact       byte-exact after strip.
  - norm_exact  exact after SQL normalization (lowercase + collapse all
                whitespace). For SQL this is the correctness-relevant match:
                keywords are case-insensitive and whitespace is irrelevant, so a
                reconstruction that differs only in case/spacing is still correct.

(No execution verification yet — that needs the Spider DBs. Reconstruction
fidelity is the Milestone-1 gate; execution pass-rate is the follow-up.)

Usage:
    uv run python -m scripts.sql_reconstruction_baseline
    uv run python -m scripts.sql_reconstruction_baseline --model facebook/bart-base
    uv run python -m scripts.sql_reconstruction_baseline --model Salesforce/codet5-base
"""
from __future__ import annotations

import argparse
import re
import time

import torch
from datasets import load_dataset
from torchmetrics.text import CharErrorRate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def normalize_sql(s: str) -> str:
    """Lowercase + collapse all whitespace runs to single spaces, stripped.

    Captures the formatting/case freedom SQL correctness tolerates (keywords are
    case-insensitive, whitespace is irrelevant). String/identifier case can
    matter in principle, so this is an upper-bound proxy for correctness, paired
    with raw `exact` as the lower bound.
    """
    return re.sub(r"\s+", " ", s).strip().lower()


@torch.no_grad()
def reconstruct_batch(model, tokenizer, texts, device, max_length, num_beams):
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    ).to(device)
    out_ids = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=(num_beams > 1),
    )
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


@torch.no_grad()
def eval_queries(model, tokenizer, queries, device, max_length, batch_size, num_beams):
    n = len(queries)
    all_preds, all_golds = [], []
    exact = norm_exact = 0
    samples = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        gold = queries[start:start + batch_size]
        preds = reconstruct_batch(model, tokenizer, gold, device, max_length, num_beams)
        for pred, g in zip(preds, gold):
            all_preds.append(pred.strip())
            all_golds.append(g.strip())
            if pred.strip() == g.strip():
                exact += 1
            if normalize_sql(pred) == normalize_sql(g):
                norm_exact += 1
            if len(samples) < 4:
                samples.append((g, pred))
        print(f"    {min(start + batch_size, n)}/{n}  ({time.time() - t0:.1f}s)", flush=True)
    cer = CharErrorRate()(all_preds, all_golds).item()
    return {
        "cer": cer,
        "exact": exact / max(n, 1),
        "norm_exact": norm_exact / max(n, 1),
        "n": n,
        "samples": samples,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/bart-large")
    p.add_argument("--split", default="validation")
    p.add_argument("--n-examples", type=int, default=1034)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    print(f"loading {args.model} on {args.device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)
    model.eval()
    print(f"  params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    print(f"loading Spider [{args.split}] ...", flush=True)
    ds = load_dataset("spider", split=args.split)
    queries = [ds[i]["query"] for i in range(min(args.n_examples, len(ds)))]
    lens = [len(q) for q in queries]
    print(f"  {len(queries)} queries  (chars: min={min(lens)} mean={sum(lens)//len(lens)} max={max(lens)})",
          flush=True)

    print(f"\n=== spider ({args.model}) ===", flush=True)
    r = eval_queries(model, tokenizer, queries, args.device,
                     args.max_length, args.batch_size, args.num_beams)

    print("\n" + "=" * 60)
    print(f"Vanilla AE SQL reconstruction — {args.model}")
    print(f"  (no pool, no recon; full encoder→decoder)  beams={args.num_beams}")
    print("=" * 60)
    print(f"  spider  n={r['n']}  cer={r['cer']:.4f}  "
          f"exact={r['exact']:.3f}  norm_exact={r['norm_exact']:.3f}")

    print("\n--- samples (gold | pred) ---")
    for gold, pred in r["samples"]:
        print(f"  gold: {gold!r}")
        print(f"  pred: {pred!r}")
        print()


if __name__ == "__main__":
    main()
