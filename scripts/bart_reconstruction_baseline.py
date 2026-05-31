"""Vanilla BART reconstruction baseline — no AttentionPool, no ReconstructionNet.

Establishes the *ceiling* the K-slot autoencoder has to live under. We take the
raw `BartForConditionalGeneration` (default `facebook/bart-large`) and round-trip
each MBPP / HumanEval answer straight through it:

    text ──► BART encoder ──► (full L×d encoder memory) ──► BART decoder.generate ──► text

There is no compression to K latent slots and no reconstruction net in the path —
the decoder cross-attends the *entire* encoder sequence. BART is pretrained as a
denoising autoencoder, so on clean input this is close to a copy task and should
reconstruct nearly perfectly. Whatever loss shows up here is a floor on the error
the compressed pipeline can never beat; it isolates "is the BART tokenizer +
decoder even capable of round-tripping this code" from "does our K-slot pool throw
away too much."

Metrics mirror `ired/train_autoencoder.py::eval_reconstruction` so the numbers are
directly comparable to the trained-AE eval:
  - cer        CharErrorRate of reconstruction vs gold (0 = exact).
  - exact      fraction of examples reconstructed byte-exact (stripped).
  - pass_rate  fraction whose reconstruction still passes the test fixture.

Usage:
    uv run python -m scripts.bart_reconstruction_baseline
    uv run python -m scripts.bart_reconstruction_baseline --model facebook/bart-base
    uv run python -m scripts.bart_reconstruction_baseline --n-examples 500 --num-beams 1
"""
from __future__ import annotations

import argparse
import time

import torch
from torchmetrics.text import CharErrorRate
from transformers import AutoTokenizer, BartForConditionalGeneration

from ired.data import HumanEvalDataset, MBPPDataset, verify_code_batch


@torch.no_grad()
def reconstruct_batch(model, tokenizer, texts, device, max_length, num_beams):
    """Round-trip a batch of texts through vanilla BART (encoder→decoder.generate).

    Full cross-attention over the entire encoder sequence — no latent bottleneck.
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
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
def eval_dataset(model, tokenizer, dataset, name, device, max_length,
                 n_examples, batch_size, num_beams, verify):
    """Reconstruct every example's `answer`, score CER / exact / pass-rate."""
    n = min(n_examples, len(dataset))
    all_preds: list[str] = []
    all_golds: list[str] = []
    exact = 0
    correct = 0
    total = 0
    samples: list[tuple[str, str]] = []

    t0 = time.time()
    for start in range(0, n, batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, n))]
        gold = [b["answer"] for b in batch]
        preds = reconstruct_batch(
            model, tokenizer, gold, device, max_length, num_beams
        )
        results = verify_code_batch(preds, batch) if verify else [False] * len(preds)
        for pred, g, ok in zip(preds, gold, results):
            all_preds.append(pred.strip())
            all_golds.append(g.strip())
            if pred.strip() == g.strip():
                exact += 1
            total += 1
            if ok:
                correct += 1
            if len(samples) < 3:
                samples.append((g, pred))
        done = min(start + batch_size, n)
        print(f"    [{name}] {done}/{n}  ({time.time() - t0:.1f}s)", flush=True)

    cer = CharErrorRate()(all_preds, all_golds).item()
    return {
        "cer": cer,
        "exact": exact / max(total, 1),
        "pass_rate": correct / max(total, 1),
        "n": total,
        "samples": samples,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/bart-large")
    p.add_argument("--n-examples", type=int, default=200,
                   help="examples per dataset (MBPP test has 500, HumanEval 164).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-beams", type=int, default=1,
                   help="1 = greedy. >1 enables beam search (usually lifts code "
                        "pass-rate when the right token is #2/#3).")
    p.add_argument("--max-length", type=int, default=384,
                   help="tokenizer truncation + generation cap. Matches the AE "
                        "training default (--max-a-length 384).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-verify", action="store_true",
                   help="skip code execution (CER + exact only). Faster smoke test.")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    print(f"loading {args.model} on {args.device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BartForConditionalGeneration.from_pretrained(args.model).to(args.device)
    model.eval()
    print(f"  d_model={model.config.d_model}  "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    print(f"loading MBPP test ({args.n_examples} max) + HumanEval ...", flush=True)
    mbpp = MBPPDataset(split="test", max_samples=args.n_examples, seed=args.seed)
    he = HumanEvalDataset(max_samples=args.n_examples, seed=args.seed)
    print(f"  mbpp={len(mbpp)}  humaneval={len(he)}", flush=True)

    results = {}
    for name, ds in [("mbpp", mbpp), ("humaneval", he)]:
        print(f"\n=== {name} ===", flush=True)
        results[name] = eval_dataset(
            model, tokenizer, ds, name, args.device, args.max_length,
            args.n_examples, args.batch_size, args.num_beams,
            verify=not args.no_verify,
        )

    print("\n" + "=" * 60)
    print(f"Vanilla BART reconstruction baseline — {args.model}")
    print(f"  (no AttentionPool, no ReconstructionNet; full encoder→decoder)")
    print(f"  num_beams={args.num_beams}  max_length={args.max_length}")
    print("=" * 60)
    for name, r in results.items():
        verify_str = "  (verify skipped)" if args.no_verify else f"  pass_rate={r['pass_rate']:.3f}"
        print(f"  {name:10s}  n={r['n']:4d}  cer={r['cer']:.4f}  "
              f"exact={r['exact']:.3f}{verify_str}")

    # One reconstruction sample per dataset for eyeballing whitespace fidelity.
    for name, r in results.items():
        if not r["samples"]:
            continue
        gold, pred = r["samples"][0]
        print(f"\n--- {name} sample ---")
        print(f"  gold(raw): {gold[:240]!r}")
        print(f"  pred(raw): {pred[:240]!r}")


if __name__ == "__main__":
    main()
