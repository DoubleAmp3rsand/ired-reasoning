"""Standalone rerun of the AE reconstruction eval for a single checkpoint,
computing CER both the original way (full strings) and with `_snip`-style
whitespace normalization applied to gold+pred *before* CER.

The stored checkpoint metrics use raw `.strip()`-ed CER; this script shows the
whitespace-insensitive variant alongside it. pass_rate is execution-based and
unchanged — it's reported to confirm the rerun reproduces the stored numbers.
"""
from __future__ import annotations

import argparse
import re
import resource

import torch
from torchmetrics.text import CharErrorRate


def _mem_report(tag: str) -> None:
    """Process peak RSS (works on unified memory, where nvidia-smi is N/A)."""
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    cur_gb = "-"
    if torch.cuda.is_available():
        cur_gb = f"{torch.cuda.memory_allocated() / 1e9:.2f}"
    print(f"[mem] {tag:24s} peak_rss={peak_gb:.2f}GB cuda_alloc={cur_gb}GB", flush=True)

from ired.model.autoencoder import FrozenBartAutoencoder
from ired.data import HumanEvalDataset, MBPPDataset, OpenWebTextDataset, verify_code_batch


def _norm(s: str) -> str:
    """`_snip`'s whitespace collapsing, WITHOUT the 180-char truncation —
    collapse newline+surrounding indentation to a single ⏎ marker, then
    squeeze remaining runs of spaces."""
    s = re.sub(r"[ \t]*\n[ \t]*", " ⏎ ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s


EXEC_ENABLED = True  # toggled off by --no-exec to isolate model vs subprocess


@torch.no_grad()
def eval_ds(ae, dataset, mode, device, max_a_length, n_examples, batch_size, num_beams):
    ae.eval()
    raw_preds, raw_golds, norm_preds, norm_golds = [], [], [], []
    code_correct = code_total = 0
    n = min(n_examples, len(dataset))
    for start in range(0, n, batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, n))]
        gold = [b["answer"] for b in batch]
        z = ae.encode_to_latents(gold, device, max_length=max_a_length)
        preds = ae.decode(z, max_length=max_a_length, num_beams=num_beams, src_texts=gold)
        print(f"[step] decoded batch start={start} (gen done)", flush=True)
        do_exec = mode == "code" and EXEC_ENABLED
        results = verify_code_batch(preds, batch) if do_exec else [False] * len(preds)
        if do_exec:
            print(f"[step] verified batch start={start}", flush=True)
        for pred, g, ok in zip(preds, gold, results):
            raw_preds.append(pred.strip())
            raw_golds.append(g.strip())
            norm_preds.append(_norm(pred.strip()))
            norm_golds.append(_norm(g.strip()))
            if mode == "code":
                code_total += 1
                code_correct += int(bool(ok))
    raw_cer = CharErrorRate()(raw_preds, raw_golds).item()
    norm_cer = CharErrorRate()(norm_preds, norm_golds).item()
    pass_rate = code_correct / max(code_total, 1) if mode == "code" else None
    return {"raw_cer": raw_cer, "norm_cer": norm_cer, "pass_rate": pass_rate, "n": len(raw_golds)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ae_conv_pg/ae_step15000.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n", type=int, default=None, help="override eval_n_examples (saturation test)")
    p.add_argument("--batch-size", type=int, default=None, help="override batch size")
    p.add_argument("--datasets", default="owt,mbpp,humaneval", help="comma list to limit eval")
    p.add_argument("--no-exec", action="store_true", help="skip code execution (isolate model path)")
    args = p.parse_args()
    global EXEC_ENABLED
    EXEC_ENABLED = not args.no_exec

    ck = torch.load(args.ckpt, map_location=args.device)
    c = ck["config"]
    torch.manual_seed(c.get("seed", 0))
    seed = c.get("seed", 0)
    n = args.n if args.n is not None else c["eval_n_examples"]
    bs = args.batch_size if args.batch_size is not None else c["batch_size"]
    beams = c["eval_beams"]
    maxlen = c["max_a_length"]
    want = set(args.datasets.split(","))
    _mem_report("after torch.load")

    d_ae = c["d_ae"] if c["d_ae"] > 0 else None
    ae = FrozenBartAutoencoder(
        model_name=c["model"], k=c["k"], pool_layers=c["pool_layers"],
        pool_heads=c["pool_heads"], pool_type=c["pool_type"], d_ae=d_ae,
        recon_layers=c["recon_layers"], recon_heads=c["recon_heads"],
        use_copy=c["use_copy"], train_decoder=c["unfreeze_decoder"],
    ).to(args.device)
    ae.load_ae(ck["ae"])
    _mem_report("after AE to device")

    all_specs = []
    if "owt" in want:
        owt = OpenWebTextDataset(max_samples=min(c["owt_eval_samples"], n), min_chars=c["owt_min_chars"],
                                 max_chars=c["owt_max_chars"], seed=seed + 1)
        all_specs.append(("owt", owt, "byte"))
    if "mbpp" in want:
        all_specs.append(("mbpp", MBPPDataset(split="test", max_samples=n, seed=seed), "code"))
    if "humaneval" in want:
        all_specs.append(("humaneval", HumanEvalDataset(max_samples=n, seed=seed), "code"))
    _mem_report("after datasets")

    print(f"\n=== rerun eval @ step {ck['step']} ({args.ckpt}) ===")
    print(f"{'dataset':12s} {'raw_cer':>9s} {'norm_cer':>9s} {'Δcer':>8s} {'pass':>7s} {'n':>4s}   stored_cer/pass")
    stored = {
        "owt": (ck.get("owt_recon_cer"), None),
        "mbpp": (ck.get("mbpp_cer"), ck.get("mbpp_pass")),
        "humaneval": (ck.get("humaneval_cer"), ck.get("humaneval_pass")),
    }
    for name, ds, mode in all_specs:
        r = eval_ds(ae, ds, mode, args.device, maxlen, n, bs, beams)
        _mem_report(f"after eval[{name}]")
        delta = r["norm_cer"] - r["raw_cer"]
        pr = f"{r['pass_rate']:.3f}" if r["pass_rate"] is not None else "  -  "
        s_cer, s_pass = stored[name]
        s_pass_str = f"/{s_pass:.3f}" if s_pass is not None else ""
        print(f"{name:12s} {r['raw_cer']:9.4f} {r['norm_cer']:9.4f} {delta:+8.4f} {pr:>7s} {r['n']:4d}   "
              f"{s_cer:.4f}{s_pass_str}")


if __name__ == "__main__":
    main()
