"""One-off: verify the step-5000 SQL AE exec_acc and put it on equal footing
with a *vanilla* (no pool/recon bottleneck) bart-base round-trip.

The training log compared trained `sql_exec_acc` (execution) against the
vanilla baseline's `norm_exact` (string match) — different metrics. This runs
both reconstructions through the SAME SpiderExecutionVerifier on the SAME first
64 Spider-dev examples, so exec_acc is compared to exec_acc.
"""
import argparse
import torch
from torchmetrics.text import CharErrorRate

from ired.data import SpiderText2SQLDataset, SpiderExecutionVerifier
from ired.model.autoencoder import FrozenBartAutoencoder
from transformers import BartForConditionalGeneration, AutoTokenizer


def batched(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i:i + bs]


def exec_acc(results):
    judged = [r for r in results if r is not None]
    passed = sum(1 for r in judged if r)
    return passed, len(judged)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ae_sql_conv/ae_step5000.pt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    print(f"ckpt step={ck['step']}  stored: "
          f"sql_exec_acc={ck.get('sql_exec_acc')}  sql_recon_cer={ck.get('sql_recon_cer')}")

    ds = SpiderText2SQLDataset(split="validation")
    batch = [ds[i] for i in range(args.n)]
    golds = [ex["answer"] for ex in batch]

    # --- trained AE (pool + recon + fine-tuned decoder) -------------------
    ae = FrozenBartAutoencoder(
        model_name=cfg["model"], k=cfg["k"],
        pool_layers=cfg["pool_layers"], pool_heads=cfg["pool_heads"],
        pool_type=cfg["pool_type"],
        d_ae=(None if cfg.get("d_ae", -1) is not None and cfg.get("d_ae", -1) < 0
              else cfg.get("d_ae")),
        recon_layers=cfg["recon_layers"], recon_heads=cfg["recon_heads"],
        train_decoder=cfg.get("unfreeze_decoder", False),
    ).to(args.device)
    ae.load_ae(ck["ae"])
    ae.eval()

    trained_preds = []
    with torch.no_grad():
        for chunk in batched(golds, args.batch_size):
            z = ae.encode_to_latents(chunk, args.device, max_length=args.max_length)
            trained_preds += ae.decode(z, max_length=args.max_length, num_beams=1)
    del ae
    torch.cuda.empty_cache()

    # --- vanilla pristine bart-base round-trip (no bottleneck) ------------
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    bart = BartForConditionalGeneration.from_pretrained(cfg["model"]).to(args.device).eval()
    vanilla_preds = []
    with torch.no_grad():
        for chunk in batched(golds, args.batch_size):
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_length).to(args.device)
            out = bart.generate(input_ids=enc.input_ids,
                                attention_mask=enc.attention_mask,
                                max_length=args.max_length, num_beams=1)
            vanilla_preds += tok.batch_decode(out, skip_special_tokens=True)
    del bart
    torch.cuda.empty_cache()

    # --- execution verification on the identical example set --------------
    v = SpiderExecutionVerifier(timeout=cfg.get("sql_exec_timeout", 5.0))
    v.setup([ex["_db_id"] for ex in batch])
    trained_res = v.verify_batch(trained_preds, batch)
    vanilla_res = v.verify_batch(vanilla_preds, batch)
    v.teardown()

    cer = CharErrorRate()
    t_pass, t_tot = exec_acc(trained_res)
    u_pass, u_tot = exec_acc(vanilla_res)
    t_cer = cer(trained_preds, golds).item()
    u_cer = cer(vanilla_preds, golds).item()

    print(f"\nn={args.n}  judged(trained)={t_tot}  judged(vanilla)={u_tot}  "
          f"(excluded = missing DB / gold errored)\n")
    print(f"{'':12}{'exec_acc':>12}{'cer':>10}")
    print(f"{'trained AE':12}{t_pass/max(t_tot,1):>12.4f}{t_cer:>10.4f}")
    print(f"{'vanilla':12}{u_pass/max(u_tot,1):>12.4f}{u_cer:>10.4f}")


if __name__ == "__main__":
    main()
