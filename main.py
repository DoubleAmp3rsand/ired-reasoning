"""Entry point dispatcher for IRED-reasoning experiments.

Subcommands:
  sanity      Random-tensor forward/backward check (no GSM8K download).
  train-ae    Milestone 1 — train AttentionPool reconstruction.
  train-ebm   Milestone 2 — train energy network in latent space.
  sample      Milestone 3 — measure accuracy vs. inner_steps.

Run `python main.py <subcommand> --help` for flags.
"""
from __future__ import annotations

import sys


USAGE = """Usage:
  python main.py sanity     [--model ...] [--device cpu|cuda]
  python main.py train-ae   --save-dir ckpts/ae   [more flags]
  python main.py train-ebm  --ae-ckpt ckpts/ae/pool_latest.pt --save-dir ckpts/ebm  [more flags]
  python main.py sample     --ae-ckpt ... --ebm-ckpt ...  [more flags]

Add `--help` after the subcommand for available flags.
"""


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    cmd = sys.argv.pop(1)
    if cmd in ("-h", "--help", "help"):
        print(USAGE)
        return
    if cmd == "sanity":
        from ired.sanity import main as run
    elif cmd == "train-ae":
        from ired.train_autoencoder import main as run
    elif cmd == "train-ebm":
        from ired.train_diffusion import main as run
    elif cmd == "sample":
        from ired.sample import main as run
    else:
        print(f"unknown command: {cmd}\n{USAGE}")
        sys.exit(1)
    run()


if __name__ == "__main__":
    main()
