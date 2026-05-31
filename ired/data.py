"""Dataset loaders for the IRED reasoning project.

The task substrate is **ZebraLogic** — natural-language logic-grid puzzles that
are NL-native, exactly verifiable, and AE-friendly. The answer surface is
low-entropy, fixed-format, and built from OWT-frequent common nouns, exactly the
regime the frozen-anchor AE reconstructs well.

Natural-language corpus (AE pretraining mix):

- `OpenWebTextDataset` — generic natural-language pretraining corpus for the
  frozen-BART autoencoder.

ZebraLogic corpus (the current task):

- `ZebraLogicDataset` — `WildEval/ZebraLogic` (grid_mode), 1000 NL logic-grid
  puzzles, 2×2 to 6×6, programmatically generated with a unique solution. The
  **held-out eval target**: the AE never trains on it (anchoring, gensis §2.2).
- `ZebraLogicVerifier` — exact-match verifier for decoded grids against the
  gold solution. Unlike the old Spider/SQL path there is no database lifecycle:
  the unique gold solution ships with every example, so verification is a pure
  structural comparison.
"""

from __future__ import annotations

import os
import random
import re

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# OpenWebText — AE pretraining corpus
# ---------------------------------------------------------------------------

class OpenWebTextDataset(Dataset):
    """A filtered slice of OpenWebText for autoencoder pretraining.

    `Skylion007/openwebtext` is the public reproduction of GPT-2's training
    data: ~8M web documents scraped from Reddit-cited URLs (~41GB full
    download). We use HF streaming to avoid the download, iterate with a
    length filter, and materialize the first `max_samples` accepted docs
    into memory.

    Returned record: `{"question": "", "answer": text}` so `collate` and
    the AE encode path are identical to the task datasets below.
    """

    def __init__(
        self,
        max_samples: int = 100_000,
        min_chars: int = 200,
        max_chars: int = 4_000,
        seed: int = 0,
    ):
        ds = load_dataset(
            "Skylion007/openwebtext",
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        self.texts: list[str] = []
        for ex in ds:
            t = (ex.get("text") or "").strip()
            if len(t) < min_chars:
                continue
            if len(t) > max_chars:
                t = t[:max_chars]
            self.texts.append(t)
            if len(self.texts) >= max_samples:
                break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"question": "", "answer": self.texts[int(idx)]}


# ---------------------------------------------------------------------------
# ZebraLogic — natural-language logic-grid puzzles (the current eval target)
# ---------------------------------------------------------------------------
#
# Why ZebraLogic: the AE round-trip is the binding ceiling for any
# execution-graded structured target. ZebraLogic keeps the two properties we
# need — an irreducibly *natural-language* problem (so the language AE is
# load-bearing) and an exactly *verifiable* answer — while the answer surface is
# low-entropy, fixed-format, and built from OWT-frequent common nouns, exactly
# the regime the frozen-anchor AE reconstructs well. It is, in effect, "Sudoku
# stated in English."
#
# Answer serialization: one line per house, `House <n>: Attr=val, Attr=val, …`.
# Fixed structure + small vocab → AE-friendly; trivially re-parseable for the
# exact-match verifier. (Values are assumed comma-free, which holds for the
# ZebraLogic value pools; a stray comma inside a value is a known limitation.)

_ZEBRA_LINE_RE = re.compile(r"house\s+(\w+)\s*:\s*(.*)", re.IGNORECASE)


def zebra_normalize(s) -> str:
    """Collapse whitespace + lowercase — the cell-comparison canonical form."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def serialize_zebra_solution(solution: dict) -> str:
    """Gold solution dict ({header, rows}) → canonical answer string.

    `header[0]` is always "House" and `row[0]` the house number; the remaining
    columns are the attribute categories and their assigned values.
    """
    header = solution["header"]
    attrs = header[1:]
    lines = []
    for row in solution["rows"]:
        house = str(row[0]).strip()
        cells = ", ".join(
            f"{a}={str(v).strip()}" for a, v in zip(attrs, row[1:])
        )
        lines.append(f"House {house}: {cells}")
    return "\n".join(lines)


def parse_zebra_solution(text: str):
    """Decoded answer string → {house: {attr: value}} (all normalized), or
    None if nothing parseable was found. Tolerant of attribute reordering and
    extra/missing whitespace — only the (house, attr)→value mapping matters."""
    mapping: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        m = _ZEBRA_LINE_RE.match(line.strip())
        if not m:
            continue
        house = zebra_normalize(m.group(1))
        cells: dict[str, str] = {}
        for part in m.group(2).split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            cells[zebra_normalize(k)] = zebra_normalize(v)
        if cells:
            mapping[house] = cells
    return mapping or None


def zebra_gold_mapping(solution: dict) -> dict[str, dict[str, str]]:
    header = solution["header"]
    attrs = header[1:]
    mapping: dict[str, dict[str, str]] = {}
    for row in solution["rows"]:
        house = zebra_normalize(row[0])
        mapping[house] = {
            zebra_normalize(a): zebra_normalize(v)
            for a, v in zip(attrs, row[1:])
        }
    return mapping


def zebra_match(pred_text: str, gold_solution: dict) -> bool:
    """Puzzle-level exact match: every gold (house, attr) cell must be present
    and equal in the decoded answer. ZebraLogic solutions are unique, so this
    all-or-nothing check *is* the correctness metric (an unparseable or
    partially-correct grid is not a solved puzzle)."""
    gold = zebra_gold_mapping(gold_solution)
    pred = parse_zebra_solution(pred_text)
    if not pred:
        return False
    for house, cells in gold.items():
        pcells = pred.get(house)
        if pcells is None:
            return False
        for attr, val in cells.items():
            if pcells.get(attr) != val:
                return False
    return True


def zebra_match_batch(preds: list[str], batch: list[dict]) -> list[bool]:
    """`preds` paired with ZebraLogic example dicts carrying `_solution`."""
    return [zebra_match(p, ex["_solution"]) for p, ex in zip(preds, batch)]


class ZebraLogicVerifier:
    """Exact-match verifier for ZebraLogic grid solutions.

    Unlike the old Spider/SQL path there is no database/in-memory lifecycle:
    the unique gold solution ships with every example, so verification is a
    pure structural comparison of the decoded assignment against the gold grid.
    Kept as a class for interface parity with the train loop (it calls
    `verify_batch(preds, batch)`), and as the seat for future partial
    cell-accuracy scoring."""

    def verify_one(self, pred: str, solution: dict) -> bool:
        return zebra_match(pred, solution)

    def verify_batch(self, preds: list[str], batch: list[dict]) -> list[bool]:
        return [self.verify_one(p, ex["_solution"]) for p, ex in zip(preds, batch)]


class ZebraLogicDataset(Dataset):
    """`WildEval/ZebraLogic` (grid_mode) — 1000 NL logic-grid puzzles, 2×2 to
    6×6, programmatically generated with a unique solution. The **held-out eval
    target**: the AE never trains on it (anchoring, gensis §2.2).

    Note the `allenai/ZebraLogicBench` mirror ships the solution **redacted**
    (`___`) as a leaderboard guard — we load `WildEval/ZebraLogic`, which keeps
    the gold grid, and still skip any redacted record defensively.

    Field mapping:
      - `question`  = `puzzle`, the NL clues (the EBM's conditioning input).
      - `answer`    = serialized gold grid — the text the AE round-trips.
      - `_solution` = raw {header, rows} dict, for the exact-match verifier.
      - `_size`/`_id` = puzzle dimensions / id (size filtering, logging).
    """

    def __init__(
        self,
        split: str = "test",
        max_samples: int | None = None,
        min_size: int = 2,
        max_size: int = 6,
        seed: int = 0,
    ):
        ds = load_dataset("WildEval/ZebraLogic", "grid_mode", split=split)
        ds = ds.shuffle(seed=seed)
        self.examples: list[dict] = []
        for ex in ds:
            sol = ex.get("solution")
            if not sol or "___" in str(sol):       # redacted mirror guard
                continue
            houses = len(sol.get("rows", []))
            if houses < min_size or houses > max_size:
                continue
            self.examples.append({
                "question": (ex.get("puzzle") or "").strip(),
                "answer": serialize_zebra_solution(sol),
                "_solution": sol,
                "_size": ex.get("size"),
                "_id": ex.get("id"),
            })
            if max_samples is not None and len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[int(idx)]


# ---------------------------------------------------------------------------
# Mixed dataset — generic, reused when training on multiple corpora
# ---------------------------------------------------------------------------

class MixedDataset(Dataset):
    """Probabilistic mixture of multiple datasets.

    Each `__getitem__(idx)` deterministically picks one of the underlying
    datasets according to `weights`, then samples an example from it using
    `idx` as the RNG seed. Determinism per-index keeps DataLoader workers
    consistent without needing shared state.
    """

    def __init__(self, datasets, weights, length: int = 50_000):
        assert len(datasets) == len(weights) and len(datasets) > 0
        self.datasets = list(datasets)
        total = float(sum(weights))
        self.cum_weights = []
        acc = 0.0
        for w in weights:
            acc += float(w) / total
            self.cum_weights.append(acc)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = random.Random(int(idx))
        r = rng.random()
        d_idx = 0
        for i, c in enumerate(self.cum_weights):
            if r <= c:
                d_idx = i
                break
        d = self.datasets[d_idx]
        return d[rng.randrange(len(d))]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate(batch):
    return {
        "question": [b["question"] for b in batch],
        "answer": [b["answer"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Pre-tokenization wrapper — move tokenization off the train hot path
# ---------------------------------------------------------------------------

class PreTokenizedDataset(Dataset):
    """Wraps a base dataset and tokenizes question/answer once at __init__.

    The main-thread tokenizer call in each train step (BART byte-level BPE in
    `FrozenBartAutoencoder._tokenize`) is single-threaded Python that the GPU
    sits idle through. For static corpora the tokenization is identical every
    epoch, so we do it once up front and have `__getitem__` return the
    pre-tokenized id lists alongside the original text.

    Pair with `collate_pretokenized` to pad-stack into batched tensors.

    `fields` selects which text fields to pre-tokenize. AE training only
    needs the answer; EBM/actor training needs both question and answer.
    """

    def __init__(
        self,
        base: Dataset,
        tokenizer,
        max_q_length: int = 512,
        max_a_length: int = 384,
        fields: tuple[str, ...] = ("question", "answer"),
        chunk_size: int = 1024,
    ):
        self.base = base
        self.pad_id = tokenizer.pad_token_id
        self.fields = tuple(fields)
        self._cache: dict[str, list[np.ndarray]] = {}
        n = len(base)
        for f in self.fields:
            max_len = max_q_length if f == "question" else max_a_length
            cache: list[np.ndarray] = []
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                texts = [base[i][f] for i in range(start, stop)]
                enc = tokenizer(
                    texts,
                    padding=False,
                    truncation=True,
                    max_length=max_len,
                )
                cache.extend(np.asarray(ids, dtype=np.int32) for ids in enc["input_ids"])
                del texts, enc
            self._cache[f] = cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        idx = int(idx)
        ex = dict(self.base[idx])
        for f in self.fields:
            ex[f + "_ids"] = self._cache[f][idx]
        return ex


def make_collate_pretokenized(pad_id: int):
    """Returns a collate_fn that pad-stacks the *_ids arrays into tensors.

    Output batch keys:
      question, answer                          : text lists (unchanged)
      question_input_ids, question_attention_mask : (B, Lq) tensors (if present)
      answer_input_ids,   answer_attention_mask   : (B, La) tensors (if present)
    """
    def _collate(batch):
        out = {
            "question": [b["question"] for b in batch],
            "answer": [b["answer"] for b in batch],
        }
        for f in ("question", "answer"):
            ids_key = f + "_ids"
            if ids_key not in batch[0]:
                continue
            seqs = [torch.as_tensor(b[ids_key], dtype=torch.long) for b in batch]
            ids = torch.nn.utils.rnn.pad_sequence(
                seqs, batch_first=True, padding_value=pad_id,
            )
            mask = (ids != pad_id).long()
            out[f + "_input_ids"] = ids
            out[f + "_attention_mask"] = mask
        return out
    return _collate
