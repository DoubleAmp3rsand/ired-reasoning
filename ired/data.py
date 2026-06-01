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
- `SyntheticZebraGridDataset` — random ZebraLogic-style solution grids for AE
  format exposure. Contamination-free by construction (random assignments over
  generic value pools), so it can be mixed into AE training without touching the
  anchoring property. Its sole job is to teach the frozen-BART AE the grid
  surface form — the anchoring-safe analog of mixing synthetic SQL.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from difflib import SequenceMatcher

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
    """Collapse whitespace + lowercase + strip surrounding punctuation — the
    cell-comparison canonical form. The surrounding-punctuation strip clears the
    transduction scaffolding noise the AE emits (`"arnold`, `/soup`, `water `)
    so it doesn't fail a solved cell (gensis §5.5); internal spaces/hyphens (e.g.
    "grilled cheese") are preserved."""
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)


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
            # Split into leading attribute token + value on whatever separator
            # survived ('=', or a corrupted '/' '-' or bare space) — so a mangled
            # delimiter degrades to a snappable key/value, not a dropped cell.
            km = re.match(r"[^a-z0-9]*([a-z0-9][a-z0-9 -]*?)[^a-z0-9]+(.+)$",
                          part.strip(), re.IGNORECASE)
            if not km:
                continue
            cells[zebra_normalize(km.group(1))] = zebra_normalize(km.group(2))
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


def zebra_gold_alphabet(solution: dict):
    """Closed answer alphabet for a puzzle: legal house ids, legal attribute
    keys, and the legal value set per attribute — all normalized. This is the
    `{house, attr, value}` vocabulary the puzzle itself fixes (and what makes it
    uniquely solvable), so snapping decoded cells onto it (gensis §5.5) repairs
    transduction noise without inventing answers."""
    header = solution["header"]
    attrs = header[1:]
    legal_attrs = [zebra_normalize(a) for a in attrs]
    legal_vals: dict[str, set] = {za: set() for za in legal_attrs}
    for row in solution["rows"]:
        for a, v in zip(attrs, row[1:]):
            legal_vals[zebra_normalize(a)].add(zebra_normalize(v))
    legal_houses = [zebra_normalize(row[0]) for row in solution["rows"]]
    return legal_attrs, legal_vals, legal_houses


def _snap_to_legal(token: str, candidates, *, threshold: float = 0.6,
                   margin: float = 0.15) -> str:
    """Snap a decoded token to its nearest legal candidate (gensis §5.5).

    Returns the token unchanged when it is already legal, or when the match is
    *ambiguous* — nearest below `threshold`, or within `margin` of the runner-up.
    That ambiguity guard is the §5.2 protection: transduction corruption is
    off-vocabulary so it snaps cleanly, but a genuine reasoning error is a
    *different legal value* (exact, so returned as-is → still fails) or a true
    near-miss between two legal values (ambiguous → left raw → still fails). It
    forgives surface noise without laundering a wrong assignment."""
    cand = list(candidates)
    if not cand or token in cand:
        return token
    scored = sorted((SequenceMatcher(None, token, c).ratio(), c) for c in cand)
    best_r, best_c = scored[-1]
    second_r = scored[-2][0] if len(scored) > 1 else 0.0
    if best_r >= threshold and (best_r - second_r) >= margin:
        return best_c
    return token


def zebra_match(pred_text: str, gold_solution: dict, *, snap: bool = True) -> bool:
    """Puzzle-level exact match: every gold (house, attr) cell must be present
    and equal in the decoded answer. ZebraLogic solutions are unique, so this
    all-or-nothing check *is* the correctness metric (an unparseable or
    partially-correct grid is not a solved puzzle).

    With `snap=True` (default) each decoded house/attr/value is first snapped to
    the puzzle's known alphabet (gensis §5.5) so serialization noise doesn't fail
    a solved grid; `snap=False` recovers the strict byte-literal metric."""
    gold = zebra_gold_mapping(gold_solution)
    pred = parse_zebra_solution(pred_text)
    if not pred:
        return False
    if snap:
        legal_attrs, legal_vals, legal_houses = zebra_gold_alphabet(gold_solution)
        snapped: dict[str, dict[str, str]] = {}
        for house, cells in pred.items():
            sh = _snap_to_legal(house, legal_houses)
            scells: dict[str, str] = {}
            for k, v in cells.items():
                sk = _snap_to_legal(k, legal_attrs)
                sv = _snap_to_legal(v, legal_vals.get(sk, ()))
                scells[sk] = sv
            snapped[sh] = scells
        pred = snapped
    for house, cells in gold.items():
        pcells = pred.get(house)
        if pcells is None:
            return False
        for attr, val in cells.items():
            if pcells.get(attr) != val:
                return False
    return True


def zebra_match_batch(preds: list[str], batch: list[dict], *,
                      snap: bool = True) -> list[bool]:
    """`preds` paired with ZebraLogic example dicts carrying `_solution`."""
    return [zebra_match(p, ex["_solution"], snap=snap)
            for p, ex in zip(preds, batch)]


class ZebraLogicVerifier:
    """Exact-match verifier for ZebraLogic grid solutions.

    Unlike the old Spider/SQL path there is no database/in-memory lifecycle:
    the unique gold solution ships with every example, so verification is a
    pure structural comparison of the decoded assignment against the gold grid.
    Kept as a class for interface parity with the train loop (it calls
    `verify_batch(preds, batch)`), and as the seat for future partial
    cell-accuracy scoring."""

    def __init__(self, snap: bool = True):
        # snap=True grades the closed-vocabulary assignment (gensis §5.5);
        # snap=False is the strict byte-literal metric.
        self.snap = snap

    def verify_one(self, pred: str, solution: dict) -> bool:
        return zebra_match(pred, solution, snap=self.snap)

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
# Synthetic ZebraLogic grids — format exposure for AE training
# ---------------------------------------------------------------------------

# Value pools for synthetic grids — modelled on ZebraLogic's categories so the
# surface form (tokens + structure) matches, while the *assignments* are random
# and therefore share no puzzle with the eval set.
_ZEBRA_SYNTH_POOLS = {
    "Name": ["Peter", "Alice", "Bob", "Eric", "Arnold", "Carol", "Samuel", "Diana"],
    "Nationality": ["norwegian", "german", "dane", "brit", "swede", "mexican", "chinese", "french"],
    "BookGenre": ["mystery", "science fiction", "romance", "fantasy", "biography", "poetry", "history", "thriller"],
    "Food": ["pizza", "grilled cheese", "spaghetti", "stew", "soup", "stir fry", "tacos", "sushi"],
    "Color": ["red", "green", "blue", "yellow", "white", "purple", "brown", "orange"],
    "Animal": ["dog", "cat", "bird", "fish", "horse", "rabbit", "zebra", "cow"],
    "Drink": ["water", "tea", "coffee", "milk", "juice", "wine", "beer", "soda"],
    "Hobby": ["painting", "cooking", "gardening", "photography", "cycling", "knitting", "fishing", "chess"],
}


class SyntheticZebraGridDataset(Dataset):
    """Random ZebraLogic-style solution grids for AE format exposure.

    Contamination-free by construction (random assignments over generic value
    pools — never the eval puzzles), so it can be mixed into AE training without
    touching the anchoring property. Only the **answer** serialization is
    produced (there are no clues); its sole job is to teach the frozen-BART AE
    the grid surface form.
    """

    def __init__(
        self,
        max_samples: int = 50_000,
        min_size: int = 2,
        max_size: int = 6,
        min_attrs: int = 2,
        max_attrs: int = 6,
        seed: int = 0,
    ):
        rng = random.Random(seed)
        cats = list(_ZEBRA_SYNTH_POOLS.keys())
        max_attrs = min(max_attrs, len(cats))
        self.examples: list[dict] = []
        for _ in range(max_samples):
            n = rng.randint(min_size, max_size)
            a = rng.randint(min_attrs, max_attrs)
            chosen = rng.sample(cats, a)
            colvals = {c: rng.sample(_ZEBRA_SYNTH_POOLS[c], n) for c in chosen}
            rows = [[str(i + 1)] + [colvals[c][i] for c in chosen] for i in range(n)]
            sol = {"header": ["House"] + chosen, "rows": rows}
            self.examples.append({
                "question": "",
                "answer": serialize_zebra_solution(sol),
                "_solution": sol,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[int(idx)]


# ---------------------------------------------------------------------------
# Synthetic ZebraLogic puzzles WITH clues — the EBM solver-training corpus
# ---------------------------------------------------------------------------

class ClueZebraGridDataset(Dataset):
    """Full logic-grid puzzles (NL clues + unique solution) for EBM training.

    `WildEval/ZebraLogic` ships only a 1000-puzzle *test* split — no training
    corpus — and `SyntheticZebraGridDataset` has no clues, so neither can train a
    solver. This dataset closes that gap: it samples a random solution grid (over
    generic value pools, so it is eval-disjoint by construction, like
    `SyntheticZebraGridDataset`) and runs the vendored `generate_puzzle`
    (`ired/puzzle_gen.py`) to obtain a minimal, uniqueness-verified clue set,
    rendered to ZebraLogic-style **prose** (gensis §5.5).

    Because the clues come from a *different* generator than the WildEval eval,
    generator→WildEval transfer measures solving rather than generator-fitting
    (the §7 protocol). Record fields match `ZebraLogicDataset`:
      - `question`  = prose clue block (EBM conditioning).
      - `answer`    = serialized gold grid (the AE round-trips this).
      - `_solution` = {header, rows} dict for the exact-match verifier.
      - `_size`     = number of houses (for size-holdout eval).

    Generation is slow (clue minimization is ~tens of ms/puzzle), so the
    materialized corpus is cached to disk keyed by every generation parameter
    (and the value pools); a second run with the same config loads instantly.
    Cache lives under `cache_dir` (default `~/.cache/ired-reasoning/clue_zebra`);
    pass `use_cache=False` to disable.
    """

    _CACHE_FIELDS = (
        "max_samples", "min_size", "max_size", "min_attrs", "max_attrs",
        "min_level", "max_level", "minimal_conditions",
        "max_seconds_for_minimizing", "prose", "seed",
    )

    def __init__(
        self,
        max_samples: int = 20_000,
        min_size: int = 2,
        max_size: int = 4,
        min_attrs: int = 3,
        max_attrs: int = 5,
        min_level: int = 5,
        max_level: int = 8,
        minimal_conditions: bool = True,
        max_seconds_for_minimizing: float = 2.0,
        prose: bool = True,
        seed: int = 0,
        cache_dir: str | None = None,
        use_cache: bool = True,
    ):
        from .puzzle_gen import generate_puzzle, render_premises

        cats = list(_ZEBRA_SYNTH_POOLS.keys())
        max_attrs = min(max_attrs, len(cats))
        self.examples: list[dict] = []

        # --- disk cache lookup -------------------------------------------------
        params = {f: locals()[f] for f in self._CACHE_FIELDS}
        cpath = None
        if use_cache:
            cdir = cache_dir if cache_dir is not None else os.path.join(
                os.path.expanduser("~"), ".cache", "ired-reasoning", "clue_zebra")
            payload = json.dumps([params, _ZEBRA_SYNTH_POOLS], sort_keys=True)
            digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
            cpath = os.path.join(cdir, f"clue_{digest}.json")
            if os.path.exists(cpath):
                try:
                    with open(cpath) as f:
                        self.examples = json.load(f)
                except (json.JSONDecodeError, OSError):
                    self.examples = []           # corrupt cache → regenerate
            if self.examples:
                return

        # --- generate ----------------------------------------------------------
        rng = random.Random(seed)
        random.seed(seed)                       # generate_puzzle uses global RNG
        attempts = 0
        while len(self.examples) < max_samples and attempts < max_samples * 4:
            attempts += 1
            n = rng.randint(min_size, max_size)
            a = rng.randint(min_attrs, min(max_attrs, len(cats)))
            level = rng.randint(min_level, max_level)
            chosen = rng.sample(cats, a)
            # rows = attributes, cols = houses (the generator's table layout)
            table = [[c] + rng.sample(_ZEBRA_SYNTH_POOLS[c], n) for c in chosen]
            try:
                premises = generate_puzzle(
                    table, level=level,
                    minimal_conditions=minimal_conditions,
                    max_seconds_for_minimizing=max_seconds_for_minimizing,
                )
            except ValueError:
                continue                        # size/level combo rejected upstream
            if not premises:
                continue
            question = render_premises(premises, prose=prose)
            # transpose attribute-major table → house-major solution grid
            sol = {
                "header": ["House"] + chosen,
                "rows": [
                    [str(p + 1)] + [table[i][p + 1] for i in range(len(table))]
                    for p in range(n)
                ],
            }
            self.examples.append({
                "question": question,
                "answer": serialize_zebra_solution(sol),
                "_solution": sol,
                "_size": n,
            })

        # --- persist cache (best-effort, atomic) -------------------------------
        if cpath:
            try:
                os.makedirs(os.path.dirname(cpath), exist_ok=True)
                tmp = f"{cpath}.tmp{os.getpid()}"
                with open(tmp, "w") as f:
                    json.dump(self.examples, f)
                os.replace(tmp, cpath)
            except OSError:
                pass                            # caching is best-effort

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
