"""Dataset loaders for the IRED reasoning project.

Four corpora live here:

- `OpenWebTextDataset` — generic natural-language pretraining corpus for the
  frozen-BART autoencoder.
- `CodeSearchNetDataset` — Python functions with docstrings, scraped from
  GitHub in 2019. Predates both MBPP (2021) and HumanEval (2021), so
  contamination of the EBM eval sets is structurally impossible. Used as a
  code-domain slice in the AE pretraining mix.
- `MBPPDataset` — Mostly Basic Python Problems. Each example is a natural-
  language task description plus a canonical Python solution and a list of
  `assert` tests. Primary EBM-training corpus.
- `HumanEvalDataset` — OpenAI's HumanEval. Function signature + docstring as
  the prompt; canonical body + a `check(candidate)` test fixture as the
  verifier. Held-out eval corpus.

The EBM is trained on MBPP (or a `MixedDataset` if more sources are added)
and evaluated on MBPP test + HumanEval separately so per-corpus
generalization is visible. Correctness goes through `verify_code`, which
executes the decoded answer against the per-example test fixture in a
subprocess with a timeout.
"""
from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

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
            trust_remote_code=True,
        )
        # Streaming datasets support .shuffle with a buffer; this isn't a true
        # global shuffle but is enough to avoid topic-clustered batches.
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
# CodeSearchNet — Python code slice for AE pretraining
# ---------------------------------------------------------------------------

class CodeSearchNetDataset(Dataset):
    """Python functions from CodeSearchNet for autoencoder pretraining.

    `code_search_net` (Husain et al. 2019) is a GitHub scrape predating MBPP
    (2021) and HumanEval (2021), so the AE pretraining slice cannot contain
    the canonical solutions of either eval benchmark — contamination is
    structurally ruled out by the temporal cutoff. Mix this alongside
    OpenWebText to teach the AE Python syntax / whitespace / docstring
    structure without leaking from the EBM eval sets.

    We pull `whole_func_string` (signature + docstring + body) because it
    matches MBPP's "answer is a complete `def`" shape — the same target
    distribution the EBM is supervised to denoise toward.

    Returned record: `{"question": "", "answer": text}` so `collate` and the
    AE encode path are identical to the OWT / task datasets.
    """

    def __init__(
        self,
        max_samples: int = 50_000,
        min_chars: int = 100,
        max_chars: int = 4_000,
        seed: int = 0,
    ):
        ds = load_dataset(
            "code_search_net",
            "python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        self.texts: list[str] = []
        for ex in ds:
            t = (ex.get("whole_func_string") or "").strip()
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
# Code-execution verifier
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_code(text: str) -> str:
    """Pull a Python code block out of decoded text.

    The decoder may emit raw code, markdown-fenced code, or code with leading
    prose. Strategy:
      1. If there's a ```python ... ``` fence, take its body.
      2. Otherwise, return the stripped text as-is — the verifier will fail
         loudly if it isn't parseable Python.
    """
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _run_python_script(script: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Execute `script` as a standalone Python file and return (passed, stderr).

    `passed` is True iff the subprocess exits 0 within `timeout`. `stderr` is
    a stderr snippet useful for diagnostics; empty on success. Uses a temp
    file rather than `python -c` so tracebacks have real line numbers.

    Safety note: this is *not* a sandbox. It runs arbitrary decoded code with
    the same privileges as the parent process. Acceptable for offline eval on
    benchmark corpora (MBPP/HumanEval); do not point it at adversarial input.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = (r.returncode == 0)
        err = "" if ok else (r.stderr or "")[-400:]
        return ok, err
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def verify_code(pred_text: str, example: dict, timeout: float = 5.0) -> bool:
    """Execute decoded code against the example's test fixture.

    `example` carries `_test_script` (str) — a Python source template with a
    `{code}` placeholder. We substitute the extracted prediction, run the
    result, and return pass/fail.
    """
    template = example.get("_test_script")
    if not template:
        return False
    code = extract_code(pred_text)
    # Indent the prediction if the template expects it (rare; default
    # placement is at module top level).
    script = template.replace("{code}", code)
    ok, _ = _run_python_script(script, timeout=timeout)
    return ok


def verify_code_batch(
    preds: list[str],
    examples: list[dict],
    timeout: float = 5.0,
    max_workers: int = 4,
) -> list[bool]:
    """Parallel version of `verify_code` over a batch. Each call still forks
    its own `python` subprocess; a thread pool lets the parent wait on many
    at once (subprocess.run drops the GIL while blocked on the child).
    """
    if not preds:
        return []
    if max_workers <= 1 or len(preds) == 1:
        return [verify_code(p, e, timeout=timeout) for p, e in zip(preds, examples)]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(
            lambda pe: verify_code(pe[0], pe[1], timeout=timeout),
            zip(preds, examples),
        ))


# ---------------------------------------------------------------------------
# MBPP — Mostly Basic Python Problems
# ---------------------------------------------------------------------------


def _mbpp_test_script(test_setup: str, test_list: list[str]) -> str:
    """Build a test script template for MBPP. The model's code is spliced
    into `{code}`; assertions follow.
    """
    setup = (test_setup or "").strip()
    tests = "\n".join(t for t in test_list if t)
    parts = []
    if setup:
        parts.append(setup)
    parts.append("{code}")
    parts.append(tests)
    return "\n".join(parts) + "\n"


class MBPPDataset(Dataset):
    """MBPP — Mostly Basic Python Problems.

    HuggingFace: `mbpp`, config `full` (974 examples) or `sanitized` (427).
    We default to `full` for training volume; sanitized is cleaner if
    annotation noise becomes a problem.

    Standard MBPP splits (full):
      - train      :  374 examples (task_id 601–974)
      - validation :   90 examples (task_id 511–600)
      - test       :  500 examples (task_id  11–510)
      - prompt     :   10 examples (task_id   1– 10), reserved for few-shot

    Our `question` field is the description, optionally suffixed with the
    *first* assertion so the function signature is unambiguous (a description
    like "Write a function to compute the LCM" doesn't tell you the function
    name or argument order — the assertion does).

    Our `answer` field is the canonical solution `code`, which includes the
    full `def fn(...):` block.

    `_test_script` is the verifier template described above.
    """

    def __init__(
        self,
        split: str = "train",
        config: str = "full",
        max_samples: int | None = None,
        seed: int = 0,
        include_signature_in_question: bool = True,
    ):
        assert split in ("train", "validation", "test", "prompt")
        ds = load_dataset("mbpp", config, split=split)
        ds = ds.shuffle(seed=seed)
        self.examples: list[dict] = []
        for ex in ds:
            text = (ex.get("text") or "").strip()
            code = (ex.get("code") or "").strip()
            test_list = list(ex.get("test_list") or [])
            test_setup = ex.get("test_setup_code") or ""
            if not text or not code or not test_list:
                continue

            if include_signature_in_question and test_list:
                q_text = f"{text}\nExample: {test_list[0].strip()}"
            else:
                q_text = text

            self.examples.append({
                "question": q_text,
                "answer": code,
                "_test_script": _mbpp_test_script(test_setup, test_list),
                "_task_id": ex.get("task_id"),
            })
            if max_samples is not None and len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[int(idx)]
        # collate only keeps question/answer; metadata is read via dataset[idx]
        # in eval loops, mirroring the BBH-era pattern.
        return ex


# ---------------------------------------------------------------------------
# HumanEval — held-out eval corpus
# ---------------------------------------------------------------------------


def _humaneval_test_script(test: str, entry_point: str) -> str:
    """HumanEval test fixture template. Body convention: the model's
    prediction is a complete function (signature + body), and we then invoke
    `check(entry_point)` against the gold test suite.
    """
    return f"{{code}}\n{test}\ncheck({entry_point})\n"


class HumanEvalDataset(Dataset):
    """OpenAI HumanEval — 164 function-completion problems.

    HuggingFace: `openai_humaneval`. Single split `test`; no upstream
    train/val. We use this as the held-out eval corpus only; never train on it.

    Field mapping:
      - `question` = the prompt (signature + docstring). Already conveys the
        function name and expected behavior.
      - `answer`   = prompt + canonical_solution — the *full* executable
        function. This matches MBPP's "answer is a complete `def`" convention
        so the EBM sees uniform target shape across corpora.
      - `_test_script` is the verifier template.
    """

    def __init__(self, max_samples: int | None = None, seed: int = 0):
        ds = load_dataset("openai_humaneval", split="test")
        ds = ds.shuffle(seed=seed)
        self.examples: list[dict] = []
        for ex in ds:
            prompt = ex.get("prompt") or ""
            canonical = ex.get("canonical_solution") or ""
            test = ex.get("test") or ""
            entry_point = ex.get("entry_point") or ""
            if not prompt or not canonical or not test or not entry_point:
                continue
            answer = prompt + canonical
            self.examples.append({
                "question": prompt.strip(),
                "answer": answer.rstrip() + "\n",
                "_test_script": _humaneval_test_script(test, entry_point),
                "_task_id": ex.get("task_id"),
                "_entry_point": entry_point,
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
    sits idle through. For static corpora (MBPP, HumanEval, OpenWebText,
    CodeSearchNet) the tokenization is identical every epoch, so we do it
    once up front and have `__getitem__` return the pre-tokenized id lists
    alongside the original text.

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
            # Stream through in small chunks — calling the HF fast tokenizer
            # with the full corpus at once can spike RSS hard enough to OOM
            # WSL on large datasets. Per-chunk memory stays bounded.
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                texts = [base[i][f] for i in range(start, stop)]
                enc = tokenizer(
                    texts,
                    padding=False,
                    truncation=True,
                    max_length=max_len,
                )
                # int32 keeps the cache compact (~4× smaller than Python int lists).
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
