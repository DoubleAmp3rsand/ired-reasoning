"""Dataset loaders for the IRED reasoning project.

The task substrate is **text-to-SQL**, not Python. Measured vanilla-AE
round-trip floors (`scripts/{bart,sql}_reconstruction_baseline.py`) showed
the frozen autoencoder cannot reconstruct Python at all — its dominant
failure is whitespace/newline collapse, and Python whitespace is syntax, so
every drift is fatal (bart-base MBPP/HumanEval: 0% exact, CER ~0.9). SQL is
whitespace- and case-insensitive, so the same AE mangling is harmless: the
*untrained* bart-base floor on Spider is already 44% normalized-exact / 0.16
CER. That gap is why the structured-generation target moved from code to SQL —
it keeps an external execution verifier and a high-volume latent while
dropping the surface form the AE can't preserve.

Natural-language / code corpora (AE pretraining mix):

- `OpenWebTextDataset` — generic natural-language pretraining corpus for the
  frozen-BART autoencoder.
- `CodeSearchNetDataset` — Python functions, GitHub 2019 scrape. Legacy code
  slice; retained but no longer the primary domain.

Text-to-SQL corpora (the current task):

- `GretelText2SQLDataset` — `gretelai/synthetic_text_to_sql`, 100k fully
  *synthetic* NL→SQL pairs across ~100 domains. **AE-training SQL source**,
  mixed 50/50 with OpenWebText. Synthetic generation (Gretel Navigator) over
  disjoint database schemas means it shares no examples with Spider's
  human-annotated corpus — example-level contamination of the eval set is
  structurally ruled out, the same guarantee CodeSearchNet got from its 2019
  temporal cutoff.
- `SpiderText2SQLDataset` — Yale's Spider, human-annotated NL→SQL over 200
  databases. **Held-out eval source.** The AE never trains on it, preserving
  the gensis §2.2 anchoring property (the latent space is not shaped by the
  eval distribution).

Legacy Python task corpora (kept for the code-execution path / ablations):

- `MBPPDataset`, `HumanEvalDataset` — execution-verified Python benchmarks.

SQL reconstruction is scored with `normalize_sql` + `sql_match_batch`
(whitespace/case-insensitive string match — the correctness-relevant
comparison for SQL). Python correctness still goes through `verify_code`,
which executes the decoded answer against a per-example test fixture in a
contained subprocess. Execution accuracy against the Spider databases
(result-set equality) is the eventual hard SQL metric and is not yet wired.
"""
from __future__ import annotations

import os
import random
import re
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import zipfile
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


_SYSTEMD_RUN = shutil.which("systemd-run")
_systemd_scope_ok: bool | None = None  # cached probe result


def _can_use_systemd_scope() -> bool:
    """One-time probe: can we open a transient `--user` cgroup scope? Needs a
    running user systemd manager + session bus. Cached after first call."""
    global _systemd_scope_ok
    if _systemd_scope_ok is None:
        _systemd_scope_ok = False
        if _SYSTEMD_RUN:
            try:
                p = subprocess.run(
                    [_SYSTEMD_RUN, "--user", "--scope", "--quiet", "--collect",
                     "-p", "MemoryMax=64M", "-p", "TasksMax=8", "--", "true"],
                    capture_output=True, timeout=10,
                )
                _systemd_scope_ok = (p.returncode == 0)
            except Exception:
                _systemd_scope_ok = False
    return _systemd_scope_ok


def _run_python_script(
    script: str, timeout: float = 5.0, mem_limit_gb: float = 2.0, max_tasks: int = 8
) -> tuple[bool, str]:
    """Execute `script` as a standalone Python file and return (passed, stderr).

    `passed` is True iff the subprocess exits 0 within `timeout`. `stderr` is
    a stderr snippet useful for diagnostics; empty on success. Uses a temp
    file rather than `python -c` so tracebacks have real line numbers.

    Containment. A single verification must not be able to multiply itself:
    `timeout` bounds wall-clock but NOT memory, and a mis-reconstructed
    prediction can `fork()` (multiprocessing / recursion) to spawn many
    children that individually respect a per-process cap yet collectively
    exhaust host RAM. So we bound *this whole invocation's* subtree:

      - `MemoryMax` (cgroup) caps total memory of the program AND anything it
        spawns — `mem_limit_gb`, set <= 0 to disable.
      - `TasksMax=max_tasks` (small) stops it spawning more than a handful of
        processes/threads, i.e. forbids fork bombs while allowing a couple of
        library threads.
      - `RuntimeMaxSec` makes *systemd itself* tear the scope down, so a
        timed-out forking program can't be orphaned and accumulate across
        calls (the failure mode that previously summed to a host OOM).

    The caller (`verify_code_batch`) caps how many of these run at once, so the
    eval's total ceiling is `max_workers * mem_limit_gb`, deterministically.
    When systemd scopes aren't available we fall back to per-process RLIMIT_AS
    (memory) + RLIMIT_NPROC (no new processes) — weaker (NPROC is per-UID) but
    still blocks the single-process runaway and most forking.

    Safety note: this is *not* a full sandbox. It runs arbitrary decoded code
    with the same privileges as the parent process. Acceptable for offline
    eval on benchmark corpora (MBPP/HumanEval); do not point it at adversarial
    input.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        path = f.name

    cmd = [sys.executable, path]
    preexec = None
    if mem_limit_gb and mem_limit_gb > 0:
        if _can_use_systemd_scope():
            cap_mb = max(int(mem_limit_gb * 1024), 64)
            runtime_max = max(int(timeout) + 2, 2)  # systemd-side teardown backstop
            cmd = [
                _SYSTEMD_RUN, "--user", "--scope", "--quiet", "--collect",
                "-p", f"MemoryMax={cap_mb}M", "-p", "MemorySwapMax=0",
                "-p", f"TasksMax={max_tasks}", "-p", f"RuntimeMaxSec={runtime_max}",
                "--", *cmd,
            ]
        else:
            nbytes = int(mem_limit_gb * 1024 ** 3)

            def preexec():  # runs in the child, after fork, before exec
                resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
                # Forbid this process from forking — bounds the subtree to 1.
                resource.setrlimit(resource.RLIMIT_NPROC, (max_tasks, max_tasks))

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=preexec,
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


def verify_code(
    pred_text: str, example: dict, timeout: float = 5.0,
    mem_limit_gb: float = 2.0, max_tasks: int = 8,
) -> bool:
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
    ok, _ = _run_python_script(
        script, timeout=timeout, mem_limit_gb=mem_limit_gb, max_tasks=max_tasks
    )
    return ok


def verify_code_batch(
    preds: list[str],
    examples: list[dict],
    timeout: float = 5.0,
    max_workers: int = 4,
    mem_limit_gb: float = 2.0,
    max_tasks: int = 8,
) -> list[bool]:
    """Verify a batch of decoded programs under a fixed execution budget.

    `max_workers` is a hard ceiling on how many verification subprocesses run
    at once; entries are evaluated iteratively through that fixed pool rather
    than fanning out one child per entry. Combined with the per-invocation
    containment in `_run_python_script` (each program can't fork past
    `max_tasks` and can't allocate past `mem_limit_gb`), the eval's total
    resource ceiling is deterministic:

        peak processes <= max_workers * max_tasks
        peak memory    <= max_workers * mem_limit_gb

    so no reconstruction — however pathological — can OOM the host. This
    matters because eval runs the *model's generated* code, which (unlike the
    gold code seen in training) can be arbitrary.
    """
    if not preds:
        return []
    if max_workers <= 1 or len(preds) == 1:
        return [verify_code(p, e, timeout=timeout, mem_limit_gb=mem_limit_gb,
                            max_tasks=max_tasks)
                for p, e in zip(preds, examples)]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(
            lambda pe: verify_code(pe[0], pe[1], timeout=timeout,
                                   mem_limit_gb=mem_limit_gb, max_tasks=max_tasks),
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
# Text-to-SQL — primary task substrate
# ---------------------------------------------------------------------------

_SQL_WS_RE = re.compile(r"\s+")


def normalize_sql(s: str) -> str:
    """Lowercase + collapse all whitespace runs to a single space, stripped.

    Captures the formatting freedom SQL correctness tolerates: keywords are
    case-insensitive and whitespace between tokens is irrelevant, so a
    reconstruction that differs only in case/spacing is still the same query.
    String/identifier case can matter in principle, so normalized-exact is an
    upper-bound proxy for correctness — pair it with raw exact (lower bound)
    and, eventually, execution accuracy (the ground truth).
    """
    return _SQL_WS_RE.sub(" ", s).strip().lower()


def sql_match(pred: str, gold: str, normalize: bool = True) -> bool:
    """Reconstruction match for a single (pred, gold) SQL pair."""
    if normalize:
        return normalize_sql(pred) == normalize_sql(gold)
    return pred.strip() == gold.strip()


def sql_match_batch(
    preds: list[str], golds: list[str], normalize: bool = True
) -> list[bool]:
    """Batch SQL reconstruction match — the SQL analog of `verify_code_batch`.

    Returns one bool per pair. `normalize=True` (default) is the
    whitespace/case-insensitive comparison that tracks SQL correctness; set
    `normalize=False` for raw byte-exact. (No subprocess: a string compare is
    enough until execution-against-the-DB accuracy is wired, at which point a
    `verify_sql_batch` mirroring `verify_code_batch` will live here.)
    """
    return [sql_match(p, g, normalize=normalize) for p, g in zip(preds, golds)]


class GretelText2SQLDataset(Dataset):
    """`gretelai/synthetic_text_to_sql` — synthetic NL→SQL, the AE-training
    SQL source.

    100k fully synthetic examples (generated by Gretel Navigator) spanning
    ~100 domains with schema context. Because the queries are model-generated
    over synthetic schemas — never sourced from Spider's human-annotated
    databases — there is no example-level overlap with the Spider eval set:
    contamination is ruled out by construction, the structural analog of
    CodeSearchNet's 2019-vs-2021 temporal cutoff.

    Mixed 50/50 with `OpenWebTextDataset` to teach the frozen-BART autoencoder
    SQL surface structure without ever showing it the eval distribution
    (gensis §2.2 anchoring).

    Field mapping:
      - `answer`   = the `sql` query — the text the AE round-trips.
      - `question` = `sql_prompt`, the NL request (unused by the AE; the EBM's
        conditioning later).
      - `_schema`  = `sql_context` (CREATE TABLE + seed rows), kept for future
        schema-grounded conditioning / execution.

    Unlike the prose/code corpora, over-long entries are **skipped, not
    truncated** — truncating mid-query yields invalid SQL, which would poison
    both reconstruction targets and any future execution check.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: int = 50_000,
        min_chars: int = 20,
        max_chars: int = 1_000,
        seed: int = 0,
    ):
        ds = load_dataset("gretelai/synthetic_text_to_sql", split=split)
        ds = ds.shuffle(seed=seed)
        self.examples: list[dict] = []
        for ex in ds:
            sql = (ex.get("sql") or "").strip()
            if len(sql) < min_chars or len(sql) > max_chars:
                continue
            self.examples.append({
                "question": (ex.get("sql_prompt") or "").strip(),
                "answer": sql,
                "_schema": (ex.get("sql_context") or "").strip(),
                "_domain": ex.get("domain"),
            })
            if max_samples is not None and len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[int(idx)]


class SpiderText2SQLDataset(Dataset):
    """Yale Spider — human-annotated NL→SQL, the held-out eval source.

    HuggingFace `spider`: train=7000, validation=1034, over 200 databases /
    138 domains. We use the validation split as the AE/EBM eval corpus and
    never train the AE on it (anchoring, gensis §2.2). Independent of the
    synthetic Gretel training source (different provenance, disjoint schemas),
    so reconstruction here is genuine held-out generalization.

    Field mapping:
      - `answer`   = `query`, the gold SQL the AE must round-trip.
      - `question` = the NL `question`.
      - `_db_id`   = the database id, needed for the eventual execution
        verifier (run pred vs gold against the DB, compare result sets).

    Over-long queries are skipped rather than truncated (invalid-SQL guard),
    same as the Gretel source.
    """

    def __init__(
        self,
        split: str = "validation",
        max_samples: int | None = None,
        min_chars: int = 10,
        max_chars: int = 1_000,
        seed: int = 0,
    ):
        assert split in ("train", "validation")
        ds = load_dataset("spider", split=split)
        ds = ds.shuffle(seed=seed)
        self.examples: list[dict] = []
        for ex in ds:
            query = (ex.get("query") or "").strip()
            question = (ex.get("question") or "").strip()
            if not query or not question:
                continue
            if len(query) < min_chars or len(query) > max_chars:
                continue
            self.examples.append({
                "question": question,
                "answer": query,
                "_db_id": ex.get("db_id"),
            })
            if max_samples is not None and len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[int(idx)]


# ---------------------------------------------------------------------------
# Spider execution verifier — result-set equality against the real databases
# ---------------------------------------------------------------------------

# The HF `spider` dataset ships only queries/questions; the SQLite databases
# come from this mirror of the official Spider release (data.zip, ~105 MB,
# contains database/<db_id>/<db_id>.sqlite).
_SPIDER_DB_REPO = "SALT-NLP/spider_VALUE"
_SPIDER_DB_FILE = "data.zip"
_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _spider_first_statement(sql: str) -> str:
    """First SQL statement, stripped of a trailing semicolon.

    Spider gold and (almost all) model predictions are a single statement;
    splitting on the first ';' guards against a stray trailing statement
    confusing sqlite3 (one statement per `execute`). Semicolons inside string
    literals are rare here and accepted as a known limitation.
    """
    s = sql.strip()
    if ";" in s:
        s = s.split(";", 1)[0]
    return s.strip()


def _canon_cell(c):
    """Canonicalize one result cell so trivially-equal values compare equal."""
    if isinstance(c, bytes):
        return c.decode("utf-8", "replace")
    if isinstance(c, float):
        # Collapse int-valued floats and fp noise so 3, 3.0, 2.9999999 agree.
        return round(c, 6)
    return c


def _canon_rows(rows, ordered: bool):
    """Comparable form of a result set: multiset of canonical row-tuples,
    order-sensitive only when the gold query had an ORDER BY."""
    canon = [tuple(_canon_cell(c) for c in row) for row in rows]
    return canon if ordered else sorted(canon, key=repr)


class SpiderExecutionVerifier:
    """Execution-accuracy verifier for Spider SQL.

    Runs the predicted and gold queries against the **real** Spider SQLite
    database for the example and compares result sets ("table differences").
    A prediction passes iff it executes without error and returns the same
    result set as the gold; an invalid/erroring prediction fails.

    Lifecycle (mirrors the training eval loop; see `train_autoencoder`):

        v = SpiderExecutionVerifier()        # once at training start: fetch DBs
        ...
        v.setup(db_ids)                       # at each eval-loop entry: POPULATE
        results = v.verify_batch(preds, batch)
        v.teardown()                          # at eval-loop exit: DEPOPULATE

    POPULATE loads each needed database from disk into a persistent in-memory
    SQLite connection (the "template"). Each query then runs against a *fresh
    clone* of that template, so a destructive prediction (`DROP`/`DELETE`/
    `UPDATE` — model output is arbitrary) can only corrupt a throwaway copy,
    never the on-disk file or another example. DEPOPULATE closes every
    in-memory connection so the databases don't sit in RAM between evals.

    Comparison is the standard execution-match proxy: rows compared as a
    multiset (order-insensitive) unless the gold has `ORDER BY` (then ordered).
    Column order is compared as-is — the full Spider test-suite's
    column-permutation handling is not reproduced, so this is a slight
    under-count, not an over-count. Unverifiable examples (missing DB file, or
    a gold query that itself errors) return `None` and are excluded from the
    pass/total denominator rather than scored as failures.

    Safety: runs arbitrary model SQL in-process against an in-memory copy with
    a watchdog `interrupt()` timeout and a fetched-row cap. It cannot corrupt
    disk, but it is not a CPU/RAM sandbox — fine for offline Spider eval.
    """

    def __init__(self, db_root: str | None = None, timeout: float = 5.0,
                 max_rows: int = 100_000, cache_dir: str | None = None):
        self.timeout = timeout
        self.max_rows = max_rows
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/ired_spider")
        self.db_root = db_root or self._ensure_dbs()
        self._templates: dict[str, sqlite3.Connection] = {}
        self._gold_cache: dict[tuple, object] = {}
        self.missing: set[str] = set()

    # -- database sourcing --------------------------------------------------
    def _ensure_dbs(self) -> str:
        """Download + extract the Spider SQLite databases once; return the
        `database/` root. Cached under `cache_dir`."""
        extracted = os.path.join(self.cache_dir, "extracted")
        db_root = self._find_db_root(extracted)
        if db_root:
            return db_root
        from huggingface_hub import hf_hub_download
        os.makedirs(self.cache_dir, exist_ok=True)
        zpath = hf_hub_download(_SPIDER_DB_REPO, _SPIDER_DB_FILE,
                                repo_type="dataset", local_dir=self.cache_dir)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(extracted)
        db_root = self._find_db_root(extracted)
        if not db_root:
            raise RuntimeError(f"Spider databases not found after extracting {zpath}")
        return db_root

    @staticmethod
    def _find_db_root(extracted: str) -> str | None:
        if not os.path.isdir(extracted):
            return None
        for dirpath, dirnames, _ in os.walk(extracted):
            if os.path.basename(dirpath) == "database" and dirnames:
                return dirpath
        return None

    def _db_path(self, db_id: str) -> str:
        return os.path.join(self.db_root, db_id, f"{db_id}.sqlite")

    # -- populate / depopulate ---------------------------------------------
    def setup(self, db_ids) -> None:
        """Load each unique db_id into a persistent in-memory template DB."""
        self.missing.clear()
        for db_id in set(db_ids):
            if not db_id or db_id in self._templates:
                continue
            path = self._db_path(db_id)
            if not os.path.isfile(path):
                self.missing.add(db_id)
                continue
            src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            mem = sqlite3.connect(":memory:")
            src.backup(mem)
            src.close()
            self._templates[db_id] = mem

    def teardown(self) -> None:
        for con in self._templates.values():
            try:
                con.close()
            except Exception:
                pass
        self._templates.clear()
        self._gold_cache.clear()
        self.missing.clear()

    # -- execution ----------------------------------------------------------
    def _clone(self, db_id: str) -> sqlite3.Connection:
        dst = sqlite3.connect(":memory:")
        dst.text_factory = lambda b: b.decode("utf-8", "replace")
        self._templates[db_id].backup(dst)
        return dst

    def _execute(self, conn: sqlite3.Connection, sql: str):
        """(ok, rows) — run one statement with an interrupt timeout + row cap."""
        sql = _spider_first_statement(sql)
        if not sql:
            return False, None
        timer = threading.Timer(self.timeout, conn.interrupt)
        timer.start()
        try:
            cur = conn.execute(sql)
            return True, cur.fetchmany(self.max_rows)
        except Exception:
            return False, None
        finally:
            timer.cancel()

    def _gold_result(self, db_id: str, gold: str, ordered: bool):
        key = (db_id, gold)
        if key not in self._gold_cache:
            con = self._clone(db_id)
            try:
                ok, rows = self._execute(con, gold)
            finally:
                con.close()
            self._gold_cache[key] = _canon_rows(rows, ordered) if ok else None
        return self._gold_cache[key]

    def verify_one(self, pred: str, gold: str, db_id: str):
        """True/False pass, or None when the example can't be judged."""
        if not db_id or db_id not in self._templates:
            return None                       # missing DB -> unverifiable
        ordered = bool(_ORDER_BY_RE.search(gold))
        gold_canon = self._gold_result(db_id, gold, ordered)
        if gold_canon is None:
            return None                       # gold itself errored -> unjudgeable
        con = self._clone(db_id)
        try:
            ok, rows = self._execute(con, pred)
        finally:
            con.close()
        if not ok:
            return False                      # invalid / erroring prediction
        return _canon_rows(rows, ordered) == gold_canon

    def verify_batch(self, preds: list[str], batch: list[dict]):
        """`preds` paired with `batch` (Spider example dicts carrying `_db_id`
        and gold `answer`). Returns list[bool | None] — None = excluded."""
        return [
            self.verify_one(p, ex.get("answer", ""), ex.get("_db_id"))
            for p, ex in zip(preds, batch)
        ]


# ---------------------------------------------------------------------------
# ZebraLogic — natural-language logic-grid puzzles (the current eval target)
# ---------------------------------------------------------------------------
#
# Why ZebraLogic over SQL: the AE round-trip is the binding ceiling for any
# execution-graded structured target (vanilla bart-base on Spider already beats
# the trained AE on exec-acc — the lossy latent drops the load-bearing tokens
# SQL needs and copy/schema tricks would hollow the latent the EBM conditions
# on). ZebraLogic keeps the two properties we need — an irreducibly
# *natural-language* problem (so the language AE is load-bearing, not a wrapper
# over a one-hot grid the way IRED's original CSP encoders were) and an exactly
# *verifiable* answer — while the answer's surface is low-entropy, fixed-format,
# and built from OWT-frequent common nouns, exactly the regime the frozen-anchor
# AE reconstructs well. It is, in effect, "Sudoku stated in English."
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

    Unlike `SpiderExecutionVerifier` there is no database/in-memory lifecycle:
    the unique gold solution ships with every example, so verification is a
    pure structural comparison of the decoded assignment against the gold grid.
    Kept as a class for interface parity with the SQL path (the train loop
    calls `verify_batch(preds, batch)`), and as the seat for future partial
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
    the grid surface form, the anchoring-safe analog of mixing synthetic SQL.
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
