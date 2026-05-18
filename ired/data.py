"""GSM8K loader.

answer_mode='final' returns just the numeric answer (a few tokens) — the
recommended setup for the first experiment in `gensis.md` Section 8.
answer_mode='full' returns the full reasoning trace + answer.
"""
from __future__ import annotations

import random
import re

from datasets import load_dataset
from torch.utils.data import Dataset


_FINAL_RE = re.compile(r"####\s*([\-\d,\.]+)")


def extract_final_answer(answer_text: str) -> str:
    m = _FINAL_RE.search(answer_text)
    if not m:
        return answer_text.strip()
    return m.group(1).replace(",", "").strip()


class GSM8KDataset(Dataset):
    def __init__(self, split: str = "train", answer_mode: str = "final"):
        assert split in ("train", "test")
        assert answer_mode in ("final", "full")
        self.data = load_dataset("gsm8k", "main", split=split)
        self.answer_mode = answer_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[int(idx)]
        q = ex["question"]
        if self.answer_mode == "final":
            a = extract_final_answer(ex["answer"])
        else:
            a = ex["answer"]
        return {"question": q, "answer": a}


class WikiTextDataset(Dataset):
    """A filtered slice of WikiText-103 for autoencoder pre-training.

    We train the AE on generic prose so its latent space isn't fitted to the
    same GSM8K examples the diffusion model later sees (gensis.md §8). The
    test loader stays GSM8K so the final_acc gate remains the eval signal.

    `wikitext-103-raw-v1` is ~500 MB, line-structured, and includes lots of
    empty lines and section headers (` = Title = `). We filter to lines with
    at least `min_chars` characters, then keep the first `max_samples`
    (shuffled with `seed`). Texts are returned as `answer` so `collate` and
    the AE encode path are identical to the GSM8K loader.
    """

    def __init__(
        self,
        max_samples: int = 50_000,
        min_chars: int = 200,
        max_chars: int = 4_000,
        seed: int = 0,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        split: str = "train",
    ):
        ds = load_dataset(dataset_name, dataset_config, split=split)
        ds = ds.shuffle(seed=seed)
        self.texts: list[str] = []
        for ex in ds:
            t = ex["text"].strip()
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


def collate(batch):
    return {
        "question": [b["question"] for b in batch],
        "answer": [b["answer"] for b in batch],
    }
