"""GSM8K loader.

answer_mode='final' returns just the numeric answer (a few tokens) — the
recommended setup for the first experiment in `gensis.md` Section 8.
answer_mode='full' returns the full reasoning trace + answer.
"""
from __future__ import annotations

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


def collate(batch):
    return {
        "question": [b["question"] for b in batch],
        "answer": [b["answer"] for b in batch],
    }
