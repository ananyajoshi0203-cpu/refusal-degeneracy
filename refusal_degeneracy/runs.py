"""Persisting generation runs as JSONL under a results root."""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import torch

T = TypeVar("T")


def results_root() -> Path:
    return Path(os.environ.get("RD_RESULTS", "results"))


def run_path(model_name: str, run: str) -> Path:
    path = results_root() / model_name.split("/")[-1] / f"{run}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_run(model_name, run, prompts, replies, split, meta=None) -> Path:
    path = run_path(model_name, run)
    with path.open("w") as f:
        for prompt, reply in zip(prompts, replies, strict=True):
            f.write(json.dumps({"split": split, "prompt": prompt, "reply": reply, "meta": meta or {}}) + "\n")
    return path


def load_run(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def completed_replies(model_name: str, run: str, prompts: Sequence[str]) -> list[str] | None:
    """Replies of a saved run, or None unless it covers exactly ``prompts`` in order. A run cut
    off mid-write, or saved under a different ``--limit``, is not reused."""
    path = run_path(model_name, run)
    if not path.exists():
        return None
    rows = load_run(path)
    if [row["prompt"] for row in rows] != list(prompts):
        return None
    return [row["reply"] for row in rows]


def checkpointed(path: Path, resume: bool, compute: Callable[[], T]) -> T:
    """Load ``path`` when resuming and it exists; otherwise compute and save it before returning,
    so a session that dies later keeps every stage that finished."""
    if resume and path.exists():
        return torch.load(path, weights_only=False)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    return value
