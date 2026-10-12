"""Persisting generation runs as JSONL under a results root."""
from __future__ import annotations

import json
import os
from pathlib import Path


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
