"""Prompt datasets and leak-free Find / Validate / Test splits.

Sources and sizes follow Arditi et al. (2024) and Kazemi et al. (2026). Prompts used to
find or validate an intervention never appear in a test set; JailbreakBench reuses AdvBench
and HarmBench prompts, so exact and near-duplicate overlaps are removed before sampling.
"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher
from pathlib import Path

from .config import SplitConfig
from .runs import results_root

TEXT_COLUMNS = ("prompt", "Goal", "goal", "instruction")

RawSets = dict[str, list[str]]
Splits = dict[str, list[str]]


def _text_column(dataset) -> str:
    for column in TEXT_COLUMNS:
        if column in dataset.column_names:
            return column
    raise KeyError(f"no prompt column in {dataset.column_names}")


def _texts(dataset) -> list[str]:
    column = _text_column(dataset)
    return [t.strip() for t in dataset[column] if t and t.strip()]


def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def load_all() -> RawSets:
    from datasets import load_dataset

    alpaca = load_dataset("tatsu-lab/alpaca", split="train").filter(lambda r: r["input"].strip() == "")
    return {
        "advbench": _texts(load_dataset("walledai/AdvBench", split="train")),
        "maliciousinstruct": _texts(load_dataset("walledai/MaliciousInstruct", split="train")),
        "harmbench": _texts(load_dataset("walledai/HarmBench", "standard", split="train")),
        "jbb": _texts(load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")),
        "strongreject": _texts(load_dataset("walledai/StrongREJECT", split="train")),
        "alpaca": _texts(alpaca),
    }


def _near_duplicate(a: str, b: str, threshold: float = 0.9) -> bool:
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    matcher = SequenceMatcher(None, na, nb)
    return matcher.real_quick_ratio() >= threshold and matcher.ratio() >= threshold


def remove_overlap(pool: Sequence[str], blocked: Sequence[str]) -> tuple[list[str], list[str]]:
    kept, dropped = [], []
    for prompt in pool:
        (dropped if any(_near_duplicate(prompt, b) for b in blocked) else kept).append(prompt)
    return kept, dropped


def dedupe(prompts: Sequence[str]) -> list[str]:
    seen, out = set(), []
    for prompt in prompts:
        key = normalize(prompt)
        if key not in seen:
            seen.add(key)
            out.append(prompt)
    return out


def check_no_leaks(splits: Splits) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if a.startswith("test") and b.startswith("test"):
                continue
            shared = {normalize(x) for x in splits[a]} & {normalize(x) for x in splits[b]}
            if shared:
                raise AssertionError(f"{len(shared)} prompts shared between {a} and {b}")


def build_splits(
    config: SplitConfig | None = None, loader: Callable[[], RawSets] = load_all
) -> tuple[Splits, dict]:
    config = config or SplitConfig()
    raw = loader()
    rng = random.Random(config.seed)
    report: dict = {}

    test_jbb, test_sr = dedupe(raw["jbb"]), dedupe(raw["strongreject"])
    tests = test_jbb + test_sr

    val_harmful, dropped = remove_overlap(dedupe(raw["harmbench"]), tests)
    report["harmbench_dropped_overlap_with_tests"] = len(dropped)

    pool, dropped = remove_overlap(dedupe(raw["advbench"] + raw["maliciousinstruct"]), tests + val_harmful)
    report["find_pool_dropped_overlap"] = len(dropped)
    report["find_pool_size"] = len(pool)
    find_harmful = rng.sample(pool, config.n_find)

    picked = rng.sample(dedupe(raw["alpaca"]), config.n_find + config.n_val_harmless)
    find_harmless, val_harmless = picked[: config.n_find], picked[config.n_find :]

    splits: Splits = {
        "find_harmful": find_harmful,
        "find_harmless": find_harmless,
        "val_harmful": val_harmful,
        "val_harmless": val_harmless,
        "test_jbb": test_jbb,
        "test_strongreject": test_sr,
    }
    check_no_leaks(splits)
    report["sizes"] = {k: len(v) for k, v in splits.items()}
    report["seed"] = config.seed
    return splits, report


def save_splits(splits: Splits, report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"report": report, "splits": splits}, indent=1))
    return path


def load_splits(path: str | Path) -> Splits:
    return json.loads(Path(path).read_text())["splits"]


def get_splits(config: SplitConfig | None = None) -> Splits:
    path = results_root() / "splits.json"
    if path.exists():
        return load_splits(path)
    splits, report = build_splits(config)
    save_splits(splits, report, path)
    return splits
