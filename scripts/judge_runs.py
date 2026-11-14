"""Judge every saved reply file of a weeks 1-2 run with Llama-Guard-3-8B.

Attack success rate (ASR) is the share of replies the guard labels unsafe, the metric DSA
reports in Table V, so our undefended numbers can be checked against theirs. The keyword
refusal rate is reported beside it; where the two disagree, read the replies.

    python scripts/judge_runs.py --run results/qwen2.5-7b
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from refusal_degeneracy.judge import guard_unsafe, load_guard, refusal_rate
from refusal_degeneracy.runs import load_run

DEFAULT_GUARD = "meta-llama/Llama-Guard-3-8B"

Judge = Callable[[Sequence[str], Sequence[str]], list[bool]]


@dataclass(frozen=True)
class RunVerdict:
    test_set: str
    intervention: str
    n: int
    asr: float
    keyword_refusal_rate: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="output directory of run_week1.py")
    parser.add_argument("--guard", default=DEFAULT_GUARD, help="guard model")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def reply_files(run: Path) -> list[Path]:
    """Reply files are ``<run>/<model>/<test set>__<intervention>.jsonl``."""
    return sorted(path for path in run.glob("*/*.jsonl") if "__" in path.stem)


def judge_files(paths: Iterable[Path], judge: Judge) -> list[RunVerdict]:
    verdicts = []
    for path in paths:
        test_set, intervention = path.stem.split("__", 1)
        rows = load_run(path)
        prompts, replies = [row["prompt"] for row in rows], [row["reply"] for row in rows]
        unsafe = judge(prompts, replies)
        verdicts.append(
            RunVerdict(test_set, intervention, len(rows), sum(unsafe) / len(rows), refusal_rate(replies))
        )
    return verdicts


def main() -> None:
    args = build_parser().parse_args()
    guard, tokenizer = load_guard(args.guard)

    def judge(prompts: Sequence[str], replies: Sequence[str]) -> list[bool]:
        return guard_unsafe(guard, tokenizer, prompts, replies, args.batch_size)

    verdicts = judge_files(reply_files(args.run), judge)
    table = pd.DataFrame([asdict(verdict) for verdict in verdicts])
    table.to_csv(args.run / "asr.csv", index=False)
    (args.run / "asr.json").write_text(json.dumps([asdict(verdict) for verdict in verdicts], indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
