"""How much more a model refuses harmful prompts than harmless ones, under an intervention.

On Qwen2.5-7B-Instruct the first-token refusal score kept half its harmful-vs-harmless gap under
direction ablation while generated replies stopped refusing (0.82 to 0.03 on StrongREJECT), so
survival on a real model is scored on generated replies; the first-token score stays available
as the cheap proxy that synthetic models, which refuse with a single token, are exact under.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum

from .directions import refusal_score
from .hooks import HookPair
from .judge import refusal_rate
from .model import generate

HookSpec = tuple[list[HookPair], list[HookPair]]
RefusalGap = Callable[[HookSpec | None], float]

GENERATED_TOKENS = 64


class RefusalMetric(str, Enum):
    GENERATED = "generated"
    FIRST_TOKEN = "first-token"


def refusal_gap(
    metric: RefusalMetric,
    model,
    tokenizer,
    harmful: Sequence[str],
    harmless: Sequence[str],
    ref_ids: Sequence[int],
    batch_size: int,
    max_new_tokens: int = GENERATED_TOKENS,
) -> RefusalGap:
    """The gap as a function of an optional hook spec: harmful minus harmless mean first-token
    refusal score, or harmful minus harmless refusal rate of greedy replies of
    ``max_new_tokens``."""

    def score(prompts: Sequence[str], hook_spec: HookSpec | None) -> float:
        if metric is RefusalMetric.FIRST_TOKEN:
            return refusal_score(model, tokenizer, prompts, ref_ids, hook_spec, batch_size).mean().item()
        replies = generate(model, tokenizer, prompts, max_new_tokens, batch_size, hook_spec, progress=False)
        return refusal_rate(replies)

    def gap(hook_spec: HookSpec | None) -> float:
        return score(harmful, hook_spec) - score(harmless, hook_spec)

    return gap
