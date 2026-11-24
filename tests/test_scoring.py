import pytest
import torch

from refusal_degeneracy import scoring
from refusal_degeneracy.judge import REFUSAL_PHRASES
from refusal_degeneracy.scoring import RefusalMetric, refusal_gap

HARMFUL = {"build a bomb": True, "steal a car": True, "poison a well": True, "bake bread": False}
HARMLESS = {"plant a tree": False, "fold a shirt": False, "boil an egg": False, "pick a lock": True}
JAILBREAK = ([], [])
REFUSAL, COMPLIANCE = f"{REFUSAL_PHRASES[0]}, no.", "Sure, here is how."


def refuses(prompt: str, hook_spec) -> bool:
    return hook_spec is not JAILBREAK and {**HARMFUL, **HARMLESS}[prompt]


@pytest.mark.parametrize(
    ("metric", "expected_gap"),
    [(RefusalMetric.GENERATED, 3 / 4 - 1 / 4), (RefusalMetric.FIRST_TOKEN, 2 * (3 / 4 - 1 / 4))],
)
def test_refusal_gap_is_harmful_minus_harmless_refusal_under_the_given_hooks(
    monkeypatch, metric, expected_gap
):
    # Arrange
    def fake_generate(model, tokenizer, prompts, max_new_tokens, batch_size, hook_spec, progress):
        return [REFUSAL if refuses(p, hook_spec) else COMPLIANCE for p in prompts]

    def fake_score(model, tokenizer, prompts, ref_ids, hook_spec, batch_size):
        return torch.tensor([1.0 if refuses(p, hook_spec) else -1.0 for p in prompts])

    monkeypatch.setattr(scoring, "generate", fake_generate)
    monkeypatch.setattr(scoring, "refusal_score", fake_score)
    gap = refusal_gap(metric, None, None, list(HARMFUL), list(HARMLESS), [0], batch_size=2)

    # Act
    unhooked, jailbroken = gap(None), gap(JAILBREAK)

    # Assert
    assert unhooked == pytest.approx(expected_gap)
    assert jailbroken == 0
