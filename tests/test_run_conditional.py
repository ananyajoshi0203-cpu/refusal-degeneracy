import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from refusal_degeneracy.eval import last_token_residuals
from refusal_degeneracy.hooks import hooks
from refusal_degeneracy.synthetic import (
    SyntheticDefense,
    implant_defense,
    refusal_gap,
    toy_model,
    toy_prompts,
)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_conditional.py"
CATEGORY_MARKERS, HARMLESS_MARKER, REFUSAL_IDS = (1, 2, 9), 10, (3, 4, 5)


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("run_conditional", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(script, k: int, retained: float, kl: float):
    return script.ClusterAblation(k, [1] * k, retained, 1.0, kl, 0.0)


@pytest.mark.parametrize(
    ("sweep", "kl_max", "expected"),
    [
        ([(2, 0.6, 0.01), (4, 0.1, 0.05), (8, 0.0, 0.5)], 0.1, 4),
        ([(2, 0.6, 0.01), (4, 0.1, 0.05), (8, 0.0, 0.5)], 1.0, 8),
        ([(2, 0.6, 0.2), (4, 0.1, 0.3)], 0.1, None),
    ],
    ids=["best-within-budget", "budget-admits-all", "none-within-budget"],
)
def test_choose_k_removes_the_most_refusal_within_the_kl_budget(script, sweep, kl_max, expected):
    # Arrange
    rows = [row(script, k, retained, kl) for k, retained, kl in sweep]

    # Act
    chosen = script.choose_k(rows, kl_max)

    # Assert
    assert chosen == expected


def test_sweep_clusters_breaks_a_degenerate_defense_that_random_directions_leave_alone(script):
    # Arrange
    model = toy_model(0)
    *harmful, harmless = toy_prompts(model, 0, [*CATEGORY_MARKERS, HARMLESS_MARKER])
    defended = implant_defense(model, SyntheticDefense(2, 2, 1.0, 1.0), harmful, harmless, REFUSAL_IDS)
    harmful_ids = torch.cat(harmful)
    last = defended.config.num_hidden_layers - 1
    harmful_resid = last_token_residuals(defended, harmful_ids, None)[:, last]
    harmless_resid = last_token_residuals(defended, harmless, None)[:, last]
    base = defended(input_ids=harmless).logits[:, -1].log_softmax(-1)

    def gap(hook_spec) -> float:
        return refusal_gap(defended, harmful_ids, harmless, REFUSAL_IDS, hook_spec)

    def kl(hook_spec) -> float:
        with hooks(*hook_spec):
            ablated = defended(input_ids=harmless).logits[:, -1].log_softmax(-1)
        return (base.exp() * (base - ablated)).sum(-1).mean().item()

    # Act
    full_gap, rows = script.sweep_clusters(defended, harmful_resid, harmless_resid, [3, 6], gap, kl)

    # Assert
    assert full_gap > 2
    for swept in rows:
        assert sum(swept.cluster_sizes) == len(harmful_ids)
        assert swept.retained_refusal < 0.25 < swept.retained_random_control
        assert swept.kl >= 0
