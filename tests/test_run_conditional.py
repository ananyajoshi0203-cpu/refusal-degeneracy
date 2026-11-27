import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from refusal_degeneracy.eval import last_token_residuals
from refusal_degeneracy.hooks import direction_ablation, hooks
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


@pytest.mark.parametrize(
    ("points", "kl_max", "expected"),
    [
        ([("direction", 0.6, 0.01), ("cluster", 0.1, 0.3), ("direction", 0.0, 1.2)], 0.3, 1),
        ([("direction", 0.6, 0.01), ("cluster", 0.1, 0.3), ("direction", 0.0, 1.2)], 1.2, 2),
        ([("direction", 0.6, 0.01), ("cluster", 0.1, 0.3)], 0.01, 0),
        ([("direction", 0.6, 0.2), ("cluster", 0.1, 0.3)], 0.1, None),
    ],
    ids=["best-within-budget", "budget-admits-all", "budget-is-inclusive", "none-within-budget"],
)
def test_strongest_within_leaves_the_least_refusal_within_the_kl_budget(script, points, kl_max, expected):
    # Arrange
    attacks = [
        script.AttackPoint(script.Attack(attack), layer, 1, retained, kl)
        for layer, (attack, retained, kl) in enumerate(points)
    ]

    # Act
    strongest = script.strongest_within(attacks, kl_max)

    # Assert
    assert strongest == (attacks[expected] if expected is not None else None)


def test_sweep_layers_leaves_degenerate_refusal_that_it_removes_from_a_redundant_defense(script):
    # Arrange
    model = toy_model(0)
    *harmful, harmless = toy_prompts(model, 0, [*CATEGORY_MARKERS, HARMLESS_MARKER])
    harmful_ids = torch.cat(harmful)
    last = model.config.num_hidden_layers - 1
    defenses = {degeneracy: SyntheticDefense(2, 2, degeneracy, 1.0) for degeneracy in (0.0, 1.0)}
    defended = {
        degeneracy: implant_defense(toy_model(0), defense, harmful, harmless, REFUSAL_IDS)
        for degeneracy, defense in defenses.items()
    }
    directions = {
        degeneracy: last_token_residuals(net, harmful_ids, None).mean(0)
        - last_token_residuals(net, harmless, None).mean(0)
        for degeneracy, net in defended.items()
    }

    def gap_of(net):
        return lambda hook_spec: refusal_gap(net, harmful_ids, harmless, REFUSAL_IDS, hook_spec)

    def kl(hook_spec) -> float:
        return float(len(hook_spec[0]) + len(hook_spec[1]))

    # Act
    swept = {
        degeneracy: script.sweep_layers(
            net,
            directions[degeneracy],
            [defenses[degeneracy].layer, last],
            gap_of(net)(None),
            gap_of(net),
            kl,
        )
        for degeneracy, net in defended.items()
    }

    # Assert
    for degeneracy, points in swept.items():
        gap, net = gap_of(defended[degeneracy]), defended[degeneracy]
        expected = gap(direction_ablation(net, directions[degeneracy][last])) / gap(None)
        assert [point.layer for point in points] == [defenses[degeneracy].layer, last]
        assert all(point.attack is script.Attack.DIRECTION and point.k == 1 for point in points)
        assert all(point.kl > 0 for point in points)
        assert points[-1].retained_refusal == pytest.approx(expected)
    assert swept[0.0][-1].retained_refusal < 0.25 < swept[1.0][-1].retained_refusal


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
