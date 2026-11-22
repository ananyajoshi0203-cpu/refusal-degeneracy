import importlib.util
from pathlib import Path

import pytest
import torch

from refusal_degeneracy.geometry import unit
from refusal_degeneracy.hooks import hooks, neuron_write_ablation, neuron_zero
from refusal_degeneracy.synthetic import (
    SyntheticDefense,
    category_detectors,
    defense_writes,
    evaluate_defense,
    toy_model,
    toy_prompts,
)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_synthetic.py"
NEURONS = [(1, 3), (1, 8), (2, 5)]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("run_synthetic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("degeneracy", [0.0, 1.0])
def test_defense_writes_run_from_one_shared_direction_to_one_per_category(degeneracy):
    per_category = torch.randn(3, 32)
    expected = (
        unit(per_category.mean(0)).expand(3, -1)
        if degeneracy == 0.0
        else per_category / per_category.norm(dim=-1, keepdim=True)
    )

    writes = defense_writes(per_category, degeneracy)

    assert torch.allclose(writes, expected, atol=1e-6)


def test_category_detectors_fire_on_their_own_category_only():
    harmful_means, harmless_mean = torch.randn(3, 32), torch.randn(32)
    expected = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1)

    detectors = category_detectors(harmful_means, harmless_mean)

    responses = detectors @ torch.cat([harmful_means, harmless_mean[None]]).T
    assert torch.allclose(responses, expected, atol=1e-5)


def test_removing_whole_writes_matches_zeroing_the_neurons(tiny_model, batch):
    with torch.no_grad(), hooks(*neuron_zero(tiny_model, NEURONS)):
        expected = tiny_model(**batch).logits

    with torch.no_grad(), hooks(*neuron_write_ablation(tiny_model, NEURONS)):
        logits = tiny_model(**batch).logits

    assert torch.allclose(logits, expected, atol=1e-5)


def test_removing_writes_along_a_direction_changes_the_mlp_output_only_along_it(tiny_model, batch):
    direction = unit(torch.randn(32))
    down_proj = tiny_model.model.layers[1].mlp.down_proj
    outputs = {}

    def keep(name):
        def post(module, inp, out):
            outputs[name] = out

        return post

    with torch.no_grad(), hooks(post=[(down_proj, keep("base"))]):
        tiny_model(**batch)
    pre, post = neuron_write_ablation(tiny_model, NEURONS[:2], direction)
    with torch.no_grad(), hooks(pre=pre, post=[*post, (down_proj, keep("edited"))]):
        tiny_model(**batch)

    change = outputs["base"] - outputs["edited"]
    off_direction = change - (change @ direction).unsqueeze(-1) * direction
    assert change.norm() > 0
    assert off_direction.norm() < 1e-4 * change.norm()


@pytest.mark.parametrize(
    ("degeneracy", "survives_ablation"), [(0.0, False), (1.0, True)], ids=["redundant", "degenerate"]
)
def test_only_the_degenerate_defense_survives_one_direction_and_neither_survives_its_clusters(
    script, degeneracy, survives_ablation
):
    model = toy_model(seed=0)
    *harmful, harmless = toy_prompts(model, 0, [*script.CATEGORY_MARKERS, script.HARMLESS_MARKER])
    defense = SyntheticDefense(script.DEFENSE_LAYER, script.NEURONS_PER_CATEGORY, degeneracy, write_gain=1.0)

    outcome = evaluate_defense(model, defense, harmful, harmless, script.REFUSAL_IDS)

    assert outcome.refusal_gap > script.MIN_REFUSAL_GAP
    assert (outcome.retained_refusal > 0.25) == survives_ablation
    assert (outcome.counterfactual_share < 0.5) == survives_ablation
    assert (outcome.retained_random_control > 0.25) == survives_ablation
    assert outcome.retained_conditional < 0.25


def test_counterfactual_share_ranks_defenses_by_how_much_refusal_survives(script):
    args = script.build_parser().parse_args(["--seeds", "2", "--write-gains", "1", "2"])

    correlations = script.rank_correlations(script.sweep(args))

    assert correlations["counterfactual_share"] < -0.8
    gradient_and_weight_measures = (correlations["direction_share"], correlations["weighted_cos2"])
    assert correlations["counterfactual_share"] < min(gradient_and_weight_measures)
