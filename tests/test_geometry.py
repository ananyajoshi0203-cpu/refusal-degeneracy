import math

import pytest
import torch

from refusal_degeneracy.eval import last_token_residuals
from refusal_degeneracy.geometry import (
    RoutedAttribution,
    direction_cosines,
    neuron_writes,
    participation_ratio,
    routed_attribution_batch,
    set_geometry,
    summarize,
)
from refusal_degeneracy.synthetic import (
    SyntheticDefense,
    implant_defense,
    refusal_gap,
    toy_model,
    toy_prompts,
)

REFUSAL_IDS = [3, 4]


def orthonormal(n: int, d: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, n))
    return q.T


@pytest.mark.parametrize(("along", "across"), [(1.0, 0.0), (0.0, 1.0), (3.0, 4.0), (-1.0, 1.0)])
def test_direction_cosines_measure_the_share_of_a_write_along_the_direction(along, across):
    r, orth = orthonormal(2, 32)
    writes = (along * r + across * orth)[None]

    cosine = direction_cosines(writes, 5.0 * r)

    assert cosine.item() == pytest.approx(along / math.hypot(along, across), abs=1e-6)


@pytest.mark.parametrize(("n_distinct", "copies"), [(1, 6), (3, 1), (3, 2), (6, 1)])
def test_participation_ratio_counts_distinct_directions(n_distinct, copies):
    vectors = orthonormal(n_distinct, 32).repeat(copies, 1)

    ratio = participation_ratio(vectors)

    assert ratio == pytest.approx(n_distinct, rel=1e-4)


def test_through_direction_over_a_full_basis_recovers_total_attribution(tiny_model, batch, dims):
    basis = orthonormal(dims["d_model"], dims["d_model"])

    parts = [routed_attribution_batch(tiny_model, batch, REFUSAL_IDS, e) for e in basis]

    total = parts[0].total
    assert torch.allclose(sum(p.through_direction for p in parts), total, atol=1e-4 * total.abs().max())


@pytest.mark.parametrize("share", [0.0, 0.25, 1.0])
def test_summarize_reports_the_share_of_top_attribution_through_the_direction(tiny_model, dims, share):
    k = 3
    total = torch.zeros(dims["n_layers"], dims["d_mlp"])
    top = [(1, 5), (2, 7), (3, 9)]
    for rank, (layer, index) in enumerate(top):
        total[layer, index] = 10.0 - rank
    direction = neuron_writes(tiny_model, top[:1])[0]

    summary = summarize(tiny_model, RoutedAttribution(total, share * total), direction, k)

    assert [(n.layer, n.index) for n in summary.neurons] == top
    assert summary.direction_share == pytest.approx(share)
    assert summary.neurons[0].cosine == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("degeneracy", "runs_through_direction"), [(0.0, True), (1.0, False)], ids=["redundant", "degenerate"]
)
def test_set_geometry_of_implanted_neurons_reflects_their_redundancy(degeneracy, runs_through_direction):
    model = toy_model(seed=0)
    *harmful, harmless = toy_prompts(model, 0, [1, 2, 9, 10])
    ref_ids = [3, 4, 5]
    defense = SyntheticDefense(layer=2, neurons_per_category=2, degeneracy=degeneracy, write_gain=1.0)
    defended = implant_defense(model, defense, harmful, harmless, ref_ids)
    harmful_ids = torch.cat(harmful)
    direction = (
        last_token_residuals(defended, harmful_ids, None)[:, -1].mean(0)
        - last_token_residuals(defended, harmless, None)[:, -1].mean(0)
    )
    implanted = [index for category in range(len(harmful)) for index in defense.neurons(category)]
    scores = torch.zeros(4, 64)
    scores[defense.layer, implanted] = 1.0

    geometry = set_geometry(
        lambda hook_spec: refusal_gap(defended, harmful_ids, harmless, ref_ids, hook_spec),
        defended, scores, direction, k=len(implanted),
    )

    assert (geometry.counterfactual_share > 0.8) == runs_through_direction
    assert (geometry.weighted_cos2 > 0.9) == runs_through_direction
