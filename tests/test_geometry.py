import math

import pytest
import torch

from refusal_degeneracy.geometry import (
    RoutedAttribution,
    direction_cosines,
    neuron_writes,
    participation_ratio,
    routed_attribution_batch,
    summarize,
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
