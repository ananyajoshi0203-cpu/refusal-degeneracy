import pytest
import torch

from refusal_degeneracy.directions import (
    candidate_layers,
    diff_in_means,
    kl_div,
    refusal_score_from_logits,
)


def test_diff_in_means_shape_and_value():
    harmful = torch.randn(20, 4, 32) + 0.5
    harmless = torch.randn(20, 4, 32) - 0.5
    directions = diff_in_means(harmful, harmless)
    assert directions.shape == (4, 32)
    assert torch.allclose(directions, harmful.mean(0) - harmless.mean(0))


def test_refusal_score_monotonic():
    logits = torch.zeros(2, 10)
    logits[0, [3, 4]] = 5.0
    scores = refusal_score_from_logits(logits, [3, 4])
    assert scores[0] > scores[1]


def test_kl_div_zero_for_identical():
    logits = torch.randn(3, 10)
    assert kl_div(logits, logits).abs().max() < 1e-6


def test_kl_div_nonnegative():
    a, b = torch.randn(3, 10), torch.randn(3, 10)
    assert (kl_div(a, b) >= -1e-6).all()


@pytest.mark.parametrize(
    ("zero_layers", "expected"),
    [((), [0, 1, 2, 3, 4, 5, 6, 7]), ((0,), [1, 2, 3, 4, 5, 6, 7]), ((0, 3), [1, 2, 4, 5, 6, 7])],
)
def test_candidate_layers_skip_zero_directions_and_the_last_fifth(zero_layers, expected):
    n_layers = 10
    directions = torch.randn(n_layers, 32)
    directions[list(zero_layers)] = 0.0

    candidates = candidate_layers(directions, skip_last_frac=0.2)

    assert candidates == expected
