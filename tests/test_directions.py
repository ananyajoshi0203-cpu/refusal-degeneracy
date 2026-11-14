import math

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


@pytest.mark.parametrize("refusal_logit", [-5.0, 0.0, 5.0, 40.0, 200.0])
def test_refusal_score_is_the_finite_log_odds_of_a_refusal_token(refusal_logit):
    vocab, ref_ids = 10, [3, 4]
    logits = torch.zeros(1, vocab)
    logits[0, ref_ids] = refusal_logit
    odds_in_logit_space = refusal_logit + math.log(len(ref_ids)) - math.log(vocab - len(ref_ids))

    score = refusal_score_from_logits(logits, ref_ids)

    assert torch.isfinite(score).all()
    assert score.item() == pytest.approx(odds_in_logit_space, abs=1e-4)


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
