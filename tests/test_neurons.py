import torch

from refusal_degeneracy.neurons import top_neurons


def test_top_neurons_picks_largest():
    scores = torch.zeros(4, 64)
    scores[2, 10] = 9.0
    scores[0, 1] = 8.0
    picked = top_neurons(scores, 2)
    assert picked[0] == (2, 10)
    assert picked[1] == (0, 1)


def test_top_neurons_count():
    assert len(top_neurons(torch.randn(4, 64), 5)) == 5
