import torch

from refusal_degeneracy.eval import measure_direction_removal, measure_neuron_pin


def test_direction_removal_eval(tiny_model):
    torch.manual_seed(2)
    harmful = torch.randint(1, 64, (6, 6))
    harmless = torch.randint(1, 64, (6, 6))
    result = measure_direction_removal(tiny_model, harmful, harmless)
    assert result["alignment_after"] < 1e-3
    assert result["alignment_after"] < result["alignment_before"]


def test_neuron_pin_eval(tiny_model):
    ids = torch.randint(1, 64, (4, 6))
    assert measure_neuron_pin(tiny_model, ids, layer=1, index=3, value=2.0) < 1e-5
