import torch

from refusal_degeneracy.hooks import (
    direction_ablation,
    hooks,
    multi_direction_ablation,
    neuron_pin,
    neuron_zero,
    project_out,
)


def _read_layer_inputs(model, batch):
    store = {}

    def save(i):
        def pre(module, args, kwargs):
            store[i] = (args[0] if args else kwargs["hidden_states"]).clone()

        return pre

    return store, [(layer, save(i)) for i, layer in enumerate(model.model.layers)]


def test_project_out_removes_component():
    x = torch.randn(4, 32)
    r = torch.randn(32)
    out = project_out(x, r / r.norm())
    assert torch.allclose((out @ (r / r.norm())), torch.zeros(4), atol=1e-5)


def test_direction_ablation_zeroes_residual(tiny_model, batch):
    r = torch.randn(32)
    unit = r / r.norm()
    store, read = _read_layer_inputs(tiny_model, batch)
    pre, post = direction_ablation(tiny_model, r)
    with hooks(pre=pre + read, post=post):
        tiny_model(**batch)
    for i in range(1, len(tiny_model.model.layers)):
        assert (store[i].float() @ unit).abs().max() < 1e-3


def test_multi_direction_ablation_zeroes_subspace(tiny_model, batch):
    rs = torch.randn(2, 32)
    q, _ = torch.linalg.qr(rs.float().T)
    store, read = _read_layer_inputs(tiny_model, batch)
    pre, post = multi_direction_ablation(tiny_model, rs)
    with hooks(pre=pre + read, post=post):
        tiny_model(**batch)
    for i in range(1, len(tiny_model.model.layers)):
        assert (store[i].float() @ q).abs().max() < 1e-3


def test_neuron_pin_holds_value(tiny_model, batch):
    seen = {}

    def watch(module, args, kwargs):
        seen["v"] = args[0][..., 7].clone()

    pre, _ = neuron_pin(tiny_model, 2, 7, 3.5)
    with hooks(pre=[*pre, (tiny_model.model.layers[2].mlp.down_proj, watch)]):
        tiny_model(**batch)
    assert torch.allclose(seen["v"], torch.full_like(seen["v"], 3.5))


def test_neuron_zero_zeroes_selected(tiny_model, batch):
    seen = {}

    def watch(module, args, kwargs):
        seen["v"] = args[0][..., [2, 5]].abs().max().item()

    pre, _ = neuron_zero(tiny_model, [(1, 2), (1, 5), (3, 9)])
    with hooks(pre=[*pre, (tiny_model.model.layers[1].mlp.down_proj, watch)]):
        tiny_model(**batch)
    assert seen["v"] == 0.0


def test_hooks_are_removed(tiny_model, batch):
    before = {id(m): len(m._forward_pre_hooks) for m in tiny_model.modules()}
    pre, post = direction_ablation(tiny_model, torch.randn(32))
    with hooks(pre=pre, post=post):
        pass
    after = {id(m): len(m._forward_pre_hooks) for m in tiny_model.modules()}
    assert before == after
