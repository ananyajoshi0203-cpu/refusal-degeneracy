"""Forward hooks for reading and editing residual-stream and MLP-neuron activations."""
from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator

import torch
from torch import Tensor
from torch.nn import Module

HookPair = tuple[Module, object]


@contextlib.contextmanager
def hooks(pre: Iterable[HookPair] = (), post: Iterable[HookPair] = ()) -> Iterator[None]:
    handles = []
    try:
        for module, fn in pre:
            handles.append(module.register_forward_pre_hook(fn, with_kwargs=True))
        for module, fn in post:
            handles.append(module.register_forward_hook(fn))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _get_hidden(args, kwargs) -> Tensor:
    return args[0] if args else kwargs["hidden_states"]


def _set_hidden(args, kwargs, x: Tensor):
    if args:
        return (x, *args[1:]), kwargs
    kwargs["hidden_states"] = x
    return args, kwargs


def project_out(x: Tensor, r: Tensor) -> Tensor:
    r = r.to(device=x.device, dtype=x.dtype)
    return x - (x @ r).unsqueeze(-1) * r


def _residual_writers(model) -> list[Module]:
    writers = []
    for layer in model.model.layers:
        writers += [layer.self_attn, layer.mlp]
    return writers


def direction_ablation(model, r: Tensor) -> tuple[list[HookPair], list[HookPair]]:
    """Project direction ``r`` out of every residual-stream read and write."""
    unit = r.float() / r.float().norm()

    def pre(module, args, kwargs):
        return _set_hidden(args, kwargs, project_out(_get_hidden(args, kwargs), unit))

    def post(module, inp, out):
        if isinstance(out, tuple):
            return (project_out(out[0], unit), *out[1:])
        return project_out(out, unit)

    pre_hooks = [(layer, pre) for layer in model.model.layers]
    post_hooks = [(writer, post) for writer in _residual_writers(model)]
    return pre_hooks, post_hooks


def multi_direction_ablation(model, directions: Tensor) -> tuple[list[HookPair], list[HookPair]]:
    """Project an orthonormal basis of ``directions`` [k, d] out of the residual stream."""
    basis, _ = torch.linalg.qr(directions.float().T)

    def proj(x: Tensor) -> Tensor:
        q = basis.to(device=x.device, dtype=x.dtype)
        return x - (x @ q) @ q.T

    def pre(module, args, kwargs):
        return _set_hidden(args, kwargs, proj(_get_hidden(args, kwargs)))

    def post(module, inp, out):
        if isinstance(out, tuple):
            return (proj(out[0]), *out[1:])
        return proj(out)

    pre_hooks = [(layer, pre) for layer in model.model.layers]
    post_hooks = [(writer, post) for writer in _residual_writers(model)]
    return pre_hooks, post_hooks


def activation_addition(model, layer: int, vec: Tensor) -> tuple[list[HookPair], list[HookPair]]:
    def pre(module, args, kwargs):
        x = _get_hidden(args, kwargs)
        return _set_hidden(args, kwargs, x + vec.to(device=x.device, dtype=x.dtype))

    return [(model.model.layers[layer], pre)], []


def neuron_pin(model, layer: int, index: int, value: float) -> tuple[list[HookPair], list[HookPair]]:
    def pre(module, args, kwargs):
        x = args[0].clone()
        x[..., index] = value
        return (x, *args[1:]), kwargs

    return [(model.model.layers[layer].mlp.down_proj, pre)], []


def neuron_zero(model, neurons: Iterable[tuple[int, int]]) -> tuple[list[HookPair], list[HookPair]]:
    by_layer: dict[int, list[int]] = {}
    for layer, index in neurons:
        by_layer.setdefault(layer, []).append(index)

    pre_hooks = []
    for layer, indices in by_layer.items():
        index_tensor = torch.tensor(indices)

        def pre(module, args, kwargs, index_tensor=index_tensor):
            x = args[0].clone()
            x[..., index_tensor.to(x.device)] = 0
            return (x, *args[1:]), kwargs

        pre_hooks.append((model.model.layers[layer].mlp.down_proj, pre))
    return pre_hooks, []


def neuron_write_ablation(
    model, neurons: Iterable[tuple[int, int]], direction: Tensor | None = None
) -> tuple[list[HookPair], list[HookPair]]:
    """Subtract the residual writes of ``neurons``: all of each write when ``direction`` is None,
    otherwise only its component along ``direction``, leaving the rest of every write intact."""
    by_layer: dict[int, list[int]] = {}
    for layer, index in neurons:
        by_layer.setdefault(layer, []).append(index)
    unit = None if direction is None else direction.float() / direction.float().norm()

    post_hooks = []
    for layer, indices in by_layer.items():
        index_tensor = torch.tensor(indices)

        def post(module, inp, out, index_tensor=index_tensor):
            index = index_tensor.to(out.device)
            writes = inp[0][..., index] @ module.weight[:, index].T
            if unit is not None:
                r = unit.to(device=out.device, dtype=out.dtype)
                writes = (writes @ r).unsqueeze(-1) * r
            return out - writes

        post_hooks.append((model.model.layers[layer].mlp.down_proj, post))
    return [], post_hooks
