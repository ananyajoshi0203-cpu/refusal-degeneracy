"""Model-agnostic functional checks that the intervention code does what it claims.

These are behavioural regression evals (runnable on CPU with a random model), distinct from
unit tests: they exercise the cache -> difference-in-means -> ablation pipeline end to end and
assert the targeted structure is actually removed from the residual stream.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .hooks import direction_ablation, hooks, neuron_pin
from .model import decoder_layers


def _last_token_residuals(model, input_ids: Tensor, attention_mask: Tensor | None, hook_spec=None) -> Tensor:
    store: dict[int, Tensor] = {}
    n_layers = len(decoder_layers(model))

    def save(i):
        def pre(module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            store[i] = x[:, -1, :].float()

        return pre

    read_hooks = [(decoder_layers(model)[i], save(i)) for i in range(n_layers)]
    pre, post = (hook_spec if hook_spec else ([], []))
    with torch.no_grad(), hooks(pre=pre + read_hooks, post=post):
        model(input_ids=input_ids, attention_mask=attention_mask)
    return torch.stack([store[i] for i in range(n_layers)], dim=1)


def mean_alignment(resid: Tensor, direction: Tensor) -> float:
    unit = direction.float() / direction.float().norm()
    return (resid.float() @ unit).abs().mean().item()


def measure_direction_removal(
    model, harmful_ids: Tensor, harmless_ids: Tensor, attention_mask: Tensor | None = None
) -> dict[str, float]:
    """Cache residuals, build the diff-in-means direction at the strongest layer, and report the
    mean |component| along it before and after ablation. ``after`` should be ~0."""
    harmful = _last_token_residuals(model, harmful_ids, attention_mask)
    harmless = _last_token_residuals(model, harmless_ids, attention_mask)
    directions = harmful.mean(0) - harmless.mean(0)
    layer = int(directions.norm(dim=-1).argmax())
    direction = directions[layer]

    before = mean_alignment(harmful[:, layer], direction)
    ablated = _last_token_residuals(model, harmful_ids, attention_mask, direction_ablation(model, direction))
    after = mean_alignment(ablated[:, layer], direction)
    return {"layer": layer, "alignment_before": before, "alignment_after": after}


def measure_neuron_pin(model, input_ids: Tensor, layer: int, index: int, value: float) -> float:
    """Return the maximum deviation of neuron ``index`` from ``value`` under a pin (should be 0)."""
    captured: dict[str, Tensor] = {}

    def watch(module, args, kwargs):
        captured["x"] = args[0][..., index].clone()

    pre, post = neuron_pin(model, layer, index, value)
    with torch.no_grad(), hooks(pre=[*pre, (decoder_layers(model)[layer].mlp.down_proj, watch)]):
        model(input_ids=input_ids)
    return (captured["x"] - value).abs().max().item()
