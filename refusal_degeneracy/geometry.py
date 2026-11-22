"""Write geometry of safety neurons relative to the refusal direction.

Direction ablation removes only the part of a neuron's effect that passes through the refusal
direction. If the neurons a defense relies on all write along that one direction (redundancy),
one ablation removes all of them at once. If they write distinct directions that each drive
refusal (degeneracy), most of their effect survives it. ``direction_share`` measures which of
the two a model is, to first order, before running the attack.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .hooks import neuron_write_ablation
from .model import batches, decoder_layers, encode
from .neurons import last_token_grads, top_neurons


@dataclass(frozen=True)
class RoutedAttribution:
    """Per-neuron refusal attribution ``[L, d_mlp]``, and the part of it carried by one direction."""

    total: Tensor
    through_direction: Tensor

    def __sub__(self, other: RoutedAttribution) -> RoutedAttribution:
        return RoutedAttribution(self.total - other.total, self.through_direction - other.through_direction)


@dataclass(frozen=True)
class NeuronGeometry:
    layer: int
    index: int
    attribution: float
    cosine: float
    direction_share: float


@dataclass(frozen=True)
class GeometrySummary:
    neurons: list[NeuronGeometry]
    direction_share: float
    weighted_cos2: float
    participation_ratio: float


@dataclass(frozen=True)
class SetGeometry:
    """Geometry of one set of safety neurons, the ``k`` most attributed."""

    k: int
    counterfactual_share: float
    weighted_cos2: float
    participation_ratio: float


def unit(vector: Tensor) -> Tensor:
    return vector.float() / vector.float().norm()


def neuron_writes(model, neurons: Sequence[tuple[int, int]]) -> Tensor:
    """Residual-stream write vector of each ``(layer, index)`` neuron, ``[k, d_model]``."""
    layers = decoder_layers(model)
    return torch.stack([layers[layer].mlp.down_proj.weight[:, index].float() for layer, index in neurons])


def direction_cosines(writes: Tensor, direction: Tensor) -> Tensor:
    return (writes.float() @ unit(direction)) / writes.float().norm(dim=-1)


def participation_ratio(vectors: Tensor) -> float:
    """Effective number of independent directions spanned by the rows of ``vectors``: 1 for
    copies of one vector, ``k`` for ``k`` orthogonal vectors of equal norm."""
    energy = torch.linalg.svdvals(vectors.float()) ** 2
    return float(energy.sum() ** 2 / (energy**2).sum())


def routed_attribution_batch(
    model, inputs: Mapping[str, Tensor], ref_ids: Sequence[int], direction: Tensor
) -> RoutedAttribution:
    """Gradient x activation per neuron, summed over the batch, split by whether the neuron's
    write reaches the refusal score through ``direction``. Summed over an orthonormal basis of
    the residual stream, ``through_direction`` recovers ``total``."""
    grads = last_token_grads(model, inputs, ref_ids)
    r = unit(direction).to(grads.acts[0].device)
    total, through = [], []
    for layer, act, act_grad, out_grad in zip(
        decoder_layers(model), grads.acts, grads.act_grads, grads.out_grads, strict=True
    ):
        write_along_r = r @ layer.mlp.down_proj.weight.float()
        grad_along_r = out_grad @ r
        total.append((act * act_grad).sum(0))
        through.append((act * grad_along_r[:, None]).sum(0) * write_along_r)
    return RoutedAttribution(torch.stack(total).cpu(), torch.stack(through).cpu())


def routed_attribution(
    model,
    tokenizer,
    prompts: Sequence[str],
    ref_ids: Sequence[int],
    direction: Tensor,
    batch_size: int = 4,
    progress: bool = True,
) -> RoutedAttribution:
    """Mean ``routed_attribution_batch`` over ``prompts``."""
    total, through = None, None
    for batch in batches(prompts, batch_size, progress, "routed attribution"):
        part = routed_attribution_batch(model, encode(tokenizer, batch, model.device), ref_ids, direction)
        total = part.total if total is None else total + part.total
        through = part.through_direction if through is None else through + part.through_direction
    return RoutedAttribution(total / len(prompts), through / len(prompts))


def summarize(model, routed: RoutedAttribution, direction: Tensor, k: int) -> GeometrySummary:
    """Geometry of the ``k`` neurons with the largest refusal attribution.

    ``direction_share`` is the fraction of their combined attribution that runs through
    ``direction``, which is what direction ablation removes to first order. ``weighted_cos2``
    is the same question asked of the weights alone. ``participation_ratio`` counts the distinct
    directions their attribution-weighted writes span."""
    neurons = top_neurons(routed.total, k)
    writes = neuron_writes(model, neurons)
    cosines = direction_cosines(writes, direction)
    attributions = torch.tensor([float(routed.total[layer, index]) for layer, index in neurons])
    through = torch.tensor([float(routed.through_direction[layer, index]) for layer, index in neurons])
    weights = attributions.clamp(min=0)
    per_neuron = [
        NeuronGeometry(layer, index, float(attribution), float(cosine), float(part / attribution))
        for (layer, index), attribution, cosine, part in zip(
            neurons, attributions, cosines, through, strict=True
        )
    ]
    weighted_unit_writes = weights[:, None] * writes / writes.norm(dim=-1, keepdim=True)
    return GeometrySummary(
        neurons=per_neuron,
        direction_share=float(through.sum() / attributions.sum()),
        weighted_cos2=float((weights * cosines**2).sum() / weights.sum()),
        participation_ratio=participation_ratio(weighted_unit_writes),
    )


def counterfactual_direction_share(
    refusal_gap: Callable[[tuple | None], float],
    model,
    neurons: Sequence[tuple[int, int]],
    direction: Tensor,
) -> float:
    """Share of the ``neurons``' joint effect on the harmful-vs-harmless refusal gap that runs
    through ``direction``: the gap lost by removing only the direction's part of their writes,
    over the gap lost by removing their writes entirely.

    Unlike the gradient ``direction_share`` this needs no linearization, so it stays right when a
    strong write saturates the final norm (where gradient x activation can even flip sign) and
    when the neurons' writes interact. ``refusal_gap`` maps an optional hook spec to the gap."""
    gap = refusal_gap(None)
    removed_entirely = gap - refusal_gap(neuron_write_ablation(model, neurons))
    removed_along_direction = gap - refusal_gap(neuron_write_ablation(model, neurons, direction))
    return removed_along_direction / removed_entirely


def set_geometry(
    refusal_gap: Callable[[tuple | None], float],
    model,
    scores: Tensor,
    direction: Tensor,
    k: int,
) -> SetGeometry:
    """Weight and counterfactual geometry of the ``k`` neurons with the highest ``scores``."""
    neurons = top_neurons(scores, k)
    writes = neuron_writes(model, neurons)
    weights = torch.tensor([float(scores[layer, index]) for layer, index in neurons]).clamp(min=0)
    cosines = direction_cosines(writes, direction)
    unit_writes = writes / writes.norm(dim=-1, keepdim=True)
    return SetGeometry(
        k=k,
        counterfactual_share=counterfactual_direction_share(refusal_gap, model, neurons, direction),
        weighted_cos2=float((weights * cosines**2).sum() / weights.sum()),
        participation_ratio=participation_ratio(weights[:, None] * unit_writes),
    )
