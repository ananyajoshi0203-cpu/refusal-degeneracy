"""Controlled defenses implanted into a model, to check that write geometry predicts whether a
neuron-level defense survives direction ablation.

Harmful prompts come in categories. ``implant_defense`` rewires one group of MLP neurons per
category to detect that category and raise refusal. ``degeneracy`` sets what each group writes:
at 0 every group writes one shared refusal direction (redundancy); at 1 the group for category
``c`` raises its own refusal token ``ref_ids[c]`` through its own direction (degeneracy).

The mechanisms must be input-conditional to be degenerate in the sense that matters here. Neurons
that all fire together on every harmful prompt sum to one write, the mean, which a
difference-in-means ablation removes however different the individual writes are.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor

from .directions import refusal_score_from_logits
from .eval import last_token_residuals
from .geometry import counterfactual_direction_share, routed_attribution_batch, summarize, unit
from .hooks import direction_ablation, hooks
from .model import decoder_layers


@dataclass(frozen=True)
class SyntheticDefense:
    layer: int
    neurons_per_category: int
    degeneracy: float
    write_gain: float

    def neurons(self, category: int) -> list[int]:
        start = category * self.neurons_per_category
        return list(range(start, start + self.neurons_per_category))


@dataclass(frozen=True)
class DefenseOutcome:
    degeneracy: float
    direction_share: float
    weighted_cos2: float
    participation_ratio: float
    counterfactual_share: float
    refusal_gap: float
    retained_refusal: float


def toy_model(seed: int, unembed_scale: float = 50.0):
    """Small random Llama whose unembedding is scaled up. At the default init (std 0.02) the
    final norm caps every logit near zero, so no write can change the next token much."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    with torch.no_grad():
        model.lm_head.weight *= unembed_scale
    return model


def toy_prompts(model, seed: int, last_tokens: Sequence[int], n: int = 12, length: int = 8) -> list[Tensor]:
    """One batch of random prompts per entry of ``last_tokens``, each ending on that token, which
    is what marks the category. Body tokens avoid every marker and refusal token."""
    generator = torch.Generator().manual_seed(seed)
    reserved = max(last_tokens) + 1
    prompts = []
    for last in last_tokens:
        ids = torch.randint(reserved, model.config.vocab_size, (n, length), generator=generator)
        ids[:, -1] = last
        prompts.append(ids)
    return prompts


def refusal_unembeddings(model, ref_ids: Sequence[int]) -> Tensor:
    """Residual direction that raises each refusal token's logit against the vocabulary mean."""
    unembed = model.get_output_embeddings().weight.float()
    return unembed[list(ref_ids)] - unembed.mean(0)


def defense_writes(per_category: Tensor, degeneracy: float) -> Tensor:
    """``[n_categories, d]`` unit writes mixing the shared refusal direction with each category's
    own direction ``per_category[c]`` in proportion ``degeneracy``."""
    own = per_category / per_category.norm(dim=-1, keepdim=True)
    mixed = (1 - degeneracy) * unit(per_category.mean(0)) + degeneracy * own
    return mixed / mixed.norm(dim=-1, keepdim=True)


def category_detectors(harmful_means: Tensor, harmless_mean: Tensor) -> Tensor:
    """``[n_categories, d]`` readouts ``v`` with ``v[c] . harmful_means[c'] = [c == c']`` and
    ``v[c] . harmless_mean = 0``: the least-norm linear detector for each category alone."""
    targets = torch.cat([torch.eye(len(harmful_means)), torch.zeros(len(harmful_means), 1)], dim=1)
    means = torch.cat([harmful_means, harmless_mean[None]])
    return targets @ torch.linalg.pinv(means.T)


def mlp_inputs(model, input_ids: Tensor, layer: int) -> Tensor:
    """Last-token input to ``layer``'s MLP (after its pre-MLP norm), ``[n, d]``."""
    captured: dict[str, Tensor] = {}

    def keep(module, args, kwargs):
        captured["x"] = args[0][:, -1].float()

    with torch.no_grad(), hooks(pre=[(decoder_layers(model)[layer].mlp, keep)]):
        model(input_ids=input_ids)
    return captured["x"]


def implant_defense(
    model,
    defense: SyntheticDefense,
    harmful_by_category: Sequence[Tensor],
    harmless_ids: Tensor,
    ref_ids: Sequence[int],
):
    """Copy of ``model`` whose neuron group for each category detects that category at
    ``defense.layer`` and writes refusal according to ``defense.degeneracy``."""
    defended = copy.deepcopy(model)
    mlp = decoder_layers(defended)[defense.layer].mlp
    harmful_means = torch.stack(
        [mlp_inputs(model, ids, defense.layer).mean(0) for ids in harmful_by_category]
    )
    detectors = category_detectors(harmful_means, mlp_inputs(model, harmless_ids, defense.layer).mean(0))
    writes = defense_writes(refusal_unembeddings(model, ref_ids), defense.degeneracy)
    with torch.no_grad():
        for category, (detector, write) in enumerate(zip(detectors, writes, strict=True)):
            index = torch.tensor(defense.neurons(category))
            mlp.gate_proj.weight[index] = detector
            mlp.up_proj.weight[index] = detector
            mlp.down_proj.weight[:, index] = defense.write_gain * write[:, None]
    return defended


@torch.no_grad()
def mean_refusal_score(model, input_ids: Tensor, ref_ids: Sequence[int], hook_spec=None) -> float:
    with hooks(*hook_spec) if hook_spec else nullcontext():
        logits = model(input_ids=input_ids).logits[:, -1, :].float()
    return refusal_score_from_logits(logits, ref_ids).mean().item()


def refusal_gap(
    model, harmful_ids: Tensor, harmless_ids: Tensor, ref_ids: Sequence[int], hook_spec=None
) -> float:
    return mean_refusal_score(model, harmful_ids, ref_ids, hook_spec) - mean_refusal_score(
        model, harmless_ids, ref_ids, hook_spec
    )


def evaluate_defense(
    model,
    defense: SyntheticDefense,
    harmful_by_category: Sequence[Tensor],
    harmless_ids: Tensor,
    ref_ids: Sequence[int],
) -> DefenseOutcome:
    """Implant ``defense``, find the refusal direction the usual way (difference in means at the
    last layer's input), then compare the predicted ``direction_share`` with the fraction of the
    harmful-vs-harmless refusal gap that survives ablating that direction."""
    defended = implant_defense(model, defense, harmful_by_category, harmless_ids, ref_ids)
    harmful_ids = torch.cat(list(harmful_by_category))
    last = len(decoder_layers(defended)) - 1
    direction = (
        last_token_residuals(defended, harmful_ids, None)[:, last].mean(0)
        - last_token_residuals(defended, harmless_ids, None)[:, last].mean(0)
    )
    routed = routed_attribution_batch(defended, {"input_ids": harmful_ids}, ref_ids, direction) - (
        routed_attribution_batch(defended, {"input_ids": harmless_ids}, ref_ids, direction)
    )
    n_implanted = defense.neurons_per_category * len(harmful_by_category)
    geometry = summarize(defended, routed, direction, n_implanted)

    def gap_under(hook_spec) -> float:
        return refusal_gap(defended, harmful_ids, harmless_ids, ref_ids, hook_spec)

    categories = range(len(harmful_by_category))
    implanted = [(defense.layer, index) for category in categories for index in defense.neurons(category)]
    gap = gap_under(None)
    ablated_gap = gap_under(direction_ablation(defended, direction))
    return DefenseOutcome(
        degeneracy=defense.degeneracy,
        direction_share=geometry.direction_share,
        weighted_cos2=geometry.weighted_cos2,
        participation_ratio=geometry.participation_ratio,
        counterfactual_share=counterfactual_direction_share(gap_under, defended, implanted, direction),
        refusal_gap=gap,
        retained_refusal=ablated_gap / gap,
    )
