"""Rank MLP neurons by gradient-times-activation and select the refusal neuron."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd
import torch
from torch import Tensor

from .directions import kl_div, last_logits, refusal_score, refusal_score_from_logits
from .hooks import hooks, neuron_pin
from .model import batches, decoder_layers, encode


@dataclass(frozen=True)
class LastTokenGrads:
    """Per-layer quantities at the last prompt token, each a list over layers."""

    acts: list[Tensor]
    act_grads: list[Tensor]
    out_grads: list[Tensor]


def last_token_grads(model, inputs: Mapping[str, Tensor], ref_ids: Sequence[int]) -> LastTokenGrads:
    """Backpropagate the summed refusal score of one batch to every MLP neuron (``acts`` and
    ``act_grads``, ``[n, d_mlp]``) and to every MLP residual write (``out_grads``, ``[n, d_model]``)."""
    layers = decoder_layers(model)
    acts: dict[int, Tensor] = {}
    outs: dict[int, Tensor] = {}

    def keep_act(i):
        def pre(module, args, kwargs):
            args[0].retain_grad()
            acts[i] = args[0]

        return pre

    def keep_out(i):
        def post(module, inp, out):
            out.retain_grad()
            outs[i] = out

        return post

    emb = model.get_input_embeddings()(inputs["input_ids"]).detach().requires_grad_(True)
    with torch.enable_grad(), hooks(
        pre=[(layer.mlp.down_proj, keep_act(i)) for i, layer in enumerate(layers)],
        post=[(layer.mlp.down_proj, keep_out(i)) for i, layer in enumerate(layers)],
    ):
        logits = model(inputs_embeds=emb, attention_mask=inputs.get("attention_mask")).logits[:, -1, :]
        refusal_score_from_logits(logits.float(), ref_ids).sum().backward()
    order = range(len(layers))
    return LastTokenGrads(
        acts=[acts[i][:, -1].detach().float() for i in order],
        act_grads=[acts[i].grad[:, -1].float() for i in order],
        out_grads=[outs[i].grad[:, -1].float() for i in order],
    )


def _attribution(model, tokenizer, prompts, ref_ids, batch_size, progress) -> Tensor:
    total, count = None, 0
    for batch in batches(prompts, batch_size, progress, "attribution"):
        grads = last_token_grads(model, encode(tokenizer, batch, model.device), ref_ids)
        per_layer = zip(grads.acts, grads.act_grads, strict=True)
        attribution = torch.stack([(act * act_grad).sum(0) for act, act_grad in per_layer]).cpu()
        total = attribution if total is None else total + attribution
        count += len(batch)
    return total / count


def score_neurons(
    model, tokenizer, harmful, harmless, ref_ids, batch_size: int = 4, progress: bool = True
) -> Tensor:
    """Return ``[L, d_mlp]`` scores; large positive pushes the model toward refusing harmful prompts."""
    return _attribution(model, tokenizer, harmful, ref_ids, batch_size, progress) - _attribution(
        model, tokenizer, harmless, ref_ids, batch_size, progress
    )


def top_neurons(scores: Tensor, k: int) -> list[tuple[int, int]]:
    d_mlp = scores.shape[1]
    idx = scores.flatten().topk(k).indices
    return [(int(i // d_mlp), int(i % d_mlp)) for i in idx]


def select_neuron(
    model,
    tokenizer,
    scores: Tensor,
    mean_act_harmful: Tensor,
    val_harmful: Sequence[str],
    val_harmless: Sequence[str],
    ref_ids: Sequence[int],
    top: int = 5,
    magnitudes: Sequence[float] = (0, 2, 5, 10, 20, 50, 100),
    n: int = 32,
    kl_max: float = 0.5,
) -> tuple[tuple[int, int, float], pd.DataFrame]:
    """Sweep the top candidate neurons and pin values; return the strongest safe suppression."""
    vh, vb = val_harmful[:n], val_harmless[:n]
    base = last_logits(model, tokenizer, vb)
    rows = []
    for layer, index in top_neurons(scores, top):
        sign = 1.0 if mean_act_harmful[layer, index] > 0 else -1.0
        for magnitude in magnitudes:
            value = -sign * magnitude
            spec = neuron_pin(model, layer, index, value)
            rows.append(
                {
                    "layer": layer,
                    "index": index,
                    "value": value,
                    "score": float(scores[layer, index]),
                    "ablate_score": refusal_score(model, tokenizer, vh, ref_ids, spec).mean().item(),
                    "kl": kl_div(base, last_logits(model, tokenizer, vb, spec)).mean().item(),
                }
            )
    df = pd.DataFrame(rows)
    ok = df[df.kl < kl_max]
    best = (ok if len(ok) else df).sort_values("ablate_score").iloc[0]
    return (int(best.layer), int(best["index"]), float(best.value)), df
