"""Rank MLP neurons by gradient-times-activation and select the refusal neuron."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
from torch import Tensor

from .directions import kl_div, last_logits, refusal_score, refusal_score_from_logits
from .hooks import hooks, neuron_pin
from .model import batches, decoder_layers, encode


def _attribution(model, tokenizer, prompts, ref_ids, batch_size, progress) -> Tensor:
    n_layers = len(decoder_layers(model))
    total, count = None, 0
    for batch in batches(prompts, batch_size, progress, "attribution"):
        enc = encode(tokenizer, batch, model.device)
        saved: dict[int, Tensor] = {}

        def keep(i, saved=saved):
            def pre(module, args, kwargs):
                args[0].retain_grad()
                saved[i] = args[0]

            return pre

        emb = model.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
        with torch.enable_grad(), hooks(
            pre=[(decoder_layers(model)[i].mlp.down_proj, keep(i)) for i in range(n_layers)]
        ):
            logits = model(inputs_embeds=emb, attention_mask=enc["attention_mask"]).logits[:, -1, :].float()
            refusal_score_from_logits(logits, ref_ids).sum().backward()
        attribution = torch.stack(
            [(saved[i][:, -1] * saved[i].grad[:, -1]).float().sum(0) for i in range(n_layers)]
        ).cpu()
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
