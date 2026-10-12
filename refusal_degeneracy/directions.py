"""Refusal score, difference-in-means directions, and best-layer selection."""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import pandas as pd
import torch
from torch import Tensor

from .hooks import activation_addition, direction_ablation, hooks
from .model import batches, encode


def refusal_token_ids(tokenizer, words: Sequence[str] = ("I", "As")) -> list[int]:
    ids = set()
    for word in words:
        for variant in (word, " " + word):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
    return sorted(ids)


@torch.no_grad()
def last_logits(model, tokenizer, prompts: Sequence[str], hook_spec=None, batch_size: int = 16) -> Tensor:
    out = []
    ctx = hooks(*hook_spec) if hook_spec else nullcontext()
    with ctx:
        for batch in batches(prompts, batch_size):
            enc = encode(tokenizer, batch, model.device)
            out.append(model(**enc).logits[:, -1, :].float().cpu())
    return torch.cat(out)


def refusal_score_from_logits(logits: Tensor, ref_ids: Sequence[int]) -> Tensor:
    p = logits.softmax(-1)[:, ref_ids].sum(-1).clamp(1e-8, 1 - 1e-8)
    return torch.log(p) - torch.log1p(-p)


def refusal_score(model, tokenizer, prompts, ref_ids, hook_spec=None, batch_size: int = 16) -> Tensor:
    return refusal_score_from_logits(
        last_logits(model, tokenizer, prompts, hook_spec, batch_size), ref_ids
    )


def kl_div(base_logits: Tensor, new_logits: Tensor) -> Tensor:
    p = base_logits.log_softmax(-1)
    q = new_logits.log_softmax(-1)
    return (p.exp() * (p - q)).sum(-1)


def diff_in_means(resid_harmful: Tensor, resid_harmless: Tensor) -> Tensor:
    return resid_harmful.mean(0) - resid_harmless.mean(0)


def select_layer(
    model,
    tokenizer,
    directions: Tensor,
    val_harmful: Sequence[str],
    val_harmless: Sequence[str],
    ref_ids: Sequence[int],
    n: int = 32,
    skip_last_frac: float = 0.2,
    kl_max: float = 0.1,
    progress: bool = True,
) -> tuple[int, pd.DataFrame]:
    """Pick the layer whose direction most reduces refusal while keeping harmless output stable."""
    n_layers = directions.shape[0]
    vh, vb = val_harmful[:n], val_harmless[:n]
    base = last_logits(model, tokenizer, vb)
    rows = []
    candidates = range(int(n_layers * (1 - skip_last_frac)))
    if progress:
        from tqdm.auto import tqdm

        candidates = tqdm(candidates, desc="layers")
    for layer in candidates:
        ablate = direction_ablation(model, directions[layer])
        add = activation_addition(model, layer, directions[layer])
        rows.append(
            {
                "layer": layer,
                "ablate_score": refusal_score(model, tokenizer, vh, ref_ids, ablate).mean().item(),
                "add_score": refusal_score(model, tokenizer, vb, ref_ids, add).mean().item(),
                "kl": kl_div(base, last_logits(model, tokenizer, vb, ablate)).mean().item(),
            }
        )
    df = pd.DataFrame(rows)
    ok = df[(df.add_score > 0) & (df.kl < kl_max)]
    best = (ok if len(ok) else df).sort_values("ablate_score").iloc[0]
    return int(best.layer), df
