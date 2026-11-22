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
    """Log-odds that the next token is a refusal token, computed in logit space: going through
    probabilities saturates at +inf once a refusal is confident enough to round p to 1.0."""
    is_refusal = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    is_refusal[list(ref_ids)] = True
    logits = logits.float()
    return logits[:, is_refusal].logsumexp(-1) - logits[:, ~is_refusal].logsumexp(-1)


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


def candidate_layers(directions: Tensor, skip_last_frac: float = 0.2, min_norm: float = 1e-6) -> list[int]:
    """Layers eligible for selection: outside the final ``skip_last_frac`` and with a direction to
    normalize. A layer whose inputs are identical for every prompt (layer 0 at the last token,
    which is always the same chat-template token) has a zero direction and would ablate as NaN."""
    n_eligible = int(directions.shape[0] * (1 - skip_last_frac))
    norms = directions[:n_eligible].float().norm(dim=-1)
    return [layer for layer in range(n_eligible) if norms[layer] > min_norm]


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
    vh, vb = val_harmful[:n], val_harmless[:n]
    base = last_logits(model, tokenizer, vb)
    rows = []
    candidates = candidate_layers(directions, skip_last_frac)
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


def kmeans(points: Tensor, k: int, seed: int, iters: int = 50) -> Tensor:
    """Cluster labels ``[n]`` for ``points [n, d]``; Lloyd's algorithm from k-means++ seeds."""
    generator = torch.Generator().manual_seed(seed)
    points = points.float()
    centers = points[torch.randint(len(points), (1,), generator=generator)]
    while len(centers) < k:
        distance = torch.cdist(points, centers).min(1).values ** 2
        pick = torch.multinomial(distance / distance.sum(), 1, generator=generator)
        centers = torch.cat([centers, points[pick]])
    for _ in range(iters):
        labels = torch.cdist(points, centers).argmin(1)
        centers = torch.stack(
            [points[labels == c].mean(0) if (labels == c).any() else centers[c] for c in range(k)]
        )
    return torch.cdist(points, centers).argmin(1)


def conditional_directions(harmful: Tensor, harmless: Tensor, k: int, seed: int = 0) -> Tensor:
    """``[k + 1, d]``: the global difference-in-means refusal direction, then one per k-means
    cluster of the harmful activations, each that cluster's mean minus the harmless mean. A
    defense that refuses through input-specific directions keeps refusing when only the global
    one is ablated; ablating their span removes every one of them."""
    harmless_mean = harmless.float().mean(0)
    labels = kmeans(harmful, k, seed)
    per_cluster = [harmful[labels == c].float().mean(0) - harmless_mean for c in labels.unique()]
    return torch.stack([harmful.float().mean(0) - harmless_mean, *per_cluster])
