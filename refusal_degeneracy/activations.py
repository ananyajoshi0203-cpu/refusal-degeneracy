"""Cache residual-stream and MLP-neuron activations at the final prompt token."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from .hooks import hooks
from .model import batches, decoder_layers, encode


@torch.no_grad()
def cache_last_token(
    model, tokenizer, prompts: Sequence[str], batch_size: int = 16, progress: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(resid [n, L, d], mlp [n, L, d_mlp])`` at the last token of each prompt."""
    n_layers = len(decoder_layers(model))
    resid, mlp, store = [], [], {}

    def save_resid(i):
        def pre(module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            store[("r", i)] = x[:, -1, :].float().cpu()

        return pre

    def save_mlp(i):
        def pre(module, args, kwargs):
            store[("m", i)] = args[0][:, -1, :].float().cpu()

        return pre

    pre_hooks = [(decoder_layers(model)[i], save_resid(i)) for i in range(n_layers)]
    pre_hooks += [(decoder_layers(model)[i].mlp.down_proj, save_mlp(i)) for i in range(n_layers)]
    with hooks(pre=pre_hooks):
        for batch in batches(prompts, batch_size, progress, "cache"):
            enc = encode(tokenizer, batch, model.device)
            model(**enc)
            resid.append(torch.stack([store[("r", i)] for i in range(n_layers)], dim=1))
            mlp.append(torch.stack([store[("m", i)] for i in range(n_layers)], dim=1).half())
    return torch.cat(resid), torch.cat(mlp)
