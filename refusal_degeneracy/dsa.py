"""Distributed Safety Alignment (DSA, arXiv 2608.01414), reimplemented from the paper.

DSA localizes the FFN neurons that carry refusal (Eq. 4), then fine-tunes the FFN with those
neurons silenced and random dropout on the rest (Eq. 9), so refusal has to be relearned by
neurons outside the core set (Eq. 13). The paper releases no code; choices it leaves open are
named fields of ``DSAConfig`` or noted where they are made.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import cycle

import torch
import torch.nn.functional as F
from torch import Tensor

from .hooks import HookPair, hooks
from .model import chat, decoder_layers
from .neurons import top_neurons

IGNORE = -100


@dataclass(frozen=True)
class ResponseExample:
    prompt: str
    response: str


@dataclass(frozen=True)
class TokenizedExample:
    """One chat-formatted prompt and its target response; only the response is trained on."""

    input_ids: Tensor
    response_start: int

    @property
    def labels(self) -> Tensor:
        labels = self.input_ids.clone()
        labels[: self.response_start] = IGNORE
        return labels


@dataclass(frozen=True)
class DSAConfig:
    core_size: int = 8000
    dropout: float = 0.15
    lambda_gen: float = 0.5
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    max_length: int = 1024
    seed: int = 42


@dataclass(frozen=True)
class StepLoss:
    safe: float
    general: float


@dataclass
class TrainingLog:
    steps: list[StepLoss] = field(default_factory=list)


def tokenize_example(tokenizer, example: ResponseExample, max_length: int) -> TokenizedExample:
    prompt_ids = tokenizer(chat(tokenizer, example.prompt), add_special_tokens=False).input_ids
    response_ids = tokenizer(example.response, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    ids = (prompt_ids + response_ids)[:max_length]
    return TokenizedExample(torch.tensor(ids), response_start=min(len(prompt_ids), len(ids)))


def collate(examples: Sequence[TokenizedExample], device: torch.device | str) -> dict[str, Tensor]:
    """Right-pad a batch; padding is masked out of attention and out of the labels."""
    width = max(len(example.input_ids) for example in examples)
    input_ids = torch.zeros(len(examples), width, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    attention_mask = torch.zeros(len(examples), width, dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = example.input_ids
        labels[row, :length] = example.labels
        attention_mask[row, :length] = 1
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
    }


def response_nll(logits: Tensor, labels: Tensor) -> Tensor:
    """Summed negative log-likelihood of the response tokens, per example (Eqs. 2 and 10)."""
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    per_token = F.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels, ignore_index=IGNORE, reduction="none"
    )
    return per_token.sum(-1)


def dsa_scores(model, examples: Sequence[TokenizedExample], batch_size: int = 1) -> Tensor:
    """Eq. 4: per neuron, the refusal loss it prevents, ``max(0, -a * dL/da)``, averaged over the
    response tokens of each example and then over examples. Returns ``[L, d_mlp]``.

    A token's activation counts when it predicts a response token, the positions the loss of
    Eq. 2 is taken over."""
    layers = decoder_layers(model)
    total = torch.zeros(len(layers), layers[0].mlp.down_proj.in_features)
    for start in range(0, len(examples), batch_size):
        batch = collate(examples[start : start + batch_size], model.device)
        acts: dict[int, Tensor] = {}
        keep = [(layer.mlp.down_proj, _retain_input(acts, i)) for i, layer in enumerate(layers)]
        emb = model.get_input_embeddings()(batch["input_ids"]).detach().requires_grad_(True)
        with torch.enable_grad(), hooks(pre=keep):
            logits = model(inputs_embeds=emb, attention_mask=batch["attention_mask"]).logits
            response_nll(logits, batch["labels"]).sum().backward()

        predicts_response = torch.zeros_like(batch["labels"], dtype=torch.float)
        predicts_response[:, :-1] = (batch["labels"][:, 1:] != IGNORE).float()
        weights = predicts_response / predicts_response.sum(-1, keepdim=True).clamp(min=1)
        total += torch.stack(
            [
                torch.einsum("bt,btd->d", weights, (-acts[i].detach() * acts[i].grad).float().clamp(min=0))
                for i in range(len(layers))
            ]
        ).cpu()
    return total / len(examples)


def _retain_input(store: dict[int, Tensor], i: int):
    def pre(module, args, kwargs):
        args[0].retain_grad()
        store[i] = args[0]

    return pre


def core_neurons(scores: Tensor, size: int) -> list[tuple[int, int]]:
    """Eqs. 5-6: the ``size`` highest-scoring neurons, ranked globally across layers."""
    return top_neurons(scores, size)


def core_dropout(
    model, core: Iterable[tuple[int, int]], dropout: float, generator: torch.Generator | None = None
) -> tuple[list[HookPair], list[HookPair]]:
    """Eq. 9: silence the core neurons and drop each other neuron-token pair with probability
    ``dropout``, rescaling survivors by ``1 / (1 - dropout)``. The mask is resampled per forward;
    ``generator`` must live on the model's device."""
    layers = decoder_layers(model)
    keep = torch.ones(len(layers), layers[0].mlp.down_proj.in_features)
    for layer, index in core:
        keep[layer, index] = 0

    def perturb(i: int):
        def pre(module, args, kwargs):
            x = args[0]
            mask = keep[i].to(device=x.device, dtype=x.dtype)
            if dropout > 0:
                survive = torch.rand(x.shape, generator=generator, device=x.device)
                mask = mask * (survive >= dropout).to(x.dtype) / (1 - dropout)
            return (x * mask, *args[1:]), kwargs

        return pre

    return [(layer.mlp.down_proj, perturb(i)) for i, layer in enumerate(layers)], []


def ffn_parameters(model) -> list[Tensor]:
    """theta_ff: the gate, up and down projections of every decoder layer (75% of Qwen2.5-7B,
    against the paper's stated 74.88% trained)."""
    return [
        projection.weight
        for layer in decoder_layers(model)
        for projection in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
    ]


def train_dsa(
    model,
    safe: Sequence[TokenizedExample],
    general: Sequence[TokenizedExample],
    core: Iterable[tuple[int, int]],
    config: DSAConfig,
) -> TrainingLog:
    """Eq. 13 with batch size 1 and constant LR (Appendix B). Each step pairs one refusal example
    with one general example, cycling the general set, so an epoch is one pass over ``safe``."""
    order_generator = torch.Generator().manual_seed(config.seed)
    dropout_generator = torch.Generator(device=model.device).manual_seed(config.seed)
    parameters = ffn_parameters(model)
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    log = TrainingLog()
    general_stream = cycle(general)
    try:
        with torch.enable_grad(), hooks(*core_dropout(model, list(core), config.dropout, dropout_generator)):
            for _ in range(config.epochs):
                order = torch.randperm(len(safe), generator=order_generator).tolist()
                for index in order:
                    safe_loss = _loss(model, safe[index])
                    general_loss = _loss(model, next(general_stream))
                    optimizer.zero_grad(set_to_none=True)
                    (safe_loss + config.lambda_gen * general_loss).backward()
                    optimizer.step()
                    log.steps.append(StepLoss(safe=safe_loss.item(), general=general_loss.item()))
    finally:
        for parameter in parameters:
            parameter.requires_grad_(False)
        optimizer.zero_grad(set_to_none=True)
    return log


def _loss(model, example: TokenizedExample) -> Tensor:
    batch = collate([example], model.device)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return response_nll(logits, batch["labels"]).sum()


@torch.no_grad()
def mean_response_nll(model, examples: Sequence[TokenizedExample], hook_spec=None) -> float:
    total = 0.0
    with hooks(*hook_spec) if hook_spec else hooks():
        for example in examples:
            total += _loss(model, example).item()
    return total / len(examples)
