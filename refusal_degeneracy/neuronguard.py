"""NeuronGuard (arXiv 2608.23959), reimplemented from the paper.

While a user fine-tunes on their task, NeuronGuard keeps refusing harmful prompts both with and
without the neurons a linear probe marks as safety neurons (Eqs. 3-6), and resolves conflicts
between the task and safety gradients by projection (Eqs. 8-9). The probes are refit every
``refresh_every`` steps, so refusal keeps moving out of whatever neurons currently carry it.
The paper releases no code; choices it leaves open are named fields of ``NeuronGuardConfig``
or noted where they are made.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import cycle

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter

from .dsa import IGNORE, TokenizedExample, collate, response_nll
from .hooks import HookPair, hooks, neuron_zero
from .model import decoder_layers

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class NeuronGuardConfig:
    fraction: float = 0.05
    refresh_every: int = 200
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lr: float = 1e-5
    epochs: int = 10
    probe_steps: int = 300
    probe_lr: float = 0.05
    probe_weight_decay: float = 1e-3
    seed: int = 0


@dataclass(frozen=True)
class SafetyLoss:
    refusal: Tensor
    ablated_refusal: Tensor
    consistency: Tensor

    @property
    def total(self) -> Tensor:
        return self.refusal + self.ablated_refusal + self.consistency


@dataclass(frozen=True)
class StepLoss:
    user: float
    refusal: float
    ablated_refusal: float
    consistency: float
    conflict: bool


@dataclass
class TrainingLog:
    steps: list[StepLoss] = field(default_factory=list)
    safety_neurons: list[list[tuple[int, int]]] = field(default_factory=list)


@dataclass
class LoRA:
    """Low-rank adapters attached as forward hooks, so the module tree and every other hook on it
    stay as they were. ``B`` starts at zero, so the adapted model starts as the base model."""

    modules: list[Linear]
    down: list[Parameter]
    up: list[Parameter]
    scale: float

    @property
    def parameters(self) -> list[Parameter]:
        return [*self.down, *self.up]

    def hook_spec(self) -> tuple[list[HookPair], list[HookPair]]:
        def adapt(down: Parameter, up: Parameter):
            def post(module, inp, out):
                return out + (inp[0] @ down.T.to(inp[0].dtype)) @ up.T.to(inp[0].dtype) * self.scale

            return post

        return [], [
            (module, adapt(down, up))
            for module, down, up in zip(self.modules, self.down, self.up, strict=True)
        ]

    @torch.no_grad()
    def merge(self) -> None:
        for module, down, up in zip(self.modules, self.down, self.up, strict=True):
            module.weight += (up @ down * self.scale).to(module.weight.dtype)


def attach_lora(
    model, rank: int, alpha: float, generator: torch.Generator, targets: Sequence[str] = LORA_TARGETS
) -> LoRA:
    """LoRA on every ``targets`` projection of every decoder layer; Kaiming-uniform ``A`` as in
    the original LoRA. The paper gives the rank and alpha but not the target modules."""
    modules = [
        module
        for layer in decoder_layers(model)
        for name, module in layer.named_modules()
        if isinstance(module, Linear) and name.split(".")[-1] in targets
    ]
    down, up = [], []
    for module in modules:
        bound = 1 / math.sqrt(module.in_features)
        weight = (torch.rand(rank, module.in_features, generator=generator) * 2 - 1) * bound
        down.append(Parameter(weight.to(device=module.weight.device, dtype=torch.float32)))
        up.append(Parameter(torch.zeros(module.out_features, rank, device=module.weight.device)))
    return LoRA(modules, down, up, scale=alpha / rank)


@torch.no_grad()
def last_prompt_mlp(model, prompts: Sequence[Tensor], batch_size: int = 16) -> Tensor:
    """MLP neuron activations (the down-projection input) at each prompt's last token,
    ``[L, n, d_mlp]``. The paper omits the activation site; this is where its ablation acts."""
    layers = decoder_layers(model)
    store: dict[int, Tensor] = {}

    def keep(i: int):
        def pre(module, args, kwargs):
            store[i] = args[0]

        return pre

    chunks = []
    with hooks(pre=[(layer.mlp.down_proj, keep(i)) for i, layer in enumerate(layers)]):
        for start in range(0, len(prompts), batch_size):
            group = prompts[start : start + batch_size]
            batch = collate([TokenizedExample(ids, len(ids)) for ids in group], model.device)
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            last = batch["attention_mask"].sum(-1) - 1
            rows = torch.arange(len(group), device=last.device)
            chunks.append(torch.stack([store[i][rows, last].float() for i in range(len(layers))]))
    return torch.cat(chunks, dim=1)


def fit_probe(features: Tensor, harmful: Tensor, steps: int, lr: float, weight_decay: float) -> Tensor:
    """Eq. 1: logistic regression of harmful (1) against benign (0) on one layer's
    ``[n, d]`` features; returns the weight ``W_l``. Optimizer and regularization are not
    stated in the paper; full-batch Adam with weight decay keeps separable data finite."""
    with torch.enable_grad():
        weight = torch.zeros(features.shape[1], device=features.device, requires_grad=True)
        bias = torch.zeros(1, device=features.device, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=lr, weight_decay=weight_decay)
        for _ in range(steps):
            optimizer.zero_grad()
            F.binary_cross_entropy_with_logits(features @ weight + bias, harmful.float()).backward()
            optimizer.step()
    return weight.detach().cpu()


def safety_neurons(probe_weights: Sequence[Tensor], fraction: float) -> list[tuple[int, int]]:
    """Eq. 2: per layer, the ``fraction`` of neurons with the largest signed probe weight, the
    ones whose activation most predicts a harmful prompt."""
    neurons = []
    for layer, weight in enumerate(probe_weights):
        count = max(1, round(fraction * len(weight)))
        neurons += [(layer, int(index)) for index in weight.topk(count).indices]
    return neurons


def locate_safety_neurons(
    model, harmful: Sequence[Tensor], benign: Sequence[Tensor], config: NeuronGuardConfig
) -> list[tuple[int, int]]:
    acts = last_prompt_mlp(model, [*harmful, *benign])
    labels = torch.cat([torch.ones(len(harmful)), torch.zeros(len(benign))]).to(acts.device)
    weights = [
        fit_probe(layer, labels, config.probe_steps, config.probe_lr, config.probe_weight_decay)
        for layer in acts
    ]
    return safety_neurons(weights, config.fraction)


def safety_loss(model, example: TokenizedExample, neurons: Sequence[tuple[int, int]]) -> SafetyLoss:
    """Eqs. 3-6 on one harmful prompt and its refusal: refusal with and without the safety
    neurons, plus ``KL(p_ablated || p)`` over the response tokens. All three are token means."""
    batch = collate([example], model.device)
    response = batch["labels"][:, 1:] != IGNORE
    n_tokens = response.sum()

    def forward() -> Tensor:
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

    logits = forward()
    with hooks(*neuron_zero(model, neurons)):
        ablated_logits = forward()
    log_p = logits[:, :-1].float().log_softmax(-1)[response]
    log_p_ablated = ablated_logits[:, :-1].float().log_softmax(-1)[response]
    consistency = (log_p_ablated.exp() * (log_p_ablated - log_p)).sum() / n_tokens
    return SafetyLoss(
        refusal=response_nll(logits, batch["labels"]).sum() / n_tokens,
        ablated_refusal=response_nll(ablated_logits, batch["labels"]).sum() / n_tokens,
        consistency=consistency,
    )


def user_loss(model, example: TokenizedExample) -> Tensor:
    batch = collate([example], model.device)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return response_nll(logits, batch["labels"]).sum() / (batch["labels"] != IGNORE).sum()


def combine_gradients(user: Tensor, safe: Tensor, safe_first: bool) -> Tensor:
    """Eqs. 8-9 on flattened gradients: when they conflict, the first of the (randomly ordered)
    pair loses its component along the second; the sum is the update direction."""
    if user @ safe >= 0:
        return user + safe
    first, second = (safe, user) if safe_first else (user, safe)
    return first - (first @ second) / (second @ second) * second + second


@dataclass(frozen=True)
class StepGradient:
    user: Tensor
    safe: Tensor
    combined: Tensor
    user_loss: float
    safety: SafetyLoss


def step_gradient(
    model,
    lora: LoRA,
    user_example: TokenizedExample,
    harmful_example: TokenizedExample,
    neurons: Sequence[tuple[int, int]],
    safe_first: bool,
) -> StepGradient:
    """One step of Algorithm 1: the task and safety gradients on the LoRA weights, flattened,
    and their projected combination. ``model`` must already carry ``lora``'s hooks."""
    with torch.enable_grad():
        task = user_loss(model, user_example)
        safety = safety_loss(model, harmful_example, neurons)
        user = _flatten(torch.autograd.grad(task, lora.parameters, allow_unused=True), lora.parameters)
        safe = _flatten(
            torch.autograd.grad(safety.total, lora.parameters, allow_unused=True), lora.parameters
        )
    return StepGradient(user, safe, combine_gradients(user, safe, safe_first), task.item(), safety)


def train_neuronguard(
    model,
    user: Sequence[TokenizedExample],
    harmful: Sequence[TokenizedExample],
    probe_harmful: Sequence[Tensor],
    probe_benign: Sequence[Tensor],
    config: NeuronGuardConfig,
) -> TrainingLog:
    """Algorithm 1 with one user and one harmful example per step (batch size is not stated)
    and AdamW applying the projected gradient (the paper writes plain gradient descent; at its
    LR of 1e-5 that would barely move zero-initialized LoRA weights). Merges LoRA at the end."""
    generator = torch.Generator().manual_seed(config.seed)
    lora = attach_lora(model, config.lora_rank, config.lora_alpha, generator)
    optimizer = torch.optim.AdamW(lora.parameters, lr=config.lr, weight_decay=0.0)
    harmful_stream = cycle(harmful)
    log = TrainingLog()
    neurons: list[tuple[int, int]] = []
    step = 0
    with hooks(*lora.hook_spec()):
        for _ in range(config.epochs):
            for index in torch.randperm(len(user), generator=generator).tolist():
                if step % config.refresh_every == 0:
                    neurons = locate_safety_neurons(model, probe_harmful, probe_benign, config)
                    log.safety_neurons.append(neurons)
                safe_first = bool(torch.rand(1, generator=generator) < 0.5)
                gradient = step_gradient(model, lora, user[index], next(harmful_stream), neurons, safe_first)
                _assign_gradients(gradient.combined, lora.parameters)
                optimizer.step()
                log.steps.append(
                    StepLoss(
                        user=gradient.user_loss,
                        refusal=gradient.safety.refusal.item(),
                        ablated_refusal=gradient.safety.ablated_refusal.item(),
                        consistency=gradient.safety.consistency.item(),
                        conflict=bool(gradient.user @ gradient.safe < 0),
                    )
                )
                step += 1
    lora.merge()
    return log


def _flatten(grads: Sequence[Tensor | None], parameters: Sequence[Parameter]) -> Tensor:
    return torch.cat(
        [
            (grad if grad is not None else torch.zeros_like(parameter)).flatten().float()
            for grad, parameter in zip(grads, parameters, strict=True)
        ]
    )


def _assign_gradients(flat: Tensor, parameters: Sequence[Parameter]) -> None:
    offset = 0
    for parameter in parameters:
        parameter.grad = flat[offset : offset + parameter.numel()].view_as(parameter).to(parameter.dtype)
        offset += parameter.numel()
