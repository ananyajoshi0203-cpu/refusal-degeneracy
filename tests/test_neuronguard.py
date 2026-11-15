import math

import pytest
import torch

from refusal_degeneracy.dsa import TokenizedExample
from refusal_degeneracy.hooks import hooks, neuron_zero
from refusal_degeneracy.neuronguard import (
    NeuronGuardConfig,
    attach_lora,
    combine_gradients,
    fit_probe,
    safety_loss,
    safety_neurons,
    step_gradient,
    train_neuronguard,
    user_loss,
)
from refusal_degeneracy.synthetic import toy_model, toy_prompts

HARMFUL_MARKER, TASK_MARKER = 1, 2
REFUSAL_TOKEN, TASK_ANSWER = 3, 6


def answered(prompts: torch.Tensor, answer: int) -> list[TokenizedExample]:
    return [TokenizedExample(torch.cat([p, torch.tensor([answer])]), len(p)) for p in prompts]


@pytest.mark.parametrize(("predictive", "fraction"), [((4,), 0.1), ((0, 7), 0.2), ((2, 5, 9), 0.3)])
def test_safety_neurons_are_the_ones_whose_activation_predicts_harm(predictive, fraction):
    # Arrange
    width, n = 10, 64
    generator = torch.Generator().manual_seed(0)
    harmful = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])
    layers = []
    for _ in range(2):
        features = torch.randn(n, width, generator=generator) * 0.1
        features[:, list(predictive)] += harmful[:, None]
        features[:, 1 if 1 not in predictive else 3] -= harmful
        layers.append(features)

    # Act
    weights = [fit_probe(features, harmful, steps=300, lr=0.05, weight_decay=1e-3) for features in layers]
    neurons = safety_neurons(weights, fraction)

    # Assert
    assert sorted(neurons) == sorted((layer, index) for layer in range(2) for index in predictive)


@pytest.mark.parametrize(
    ("user", "safe", "safe_first", "expected"),
    [
        ((1.0, 0.0), (0.0, 2.0), False, (1.0, 2.0)),
        ((1.0, 0.0), (-1.0, 1.0), False, (-0.5, 1.5)),
        ((1.0, 0.0), (-1.0, 1.0), True, (1.0, 1.0)),
    ],
)
def test_combine_gradients_projects_only_conflicting_pairs(user, safe, safe_first, expected):
    # Arrange
    user_grad, safe_grad = torch.tensor(user), torch.tensor(safe)

    # Act
    combined = combine_gradients(user_grad, safe_grad, safe_first)

    # Assert
    assert torch.allclose(combined, torch.tensor(expected))


@pytest.mark.parametrize("neurons", [[], [(0, 5), (1, 9), (2, 30), (3, 2)]])
def test_safety_loss_without_safety_neurons_is_twice_the_refusal_loss(neurons):
    # Arrange
    model = toy_model(0)
    (prompts,) = toy_prompts(model, 0, (HARMFUL_MARKER,), n=1)
    (example,) = answered(prompts, REFUSAL_TOKEN)
    with hooks(*neuron_zero(model, neurons)):
        expected_ablated = user_loss(model, example)

    # Act
    loss = safety_loss(model, example, neurons)

    # Assert
    assert loss.refusal.item() == pytest.approx(user_loss(model, example).item(), rel=1e-5)
    assert loss.ablated_refusal.item() == pytest.approx(expected_ablated.item(), rel=1e-5)
    assert (loss.consistency.item() == pytest.approx(0, abs=1e-6)) == (not neurons)
    assert loss.consistency.item() >= -1e-6
    terms = (loss.refusal, loss.ablated_refusal, loss.consistency)
    assert loss.total.item() == pytest.approx(sum(term.item() for term in terms), rel=1e-6)


@pytest.mark.parametrize("targets", [("q_proj", "down_proj"), ("gate_proj", "up_proj", "o_proj")])
def test_lora_hooks_and_merged_weights_compute_the_same_function(batch, targets):
    # Arrange
    model = toy_model(1)
    generator = torch.Generator().manual_seed(0)
    lora = attach_lora(model, rank=4, alpha=8, generator=generator, targets=targets)
    base = model(**batch).logits
    with hooks(*lora.hook_spec()):
        untrained = model(**batch).logits
    with torch.no_grad():
        for up in lora.up:
            up.normal_(generator=generator)
    with hooks(*lora.hook_spec()):
        adapted = model(**batch).logits

    # Act
    lora.merge()
    merged = model(**batch).logits

    # Assert
    assert torch.allclose(untrained, base)
    assert not torch.allclose(adapted, base)
    assert torch.allclose(merged, adapted, rtol=1e-4, atol=1e-3)
    assert len(lora.modules) == len(targets) * model.config.num_hidden_layers


@pytest.mark.parametrize("safe_first", [False, True])
def test_step_gradient_trains_all_three_safety_terms_and_projects_conflicts(safe_first):
    # Arrange
    model = toy_model(0)
    harmful_prompts, task_prompts = toy_prompts(model, 0, (HARMFUL_MARKER, TASK_MARKER), n=1)
    (harmful,), (task,) = answered(harmful_prompts, REFUSAL_TOKEN), answered(task_prompts, TASK_ANSWER)
    lora = attach_lora(model, rank=4, alpha=8, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for up in lora.up:
            up.normal_(std=0.1, generator=torch.Generator().manual_seed(1))

    # Act
    with hooks(*lora.hook_spec()):
        unablated = step_gradient(model, lora, task, harmful, [], safe_first)
        refusal_only = step_gradient(model, lora, harmful, task, [], safe_first).user
        ablated = step_gradient(model, lora, task, harmful, [(1, 4), (2, 9)], safe_first)

    # Assert
    assert torch.allclose(unablated.safe, 2 * refusal_only, rtol=1e-4, atol=1e-6)
    assert not torch.allclose(ablated.safe, unablated.safe)
    assert ablated.user @ ablated.safe < 0
    kept = ablated.user if safe_first else ablated.safe
    projected = ablated.combined - kept
    assert (projected @ kept).abs() < 1e-4 * projected.norm() * kept.norm()


def test_train_neuronguard_keeps_refusal_when_its_safety_neurons_are_ablated():
    # Arrange
    seed = 0
    model = toy_model(seed)
    harmful_prompts, task_prompts = toy_prompts(model, seed, (HARMFUL_MARKER, TASK_MARKER), n=8)
    harmful = answered(harmful_prompts, REFUSAL_TOKEN)
    user = answered(task_prompts, TASK_ANSWER)
    config = NeuronGuardConfig(lr=3e-3, epochs=3, refresh_every=5, lora_rank=4, lora_alpha=8)
    steps = config.epochs * len(user)
    original = {name: parameter.clone() for name, parameter in model.named_parameters()}

    # Act
    log = train_neuronguard(model, user, harmful, list(harmful_prompts), list(task_prompts), config)

    # Assert
    final_neurons = log.safety_neurons[-1]
    with hooks(*neuron_zero(model, final_neurons)):
        ablated_refusal = sum(user_loss(model, example).item() for example in harmful) / len(harmful)
    assert len(log.steps) == steps
    assert len(log.safety_neurons) == math.ceil(steps / config.refresh_every)
    assert len(final_neurons) == model.config.num_hidden_layers * round(
        config.fraction * model.config.intermediate_size
    )
    assert ablated_refusal < log.steps[0].ablated_refusal / 4
    assert sum(user_loss(model, example).item() for example in user) / len(user) < log.steps[0].user / 4
    assert not torch.equal(
        model.model.layers[0].self_attn.q_proj.weight, original["model.layers.0.self_attn.q_proj.weight"]
    )
    assert torch.equal(model.lm_head.weight, original["lm_head.weight"])
