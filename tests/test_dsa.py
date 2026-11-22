import math

import pytest
import torch

from refusal_degeneracy.dsa import (
    IGNORE,
    DSAConfig,
    TokenizedExample,
    collate,
    core_dropout,
    core_neurons,
    dsa_scores,
    mean_response_nll,
    response_nll,
    train_dsa,
)
from refusal_degeneracy.hooks import hooks
from refusal_degeneracy.synthetic import mlp_inputs, refusal_unembeddings, toy_model, toy_prompts

REFUSAL_TOKEN = 3
PROMPT_MARKER = 1
SILENT_NEURON_GAIN = 0.1


def refusal_examples(prompts: torch.Tensor, response: list[int]) -> list[TokenizedExample]:
    return [TokenizedExample(torch.cat([p, torch.tensor(response)]), len(p)) for p in prompts]


@pytest.mark.parametrize(
    ("lengths", "response_starts"), [((5,), (3,)), ((4, 7), (2, 6)), ((6, 6, 3), (1, 5, 3))]
)
def test_collate_trains_only_on_response_tokens_and_masks_padding(lengths, response_starts):
    # Arrange
    examples = [
        TokenizedExample(torch.arange(1, length + 1), start)
        for length, start in zip(lengths, response_starts, strict=True)
    ]

    # Act
    batch = collate(examples, "cpu")

    # Assert
    for row, (length, start) in enumerate(zip(lengths, response_starts, strict=True)):
        trained = (batch["labels"][row] != IGNORE).nonzero().flatten().tolist()
        assert trained == list(range(start, length))
        assert batch["attention_mask"][row].tolist() == [1] * length + [0] * (max(lengths) - length)
        assert batch["input_ids"][row, :length].tolist() == list(range(1, length + 1))


@pytest.mark.parametrize(("length", "response_start"), [(6, 5), (6, 2), (8, 1)])
def test_response_nll_sums_only_response_positions(length, response_start):
    # Arrange
    vocab, confidence = 11, 4.0
    token_nll = math.log(math.exp(confidence) + vocab - 1) - confidence
    expected = (length - response_start) * token_nll
    example = TokenizedExample(torch.arange(length) % vocab, response_start)
    logits = torch.zeros(1, length, vocab)
    logits[0, torch.arange(length - 1), example.input_ids[1:]] = confidence

    # Act
    nll = response_nll(logits, collate([example], "cpu")["labels"])

    # Assert
    assert nll.item() == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    ("seed", "layer", "index"), [(0, 1, 7), (0, 3, 50), (1, 2, 30), (2, 1, 7), (2, 3, 50)]
)
def test_dsa_scores_rank_a_neuron_that_writes_the_refusal_token_first(seed, layer, index):
    # Arrange
    model = toy_model(seed)
    (prompts,) = toy_prompts(model, seed, (PROMPT_MARKER,))
    examples = refusal_examples(prompts, [REFUSAL_TOKEN])
    fires_on_every_prompt = mlp_inputs(model, prompts, layer).mean(0)
    writes_refusal = refusal_unembeddings(model, (REFUSAL_TOKEN,))[0]
    mlp = model.model.layers[layer].mlp
    with torch.no_grad():
        mlp.gate_proj.weight[index] = 2 * fires_on_every_prompt / fires_on_every_prompt.norm() ** 2
        mlp.up_proj.weight[index] = 2 * fires_on_every_prompt / fires_on_every_prompt.norm() ** 2
        mlp.down_proj.weight[:, index] = SILENT_NEURON_GAIN * writes_refusal / writes_refusal.norm()

    # Act
    scores = dsa_scores(model, examples, batch_size=4)

    # Assert
    assert core_neurons(scores, 1) == [(layer, index)]
    assert (scores >= 0).all()


@pytest.mark.parametrize("dropout", [0.0, 0.15, 0.3])
def test_core_dropout_silences_the_core_and_rescales_survivors(tiny_model, batch, dropout):
    # Arrange
    core = [(0, 3), (0, 9), (2, 40)]
    layer = 0
    captured: dict[str, torch.Tensor] = {}

    def keep(module, args, kwargs):
        captured.setdefault("x", args[0])

    down = tiny_model.model.layers[layer].mlp.down_proj
    with hooks(pre=[(down, keep)]):
        tiny_model(**batch)
    clean = captured.pop("x")
    generator = torch.Generator().manual_seed(0)

    # Act
    with hooks(*core_dropout(tiny_model, core, dropout, generator)), hooks(pre=[(down, keep)]):
        tiny_model(**batch)
    perturbed = captured["x"]

    # Assert
    core_here = [index for core_layer, index in core if core_layer == layer]
    assert (perturbed[..., core_here] == 0).all()
    rest = torch.ones(clean.shape[-1], dtype=torch.bool)
    rest[core_here] = False
    kept = perturbed[..., rest] != 0
    assert torch.allclose(perturbed[..., rest][kept], clean[..., rest][kept] / (1 - dropout))
    assert (1 - kept.float().mean()).item() == pytest.approx(dropout, abs=0.05)


def test_train_dsa_relearns_refusal_outside_the_core_and_touches_only_the_ffn():
    # Arrange
    seed = 0
    model = toy_model(seed)
    harmful, general_prompts = toy_prompts(model, seed, (PROMPT_MARKER, PROMPT_MARKER + 1), n=8)
    safe = refusal_examples(harmful, [REFUSAL_TOKEN, REFUSAL_TOKEN + 1])
    general = refusal_examples(general_prompts, [REFUSAL_TOKEN + 2])
    core = core_neurons(dsa_scores(model, safe), 16)
    silenced = core_dropout(model, core, 0.0)
    config = DSAConfig(core_size=len(core), lr=1e-2, epochs=4)
    steps = config.epochs * len(safe)
    decay_only = (1 - config.lr * config.weight_decay) ** steps
    before = mean_response_nll(model, safe, silenced)
    original = {name: parameter.clone() for name, parameter in model.named_parameters()}

    # Act
    log = train_dsa(model, safe, general, core, config)

    # Assert
    assert len(log.steps) == steps
    assert mean_response_nll(model, safe, silenced) < before / 4
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name]) != (".mlp." in name), name
    for layer, index in core:
        down = model.model.layers[layer].mlp.down_proj.weight[:, index]
        original_down = original[f"model.layers.{layer}.mlp.down_proj.weight"][:, index]
        assert torch.allclose(down, original_down * decay_only, atol=1e-6)
    assert not any(parameter.requires_grad for parameter in model.parameters())
