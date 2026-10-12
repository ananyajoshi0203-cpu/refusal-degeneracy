# refusal-degeneracy

Testing whether neuron-level LLM safety defenses survive refusal-direction ablation.

Recent defenses redistribute refusal across many MLP neurons and report near-zero attack
success against neuron-level attacks. We ask whether that also spreads refusal across
*mechanisms* (degeneracy) or just makes more copies of one mechanism (redundancy), and
whether a single direction ablation therefore still breaks them.

## Install

```bash
pip install -e ".[dev]"
```

Experiments need a GPU and gated Hugging Face access (AdvBench, HarmBench, StrongREJECT,
and the Llama models). The library and its tests run on CPU with no downloads.

## Layout

```
refusal_degeneracy/
  config.py        experiment config (frozen dataclasses)
  model.py         loading, chat formatting, deterministic generation
  hooks.py         interventions: direction / multi-direction ablation, neuron pin / zero
  activations.py   cache residual-stream and MLP-neuron activations
  directions.py    refusal score, difference-in-means directions, layer selection
  neurons.py       gradient x activation neuron scoring and selection
  judge.py         refusal-substring and Llama-Guard-3 judging
  data.py          leak-free Find / Validate / Test splits
  eval.py          functional checks that interventions remove the targeted structure
  runs.py          JSONL run persistence
tests/             CPU unit tests + functional evals (random tiny model)
notebooks/         step-by-step reproduction (weeks 1-2)
```

## Method

Attacks and defenses are evaluated at matched granularities, following Arditi et al. (2024),
Kazemi et al. (2026), ReFAT, DSA, and NeuronGuard. Prompts used to find or validate an
intervention never appear in a test set; overlapping JailbreakBench prompts are removed before
sampling (`data.build_splits`).

## Development

```bash
make lint    # ruff
make test    # pytest (CPU, no downloads)
```

CI runs ruff and the CPU test suite on every pull request.
