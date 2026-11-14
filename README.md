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
  geometry.py      safety-neuron write geometry: how much refusal runs through one direction
  eval.py          functional checks that interventions remove the targeted structure
  runs.py          JSONL run persistence
tests/             CPU unit tests + functional evals (random tiny model)
scripts/           end-to-end pipelines driven by a config
configs/           example experiment configs
```

## Reproduce

Weeks 1-2 reproduce the refusal direction and the single-neuron attack, then evaluate both
against the held-out test sets:

```bash
python scripts/run_week1.py --config configs/qwen3-1.7b.json
python scripts/run_week1.py --config configs/qwen2.5-7b.json   # A100; matches DSA Table V
python scripts/run_week1.py --model Qwen/Qwen3-1.7B --limit 40 --logits-only  # fast check
```

Outputs (splits, directions, layer and neuron sweeps, per-set refusal rates, and the run
JSONL) are written under `RD_RESULTS` (default `results/`). Each stage is saved as soon as it
finishes; rerun with `--resume` and the same `--out` to skip finished stages and reply files.

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
