"""Fine-tune a model on SST2 under NeuronGuard (arXiv 2608.23959) and save it for the attacks.

Stages, each checkpointed under ``--out`` so ``--resume`` skips what finished: BeaverTails
probe and safety prompts, SST2 user data, then LoRA training with the safety neurons re-probed
every ``--refresh-every`` steps. The merged weights go to ``<out>/model``; run
``scripts/run_week1.py --model <out>/model`` on them next.

    python scripts/run_neuronguard.py --model Qwen/Qwen2.5-7B-Instruct \\
        --out results/qwen2.5-7b-neuronguard --splits results/qwen2.5-7b/splits.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, fields
from pathlib import Path

import torch

from refusal_degeneracy.data import load_beavertails_prompts, load_splits, load_sst2
from refusal_degeneracy.dsa import ResponseExample, tokenize_example
from refusal_degeneracy.model import chat, load_model, set_seed
from refusal_degeneracy.neuronguard import NeuronGuardConfig, train_neuronguard
from refusal_degeneracy.runs import checkpointed

N_PROBE = 750
N_SAFETY = 2000
N_USER = 2000
REFUSAL = "I'm sorry, but I cannot assist with that request."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True, help="splits.json of the undefended run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--n-probe", type=int, default=N_PROBE, help="probe prompts per class")
    parser.add_argument("--n-safety", type=int, default=N_SAFETY, help="harmful prompts trained to refuse")
    parser.add_argument("--n-user", type=int, default=N_USER, help="SST2 examples")
    for field in fields(NeuronGuardConfig):
        name = f"--{field.name.replace('_', '-')}"
        parser.add_argument(name, type=type(field.default), default=field.default)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = NeuronGuardConfig(
        **{field.name: getattr(args, field.name) for field in fields(NeuronGuardConfig)}
    )
    set_seed(config.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "neuronguard_config.json").write_text(
        json.dumps({"model": args.model, **asdict(config)}, indent=1)
    )
    model, tokenizer = load_model(args.model)

    splits = load_splits(args.splits)
    blocked = [
        prompt for name, prompts in splits.items() if not name.startswith("find") for prompt in prompts
    ]

    def beavertails() -> dict[str, list[str]]:
        harmful, benign = load_beavertails_prompts(blocked)
        rng = random.Random(config.seed)
        harmful = rng.sample(harmful, args.n_probe + args.n_safety)
        return {
            "probe_harmful": harmful[: args.n_probe],
            "safety": harmful[args.n_probe :],
            "probe_benign": rng.sample(benign, args.n_probe),
        }

    prompts = checkpointed(args.out / "beavertails.pt", args.resume, beavertails)
    user_text = checkpointed(
        args.out / "user_examples.pt", args.resume, lambda: load_sst2(args.n_user, config.seed)
    )

    def encode_prompt(prompt: str) -> torch.Tensor:
        ids = tokenizer(chat(tokenizer, prompt), add_special_tokens=False).input_ids[: args.max_length]
        return torch.tensor(ids)

    harmful = [
        tokenize_example(tokenizer, ResponseExample(p, REFUSAL), args.max_length) for p in prompts["safety"]
    ]
    user = [tokenize_example(tokenizer, ResponseExample(q, a), args.max_length) for q, a in user_text]
    probe_harmful = [encode_prompt(p) for p in prompts["probe_harmful"]]
    probe_benign = [encode_prompt(p) for p in prompts["probe_benign"]]

    log = train_neuronguard(model, user, harmful, probe_harmful, probe_benign, config)
    (args.out / "training_log.json").write_text(json.dumps([asdict(step) for step in log.steps]))
    (args.out / "safety_neurons.json").write_text(json.dumps(log.safety_neurons))
    model.save_pretrained(args.out / "model", safe_serialization=True)
    tokenizer.save_pretrained(args.out / "model")
    print(f"defended model: {args.out / 'model'}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
