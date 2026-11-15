"""Train a DSA-defended copy of a model (arXiv 2608.01414) and save it for the weeks 1-2 attacks.

Stages, each checkpointed under ``--out`` so ``--resume`` skips what finished:
refusal targets for 2,000 harmful prompts, NaturalReasoning general data, Eq. 4 neuron scores
on the localization subset, then FFN training with the core silenced. The defended weights go
to ``<out>/model``; run ``scripts/run_week1.py --model <out>/model`` on them next.

    python scripts/run_dsa.py --model Qwen/Qwen2.5-7B-Instruct --out results/qwen2.5-7b-dsa \\
        --splits results/qwen2.5-7b/splits.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, fields
from pathlib import Path

import torch

from refusal_degeneracy.data import load_dsa_harmful_pool, load_natural_reasoning, load_splits
from refusal_degeneracy.dsa import (
    DSAConfig,
    ResponseExample,
    core_neurons,
    dsa_scores,
    tokenize_example,
    train_dsa,
)
from refusal_degeneracy.judge import is_refusal
from refusal_degeneracy.model import generate, load_model, set_seed
from refusal_degeneracy.runs import checkpointed

N_SAFE = 2000
N_GENERAL = 2000
N_LOCALIZE = 1000
TARGET_TOKENS = 256
FALLBACK_REFUSAL = "I'm sorry, but I can't help with that request."


def refusal_targets(model, tokenizer, prompts: list[str], batch_size: int) -> list[ResponseExample]:
    """The model's own greedy reply where it already refuses, so training teaches refusal in the
    model's voice; a fixed refusal where it complies. The paper does not say how it built them."""
    replies = generate(model, tokenizer, prompts, max_new_tokens=TARGET_TOKENS, batch_size=batch_size)
    return [
        ResponseExample(prompt, reply if is_refusal(reply) else FALLBACK_REFUSAL)
        for prompt, reply in zip(prompts, replies, strict=True)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True, help="splits.json of the undefended run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16, help="generation batch")
    parser.add_argument("--n-safe", type=int, default=N_SAFE, help="harmful prompts trained to refuse")
    parser.add_argument("--n-general", type=int, default=N_GENERAL, help="NaturalReasoning examples")
    parser.add_argument("--n-localize", type=int, default=N_LOCALIZE, help="examples scored by Eq. 4")
    for field in fields(DSAConfig):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}", type=type(field.default), default=field.default
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DSAConfig(**{field.name: getattr(args, field.name) for field in fields(DSAConfig)})
    set_seed(config.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dsa_config.json").write_text(json.dumps({"model": args.model, **asdict(config)}, indent=1))
    model, tokenizer = load_model(args.model)

    splits = load_splits(args.splits)
    blocked = [
        prompt for name, prompts in splits.items() if not name.startswith("find") for prompt in prompts
    ]

    def harmful_prompts() -> list[str]:
        return random.Random(config.seed).sample(load_dsa_harmful_pool(blocked), args.n_safe)

    prompts = checkpointed(args.out / "harmful_prompts.pt", args.resume, harmful_prompts)
    safe_text = checkpointed(
        args.out / "safe_examples.pt",
        args.resume,
        lambda: refusal_targets(model, tokenizer, prompts, args.batch_size),
    )
    general_text = checkpointed(
        args.out / "general_examples.pt",
        args.resume,
        lambda: [ResponseExample(q, a) for q, a in load_natural_reasoning(args.n_general, config.seed)],
    )
    safe = [tokenize_example(tokenizer, example, config.max_length) for example in safe_text]
    general = [tokenize_example(tokenizer, example, config.max_length) for example in general_text]

    scores = checkpointed(
        args.out / "dsa_scores.pt",
        args.resume,
        lambda: dsa_scores(model, safe[: args.n_localize], batch_size=1),
    )
    core = core_neurons(scores, config.core_size)
    (args.out / "core_neurons.json").write_text(json.dumps(core))

    log = train_dsa(model, safe, general, core, config)
    (args.out / "training_log.json").write_text(json.dumps([asdict(step) for step in log.steps]))
    model.save_pretrained(args.out / "model", safe_serialization=True)
    tokenizer.save_pretrained(args.out / "model")
    own = sum(example.response != FALLBACK_REFUSAL for example in safe_text)
    print(f"refusal targets in the model's own words: {own}/{len(safe_text)}")
    print(f"defended model: {args.out / 'model'}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
