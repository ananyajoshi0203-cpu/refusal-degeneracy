"""Weeks 1-2 pipeline: reproduce the refusal direction and the single-neuron attack.

Runs the full chain from an ``ExperimentConfig``: leak-free splits, difference-in-means
direction with layer selection, single refusal-neuron attribution and selection, and a
refusal-rate evaluation of each intervention against the held-out test sets.

    python scripts/run_week1.py --config configs/qwen3-1.7b.json
    python scripts/run_week1.py --model Qwen/Qwen3-1.7B --limit 40 --logits-only
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from refusal_degeneracy.activations import cache_last_token
from refusal_degeneracy.config import ExperimentConfig
from refusal_degeneracy.data import get_splits
from refusal_degeneracy.directions import (
    diff_in_means,
    direction_ablation,
    refusal_score,
    refusal_token_ids,
    select_layer,
)
from refusal_degeneracy.hooks import neuron_pin
from refusal_degeneracy.judge import refusal_rate
from refusal_degeneracy.model import generate, load_model, set_seed
from refusal_degeneracy.neurons import score_neurons, select_neuron
from refusal_degeneracy.runs import results_root, save_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="ExperimentConfig JSON; defaults are used otherwise")
    parser.add_argument("--model", help="override the model name")
    parser.add_argument("--device", help="override the device (cuda / cpu)")
    parser.add_argument("--seed", type=int, help="override the split seed")
    parser.add_argument("--max-new-tokens", type=int, help="override generation length")
    parser.add_argument("--limit", type=int, help="cap test prompts per set for a quick run")
    parser.add_argument("--out", type=Path, help="results root (sets RD_RESULTS)")
    parser.add_argument(
        "--logits-only",
        action="store_true",
        help="score refusal from logits without generating text (fast, no judge)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.max_new_tokens:
        overrides["max_new_tokens"] = args.max_new_tokens
    if args.seed is not None:
        overrides["split"] = replace(config.split, seed=args.seed)
    return replace(config, **overrides) if overrides else config


def run(args: argparse.Namespace) -> dict:
    if args.out:
        import os

        os.environ["RD_RESULTS"] = str(args.out)

    config = config_from_args(args)
    set_seed(config.split.seed)

    splits = get_splits(config.split)
    test_sets = {name: splits[name] for name in splits if name.startswith("test")}
    if args.limit:
        test_sets = {name: prompts[: args.limit] for name, prompts in test_sets.items()}

    model, tokenizer = load_model(config.model, device=args.device)
    ref_ids = refusal_token_ids(tokenizer, config.refusal_words)

    resid_h, mlp_h = cache_last_token(model, tokenizer, splits["find_harmful"], config.cache_batch_size)
    resid_b, _ = cache_last_token(model, tokenizer, splits["find_harmless"], config.cache_batch_size)
    directions = diff_in_means(resid_h, resid_b)
    layer, layer_sweep = select_layer(
        model, tokenizer, directions, splits["val_harmful"], splits["val_harmless"], ref_ids,
        kl_max=config.layer_kl_max,
    )
    direction = directions[layer]

    scores = score_neurons(
        model, tokenizer, splits["find_harmful"], splits["find_harmless"], ref_ids,
        config.attribution_batch_size,
    )
    (neuron_layer, neuron_index, neuron_value), neuron_sweep = select_neuron(
        model, tokenizer, scores, mlp_h.float().mean(0),
        splits["val_harmful"], splits["val_harmless"], ref_ids,
        top=config.neuron_top_k, magnitudes=config.neuron_magnitudes, kl_max=config.neuron_kl_max,
    )

    interventions = {
        "baseline": None,
        "direction_ablation": direction_ablation(model, direction),
        "neuron_pin": neuron_pin(model, neuron_layer, neuron_index, neuron_value),
    }

    summary: dict = {
        "model": config.model,
        "layer": layer,
        "neuron": {"layer": neuron_layer, "index": neuron_index, "value": neuron_value},
        "test_sizes": {name: len(prompts) for name, prompts in test_sets.items()},
        "results": {},
    }

    for set_name, prompts in test_sets.items():
        summary["results"][set_name] = {}
        for intervention, hook_spec in interventions.items():
            if args.logits_only:
                score = refusal_score(model, tokenizer, prompts, ref_ids, hook_spec).mean().item()
                summary["results"][set_name][intervention] = {"refusal_score": score}
                continue
            replies = generate(
                model, tokenizer, prompts,
                max_new_tokens=config.max_new_tokens, batch_size=config.gen_batch_size,
                hook_spec=hook_spec,
            )
            save_run(config.model, f"{set_name}__{intervention}", prompts, replies, set_name)
            summary["results"][set_name][intervention] = {"refusal_rate": refusal_rate(replies)}

    root = results_root()
    root.mkdir(parents=True, exist_ok=True)
    config.to_json(root / "config.json")
    layer_sweep.to_csv(root / "layer_sweep.csv", index=False)
    neuron_sweep.to_csv(root / "neuron_sweep.csv", index=False)
    torch.save(directions, root / "directions.pt")
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
