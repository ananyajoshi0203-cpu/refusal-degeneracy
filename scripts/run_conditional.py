"""Attack a finished weeks 1-2 run with cluster-conditional direction ablation.

Clusters the find-split harmful activations at the run's refusal layer, ablates the span of the
global direction and one difference-in-means direction per cluster (the attack of Piras et al.
2025, with k-means in place of a SOM), and next to it as many random directions as a control.
The cluster count is chosen on the validation prompts under the run's layer KL budget; the test
sets are generated only for the chosen count, as JSONL that ``judge_runs.py`` reads.

    python scripts/run_conditional.py --run results/qwen2.5-7b
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import torch
from torch import Tensor

from refusal_degeneracy.activations import cache_last_token
from refusal_degeneracy.config import ExperimentConfig
from refusal_degeneracy.data import get_splits
from refusal_degeneracy.directions import (
    candidate_layers,
    conditional_directions,
    kl_div,
    kmeans,
    last_logits,
    refusal_token_ids,
)
from refusal_degeneracy.hooks import direction_ablation, multi_direction_ablation
from refusal_degeneracy.judge import refusal_rate
from refusal_degeneracy.model import generate, load_model
from refusal_degeneracy.runs import completed_replies, save_run
from refusal_degeneracy.scoring import HookSpec, RefusalMetric, refusal_gap
from refusal_degeneracy.stages import DirectionStage

DEFAULT_KS = (2, 4, 8, 16)
KL_BUDGETS = (0.1, 0.3, 1.0)
CONTROL_SEED = 0


@dataclass(frozen=True)
class ClusterAblation:
    k: int
    cluster_sizes: list[int]
    retained_refusal: float
    retained_random_control: float
    kl: float
    kl_random_control: float


class Attack(str, Enum):
    DIRECTION = "direction"
    CLUSTER = "cluster"


@dataclass(frozen=True)
class AttackPoint:
    """One ablation's cost and effect: ``k`` directions removed at ``layer``."""

    attack: Attack
    layer: int
    k: int
    retained_refusal: float
    kl: float


@dataclass(frozen=True)
class ConditionalReport:
    model: str
    layer: int
    metric: RefusalMetric
    n_prompts: int
    refusal_gap: float
    sweep: list[ClusterAblation]
    chosen_k: int | None
    refusal_rates: dict[str, dict[str, float]]
    layer_sweep: list[AttackPoint]
    strongest_within_kl: dict[float, AttackPoint | None]


def random_control(directions: Tensor, seed: int) -> Tensor:
    """The global direction (row 0) plus as many random directions as there are clusters."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(len(directions) - 1, directions.shape[1], generator=generator)
    return torch.cat([directions[:1].float(), noise])


def sweep_clusters(
    model,
    harmful: Tensor,
    harmless: Tensor,
    ks: Sequence[int],
    refusal_gap: Callable[[HookSpec | None], float],
    kl: Callable[[HookSpec], float],
) -> tuple[float, list[ClusterAblation]]:
    """For each ``k``: share of the refusal gap retained and the harmless KL, under the cluster
    directions and under the random control. ``harmful`` and ``harmless`` are ``[n, d]``
    activations at the refusal layer."""
    gap = refusal_gap(None)
    rows = []
    for k in ks:
        directions = conditional_directions(harmful, harmless, k)
        control = random_control(directions, CONTROL_SEED + k)
        attack, controlled = (
            multi_direction_ablation(model, directions),
            multi_direction_ablation(model, control),
        )
        rows.append(
            ClusterAblation(
                k=k,
                cluster_sizes=kmeans(harmful, k, seed=0).bincount(minlength=k).tolist(),
                retained_refusal=refusal_gap(attack) / gap,
                retained_random_control=refusal_gap(controlled) / gap,
                kl=kl(attack),
                kl_random_control=kl(controlled),
            )
        )
    return gap, rows


def sweep_layers(
    model,
    directions: Tensor,
    layers: Iterable[int],
    gap: float,
    refusal_gap: Callable[[HookSpec | None], float],
    kl: Callable[[HookSpec], float],
) -> list[AttackPoint]:
    """Single-direction ablation of each layer's difference-in-means direction, scored by the
    share of the unablated ``gap`` it leaves and its harmless KL."""
    points = []
    for layer in layers:
        attack = direction_ablation(model, directions[layer])
        points.append(AttackPoint(Attack.DIRECTION, layer, 1, refusal_gap(attack) / gap, kl(attack)))
    return points


def strongest_within(points: Iterable[AttackPoint], kl_max: float) -> AttackPoint | None:
    """The attack that leaves the least refusal among those whose harmless KL is within
    ``kl_max``; None when none is. Over a range of budgets this traces the attacker's
    refusal-vs-KL frontier, so no single budget decides whether a defense holds."""
    within = [point for point in points if point.kl <= kl_max]
    return min(within, key=lambda point: point.retained_refusal) if within else None


def choose_k(rows: Sequence[ClusterAblation], kl_max: float) -> int | None:
    """The cluster count that removes the most refusal while keeping harmless behaviour within
    ``kl_max``; None when every count exceeds it."""
    within = [row for row in rows if row.kl < kl_max]
    return min(within, key=lambda row: row.retained_refusal).k if within else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="output directory of run_week1.py")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS), help="cluster counts")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="layers for single-direction ablation (default: every candidate)",
    )
    parser.add_argument("--n", type=int, default=64, help="validation prompts per side")
    parser.add_argument(
        "--metric", type=RefusalMetric, choices=list(RefusalMetric), default=RefusalMetric.GENERATED
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", help="override the device (cuda / cpu)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ["RD_RESULTS"] = str(args.run)
    config = ExperimentConfig.from_json(args.run / "config.json")
    splits = get_splits(config.split)
    stage: DirectionStage = torch.load(args.run / "direction_stage.pt", weights_only=False)
    model, tokenizer = load_model(config.model, device=args.device)
    ref_ids = refusal_token_ids(tokenizer, config.refusal_words)

    resid_harmful, _ = cache_last_token(model, tokenizer, splits["find_harmful"], config.cache_batch_size)
    resid_harmless, _ = cache_last_token(model, tokenizer, splits["find_harmless"], config.cache_batch_size)
    val_harmful, val_harmless = splits["val_harmful"][: args.n], splits["val_harmless"][: args.n]
    base_logits = last_logits(model, tokenizer, val_harmless)

    gap_of = refusal_gap(
        args.metric, model, tokenizer, val_harmful, val_harmless, ref_ids, config.gen_batch_size
    )

    def kl(hook_spec: HookSpec) -> float:
        return kl_div(base_logits, last_logits(model, tokenizer, val_harmless, hook_spec)).mean().item()

    gap, rows = sweep_clusters(
        model, resid_harmful[:, stage.layer], resid_harmless[:, stage.layer], args.k, gap_of, kl
    )
    layers = args.layers or candidate_layers(stage.directions)
    layer_points = sweep_layers(model, stage.directions, layers, gap, gap_of, kl)
    cluster_points = [
        AttackPoint(Attack.CLUSTER, stage.layer, row.k + 1, row.retained_refusal, row.kl) for row in rows
    ]
    strongest = {budget: strongest_within([*layer_points, *cluster_points], budget) for budget in KL_BUDGETS}
    chosen = choose_k(rows, config.layer_kl_max)
    rates: dict[str, dict[str, float]] = {}
    if chosen is not None:
        directions = conditional_directions(
            resid_harmful[:, stage.layer], resid_harmless[:, stage.layer], chosen
        )
        interventions = {
            "conditional_ablation": multi_direction_ablation(model, directions),
            "random_control": multi_direction_ablation(
                model, random_control(directions, CONTROL_SEED + chosen)
            ),
        }
        for set_name, prompts in ((name, splits[name]) for name in splits if name.startswith("test")):
            rates[set_name] = {}
            for intervention, hook_spec in interventions.items():
                run_name = f"{set_name}__{intervention}"
                replies = completed_replies(config.model, run_name, prompts) if args.resume else None
                if replies is None:
                    replies = generate(
                        model,
                        tokenizer,
                        prompts,
                        max_new_tokens=config.max_new_tokens,
                        batch_size=config.gen_batch_size,
                        hook_spec=hook_spec,
                    )
                    save_run(config.model, run_name, prompts, replies, set_name, {"k": chosen})
                rates[set_name][intervention] = refusal_rate(replies)

    report = ConditionalReport(
        config.model,
        stage.layer,
        args.metric,
        len(val_harmful),
        gap,
        rows,
        chosen,
        rates,
        layer_points,
        strongest,
    )
    (args.run / "conditional.json").write_text(json.dumps(asdict(report), indent=2))
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
