"""Measure the write geometry of a model's safety neurons and how much refusal survives
direction ablation, from a finished weeks 1-2 run.

Reads the run's config, splits, refusal direction and neuron scores, then on the validation
prompts reports, for the top-k attributed neurons, the share of their effect on the refusal gap
that runs through the refusal direction, next to the measured share of the gap that survives
ablating that direction. Test sets are never touched.

    python scripts/run_geometry.py --run results/qwen2.5-7b
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from refusal_degeneracy.config import ExperimentConfig
from refusal_degeneracy.data import get_splits
from refusal_degeneracy.directions import refusal_token_ids
from refusal_degeneracy.geometry import SetGeometry, set_geometry
from refusal_degeneracy.hooks import direction_ablation
from refusal_degeneracy.model import load_model
from refusal_degeneracy.scoring import RefusalMetric, refusal_gap
from refusal_degeneracy.stages import DirectionStage, NeuronStage

DEFAULT_KS = (1, 5, 20, 100)


@dataclass(frozen=True)
class GeometryReport:
    model: str
    layer: int
    metric: RefusalMetric
    n_prompts: int
    refusal_gap: float
    retained_refusal: float
    sets: list[SetGeometry]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="output directory of run_week1.py")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS), help="safety-set sizes")
    parser.add_argument("--n", type=int, default=64, help="validation prompts per side")
    parser.add_argument(
        "--metric", type=RefusalMetric, choices=list(RefusalMetric), default=RefusalMetric.GENERATED
    )
    parser.add_argument("--device", help="override the device (cuda / cpu)")
    return parser


def measure(
    model,
    tokenizer,
    config: ExperimentConfig,
    direction_stage: DirectionStage,
    neuron_stage: NeuronStage,
    harmful: Sequence[str],
    harmless: Sequence[str],
    ks: Sequence[int],
    metric: RefusalMetric,
) -> GeometryReport:
    ref_ids = refusal_token_ids(tokenizer, config.refusal_words)
    direction = direction_stage.directions[direction_stage.layer]
    gap_of = refusal_gap(metric, model, tokenizer, harmful, harmless, ref_ids, config.gen_batch_size)
    gap = gap_of(None)
    return GeometryReport(
        model=config.model,
        layer=direction_stage.layer,
        metric=metric,
        n_prompts=len(harmful),
        refusal_gap=gap,
        retained_refusal=gap_of(direction_ablation(model, direction)) / gap,
        sets=[set_geometry(gap_of, model, neuron_stage.scores, direction, k) for k in ks],
    )


def main() -> None:
    args = build_parser().parse_args()
    os.environ["RD_RESULTS"] = str(args.run)
    config = ExperimentConfig.from_json(args.run / "config.json")
    splits = get_splits(config.split)
    direction_stage: DirectionStage = torch.load(args.run / "direction_stage.pt", weights_only=False)
    neuron_stage: NeuronStage = torch.load(args.run / "neuron_stage.pt", weights_only=False)
    model, tokenizer = load_model(config.model, device=args.device)

    report = measure(
        model, tokenizer, config, direction_stage, neuron_stage,
        splits["val_harmful"][: args.n], splits["val_harmless"][: args.n], args.k, args.metric,
    )
    (args.run / "geometry.json").write_text(json.dumps(asdict(report), indent=2))
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
