"""Validate the geometry measures on controlled defenses with a known degeneracy.

Implants per-category safety neurons into small random models, sweeping how degenerate their
writes are and how strong, and records each measure next to the fraction of the refusal gap that
survives direction ablation. A measure that predicts the attack should rank-correlate with it.

    python scripts/run_synthetic.py --out results/synthetic
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from refusal_degeneracy.synthetic import SyntheticDefense, evaluate_defense, toy_model, toy_prompts

HARMLESS_MARKER = 10
CATEGORY_MARKERS = (1, 2, 9)
REFUSAL_IDS = (3, 4, 5)
DEFENSE_LAYER = 2
NEURONS_PER_CATEGORY = 2
MIN_REFUSAL_GAP = 2.0
MEASURES = ("counterfactual_share", "direction_share", "weighted_cos2", "participation_ratio")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--write-gains", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--degeneracies", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 0.75, 1.0])
    return parser


def sweep(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for seed in range(args.seeds):
        model = toy_model(seed)
        *harmful, harmless = toy_prompts(model, seed, [*CATEGORY_MARKERS, HARMLESS_MARKER])
        for write_gain in args.write_gains:
            for degeneracy in args.degeneracies:
                defense = SyntheticDefense(DEFENSE_LAYER, NEURONS_PER_CATEGORY, degeneracy, write_gain)
                outcome = evaluate_defense(model, defense, harmful, harmless, REFUSAL_IDS)
                rows.append({"seed": seed, "write_gain": write_gain, **asdict(outcome)})
    return pd.DataFrame(rows)


def rank_correlations(results: pd.DataFrame) -> dict[str, float]:
    """Spearman correlation of each measure with retained refusal, over defenses strong enough
    to open a real refusal gap."""
    working = results[results.refusal_gap > MIN_REFUSAL_GAP]
    retained = working.retained_refusal.rank()
    return {measure: float(working[measure].rank().corr(retained)) for measure in MEASURES}


def main() -> None:
    args = build_parser().parse_args()
    results = sweep(args)
    correlations = rank_correlations(results)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "synthetic_sweep.csv", index=False)
    (args.out / "correlations.json").write_text(json.dumps(correlations, indent=2))
    print(results.round(3).to_string(index=False))
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
