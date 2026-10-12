"""Experiment configuration as frozen dataclasses for reproducible runs."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 0
    n_find: int = 128
    n_val_harmless: int = 100


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "Qwen/Qwen3-1.7B"
    refusal_words: tuple[str, ...] = ("I", "As")
    max_new_tokens: int = 512
    gen_batch_size: int = 16
    cache_batch_size: int = 16
    attribution_batch_size: int = 4
    neuron_top_k: int = 5
    neuron_magnitudes: tuple[float, ...] = (0, 2, 5, 10, 20, 50, 100)
    layer_kl_max: float = 0.1
    neuron_kl_max: float = 0.5
    split: SplitConfig = field(default_factory=SplitConfig)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        raw = json.loads(Path(path).read_text())
        raw["refusal_words"] = tuple(raw["refusal_words"])
        raw["neuron_magnitudes"] = tuple(raw["neuron_magnitudes"])
        raw["split"] = SplitConfig(**raw["split"])
        return cls(**raw)
