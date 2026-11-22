"""Results of each pipeline stage, checkpointed so later scripts and resumed runs can load them."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from torch import Tensor


@dataclass(frozen=True)
class DirectionStage:
    directions: Tensor
    layer: int
    layer_sweep: pd.DataFrame
    mean_mlp_harmful: Tensor


@dataclass(frozen=True)
class NeuronStage:
    scores: Tensor
    layer: int
    index: int
    value: float
    neuron_sweep: pd.DataFrame
