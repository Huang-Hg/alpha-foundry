"""Explicit binding preparation for the native additive weight kernel."""
from collections.abc import Callable
from dataclasses import dataclass

from foundry.types import HostArray


@dataclass(frozen=True)
class WeightKernel:
    """Host float64 buffers; update mutates only the supplied (N,) mean and sd."""

    sample: Callable[[HostArray, HostArray, int, int], HostArray]
    update: Callable[[HostArray, HostArray, int, HostArray, HostArray], None]


def prepare() -> WeightKernel:
    """Load the native kernel at explicit resource initialization."""
    from foundry.backtest import _weights

    return WeightKernel(_weights.cem_sample, _weights.cem_update)
