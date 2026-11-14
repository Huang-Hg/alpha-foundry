"""Candidate sources and batch ownership, independent of storage and devices.

The daily grid is the study's full ordered (date, asset) grid. A value at row t
is a close-decision signal/weight, executed no earlier than the next open.
Cutoffs are inclusive row indices, -1 means no observation / no fitted model.
They are producer declarations, not proof that external training was causal.
"""
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Formula:
    expression: str
    kind: Literal['formula'] = 'formula'


@dataclass(frozen=True)
class ArraySource:
    kind: Literal['signals', 'target_weights', 'target_exposure']
    digest: str
    snapshot: str
    producer: str


@dataclass(frozen=True)
class EqualWeight:
    kind: Literal['equal_weight'] = 'equal_weight'


Source = Formula | ArraySource | EqualWeight


@dataclass(frozen=True)
class ArrayBatch:
    """Explicit external-input boundary; registration takes an owned copy.

    values: NumPy/CuPy float32/float64 (B,T,S), on the engine's device or host.
    signals: NaN denotes missing; cash uses clip(0,1)/eligible_count;
    short-enabled markets use clip(-1,1)/eligible_count (at most 1x gross).
    target_weights: finite long-only fractions, zero for ineligible assets,
    each day's sum <= 1 (the balance is cash); excess within the configured
    portfolio.weight_sum_atol is normalized to 1. No leverage or shorts implied.
    target_exposure: signed notional/equity ratios, zero for ineligible assets;
    sum(abs(exposure)) <= the market allocation budget. Cash allocation still
    forbids shorts. Derivatives require this kind instead of target_weights.
    observed_through / fit_through: integer (B,T), from -1 through row t.
    producer: immutable model/program/version identifier supplied by the caller.
    dates: ISO trading dates; assets: identifiers in the exact study order.
    All arrays are borrowed until register_batch returns; callers may then
    mutate their buffers without changing the registered candidate.
    CUDA inputs must be ready on the runtime's current stream when submitted;
    producers using other streams establish that dependency before calling.
    """
    kind: Literal['signals', 'target_weights', 'target_exposure']
    ids: tuple[str, ...]
    values: object
    snapshot: str
    producer: str
    dates: tuple[str, ...]
    assets: tuple[str, ...]
    observed_through: object
    fit_through: object


class ArrayStore(Protocol):
    """Immutable, content-addressed artifacts; writes only on registration."""
    def put(self, batch: ArrayBatch) -> tuple[ArraySource, ...]: ...
    def read(self, source: ArraySource) -> ArrayBatch: ...
