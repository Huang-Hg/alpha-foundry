"""Causal time-interval folds with caller-owned resident candidate leases."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from foundry.config import frequency_nanoseconds
from foundry.study.session import ResidentPort, Session
from foundry.types import ArrayKind, ArrayUnit, DataGrid, ExecutionSchedule, FloatArray, Fold, FoldResult, FoldView, NO_INFORMATION, WalkForwardResult

Model = TypeVar('Model')


@dataclass(frozen=True)
class _FoldRows:
    """Private compiled row bounds; public fold boundaries remain UTC ns."""
    fold: Fold
    fit_start: int
    fit_stop: int
    predict_start: int
    predict_stop: int


def _boundary(times: NDArray[np.int64], row: int) -> int:
    """Return an exclusive terminal bound without inventing a future decision."""
    return int(times[row]) if row < len(times) else int(times[-1]) + 1


def _view(resident: ResidentPort, grid: DataGrid, times: NDArray[np.int64],
          names: tuple[str, ...], start: int, stop: int) -> FoldView:
    """Copy a bounded prefix so callback array bases cannot expose later data."""
    source, xp = resident.features, resident.xp
    values = {name: xp.array(source[name][start:stop], copy=True, order='C') for name in names}
    eligibility = xp.array(resident.eligibility[start:stop], copy=True, order='C')
    if xp is np:
        for value in values.values():
            value.flags.writeable = False
        eligibility.flags.writeable = False
    return FoldView(int(times[start]), int(times[stop - 1]) + 1, grid.decision_times[start:stop],
                    grid.assets, MappingProxyType(values), eligibility)


def _folds(folds: Sequence[Fold], times: NDArray[np.int64]) -> tuple[_FoldRows, ...]:
    ordered = tuple(folds)
    if not ordered:
        raise ValueError('walk-forward requires at least one fold')
    previous_stop = int(times[0])
    compiled = []
    for fold in ordered:
        bounds = (fold.fit_start, fold.fit_stop, fold.predict_start, fold.predict_stop)
        if any(type(bound) is not int for bound in bounds):
            raise TypeError('fold boundaries must be integer UTC nanoseconds')
        if not int(times[0]) <= fold.fit_start < fold.fit_stop <= fold.predict_start < fold.predict_stop <= _boundary(times, len(times)):
            raise ValueError('folds require a historical half-open fit interval followed by an in-bounds prediction interval')
        if fold.predict_start < previous_stop:
            raise ValueError('prediction folds must be ordered and nonoverlapping')
        fit_start, fit_stop, predict_start, predict_stop = (int(np.searchsorted(times, bound)) for bound in bounds)
        if fit_start == fit_stop or predict_start == predict_stop:
            raise ValueError('fit and prediction intervals must each contain a decision')
        compiled.append(_FoldRows(fold, fit_start, fit_stop, predict_start, predict_stop))
        previous_stop = fold.predict_stop
    return tuple(compiled)


def _measurement_ranges(folds: tuple[_FoldRows, ...], schedule: ExecutionSchedule,
                        decision_start: int, measurement_start: int) -> tuple[tuple[int, int], ...]:
    """Attribute complete measurement periods using the actual target events."""
    starts = []
    for fold in folds:
        updates = np.flatnonzero((schedule.decision_index >= max(fold.predict_start, decision_start))
            & (schedule.decision_index < fold.predict_stop)
            & (schedule.measurement_index >= measurement_start))
        if not len(updates):
            raise ValueError('each fold must contain a next-base-bar execution inside the evaluation interval')
        event = int(updates[0])
        period = int(schedule.measurement_index[event])
        if starts and event > 0 and schedule.measurement_index[event - 1] == period:
            raise ValueError('a fold changes models inside a statistics period; choose a finer statistics clock')
        if starts and period <= starts[-1]:
            raise ValueError('prediction folds share a statistics period; choose a finer statistics clock')
        starts.append(period)
    stops = (*starts[1:], schedule.measurement_count)
    return tuple(zip(starts, stops))


def walk_forward(session: Session, fit: Callable[[FoldView], Model],
                 predict: Callable[[Model, FoldView], FloatArray], *, folds: Sequence[Fold],
                 ids: Sequence[str], kind: ArrayKind, unit: ArrayUnit,
                 producer: str, features: Sequence[str] | None = None,
                 complexity: Sequence[float] | None = None) -> WalkForwardResult:
    """Fit each time interval once and predict from strictly causal prefixes.

    Fold intervals use UTC nanoseconds and half-open endpoints. Fit receives
    only decisions in [fit_start, fit_stop); predict receives a fresh owned
    prefix from the first fit decision through the current decision inclusive.
    Both callbacks see timestamp coordinates rather than study row numbers.
    The fitted-information cutoff is the final visible fit decision; callers
    whose labels consume later data must extend their declared fit interval.
    Unpredicted rows are missing scores or zero targets with NO_INFORMATION.
    Requires an in-process Session; remote clients register causal ArrayBatch
    outputs. Returned leases belong to the caller. Evaluation remains read-only.
    Each fold owns complete statistics periods from its first actual base-bar
    execution until the next model's first execution period, or study end.
    Later model changes inside a statistics period require a finer statistics
    clock. The first model may own its initial period's preceding flat prefix.
    Measurement bounds are period boundaries, not decision or fill times;
    gap exits, funding and blocked-position carry belong to the preceding model.
    """
    resident = session.resident
    grid, xp = resident.grid, resident.xp
    times = grid.time_ns
    count, days, assets = len(ids), len(times), len(grid.assets)
    ordered = _folds(folds, times)
    measurement_ranges = _measurement_ranges(ordered, resident.execution_schedule,
                                             resident.decision_start, resident.measurement_start)
    measurement_times = np.asarray(resident.measurement_times, dtype='datetime64[ns]').astype(np.int64)
    measurement_duration = frequency_nanoseconds(resident.settings.clock.statistics)
    names = tuple(resident.vocabulary if features is None else features)
    if not names or len(set(names)) != len(names) or not set(names) <= set(resident.features):
        raise ValueError('features must name unique available study fields')
    identities = tuple(ids)
    if not identities or any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError('walk-forward requires nonempty candidate IDs')
    if len(set(identities)) != count:
        raise ValueError('walk-forward candidate IDs must be unique')
    if kind not in ('scores', 'targets'):
        raise ValueError('unknown walk-forward output kind')
    if (kind == 'scores' and unit != 'score') or (kind == 'targets' and unit != 'equity_fraction'):
        raise ValueError('this walk-forward execution plan requires scores/score or targets/equity_fraction')
    with resident.scope():
        values = xp.full((count, days, assets), xp.nan if kind == 'scores' else 0., dtype=xp.float64)
        observed = np.full((count, days), NO_INFORMATION, dtype=np.int64)
        fitted = np.full((count, days), NO_INFORMATION, dtype=np.int64)
        for fold in ordered:
            model = fit(_view(resident, grid, times, names, fold.fit_start, fold.fit_stop))
            for row in range(fold.predict_start, fold.predict_stop):
                output = xp.asarray(predict(model, _view(resident, grid, times, names, fold.fit_start, row + 1)))
                if output.shape != (count, assets):
                    raise ValueError('prediction must return shape (candidates, assets) in study asset order')
                if output.dtype.kind != 'f' or output.dtype.itemsize not in (4, 8):
                    raise TypeError('prediction must return float32 or float64 values')
                values[:, row] = output
                observed[:, row] = times[row]
                fitted[:, row] = times[fold.fit_stop - 1]
        registered = session.register(values, kind=kind, unit=unit, ids=identities, grid=grid, producer=producer,
            observed_at=observed, fit_at=fitted, complexity=complexity)
        completed = False
        try:
            batch = resident.evaluate(registered.candidates, report=True)
            results = []
            for fold, (start, stop) in zip(ordered, measurement_ranges):
                start -= resident.measurement_start
                stop -= resident.measurement_start
                means, signs = [], []
                for row in batch.rows:
                    if row.ok and row.valid_returns:
                        returns = row.returns[start:stop]
                        mean = float(resident.to_host(returns.mean()))
                        means.append(mean)
                        signs.append(int(np.sign(mean)))
                    else:
                        means.append(None)
                        signs.append(0)
                results.append(FoldResult(fold=fold.fold,
                    measurement_start=int(measurement_times[start]),
                    measurement_stop=(int(measurement_times[stop]) if stop < len(measurement_times)
                                      else int(measurement_times[-1]) + measurement_duration),
                    mean_returns=tuple(means), signs=tuple(signs)))
            consistency = tuple(max(sum(f.signs[i] > 0 for f in results), sum(f.signs[i] < 0 for f in results))
                                / len(results) for i in range(count))
            completed = True
        finally:
            if not completed:
                session.release(registered)
    return WalkForwardResult(registered, tuple(results), consistency)
