"""Causal model fitting, daily prediction and registered out-of-fold signals."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from types import MappingProxyType
from typing import Literal, TypeVar

import numpy as np

from foundry.study.session import ResidentPort, Session
from foundry.types import FloatArray, Fold, FoldResult, FoldView, WalkForwardResult

Model = TypeVar('Model')


def _view(resident: ResidentPort, names: tuple[str, ...], start: int, stop: int) -> FoldView:
    """Copy a bounded prefix so callback array bases cannot expose later data."""
    source, xp = resident.features, resident.xp
    values = {name: xp.array(source[name][start:stop], copy=True, order='C') for name in names}
    eligibility = xp.array(resident.eligibility[start:stop], copy=True, order='C')
    if xp is np:
        for value in values.values():
            value.flags.writeable = False
        eligibility.flags.writeable = False
    grid = resident.grid
    return FoldView(start, stop, grid.dates[start:stop], grid.assets, MappingProxyType(values), eligibility)


def _folds(folds: Sequence[Fold], days: int, evaluation_start: int) -> tuple[Fold, ...]:
    ordered = tuple(folds)
    if not ordered:
        raise ValueError('walk-forward requires at least one fold')
    previous_stop = 0
    for fold in ordered:
        indices = (fold.fit_start, fold.fit_through, fold.predict_start, fold.predict_stop)
        if any(type(index) is not int for index in indices):
            raise TypeError('fold boundaries must be integer study row indices')
        if not 0 <= fold.fit_start <= fold.fit_through < fold.predict_start < fold.predict_stop <= days:
            raise ValueError('folds require a historical fit window followed by an in-bounds prediction interval')
        if fold.predict_start < previous_stop:
            raise ValueError('prediction folds must be ordered and nonoverlapping')
        if max(fold.predict_start, evaluation_start) + 1 >= min(fold.predict_stop + 1, days):
            raise ValueError('each fold must contain a next-bar execution inside the evaluation interval')
        previous_stop = fold.predict_stop
    return ordered


def walk_forward(session: Session, fit: Callable[[FoldView], Model],
                 predict: Callable[[Model, FoldView], FloatArray], *, folds: Sequence[Fold],
                 ids: Sequence[str], kind: Literal['signals', 'target_weights', 'target_exposure'],
                 producer: str, features: Sequence[str] | None = None,
                 complexity: Sequence[float] | None = None) -> WalkForwardResult:
    """Fit each fold once and predict one decision row from its causal prefix.

    Fit receives only [fit_start, fit_through]. Predict receives a fresh owned
    [fit_start, decision_row] prefix and must return floating (candidates, assets).
    Expanding folds keep fit_start fixed; rolling folds advance it explicitly.
    Unpredicted rows are missing signals or zero targets with cutoff -1.
    Registration enforces the same finite/NaN, eligibility and exposure contract
    as every ArrayBatch. The final read-only evaluation reports realized signs;
    it does not submit candidates or modify portfolio weights.
    Each fold owns returns until the next model first executes, or study end
    for the final fold. Gap exit fees, funding and blocked-position carry remain
    attributed to the preceding model instead of disappearing from diagnostics.
    """
    resident = session.resident
    grid, xp = resident.grid, resident.xp
    count, days, assets = len(ids), len(grid.dates), len(grid.assets)
    evaluation_start = days - len(resident.stat_dates)
    ordered = _folds(folds, days, evaluation_start)
    names = tuple(resident.vocabulary if features is None else features)
    if not names or len(set(names)) != len(names) or not set(names) <= set(resident.features):
        raise ValueError('features must name unique available study fields')
    identities = tuple(ids)
    if not identities or any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError('walk-forward requires nonempty candidate IDs')
    if len(set(identities)) != count:
        raise ValueError('walk-forward candidate IDs must be unique')
    if kind not in ('signals', 'target_weights', 'target_exposure'):
        raise ValueError('unknown walk-forward output kind')
    with resident.scope():
        values = xp.full((count, days, assets), xp.nan if kind == 'signals' else 0., dtype=xp.float64)
        observed = np.full((count, days), -1, dtype=np.int64)
        fitted = np.full((count, days), -1, dtype=np.int64)
        for fold in ordered:
            model = fit(_view(resident, names, fold.fit_start, fold.fit_through + 1))
            for row in range(fold.predict_start, fold.predict_stop):
                output = xp.asarray(predict(model, _view(resident, names, fold.fit_start, row + 1)))
                if output.shape != (count, assets):
                    raise ValueError('prediction must return shape (candidates, assets) in study asset order')
                if output.dtype.kind != 'f' or output.dtype.itemsize not in (4, 8):
                    raise TypeError('prediction must return float32 or float64 values')
                values[:, row] = output
                observed[:, row] = row
                fitted[:, row] = fold.fit_through
        registered = session.register(values, kind=kind, ids=identities, grid=grid, producer=producer,
            observed_through=observed, fit_through=fitted, complexity=complexity)
        batch = resident.evaluate(registered.candidates, report=True)
        results = []
        for index, fold in enumerate(ordered):
            start = max(fold.predict_start, evaluation_start) + 1
            stop = min(ordered[index + 1].predict_start + 1, days) if index + 1 < len(ordered) else days
            means, signs = [], []
            for row in batch.rows:
                if row.ok and row.valid_returns:
                    returns = row.returns[start - evaluation_start:stop - evaluation_start]
                    mean = float(resident.to_host(returns.mean()))
                    means.append(mean)
                    signs.append(int(np.sign(mean)))
                else:
                    means.append(None)
                    signs.append(0)
            results.append(FoldResult(fold, start, stop, tuple(means), tuple(signs)))
        consistency = tuple(max(sum(f.signs[i] > 0 for f in results), sum(f.signs[i] < 0 for f in results))
                            / len(results) for i in range(count))
    return WalkForwardResult(registered, tuple(results), consistency)
