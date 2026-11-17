"""Explicit UTC crypto bar artifacts and causal multi-frequency compilation.

The source span and research cutoff must contain complete statistics periods.
Bars use [open_ns, close_ns); availability identifies a published vintage.
Execution uses the first published factual bar; features use only vintages
available at the containing decision bar's close. Funding is execution-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from foundry.config import Clock, frequency_nanoseconds
from foundry.types import ClockGrid, DataContract, ExecutionSchedule, HostArray, MarketData, Study, utc_nanoseconds

if TYPE_CHECKING:
    import pandas as pd


SCHEMA = 'market-bars-v2'
OHLCV = ('open', 'high', 'low', 'close', 'volume')
MARK = ('mark_open', 'mark_high', 'mark_low', 'mark_close')


@dataclass(frozen=True)
class IntradayData:
    """Owned host decision (D,S), execution (E,S), and UTC statistics (V,) arrays."""
    panel: MarketData
    clock: ClockGrid
    schedule: ExecutionSchedule
    event_close_ns: NDArray[np.int64]
    measurement_times: NDArray[np.datetime64]
    execution: Mapping[str, HostArray]
    periods_per_year: float


def _bound(value: str, *, end: bool = False) -> int:
    if len(value) == 10:
        result = utc_nanoseconds(value + 'T00:00:00Z')
        return result + (86_400_000_000_000 if end else 0)
    return utc_nanoseconds(value)


def _table(directory: Path, artifacts: dict, name: str) -> pd.DataFrame:
    import pandas as pd

    artifact = artifacts[name]
    if artifact['path'] != name + '.parquet':
        raise ValueError(f'Bar artifact path must be {name}.parquet')
    payload = (directory / artifact['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != artifact['sha256']:
        raise ValueError(f'Bar artifact SHA256 differs: {name}')
    return pd.read_parquet(BytesIO(payload))


def _integer(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for name in columns:
        if frame[name].dtype != np.dtype('int64'):
            raise ValueError(f'{name} must contain int64 UTC nanoseconds')
        if (frame[name].to_numpy() == np.iinfo(np.int64).min).any():
            raise ValueError(f'{name} cannot contain an absent timestamp')


def _grid(frame: pd.DataFrame, times: NDArray[np.int64], assets: tuple[str, ...],
          columns: tuple[str, ...], time: str = 'open_ns') -> dict[str, HostArray]:
    return {name: np.ascontiguousarray(frame.pivot(index=time, columns='symbol', values=name)
            .reindex(index=times, columns=assets).to_numpy(dtype=np.float64)) for name in columns}


def _prices(values: Mapping[str, HostArray], names: tuple[str, ...], label: str) -> None:
    o, h, l, c = (values[name] for name in names)
    prices = np.stack((o, h, l, c))
    present = np.isfinite(prices)
    if ((present & (prices <= 0)).any() or np.isinf(prices).any()
            or (present.any(0) != present.all(0)).any()):
        raise ValueError(f'{label} OHLC must be all finite positive values or all missing')
    if ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)).any():
        raise ValueError(f'{label} OHLC bounds are inconsistent')


def load_intraday(directory: Path, study: Study, clock: Clock) -> IntradayData:
    """Truncate at the requested cutoff, then freeze complete decision bars.

    The dense funding table explicitly records zero cashflow at non-settlement
    base opens. Missing funding records are rejected, never interpreted as zero.
    Finer settlement timestamps require a finer base-bar artifact.
    """
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest['schema'] != SCHEMA or manifest['market'] != study.market
            or study.market not in ('crypto_spot', 'crypto_perpetual') or manifest['timezone'] != 'UTC'):
        raise ValueError('Unsupported intraday crypto data contract')
    base, decision, statistics = (frequency_nanoseconds(value) for value in
                                 (manifest['interval'], clock.decision, clock.statistics))
    if decision < base or decision % base or statistics < base or statistics % base:
        raise ValueError('Decision and statistics periods must be integer multiples of the base bar')
    lower, upper, dataset_end = (utc_nanoseconds(manifest['history_start']), _bound(study.asof, end=True),
                                 utc_nanoseconds(manifest['end']))
    if not lower < upper <= dataset_end or lower % statistics or upper % statistics:
        raise ValueError('Crypto history start and cutoff require complete UTC statistics periods inside the dataset')
    start = _bound(study.start)
    if not lower <= start < upper or start % statistics:
        raise ValueError('Crypto research start must be a UTC statistics boundary inside the source span')
    if lower % base or upper % base:
        raise ValueError('Crypto source bounds must align with the base bar')
    assets = tuple(manifest['symbols'])
    if not assets or len(assets) != len(set(assets)) or any(not isinstance(s, str) or not s for s in assets):
        raise ValueError('Crypto symbols must be unique nonempty strings')
    bars = _table(directory, manifest['artifacts'], 'bars')
    required = ('symbol', 'open_ns', 'close_ns', 'available_ns', *OHLCV, 'eligible')
    if not set(required).issubset(bars.columns):
        raise ValueError('Crypto bars lack required timestamp, OHLCV, or eligibility columns')
    _integer(bars, ('open_ns', 'close_ns', 'available_ns'))
    # Truncate source rows before validating values, selecting vintages or aggregating.
    bars = bars[(bars['open_ns'] >= lower) & (bars['close_ns'] <= upper)
                & (bars['available_ns'] <= upper)].copy()
    if bars.empty or not set(bars['symbol']).issubset(assets):
        raise ValueError('Crypto bars have no historical observations or unknown symbols')
    if ((bars['open_ns'] % base != 0).any() or (bars['close_ns'] != bars['open_ns'] + base).any()
            or (bars['available_ns'] < bars['close_ns']).any()):
        raise ValueError('Crypto bars must be aligned half-open base periods, published at or after close')
    if bars['eligible'].dtype != np.dtype('bool'):
        raise ValueError('Crypto eligibility must be boolean')
    if bars.duplicated(['symbol', 'open_ns', 'available_ns']).any():
        raise ValueError('Crypto bars contain duplicate publication vintages')
    events = np.arange(lower, upper, base, dtype=np.int64)
    first = bars.sort_values('available_ns', kind='stable').drop_duplicates(['symbol', 'open_ns'], keep='first')
    execution_columns = (*OHLCV, *MARK) if study.market == 'crypto_perpetual' else OHLCV
    if not set(execution_columns).issubset(first.columns):
        raise ValueError('Perpetual bars require execution-only mark OHLC')
    execution = _grid(first, events, assets, execution_columns)
    _prices(execution, ('open', 'high', 'low', 'close'), 'Trade')
    volume = execution['volume']
    if np.isinf(volume).any() or (volume < 0).any():
        raise ValueError('Crypto volume must be nonnegative or missing with an absent quote')
    if (np.isfinite(volume) != np.isfinite(execution['open'])).any():
        raise ValueError('Crypto volume and quote availability must align')
    if study.market == 'crypto_perpetual':
        _prices(execution, MARK, 'Mark')
        if not all(np.isfinite(execution[name]).all() for name in execution_columns):
            raise ValueError('Perpetual execution requires a complete finite trade/mark base-bar grid')
        funding = _table(directory, manifest['artifacts'], 'funding')
        if not {'symbol', 'settlement_ns', 'rate', 'mark'}.issubset(funding.columns):
            raise ValueError('Funding requires symbol, settlement_ns, rate, and mark')
        _integer(funding, ('settlement_ns',))
        funding = funding[(funding['settlement_ns'] >= lower) & (funding['settlement_ns'] < upper)]
        if (funding.duplicated(['symbol', 'settlement_ns']).any() or len(funding) != len(events) * len(assets)
                or set(funding['symbol']) != set(assets) or not funding['settlement_ns'].isin(events).all()):
            raise ValueError('Funding must explicitly cover the complete base-event/asset grid')
        rates = _grid(funding, events, assets, ('rate', 'mark'), 'settlement_ns')
        if not all(np.isfinite(value).all() for value in rates.values()) or (rates['mark'] <= 0).any():
            raise ValueError('Funding rate/mark must be finite and mark positive')
        execution['funding_rate'], execution['funding_mark'] = rates['rate'], rates['mark']
    first_start = ((lower + decision - 1) // decision) * decision
    decision_ns = np.arange(first_start + decision, upper + 1, decision, dtype=np.int64)
    if not len(decision_ns):
        raise ValueError('Crypto cutoff contains no complete decision bars')
    bars['decision_ns'] = (bars['open_ns'] // decision + 1) * decision
    visible = bars[(bars['available_ns'] <= bars['decision_ns']) & (bars['open_ns'] >= first_start)
                   & (bars['decision_ns'] <= upper)]
    visible = visible.sort_values('available_ns', kind='stable').drop_duplicates(['symbol', 'open_ns'], keep='last')
    feature_events = np.arange(first_start, int(decision_ns[-1]), base, dtype=np.int64)
    raw = _grid(visible, feature_events, assets, (*OHLCV, 'eligible'))
    _prices(raw, ('open', 'high', 'low', 'close'), 'Visible trade')
    if ((raw['volume'] < 0).any() or np.isinf(raw['volume']).any()
            or (np.isfinite(raw['volume']) != np.isfinite(raw['open'])).any()):
        raise ValueError('Visible crypto volume and quote availability must align and volume be nonnegative')
    width = decision // base
    cube = {name: value.reshape(len(decision_ns), width, len(assets)) for name, value in raw.items()}
    complete = np.logical_and.reduce([np.isfinite(cube[name]).all(1) for name in OHLCV])
    values = dict(open=cube['open'][:, 0], high=np.max(cube['high'], axis=1),
                  low=np.min(cube['low'], axis=1), close=cube['close'][:, -1], volume=cube['volume'].sum(1))
    values = {name: np.ascontiguousarray(np.where(complete, value, np.nan)) for name, value in values.items()}
    members = complete & (cube['eligible'][:, -1] == 1)
    contract = DataContract(**{key: manifest[key] for key in DataContract.__dataclass_fields__})
    panel = MarketData(decision_ns.view('datetime64[ns]'), assets, values, members, contract)
    grid = ClockGrid(decision_ns, 'UTC', 'continuous_utc',
                     f'{clock.decision} complete bars [open,close); vintages available_ns<=decision close; '
                     'late bars and revisions never rewrite earlier decisions')
    target = np.full(len(events), -1, dtype=np.int64)
    slots = np.searchsorted(events, decision_ns)
    accepted = slots < len(events)
    target[slots[accepted]] = np.arange(len(decision_ns), dtype=np.int64)[accepted]
    measurement = np.arange(lower, upper, statistics, dtype=np.int64)
    schedule = ExecutionSchedule(events, target, (events - lower) // statistics,
                                 len(decision_ns), len(measurement))
    return IntradayData(panel, grid, schedule, events + base, measurement.view('datetime64[ns]'), execution,
                        365. * 86_400_000_000_000 / statistics)
