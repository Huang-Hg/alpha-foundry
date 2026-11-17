"""Regular-session equity bars with explicit publication and execution rules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from foundry.config import Clock, frequency_nanoseconds
from foundry.market.intraday import IntradayData, OHLCV, SCHEMA, _bound, _grid, _integer, _prices, _table
from foundry.types import ClockGrid, DataContract, ExecutionSchedule, MarketData, Study, utc_nanoseconds

NUMERIC_RULES = ('fee_buy', 'fee_sell', 'maker_fee', 'taker_fee', 'split_factor', 'cash_dividend',
                 'min_quantity', 'limit_up', 'limit_down', 'quantity_step', 'fill_step', 'price_tick')
BOOLEAN_RULES = ('can_buy', 'can_sell', 'suspended', 'inventory_t1', 'cash_t1', 'odd_lot_sell')


def load_equity_intraday(directory: Path, study: Study, clock: Clock) -> IntradayData:
    """Build only actual regular-session events, never overnight or lunch bars.

    Rule rows are explicit source facts known no later than their execution open.
    Corporate actions occur at the specified open and prices remain unadjusted.
    Decision and statistics buckets reset at every session and count trading
    time only; shorter terminal buckets are retained at early/session closes.
    """
    import exchange_calendars as calendars
    import pandas as pd

    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    expected = {'us_equity': 'XNYS', 'a_share': 'XSHG'}[study.market]
    if (manifest['schema'] != SCHEMA or manifest['market'] != study.market
            or manifest['calendar'] != expected or manifest['prices'] != 'raw unadjusted OHLC'
            or manifest['timezone'] != 'UTC'):
        raise ValueError('Equity bars require explicit regular-session calendar and raw unadjusted prices')
    base, decision, statistics = (frequency_nanoseconds(x) for x in
                                 (manifest['interval'], clock.decision, clock.statistics))
    if min(decision, statistics) < base or decision % base or statistics % base:
        raise ValueError('Equity decision/statistics intervals must be integer base-bar multiples')
    lower, upper = utc_nanoseconds(manifest['history_start']), _bound(study.asof, end=True)
    if not lower < upper <= utc_nanoseconds(manifest['end']) or not lower <= _bound(study.start) < upper:
        raise ValueError('Equity research bounds must lie within the source span')
    assets = tuple(manifest['symbols'])
    if not assets or len(set(assets)) != len(assets) or any(not isinstance(x, str) or not x for x in assets):
        raise ValueError('Equity symbols must be unique nonempty strings')
    calendar = calendars.get_calendar(expected, start=pd.Timestamp(lower, tz='UTC').date().isoformat(),
                                      end=pd.Timestamp(upper, tz='UTC').date().isoformat())
    opens, closes, sessions = [], [], []
    for session in calendar.sessions:
        begin, finish = calendar.session_open(session).value, calendar.session_close(session).value
        if finish <= lower or begin >= upper:
            continue
        if study.market == 'us_equity' and session.date() < pd.Timestamp('2024-05-28').date():
            raise ValueError('US cash orders support the T+1 regime from 2024-05-28; earlier T+2 sessions are unsupported')
        if begin < lower or finish > upper:
            raise ValueError('Equity source bounds must contain complete trading sessions')
        break_start, break_end = calendar.session_break_start(session), calendar.session_break_end(session)
        segments = [(begin, finish)] if pd.isna(break_start) else [(begin, break_start.value), (break_end.value, finish)]
        if any((b - a) % base for a, b in segments[:-1]):
            raise ValueError('Equity base interval must divide trading segments before lunch; only the session-close bar may be partial')
        for a, b in segments:
            event = np.arange(a, b, base, dtype=np.int64)
            opens.extend(event); closes.extend(np.minimum(event + base, b))
            sessions.extend([session.value] * len(event))
    events, ends, labels = (np.asarray(x, dtype=np.int64) for x in (opens, closes, sessions))
    if not len(events):
        raise ValueError('Equity cutoff contains no complete trading segments')
    bars = _table(directory, manifest['artifacts'], 'bars')
    required = {'symbol', 'open_ns', 'close_ns', 'available_ns', 'eligible', *OHLCV}
    if not required.issubset(bars.columns):
        raise ValueError('Equity bars lack OHLCV, eligibility or publication timestamps')
    _integer(bars, ('open_ns', 'close_ns', 'available_ns'))
    bars = bars[(bars.open_ns >= lower) & (bars.close_ns <= upper) & (bars.available_ns <= upper)].copy()
    if (bars.empty or not set(bars.symbol).issubset(assets) or bars.eligible.dtype != np.dtype('bool')
            or bars.duplicated(['symbol', 'open_ns', 'available_ns']).any()):
        raise ValueError('Equity bars have invalid symbols, eligibility or duplicate vintages')
    slots = np.searchsorted(events, bars.open_ns.to_numpy())
    if (np.any(slots == len(events)) or not np.array_equal(events[slots], bars.open_ns.to_numpy())
            or not np.array_equal(ends[slots], bars.close_ns.to_numpy())
            or (bars.available_ns < bars.close_ns).any()):
        raise ValueError('Equity bars must follow factual regular-session intervals and publication order')
    first = bars.sort_values('available_ns', kind='stable').drop_duplicates(['symbol', 'open_ns'], keep='first')
    execution = _grid(first, events, assets, OHLCV)
    _prices(execution, OHLCV[:4], 'Equity execution')
    rules = _table(directory, manifest['artifacts'], 'rules')
    if not {'symbol', 'open_ns', 'available_ns', *NUMERIC_RULES, *BOOLEAN_RULES}.issubset(rules.columns):
        raise ValueError('Equity execution rules require explicit fees, actions, limits, lots and settlement')
    _integer(rules, ('open_ns', 'available_ns'))
    rules = rules[(rules.open_ns >= lower) & (rules.open_ns < upper)].copy()
    if (rules.duplicated(['symbol', 'open_ns']).any() or len(rules) != len(events) * len(assets)
            or set(rules.symbol) != set(assets) or not rules.open_ns.isin(events).all()
            or (rules.available_ns > rules.open_ns).any()):
        raise ValueError('Equity rules must explicitly cover every event/asset and be known at its open')
    if any(rules[k].dtype != np.dtype('bool') for k in BOOLEAN_RULES):
        raise TypeError('Equity execution direction and settlement rules require boolean columns')
    numeric = _grid(rules, events, assets, NUMERIC_RULES)
    if any(not np.isfinite(v).all() for v in numeric.values()):
        raise ValueError('Equity numeric rules must be finite')
    if ((numeric['split_factor'] <= 0).any() or any((numeric[k] < 0).any() for k in
            NUMERIC_RULES if k not in ('maker_fee', 'taker_fee', 'split_factor'))
            or ((numeric['limit_down'] > numeric['limit_up']) & (numeric['limit_up'] > 0)).any()):
        raise ValueError('Equity action, lot, fee or directional limit rules are invalid')
    execution.update(numeric)
    execution.update({k: v.astype(bool) for k, v in _grid(rules, events, assets, BOOLEAN_RULES).items()})
    for k, flag in [('inventory_t1', study.market == 'a_share'), ('cash_t1', study.market == 'us_equity')]:
        if not np.all(execution[k] == flag):
            raise ValueError('Equity cash-account settlement rules differ from the declared market')
    execution['session_start'] = np.broadcast_to(np.r_[True, labels[1:] != labels[:-1]][:, None], execution['open'].shape)
    execution['session_end'] = np.broadcast_to(np.r_[labels[:-1] != labels[1:], True][:, None], execution['open'].shape)
    volume = execution['volume']
    if (np.isinf(volume).any() or (volume < 0).any()
            or (np.isfinite(volume) != np.isfinite(execution['open'])).any()):
        raise ValueError('Equity volume must be nonnegative and align with quote availability')
    # Session-relative trading-time buckets omit lunch and never cross dates.
    def buckets(width):
        result = np.empty(len(events), dtype=np.int64)
        count = 0
        boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            ix = np.arange(begin, end)
            elapsed = np.r_[0, np.cumsum(ends[ix] - events[ix])[:-1]]
            local = elapsed // width
            result[ix] = local + count
            count += int(local[-1]) + 1
        return result, count
    groups, count = buckets(decision)
    boundaries = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    decision_ns = ends[boundaries[1:] - 1]
    bars['decision_ns'] = decision_ns[groups[slots]]
    visible = bars[bars.available_ns <= bars.decision_ns].sort_values('available_ns', kind='stable').drop_duplicates(['symbol', 'open_ns'], keep='last')
    raw = _grid(visible, events, assets, (*OHLCV, 'eligible'))
    _prices(raw, OHLCV[:4], 'Visible equity')
    if (np.isinf(raw['volume']).any() or (raw['volume'] < 0).any()
            or (np.isfinite(raw['volume']) != np.isfinite(raw['open'])).any()):
        raise ValueError('Visible equity volume must align with quote availability')
    values = {k: np.full((count, len(assets)), np.nan) for k in OHLCV}
    members = np.zeros((count, len(assets)), dtype=bool)
    for i, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ix = slice(begin, end)
        complete = np.logical_and.reduce([np.isfinite(raw[k][ix]).all(0) for k in OHLCV])
        reduced = (raw['open'][begin], raw['high'][ix].max(0), raw['low'][ix].min(0), raw['close'][end - 1], raw['volume'][ix].sum(0))
        for k, v in zip(OHLCV, reduced):
            values[k][i] = np.where(complete, v, np.nan)
        members[i] = complete & (raw['eligible'][end - 1] == 1)
    contract = DataContract(**{k: manifest[k] for k in DataContract.__dataclass_fields__})
    panel = MarketData(decision_ns.view('datetime64[ns]'), assets, values, members, contract)
    target = np.full(len(events), -1, dtype=np.int64)
    next_event = np.searchsorted(events, decision_ns)
    accepted = next_event < len(events)
    target[next_event[accepted]] = np.arange(count, dtype=np.int64)[accepted]
    measurement, measurements = buckets(statistics)
    times = events[np.r_[0, np.flatnonzero(measurement[1:] != measurement[:-1]) + 1]]
    annual = (252. if study.market == 'us_equity' else 242.) * measurements / len(np.unique(labels))
    grid = ClockGrid(decision_ns, str(calendar.tz), expected,
                     f'exchange_calendars {calendars.__version__}; session-relative trading-time buckets; available_ns<=decision close')
    schedule = ExecutionSchedule(events, target, measurement, count, measurements)
    return IntradayData(panel, grid, schedule, ends, times.view('datetime64[ns]'), execution, annual)
