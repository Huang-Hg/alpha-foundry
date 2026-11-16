"""Compile daily session labels into explicit absolute decision timestamps."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from foundry.types import ClockGrid


def daily_clock(dates: NDArray[np.datetime64], market: str) -> ClockGrid:
    """Resolve daily closes without inferring publication times from date labels.

    Equity artifacts use XNYS or XSHG sessions, including known special closes
    and daylight-saving changes. XNYS represents the US-equity universe's
    common regular-session close; XSHG represents mainland cash sessions.
    These are explicit daily-bar calendar assumptions, not verified per-record
    vendor availability. Missing publication/revision vintages remain disclosed
    by the data contract. Crypto date d denotes [d, d+1) UTC and becomes visible
    at its right boundary, before next-day-open execution at that boundary.
    """
    labels = np.asarray(dates)
    if (labels.dtype != np.dtype('datetime64[D]') or labels.ndim != 1
            or not labels.size or np.any(np.isnat(labels))
            or np.any(labels[1:] <= labels[:-1])):
        raise ValueError('Daily session labels must be nonempty ordered datetime64[D] values')
    nanosecond_labels = labels.astype('datetime64[ns]')
    if np.any(nanosecond_labels.astype('datetime64[D]') != labels):
        raise ValueError('Daily labels are outside the int64 nanosecond range')
    if market in ('crypto_spot', 'crypto_perpetual'):
        boundaries = labels + np.timedelta64(1, 'D')
        decision = boundaries.astype('datetime64[ns]')
        if np.any(decision.astype('datetime64[D]') != boundaries):
            raise ValueError('Daily close is outside the int64 nanosecond range')
        decision = decision.astype(np.int64)
        return ClockGrid(decision, 'UTC', 'continuous_utc',
                         'UTC daily bars [d,d+1); completed bar visible at d+1 midnight; '
                         'publication/revision vintages are limited by DataContract.publication')
    names = {'us_equity': 'XNYS', 'a_share': 'XSHG'}
    if market not in names:
        raise ValueError(f'Unknown daily clock market: {market}')
    import exchange_calendars as calendars
    import pandas as pd

    name = names[market]
    end = labels[-1] if len(labels) > 1 else labels[-1] + np.timedelta64(1, 'D')
    try:
        calendar = calendars.get_calendar(name, start=str(labels[0]), end=str(end))
    except calendars.errors.NoSessionsError as error:
        raise ValueError(f'{name} daily artifact contains no trading sessions') from error
    sessions = pd.DatetimeIndex(nanosecond_labels)
    closes = calendar.closes.reindex(sessions)
    if closes.isna().any():
        invalid = tuple(str(labels[index]) for index in np.flatnonzero(closes.isna().to_numpy()))
        raise ValueError(f'{name} daily artifact contains non-session labels: {invalid}')
    return ClockGrid(closes.array.as_unit('ns').asi8, str(calendar.tz), name,
                     f'exchange_calendars {calendars.__version__} {name} regular-session closes; '
                     'daily-bar availability assumption; exact source publication/revision '
                     'vintages are limited by DataContract.publication')
