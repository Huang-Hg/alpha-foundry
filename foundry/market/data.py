"""Local daily market artifacts. Files stop at this adapter boundary."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

from foundry.config import Settings, path
from foundry.types import DataContract, MarketData, Study
from foundry.market.pit import membership
from foundry.market.perpetual import PerpetualData, load_perpetual

if TYPE_CHECKING:
    import pandas as pd


def long_table(df: pd.DataFrame, start: str, end: str, columns: Sequence[str]) -> tuple[
        NDArray[np.datetime64], list[str], dict[str, NDArray[np.floating]], int]:
    """Materialize the historical date/asset grid; an absent observation is NaN."""
    import pandas as pd

    df = df[df['date'] <= pd.Timestamp(end)]
    dates, symbols = np.sort(df['date'].unique()), np.sort(df['symbol'].unique())
    grids = {c: np.ascontiguousarray(df.pivot(index='date', columns='symbol', values=c)
                                     .reindex(index=dates, columns=symbols).to_numpy(dtype=np.float64)) for c in columns}
    dates = dates.astype('datetime64[D]')
    return dates, list(symbols), grids, int(np.searchsorted(dates, np.datetime64(start)))


def published_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Freeze each bar at session close; late publication cannot rewrite history.

    `date` and `published` are exchange-session dates. Intraday availability
    requires an intraday clock and is not implied by this daily artifact.
    """
    available = (df[df['published'] <= df['date']].sort_values('published', kind='stable')
                 .drop_duplicates(['date', 'symbol'], keep='last'))
    return df[['date', 'symbol']].drop_duplicates().merge(available, on=['date', 'symbol'],
                                                        how='left', validate='one_to_one')


def crypto_grid(frame: pd.DataFrame, asof: str) -> tuple[
        NDArray[np.datetime64], tuple[str, ...], dict[str, NDArray[np.floating]], NDArray[np.bool_]]:
    """UTC daily spot grid, including weekends and missing calendar days.

    Membership is a historical close-decision observation, separate from quote
    history. A column with no quote before the cutoff is not a past asset.
    """
    import pandas as pd

    columns = ('open', 'high', 'low', 'close', 'volume')
    frame = frame[frame['date'] <= pd.Timestamp(asof)]
    quoted = frame['open'].notna() | frame['close'].notna()
    symbols = tuple(sorted(frame.loc[quoted, 'symbol'].unique()))
    dates = pd.date_range(frame['date'].min(), asof, freq='D')
    tables = {name: frame.pivot(index='date', columns='symbol', values=name)
              .reindex(index=dates, columns=symbols) for name in (*columns, 'eligible')}
    grids = {name: np.ascontiguousarray(tables[name].to_numpy(dtype=np.float64)) for name in columns}
    members = tables['eligible'].fillna(False).to_numpy(dtype=bool)
    return dates.to_numpy(dtype='datetime64[D]'), symbols, grids, members


def load_data(settings: Settings, study: Study) -> MarketData | PerpetualData:
    import pandas as pd
    import pyarrow.parquet as pq

    if study.market == 'crypto_perpetual':
        return load_perpetual(path(settings.market.data_dir), study)
    cfg = settings.market
    directory = path(cfg.data_dir)
    frame = pd.read_parquet(directory / 'panel.parquet')
    if study.market == 'crypto_spot':
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        contract = DataContract(**{key: manifest[key] for key in DataContract.__dataclass_fields__})
        dates, assets, grids, members = crypto_grid(frame, study.asof)
        return MarketData(dates, assets, grids, members, contract)
    if study.market == 'a_share':
        frame = published_bars(frame)
        columns = ('open', 'high', 'low', 'close', 'volume', 'exec_open', 'exec_close')
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        contract = DataContract(**{key: manifest[key] for key in DataContract.__dataclass_fields__})
    else:
        columns = ('open', 'high', 'low', 'close', 'volume', 'adj_factor')
        contract = DataContract('legacy-daily', 'stored US equity artifact', 'unrecorded',
                                'vendor total-return adjustment; historical vintages unverified',
                                'adjusted OHLC',
                                'publication timestamps and revision vintages unavailable',
                                'effective intervals; announcement timestamps unavailable')
    dates, assets, grids, _ = long_table(frame, study.start, study.asof, columns)
    intervals = pq.read_table(directory / 'membership.parquet').to_pylist()
    members = membership(dates, assets, intervals)
    dates.setflags(write=False)
    return MarketData(dates, tuple(assets), grids, members, contract)
