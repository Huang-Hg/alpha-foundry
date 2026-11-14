"""Local daily market artifacts. Files stop at this adapter boundary."""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from foundry.config import path
from foundry.contracts import DataContract, MarketData
from foundry.core.pit import membership
from foundry.adapters.perpetual_data import load_perpetual


def long_table(df, start: str, end: str, columns):
    """Materialize the historical date/asset grid; an absent observation is NaN."""
    df = df[df['date'] <= pd.Timestamp(end)]
    dates, symbols = np.sort(df['date'].unique()), np.sort(df['symbol'].unique())
    grids = {c: np.ascontiguousarray(df.pivot(index='date', columns='symbol', values=c)
                                     .reindex(index=dates, columns=symbols).to_numpy(dtype=np.float64)) for c in columns}
    dates = dates.astype('datetime64[D]')
    return dates, list(symbols), grids, int(np.searchsorted(dates, np.datetime64(start)))


def published_bars(df):
    """Freeze each bar at session close; late publication cannot rewrite history.

    `date` and `published` are exchange-session dates. Intraday availability
    requires an intraday clock and is not implied by this daily artifact.
    """
    available = (df[df['published'] <= df['date']].sort_values('published', kind='stable')
                 .drop_duplicates(['date', 'symbol'], keep='last'))
    return df[['date', 'symbol']].drop_duplicates().merge(available, on=['date', 'symbol'],
                                                        how='left', validate='one_to_one')


def crypto_grid(frame, asof):
    """UTC daily spot grid, including weekends and missing calendar days.

    Membership is a historical close-decision observation, separate from quote
    history. A column with no quote before the cutoff is not a past asset.
    """
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


def load_data(settings, study):
    if study.market == 'crypto_perpetual':
        return load_perpetual(path(settings.crypto_perpetual.data_dir), study)
    cfg = {'a_share': settings.a_share, 'us_equity': settings.us_equity,
           'crypto_spot': settings.crypto_spot}[study.market]
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
