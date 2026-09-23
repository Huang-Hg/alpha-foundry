"""Market: everything that differs between markets, and nothing else.

A market plugin declares its fields (the panel components formulas can read), loads its data artifacts into a Panel
over its research window, compiles its execution rules into an ExecutionSpec, names the universe a book may hold
on each day, and states its calendar (periods per year, Sharpe convention). Formula evaluation, books, the
execution kernel, statistics, admission and search are shared by every market.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from foundry.backtest.spec import ExecutionSpec
from foundry.core.panel import Field, Panel
from foundry.expr.evaluate import Backend


@dataclass(frozen=True)
class Window:
    """Research rows [0, end) of the loaded grid; statistics and holdings start at `start`."""
    dates: np.ndarray                  # datetime64[D], rows [0, end)
    start: int

    @property
    def stat_dates(self):
        return self.dates[self.start:]


class Market:
    name = ''
    ppy = 252.0
    geometric = True                   # Sharpe on log returns; the Quantiacs crypto convention is arithmetic
    categories: dict = {}              # category -> description, for formula authors
    cfg: object                        # the market's config section: data_dir, start, end and its execution rules

    def load(self, backend: Backend) -> Tuple[Panel, ExecutionSpec, Window]:
        raise NotImplementedError

    def universe(self, panel: Panel):
        """Bool (T,S) on the panel's device: the assets a book may hold each day."""
        raise NotImplementedError


def market(name: str) -> Market:
    return importlib.import_module(f'foundry.market.{name}').MARKET


def long_table(file, start: str, end: str, columns):
    """A long (date, symbol, field...) parquet -> (dates up to `end`, symbols, {field: (T,S)}, first research row).
    A symbol-day without a row is NaN."""
    import pandas as pd
    df = pd.read_parquet(file)
    df = df[df['date'] <= pd.Timestamp(end)]
    dates, symbols = np.sort(df['date'].unique()), np.sort(df['symbol'].unique())
    grids = {c: np.ascontiguousarray(df.pivot(index='date', columns='symbol', values=c)
                                     .reindex(index=dates, columns=symbols).to_numpy(dtype=np.float64)) for c in columns}
    dates = dates.astype('datetime64[D]')
    return dates, list(symbols), grids, int(np.searchsorted(dates, np.datetime64(start)))


EPS = 1e-12


def ohlcv_fields():
    """Scale-free daily price/volume primitives from open/high/low/close/volume (NaN where a price is missing)."""
    def rng(B, h, l):
        return h - l + EPS
    return [
        Field('rangepos', 'pricevolume', deps=('high', 'low', 'close'), doc='(c-l)/(h-l): close within the day range',
              compute=lambda B, h, l, c: (c - l) / rng(B, h, l)),
        Field('candlebody', 'pricevolume', deps=('open', 'high', 'low', 'close'), doc='(c-o)/(h-l)',
              compute=lambda B, o, h, l, c: (c - o) / rng(B, h, l)),
        Field('rangerel', 'pricevolume', deps=('high', 'low', 'close'), doc='(h-l)/c',
              compute=lambda B, h, l, c: (h - l) / c),
        Field('wickskew', 'pricevolume', deps=('open', 'high', 'low', 'close'), doc='(upper wick - lower wick)/(h-l)',
              compute=lambda B, o, h, l, c: ((h - B.xp.maximum(o, c)) - (B.xp.minimum(o, c) - l)) / rng(B, h, l)),
        Field('dvol', 'pricevolume', deps=('volume', 'close'), doc='log1p(dollar volume)',
              compute=lambda B, v, c: B.xp.log1p(v * c)),
        Field('price', 'price', formula=False, deps=('close',), compute=lambda B, c: B.xp.where(c > 0, c, B.xp.nan)),
    ]


def trend_fields(price: str = 'price', windows=(20, 60, 120)):
    """Distance to the moving average and trailing return of a price field, derived with the panel's backend."""
    out = []
    for w in windows:
        out.append(Field(f'ma_dist_{w}', 'trend', deps=(price,),
                         compute=lambda B, p, w=w: p / B.impl['ts_mean'](p, w) - 1, doc=f'price / {w}-day mean - 1'))
        out.append(Field(f'ret_{w}', 'trend', deps=(price,),
                         compute=lambda B, p, w=w: p / B.impl['ts_ref'](p, w) - 1, doc=f'{w}-day return'))
    return out
