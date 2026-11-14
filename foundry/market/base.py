"""Market: everything that differs between markets, and nothing else.

A market plugin declares its fields, compiles already-loaded data into an execution specification, names the universe a book may hold
on each day, and states its calendar (periods per year, Sharpe convention). Formula evaluation, books, the
execution kernel, statistics, admission and search are shared by every market.
"""
from __future__ import annotations

from foundry.core.panel import Field
from foundry.contracts import Backend, LoadedStudy, MarketData, Study


class Market:
    name = ''
    decision_clock = 'session_close'
    execution_clock = 'next_session_open'
    ppy = 252.0
    geometric = True                   # Market-level descriptive statistics use log returns.
    categories: dict = {}              # category -> description, for formula authors
    def compile(self, data: MarketData, backend: Backend, study: Study) -> LoadedStudy:
        raise NotImplementedError

    def universe(self, panel):
        """Bool (T,S) on the panel's device: the assets a book may hold each day."""
        raise NotImplementedError


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
