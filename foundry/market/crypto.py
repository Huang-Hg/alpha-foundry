"""Quantiacs crypto daily (Q25: the top-10 coins by market cap each day, long only).

Data: three parquet artifacts on one calendar grid, built by the local Quantiacs venue (not in git):
    panel.parquet     {open,high,low,close,vol,liquid}__{asset}
    features.parquet  {resret,...,plsd}__{asset}   price/volume primitives and causal rolling PLS composites
    market.parquet    market-wide daily series (on-chain, equity indices, BTC intraday regime, alt volatility)
Execution: Quantiacs calc_relative_return 1:1 - per-share slippage = 14-day ATR x 0.04 on buys and sells, no fees,
arithmetic Sharpe over 365 days. Universe: the day's `is_liquid` coins.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from foundry.backtest.spec import ExecutionSpec, ffill, previous
from foundry.config import CryptoConfig, path
from foundry.core.panel import Field, Panel
from foundry.market.base import Market, Window, trend_fields

SLIPPAGE = 0.04                               # Quantiacs SL_CRYPTODAILY: part of the platform's scoring, not a knob
PRIMITIVES = ('resret', 'rangepos', 'rangerel', 'candlebody', 'wickskew', 'gkvol', 'dvol')
PLS = ('plsa', 'plsb', 'plsc', 'plsd')
ONCHAIN = ('btc_txcnt', 'btc_addrs', 'btc_txvol', 'btc_exchvol', 'btc_hashrate', 'btc_minerrev', 'btc_fees',
           'btc_feepertx', 'btc_conftime', 'btc_blocksize')
VOLREGIME = ('btc_rv', 'btc_rskew', 'btc_rsdown', 'btc_jump', 'btc_ac', 'btc_vwapdev', 'btc_volhhi', 'btc_lateret',
             'alt_rv')
EQUITY = ('spx', 'ndx', 'rut', 'sox')


def slippage(open_, high, low, close):
    """Per-share cost: the 14-day average true range (NaN inside the window stays NaN), forward filled, x 0.04."""
    cp_ = previous(close)
    d1, d2, d3 = high - low, np.abs(high - cp_), np.abs(cp_ - low)
    tr = np.where(np.isnan(d1) | np.isnan(d2) | np.isnan(d3), np.nan, np.fmax(d1, np.fmax(d2, d3)))
    w = min(len(tr), 14)
    atr = np.full(tr.shape, np.nan)
    atr[w - 1:] = np.lib.stride_tricks.sliding_window_view(tr, w, axis=0).mean(axis=2)
    return ffill(atr) * SLIPPAGE


class Crypto(Market):
    name = 'crypto'
    ppy = 365.0
    geometric = False
    categories = {
        'pricevolume': 'scale-free daily price/volume primitives',
        'trend': 'momentum / trend axis (distance to moving average, trailing return)',
        'model': 'causal rolling PLS composites of the primitives (row t fits on rows <= t-2)',
        'onchain': 'BTC on-chain activity and miner economics (blockchain.com)',
        'volregime': 'BTC intraday realised-volatility regime, higher moments, microstructure; altcoin volatility',
        'equityindex': 'equity indices (risk appetite proxy)',
    }

    def __init__(self):
        self.cfg = CryptoConfig()

    def fields(self):
        f = [Field(k, 'price', formula=False) for k in ('open', 'high', 'low', 'close', 'volume', 'liquid')]
        f += [Field(k, 'pricevolume') for k in PRIMITIVES] + [Field(k, 'model') for k in PLS]
        f += [Field('amihud_log1p', 'pricevolume', formula=False),
              Field('amihud', 'pricevolume', deps=('amihud_log1p',), doc='log of log1p(|ret| / dollar volume * 1e9)',
                    compute=lambda B, a: B.xp.log(B.xp.where(a > 0, a, B.xp.nan)))]
        f += [Field(k, cat, scope='market') for cat, names in (('onchain', ONCHAIN), ('volregime', VOLREGIME),
                                                               ('equityindex', EQUITY)) for k in names]
        f += [Field('price', 'price', formula=False, deps=('close',),
                    compute=lambda B, c: B.xp.where(c > 0, c, B.xp.nan))]
        return f + trend_fields()

    def load(self, backend):
        d = path(self.cfg.data_dir)
        panel, features, market = (pq.read_table(d / f'{n}.parquet') for n in ('panel', 'features', 'market'))
        dates = panel.column('date').to_pylist()
        assert dates == features.column('date').to_pylist() == market.column('date').to_pylist()
        dates = np.array(dates, dtype='datetime64[D]')
        end = int(np.searchsorted(dates, np.datetime64(self.cfg.end), side='right'))
        assets = [n.split('__', 1)[1] for n in panel.column_names if n.startswith('close__')]

        def grid(table, name):
            return np.column_stack([table.column(f'{name}__{a}').to_numpy() for a in assets])[:end].astype(np.float64)

        src = {k: grid(panel, 'vol' if k == 'volume' else k) for k in ('open', 'high', 'low', 'close', 'volume', 'liquid')}
        src.update({k: grid(features, k) for k in PRIMITIVES + PLS})
        src['amihud_log1p'] = grid(features, 'amihud')
        T, S = end, len(assets)
        src.update({k: np.broadcast_to(market.column(k).to_numpy()[:end, None].astype(np.float64), (T, S))
                    for k in ONCHAIN + VOLREGIME + EQUITY})
        assert not any(np.isinf(v).any() for k, v in src.items()), 'dataset contains infinity'
        spec = ExecutionSpec.build(src['open'], src['close'], slippage(src['open'], src['high'], src['low'], src['close']))
        window = Window(dates[:end], int(np.searchsorted(dates, np.datetime64(self.cfg.start))))
        return Panel(dates[:end], assets, self.fields(), src, backend), spec, window

    def universe(self, panel):
        return panel['liquid'] > 0


MARKET = Crypto()
