"""The research engine: one market's research window resident on the GPU, and the batched path
formula -> value panel -> book -> net daily returns -> statistics.

Everything numerical stays on the device; the host gets report scalars. Formulas of a batch are evaluated on
N_STREAMS CUDA streams with one subtree cache per stream (rolling kernels walk the days serially and leave most SMs
idle, so independent trees overlap). Batch sizes come from [cuda] and are tightened to the free device memory at
load time, so a small card does not inherit a large card's batches.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

import cupy as cp

from foundry.backtest import cuda as bt
from foundry.config import CudaConfig, MarketConfig
from foundry.core import metrics as M
from foundry.expr.evaluate import backend, evaluate
from foundry.expr.parse import validate
from foundry.market.base import market as load_market
from foundry.ops.cuda import unsupported

N_STREAMS = 8
# Noise control: a candidate's signal is delayed by a fixed number of days in this range, chosen by its formula text.
# The signal then carries no information about the days it trades, while presence, turnover and costs are kept, so
# whatever a pipeline "finds" on delayed signals is pure selection. Only shifted forward (never rolled): rolling the
# end of the sample to its start would bring long-window statistics of the same days back in.
NULL_LAG = (365, 1095)
INVALID = 'G1 数值无效:含无穷值或全为缺失'
REPORT = ('sharpe', 'ann_ret', 'maxdd', 'cagr', 'total_ret', 'turnover', 'exposure', 'margin_bps', 'max_single_weight',
          'concentration')


def null_lag(expr: str) -> int:
    lo, hi = NULL_LAG
    return lo + int.from_bytes(hashlib.sha256(expr.encode()).digest()[:8], 'big') % (hi - lo + 1)


class Engine:
    def __init__(self, market_name: str = None):
        self.market = load_market(market_name or MarketConfig().name)
        self.backend = backend('cuda')
        self.panel, spec, window = self.market.load(self.backend)
        self.fields = {k: self.panel[k] for k in self.panel.vocabulary}
        self.vocabulary = self.panel.vocabulary
        self.dates, self.start, self.ppy = window.dates, window.start, self.market.ppy
        self.universe = self.market.universe(self.panel)
        self.slots = cp.maximum(self.universe.sum(1, keepdims=True), 1).astype(cp.float64)
        self.spec = bt.DeviceSpec(spec)
        self.years = M.Years(window.stat_dates, cp)
        self.top_days = max(1, int((len(self.dates) - self.start) * 0.01))      # concentration: best 1% of days
        self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(N_STREAMS)]
        # Live books, tree intermediates and trial reductions against the free memory, with 35% headroom.
        cfg, (free, _) = CudaConfig(), cp.cuda.runtime.memGetInfo()
        T, S = self.panel.shape
        limit = max(1, int(free * .65 / (T * S * 8 * 20)))
        self.candidate_batch = min(cfg.candidate_batch, 1 << (limit.bit_length() - 1))
        self.trial_batch = min(cfg.trial_batch, max(1, int(free * .10 / (T * 8 * 10))))
        self.precision = cfg.precision
        self._members: Dict[tuple, tuple] = {}

    # ------------------------------------------------------------ formulas
    def formulas(self, exprs: List[str], report=True, precision='f64', null_shift=False) -> List[Dict]:
        """Rows per formula: ok / error, and for a sound formula its book `_W`, net returns over the research window
        `_rr`, `_valid_rr` (every day finite, no ruin) plus either the report statistics (report=True) or just the
        max drawdown `_maxdd` (the mining path). The batch crosses to the host once."""
        ready = cp.cuda.Event()
        ready.record()
        for s in self.streams:
            s.wait_event(ready)
        caches = [{} for _ in self.streams]
        rows, ids, values = [], [], []
        for expr in exprs:
            text = str(expr).strip()
            v = validate(text, self.vocabulary)
            rows.append(dict(ok=v['ok'], error=v['error'], expr=text, precision=precision))
            if not v['ok']:
                continue
            why = unsupported(v['tree'])
            if why:
                raise RuntimeError(f'CUDA operator implementation required: {why}: {text}')
            k = len(ids) % len(self.streams)
            with self.streams[k]:
                values.append(evaluate(v['tree'].root, self.fields, self.backend, caches[k]))
            rows[-1]['tree'] = v['tree']
            ids.append(len(rows) - 1)
        for s in self.streams:
            s.synchronize()
        if not ids:
            return rows
        values = cp.stack(values)
        if null_shift:
            lag = cp.asarray([null_lag(rows[i]['expr']) for i in ids])
            at = cp.arange(values.shape[1])[None, :] - lag[:, None]
            values = cp.where((at >= 0)[:, :, None], values[cp.arange(len(ids))[:, None], cp.maximum(at, 0)], cp.nan)
        finite = (~cp.isinf(values).any(axis=(1, 2))) & cp.isfinite(values).any(axis=(1, 2))
        books = bt.book(values, self.universe, self.slots)
        if report:
            ok = cp.asnumpy(finite).tolist()
            for i, m, good in zip(ids, self.metrics(books, precision=precision), ok):
                rows[i].update(ok=good, error=None if good else INVALID, **{k: v for k, v in m.items() if k not in ('rr', 'W')},
                               _rr=m['rr'], _W=m['W'])
            return rows
        W, rt, sound, maxdd = self.returns(books, precision)
        for j, (i, (good, fine)) in enumerate(zip(ids, cp.asnumpy(cp.stack((finite, sound), axis=1)).tolist())):
            rows[i].update(ok=good, error=None if good else INVALID, _rr=rt[j], _W=W[j], _maxdd=maxdd[j], _valid_rr=fine)
        return rows

    def members(self, exprs: List[str], null_shift=False) -> List[tuple]:
        """(book, net returns) of formulas that are portfolio members, device-resident and computed once."""
        missing = list(dict.fromkeys(e for e in exprs if (e, null_shift) not in self._members))
        for lo in range(0, len(missing), self.candidate_batch):
            part = missing[lo:lo + self.candidate_batch]
            for e, r in zip(part, self.formulas(part, null_shift=null_shift)):
                assert r['ok'], (e, r['error'])
                self._members[(e, null_shift)] = (r['_W'], r['_rr'])
        return [self._members[(e, null_shift)] for e in exprs]

    def remember(self, expr: str, book, rr, null_shift=False):
        """An admitted candidate's book and returns are already known: keep them for its membership."""
        self._members[(expr, null_shift)] = (book, rr)

    def equal_weight(self):
        """The full-exposure equal-weight book over the universe (the empty portfolio's reference)."""
        return self.universe / self.slots

    # ------------------------------------------------------------ backtest and statistics
    def simulate(self, W, precision='f64'):
        return bt.simulate(W, self.spec, self.start, self.ppy, precision)

    def blend(self, W, U, owners, weights, precision):
        return bt.blend(W, U, owners, weights, self.spec, self.start, self.ppy, precision, self.trial_batch)

    def _stack(self, books):
        W = cp.ascontiguousarray(books if isinstance(books, cp.ndarray) else cp.stack([cp.asarray(b, dtype=cp.float64) for b in books]))
        W[:, :self.start] = 0                        # nothing is held before the research window opens
        return W

    def returns(self, books, precision='f64'):
        """-> (books zeroed before the window, net returns over the window, sound flags, max drawdowns)."""
        W = self._stack(books)
        rr, stats = self.simulate(W, precision)
        rt = rr[:, self.start:]
        return W, rt, M.sound(rt), stats[:, 3]

    def metrics(self, books, result=None, precision='f64') -> List[Dict]:
        """Report statistics per book; the host receives scalars, never daily series."""
        W = self._stack(books)
        rr, st = self.simulate(W, precision) if result is None else result
        rt, wt = rr[:, self.start:], W[:, self.start:]
        lg = cp.log1p(rt)
        gross = cp.abs(wt).sum(2)
        turnover = cp.abs(cp.diff(wt, axis=1)).sum(2).mean(1)
        ann = rt.mean(1) * self.ppy
        total = rt.sum(1)
        concentration = cp.where(total > 0, cp.sort(rt, axis=1)[:, -self.top_days:].sum(1) / cp.where(total > 0, total, 1), cp.inf)
        vals = cp.stack((M.sharpe(rt, self.ppy), ann, st[:, 3], M.cagr(rt, self.ppy), st[:, 4], turnover,
                         gross.mean(1), ann / cp.maximum(turnover * self.ppy, 1e-12) * 1e4, cp.abs(wt).max(axis=(1, 2)),
                         concentration), axis=1)
        y = cp.stack((self.years.sharpe(rt, self.ppy), self.years.share(gross > 1e-9), self.years.total(lg)), axis=2)
        host = cp.asnumpy(cp.concatenate((vals, y.reshape(len(W), -1), M.sound(rt)[:, None]), axis=1)).tolist()
        dates = self.dates[self.start:]
        return [dict(zip(REPORT, v[:10]),
                     yearly=[dict(year=int(year), days=int(days), sharpe=round(v[10 + j * 3], 3),
                                  attend=round(v[11 + j * 3], 3), lg=round(v[12 + j * 3], 4))
                             for j, (year, days) in enumerate(zip(self.years.years, self.years.days))],
                     span=f'{dates[0]}..{dates[-1]}', n_days=len(dates), rr=rt[i], W=W[i], _valid_rr=bool(v[-1]))
                for i, v in enumerate(host)]
