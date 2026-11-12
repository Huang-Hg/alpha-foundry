"""Portable operator backend (foundry/ops/portable.py) against the CPU reference (foundry/ops/cpu.py) on the real panel.

Every operator runs on real fields (per-asset and market-wide, with listing gaps and warm-up NaN) over windows on
both sides of each reference dispatch threshold (deque at 256, BIT ts_quantile at 32); then the production library's
formulas are evaluated end to end, derived fields included (each side computes ma_dist / ret / amihud with its own
backend). Reported per operator: NaN-position agreement and the largest relative difference |p - c| / max(|p|, |c|)
over cells finite on both sides.

Criteria. Portable follows the C++ source step by step and equals a strict-IEEE build of it bitwise: compiling
foundry/ops/src with build.sh's flags minus -ffast-math plus -ffp-contract=off gives NaN agreement 100% and relative
difference 0 on every case below (tanh 2.2e-16 from the libm; ts_mean 1 ulp, as portable mirrors the production
build's multiply by 1/w). The production build uses -ffast-math, and for the kernels in FAST_MATH its code differs
from the source: fused multiply-adds, division by a reciprocal, 4-lane vectorised window reductions. Where a result
cancels (a value near 0, a variance far below the mean squared, running sums late in a long history, skew over two
points) the difference exceeds any fixed relative tolerance, and on exactly constant inputs (a flat window, a day with
one value across assets) the reference's zero-variance tests see rounding noise. So: operators outside FAST_MATH
must agree everywhere (NaN 100%, relative difference <= 1e-12); FAST_MATH operators must agree on NaN positions
outside constant inputs, and their value differences are reported; the library formulas and the derived fields must
agree exactly.

    python.exe -X utf8 -m tests.test_portable            (Windows interpreter with the native extension)
"""
from __future__ import annotations

import time
import unittest
from collections import defaultdict

import numpy as np

from foundry.backtest.cpu import book
from foundry.expr.evaluate import backend, evaluate
from foundry.expr.parse import parse
from foundry.market.crypto import MARKET
from foundry.ops.registry import OPS, WINDOWED
from foundry.portfolio.library import Library

REL_TOL = 1e-12
FAST_MATH = frozenset({'cs_zscore', 'cs_demean', 'ts_std', 'ts_ema', 'ts_wma', 'ts_skew', 'ts_kurt', 'ts_mad',
                       'ts_slope', 'ts_zscore', 'ts_snr', 'ts_decay_exp_window', 'ts_corr', 'ts_cov', 'ts_regression'})
FIELDS = ('candlebody', 'resret', 'ma_dist_60', 'dvol', 'btc_fees', 'btc_ac', 'plsc', 'price')
PAIRS = (('candlebody', 'resret'), ('ma_dist_60', 'btc_ac'), ('dvol', 'btc_fees'), ('resret', 'plsc'))
WINDOWS = (2, 5, 20, 45, 135, 300)
CONSTS = {'add_const': (1.0, -0.3), 'mul_const': (3.0, -0.5), 'pow_const': (2.0, -2.0, 0.5, -0.5, 1.7),
          'left_tail': (0.0, 0.2), 'right_tail': (0.0, -0.1), 'hump': (0.0, 0.05)}
TS_PARAM = {'ts_decay_exp_window': ((5, 0.5), (20, 0.9), (135, 1.0)), 'ts_scale': ((5, 0.0), (45, 0.25), (300, 0.0)),
            'ts_quantile': ((5, 0.0), (20, 1.0), (60, 2.0), (60, 0.0))}
RANK_WINDOWS = (2, 5, 20, 31)                  # ts_rank beyond its causal window (31) is rejected by validate()


def constant_inputs(name, args):
    """Cells computed from exactly constant inputs: a flat window of any panel argument (windowed operators) or a
    day whose values are all equal (cross-sectional operators)."""
    panels = [a for a in args if isinstance(a, np.ndarray)]
    out = np.zeros(panels[0].shape, bool)
    if name.startswith('cs_'):
        v = panels[0]
        out |= (np.fmax.reduce(v, axis=1) == np.fmin.reduce(v, axis=1))[:, None]
    elif OPS[name].kind in WINDOWED:
        w = int(args[len(panels)])
        for a in panels:
            if len(a) >= w:
                win = np.lib.stride_tricks.sliding_window_view(a, w, axis=0)
                out[w - 1:] |= win.min(-1) == win.max(-1)
    return out


def compare(p, c, skip=None):
    """-> (NaN positions differing, of them outside `skip`, cells, max relative difference, its max outside `skip`,
    median relative difference outside `skip`), over cells finite on both sides."""
    skip = np.zeros(p.shape, bool) if skip is None else skip
    pn, cn = np.isnan(p), np.isnan(c)
    both = np.isfinite(p) & np.isfinite(c)
    infs = ~(pn | cn | both)                   # an infinity on either side must be the same infinity on both
    assert np.array_equal(p[infs], c[infs]), 'infinities differ'
    scale = np.maximum(np.abs(p), np.abs(c))
    rel = np.where(both & (scale > 0), np.abs(p - c) / np.where(scale > 0, scale, 1.0), 0.0)
    kept = rel[both & ~skip]
    return (int((pn != cn).sum()), int(((pn != cn) & ~skip).sum()), p.size, float(rel.max()),
            float(kept.max()) if kept.size else 0.0, float(np.median(kept)) if kept.size else 0.0)


class PortableParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = backend('cpu')
        cls.port = backend('portable')
        cls.panel_cpu, _, window = MARKET.load(cls.cpu)
        cls.panel_port, _, _ = MARKET.load(cls.port)
        cls.x = {k: np.ascontiguousarray(cls.panel_cpu[k]) for k in FIELDS}
        T, S = cls.panel_cpu.shape
        print(f'\n面板 {window.dates[0]}..{window.dates[-1]}  T={T} S={S}')

    def cases(self):
        """(operator, description, args) over the real fields."""
        x = self.x
        for name, op in OPS.items():
            k = op.kind
            if k == 'unary':
                for f in FIELDS:
                    yield name, f, (x[f],)
            elif k == 'binary':
                for a, b in PAIRS:
                    yield name, f'{a},{b}', (x[a], x[b])
                yield name, 'candlebody,0.1', (x['candlebody'], np.full(x['candlebody'].shape, 0.1))
            elif k == 'binary_const':
                for f in FIELDS:
                    for c in CONSTS[name]:
                        yield name, f'{f},{c:g}', (x[f], c)
            elif k == 'ternary':
                yield name, 'candlebody,resret,ma_dist_60', (x['candlebody'], x['resret'], x['ma_dist_60'])
                yield name, 'ma_dist_60,dvol,btc_ac', (x['ma_dist_60'], x['dvol'], x['btc_ac'])
            elif k == 'ts':
                for f in FIELDS:
                    for w in (RANK_WINDOWS if name == 'ts_rank' else WINDOWS):
                        yield name, f'{f},{w}', (x[f], w)
            elif k == 'ts_param':
                for f in FIELDS:
                    for w, c in TS_PARAM[name]:
                        yield name, f'{f},{w},{c:g}', (x[f], w, c)
            elif k == 'pair':
                for a, b in PAIRS:
                    for w in (5, 30, 135, 365):
                        yield name, f'{a},{b},{w}', (x[a], x[b], w)
            else:
                for a, b in PAIRS:
                    for w, rt in ((10, 0.0), (60, 1.0), (60, 2.0), (180, 3.0), (180, 4.0)):
                        yield name, f'{a},{b},{w},{rt:g}', (x[a], x[b], w, rt)

    def test_operators(self):
        # NaN diff, of them outside constant inputs, cells, max rel, max rel outside, worst median, where, calls, s
        agg = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0.0, '', 0, 0.0])
        for name, what, args in self.cases():
            c = self.cpu.impl[name](*args)
            t0 = time.perf_counter()
            with np.errstate(divide='raise', invalid='raise', over='raise'):     # portable kernels never warn
                p = self.port.impl[name](*args)
            dt = time.perf_counter() - t0
            nd, nd_out, n, rel, rel_out, med = compare(np.asarray(p, dtype=np.float64), np.asarray(c, dtype=np.float64),
                                                      constant_inputs(name, args))
            r = agg[name]
            r[0] += nd
            r[1] += nd_out
            r[2] += n
            if rel >= r[3]:
                r[3], r[6] = rel, what
            r[4] = max(r[4], rel_out)
            r[5] = max(r[5], med)
            r[7] += 1
            r[8] += dt
        print(f'\n{"算子":20s} {"类":9s} {"用例":>4s} {"NaN一致率":>10s} {"常数外NaN差":>9s} {"最大相对差":>9s} '
              f'{"常数外最大":>9s} {"最差中位":>8s} {"耗时":>6s}  最大差出处')
        for name in OPS:
            nd, nd_out, n, rel, rel_out, med, where, calls, dt = agg[name]
            print(f'{name:20s} {"fast-math" if name in FAST_MATH else "exact":9s} {calls:4d} {100 * (1 - nd / n):9.4f}% '
                  f'{nd_out:9d} {rel:10.1e} {rel_out:10.1e} {med:9.1e} {dt:5.2f}s  {where}')
        self.assertEqual(set(agg), set(OPS))
        bad = {}
        for name, (nd, nd_out, n, rel, rel_out, med, *_) in agg.items():
            ok = nd_out == 0 and (name in FAST_MATH or nd == 0 and rel <= REL_TOL)
            if not ok:
                bad[name] = dict(nan_diff=nd, nan_diff_outside_constant=nd_out, max_rel=rel, worst_median=med)
        self.assertFalse(bad, bad)

    def test_derived_fields(self):
        print()
        for k in ('price', 'amihud', 'ma_dist_20', 'ma_dist_60', 'ma_dist_120', 'ret_20', 'ret_60', 'ret_120'):
            nd, _, n, rel, _, _ = compare(self.panel_port[k], self.panel_cpu[k])
            print(f'  字段 {k:12s} NaN 位置不同 {nd} / {n} · 最大相对差 {rel:.1e}')
            self.assertEqual(nd, 0)
            self.assertLessEqual(rel, REL_TOL)

    def test_library(self):
        universe = np.asarray(MARKET.universe(self.panel_cpu))
        vocab = self.panel_cpu.vocabulary
        print()
        for i, m in enumerate(Library.load().members):
            tree = parse(m['expr'], vocab)
            names = tree.fields()
            c = evaluate(tree.root, {k: self.panel_cpu[k] for k in names}, self.cpu, {})
            p = evaluate(tree.root, {k: self.panel_port[k] for k in names}, self.port, {})
            nd, _, n, rel, _, _ = compare(p, c)
            wdiff = float(np.abs(book(p, universe) - book(c, universe)).max())
            print(f'  成员 {i} NaN 位置不同 {nd} / {n} · 最大相对差 {rel:.1e} · 成书最大差 {wdiff:.1e} · '
                  f'有值格 {int(np.isfinite(c).sum())}  {m["expr"]}')
            self.assertEqual(nd, 0)
            self.assertLessEqual(rel, REL_TOL)


if __name__ == '__main__':
    unittest.main(verbosity=2)
