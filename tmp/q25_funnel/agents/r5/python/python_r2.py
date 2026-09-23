from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r2:围绕 r1 最佳骨架 A0(20 日突破入场 + 自入场峰值回撤 25% 追踪止损 + 跌破 SMA60 出场)
# 假设:追踪止损/突破族在 2025 已接近 0(C0 是 −1.16),短板是 IS≈1.60 与熊年亏损(2018 −19% / 2022 −37%)。
#   1) 邻居:n 15/25、s 0.20/0.30、m 45/75
#   2) 市场级门只卡新开仓(不强平):熊市状态机 / EW>SMA100 / EW>SMA50 / EW 滞回
#   3) 逐币入场过滤:SMA60 上行 / 价在 SMA120 上
#   4) 路径依赖:上一笔亏损后改用 60/40 日突破;浮盈收紧止损;状态依赖止损(熊市 15% + 40 日突破)
#   5) 权益曲线过滤(组合层 / 逐币);牛市保底;与 C0 并集 / 平均;SMA 出场需连续 2 日;ATR 止损
# ==== Python 车道 R5 共用原语(由 _build.py 拼进每轮的自包含 alpha 文件)====
# 因果:rolling 右对齐 min_periods=w;状态机逐行前推只读 ≤t;截面只用当日行;缓存键含 T(截断重跑互不串)。
import cupy as np

_C = {}


def _cache(key, fn):
    if key not in _C:
        _C[key] = fn()
    return _C[key]


def _T(d):
    return d['close'].shape[0]


def _px(d):
    return _cache(('px', _T(d)), lambda: np.where(d['close'] > 0, d['close'], np.nan))


def _roll(x, w, how):
    return _cuda_rolling(x, w, how)


def _roll1(x, w, how):
    return _cuda_rolling(x, w, how)


def _lag(x, k=1):
    out = np.full(x.shape, np.nan)
    out[k:] = x[:-k]
    return out


def _ret(d):
    return _cache(('ret', _T(d)), lambda: _px(d) / _lag(_px(d)) - 1.0)


def _sma(d, w):
    return _cache(('sma', w, _T(d)), lambda: _roll(_px(d), w, 'mean'))


def _ew(d, liq):
    """(T,) 等权指数:t 日收益取 t-1 日 liquid 且收益有限的币。→ (对数累计, 日收益)"""
    def f():
        r = _ret(d)
        lp = np.zeros_like(liq)
        lp[1:] = liq[:-1]
        ok = lp & np.isfinite(r)
        ew = np.where(ok, r, 0.0).sum(1) / np.maximum(ok.sum(1), 1)
        return np.cumsum(np.log1p(ew)), ew
    return _cache(('ew', liq.shape[0]), f)


def _c0(d):
    """前沿 #1 逐币趋势闸:ma_dist_20>0 且 30 日内 ma_dist_20 上升。"""
    def f():
        m = d['ma_dist_20']
        return ((m > 0) & (m - _lag(m, 30) > 0)).astype(np.float64)
    return _cache(('c0', _T(d)), f)


def _brk(d, n):
    """n 日突破事件:close > 前 n 日最高收盘。"""
    return _cache(('brk', n, _T(d)), lambda: _px(d) > _lag(_roll(_px(d), n, 'max')))


def _below(d, m):
    return _cache(('below', m, _T(d)), lambda: _px(d) < _sma(d, m))


def _none(d):
    return np.zeros(d['close'].shape, bool)


from alpha_cuda import state2 as _smx


from backtest.ops_cuda.q25 import hysteresis as _hyst


def _mbear(d, liq, sma_in=100, dd_in=0.25, sma_out=50, rise=10):
    """市场级熊市状态机(进入 / 退出条件不同):进入 = EW < SMA_in 且距 365 日高回撤 > dd_in;
    退出 = EW > SMA_out 且 SMA_out 较 rise 天前上行。→ (T,) bool"""
    def f():
        L, _ = _ew(d, liq)
        s_in = _roll1(L, sma_in, 'mean')
        s_out = _roll1(L, sma_out, 'mean')
        ddm = L - _roll1(L, 365, 'max')
        ent = (L < s_in) & (ddm < np.log1p(-dd_in))
        ext = (L > s_out) & (s_out > _lag(s_out, rise))
        return _hyst(ent, ext)
    return _cache(('mbear', sma_in, dd_in, sma_out, rise, liq.shape[0]), f)


def _mabove(d, liq, w):
    """(T,) EW 指数在 SMA_w 之上。"""
    def f():
        L, _ = _ew(d, liq)
        return L > _roll1(L, w, 'mean')
    return _cache(('mabove', w, liq.shape[0]), f)


def _mhyst(d, liq, w=150, band=0.03):
    def f():
        L, _ = _ew(d, liq)
        s = _roll1(L, w, 'mean')
        return _hyst(L > s + band, L < s - band)
    return _cache(('mhyst', w, band, liq.shape[0]), f)


def _book_ret(d, liq, e):
    """(T,) 影子书日收益(t-1 敞口 × t 日收益 / t-1 流动名数)—— 只供权益曲线过滤的状态输入。"""
    r = np.nan_to_num(_ret(d))
    lp = np.zeros_like(liq)
    lp[1:] = liq[:-1]
    el = np.nan_to_num(_lag(np.where(liq, np.clip(e, 0, 1), 0.0)))
    return (el * r * lp).sum(1) / np.maximum(lp.sum(1), 1)


def _chanpos(d, w):
    def f():
        c = _px(d)
        hi, lo = _roll(c, w, 'max'), _roll(c, w, 'min')
        return np.where(hi > lo, (c - lo) / np.where(hi > lo, hi - lo, 1.0), np.nan)
    return _cache(('cp', w, _T(d)), f)


def _topk(score, liq, k):
    x = np.where(liq & np.isfinite(score), score, -np.inf)
    order = np.argsort(-x, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(x.shape[1])[None, :], axis=1)
    return (rank < k) & np.isfinite(score) & liq


def _A(d, n=20, s=0.25, m=60, allow=None, akey=None, n_loss=None, tight=None):
    """A 族骨架:n 日突破入场;自入场峰值回撤 s 或跌破 SMA_m 出场。allow 须配 akey(缓存键)。"""
    assert allow is None or akey is not None
    key = ('A', n, s, m, akey, n_loss, tight, _T(d))

    def f():
        ex = _none(d) if m is None else _below(d, m)
        el = None if n_loss is None else _brk(d, n_loss)
        return _smx(_px(d), _brk(d, n), ex, s, allow=allow, ent_loss=el, tight=tight)[0]
    return _cache(key, f)


def _reg(name, fn, doc=''):
    def f(data, liquid):
        return fn(data, liquid)
    f.__name__ = name
    f.__doc__ = doc
    globals()[name] = f
# ==== 共用原语结束 ====


A0 = dict(n=20, s=0.25, m=60)


def _nb(**kw):
    return lambda d, l: _A(d, **{**A0, **kw})


_reg('alpha_r2_A0', _nb(), '基线 A0')
for k, v in (('n15', dict(n=15)), ('n25', dict(n=25)), ('s20', dict(s=0.20)), ('s30', dict(s=0.30)),
             ('m45', dict(m=45)), ('m75', dict(m=75))):
    _reg(f'alpha_r2_A0_{k}', _nb(**v), f'A0 邻居 {v}')


def _gate(key, fn):
    def g(d, l):
        a = fn(d, l)
        return _A(d, **A0, allow=a if a.ndim == 2 else a[:, None], akey=key)
    return g


_reg('alpha_r2_A0_gbear', _gate('gbear', lambda d, l: ~_mbear(d, l)), '市场非熊才开新仓')
_reg('alpha_r2_A0_gbear35', _gate('gbear35', lambda d, l: ~_mbear(d, l, dd_in=0.35)), '同上,熊市回撤门 35%')
_reg('alpha_r2_A0_gew100', _gate('gew100', lambda d, l: _mabove(d, l, 100)), 'EW>SMA100 才开新仓')
_reg('alpha_r2_A0_gew50', _gate('gew50', lambda d, l: _mabove(d, l, 50)), 'EW>SMA50 才开新仓')
_reg('alpha_r2_A0_gmh', _gate('gmh', lambda d, l: _mhyst(d, l)), 'EW 滞回牛才开新仓')
_reg('alpha_r2_A0_s60up', _gate('s60up', lambda d, l: _sma(d, 60) > _lag(_sma(d, 60), 10)), 'SMA60 10 日上行才开仓')
_reg('alpha_r2_A0_ma120', _gate('ma120', lambda d, l: _px(d) > _sma(d, 120)), '价在 SMA120 上才开仓')
_reg('alpha_r2_A0_nl60', _nb(n_loss=60), '上一笔亏损后改用 60 日突破入场')
_reg('alpha_r2_A0_nl40', _nb(n_loss=40), '上一笔亏损后改用 40 日突破入场')
_reg('alpha_r2_A0_tg25', _nb(tight=(0.25, 0.10)), '止损随浮盈收紧 s=max(0.10, 0.25-0.25·gain)')
_reg('alpha_r2_A0_tg50', _nb(tight=(0.50, 0.12)), '止损随浮盈收紧 s=max(0.12, 0.25-0.5·gain)')


def _sd(d, l):
    """状态依赖:市场熊 → 40 日突破入场 + 15% 止损;非熊 → A0"""
    b = _mbear(d, l)[:, None]
    ent = np.where(b, _brk(d, 40), _brk(d, 20))
    stop = np.where(b, 0.15, 0.25) * np.ones(d['close'].shape)
    return _smx(_px(d), ent, _below(d, 60), stop)[0]


_reg('alpha_r2_A_sd', _sd, '熊市状态机切换逐币规则')


def _eq30(d, l):
    a = _A(d, **A0)
    eq = np.cumsum(np.log1p(_book_ret(d, l, a)))
    on = eq >= np.nan_to_num(_roll1(eq, 30, 'mean'), nan=-np.inf)
    return a * np.where(on, 1.0, 0.5)[:, None]


_reg('alpha_r2_A0_eq30', _eq30, '组合影子书权益 < 其 30 日均线时减半')


def _ec60(d, l):
    a = _A(d, **A0)
    pnl = np.nan_to_num(_lag(a) * np.nan_to_num(_ret(d)))
    return a * (np.nan_to_num(_roll(pnl, 60, 'sum'), nan=0.0) >= 0)


_reg('alpha_r2_A0_ec60', _ec60, '逐币:A0 在该币近 60 日影子收益 <0 则不持')


def _fl(d, l):
    a = _A(d, **A0)
    bull = (~_mbear(d, l) & _mabove(d, l, 100))[:, None]
    return np.maximum(a, 0.5 * (bull & (_px(d) > _sma(d, 60))))


_reg('alpha_r2_A0_fl', _fl, '市场牛(非熊且 EW>SMA100)时,价在 SMA60 上的币保底 0.5')
_reg('alpha_r2_A0_or_c0', lambda d, l: np.maximum(_A(d, **A0), _c0(d)), 'A0 ∪ C0')
_reg('alpha_r2_A0_avg_c0', lambda d, l: 0.5 * (_A(d, **A0) + _c0(d)), '(A0 + C0)/2')


def _2c(d, l):
    b = _below(d, 60)
    return _smx(_px(d), _brk(d, 20), b & (_lag(b.astype(float)) > 0), 0.25)[0]


_reg('alpha_r2_A0_2c', _2c, 'SMA60 出场需连续 2 日收在下方')


def _atr(d, w):
    c = _px(d)
    pc = _lag(c)
    tr = np.fmax(d['high'] - d['low'], np.fmax(np.abs(d['high'] - pc), np.abs(d['low'] - pc)))
    return _roll(tr, w, 'mean') / c


_reg('alpha_r2_A0_atr4', lambda d, l: _smx(_px(d), _brk(d, 20), _below(d, 60),
                                           np.nan_to_num(4 * _atr(d, 20), nan=0.25))[0], '止损改 4×ATR20')
_reg('alpha_r2_c0_gbear', lambda d, l: _c0(d) * ~_mbear(d, l)[:, None] + 0.5 * _c0(d) * _mbear(d, l)[:, None],
     'C0,市场熊时减半')
