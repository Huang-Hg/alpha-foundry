from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r3:新骨架 P1 = 20 日突破入场 + 自入场峰值回撤 25% + 跌破 SMA45 出场
# (r2:A0_m45 IS 1.67 / 2025 +0.35 / 2026 +0.22 / 牛年最小跟随 0.62,G2+GR 同过)
# 假设:① P1 的邻居是否整片过门;② 路径依赖的仓位管理能否在不损牛年跟随的前提下抬 IS:
#   浮盈收紧止损、SMA 出场需 2 日确认、熊市状态切换逐币规则、
#   分批加仓(突破先半仓、浮盈 x 确认后满仓 —— 假突破只亏半仓)、分批减仓(回撤过半止损距离先减半)、
#   SMA 收复再入场(牛市里被洗出后不必等新突破)
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


from backtest.ops_cuda.q25 import state as _smx


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


P1 = dict(n=20, s=0.25, m=45)


def _nb(**kw):
    return lambda d, l: _A(d, **{**P1, **kw})


_reg('alpha_r3_P1', _nb(), 'P1')
for k, v in (('m35', dict(m=35)), ('m55', dict(m=55)), ('n15', dict(n=15)), ('n25', dict(n=25)),
             ('s20', dict(s=0.20)), ('s30', dict(s=0.30))):
    _reg(f'alpha_r3_P1_{k}', _nb(**v), f'P1 邻居 {v}')
_reg('alpha_r3_P1_tg', _nb(tight=(0.25, 0.10)), 'P1 + 止损随浮盈收紧')


def _2c(d, l, m=45):
    b = _below(d, m)
    return _smx(_px(d), _brk(d, 20), b & (_lag(b.astype(float)) > 0), 0.25)


_reg('alpha_r3_P1_2c', lambda d, l: _2c(d, l)[0], 'P1,SMA45 出场需连续 2 日')


def _sd(d, l):
    b = _mbear(d, l)[:, None]
    ent = np.where(b, _brk(d, 40), _brk(d, 20))
    stop = np.where(b, 0.15, 0.25) * np.ones(d['close'].shape)
    return _smx(_px(d), ent, _below(d, 45), stop)[0]


_reg('alpha_r3_P1_sd', _sd, 'P1,市场熊时改 40 日突破 + 15% 止损')


def _pyr(x, tight=None):
    def f(d, l):
        st, dd, pg = _smx(_px(d), _brk(d, 20), _below(d, 45), 0.25, tight=tight)
        return st * np.where(pg >= x, 1.0, 0.5)
    return f


for x in (0.05, 0.10, 0.20):
    _reg(f'alpha_r3_P1_py{int(x * 100):02d}', _pyr(x), f'P1 分批加仓:突破半仓,峰值浮盈 ≥{x:.0%} 后满仓')
_reg('alpha_r3_P1_tg_py10', _pyr(0.10, (0.25, 0.10)), 'P1 + 浮盈收紧 + 分批加仓 10%')


def _so(d, l):
    st, dd, pg = _smx(_px(d), _brk(d, 20), _below(d, 45), 0.25)
    return st * np.where(dd >= 0.125, 0.5, 1.0)


_reg('alpha_r3_P1_so', _so, 'P1 分批减仓:回撤 ≥12.5% 减半')


def _reent(mkt):
    def f(d, l):
        c = _px(d)
        s = _sma(d, 45)
        x = (c > s) & (_lag(c) <= _lag(s)) & (s > _lag(s, 5))
        if mkt:
            x &= _mabove(d, l, 100)[:, None]
        return _smx(c, _brk(d, 20) | x, _below(d, 45), 0.25)[0]
    return f


_reg('alpha_r3_P1_re_m', _reent(True), 'P1 + 市场 EW>SMA100 时上穿上行 SMA45 也可入场')
_reg('alpha_r3_P1_re', _reent(False), 'P1 + 上穿上行 SMA45 也可入场')
_reg('alpha_r3_P1_avg_c0', lambda d, l: 0.5 * (_A(d, **P1) + _c0(d)), '(P1 + C0)/2')
