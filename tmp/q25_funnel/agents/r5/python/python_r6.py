from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r6:最后一轮结构探索(之后进终选 + 邻域电池)
# r5:崩盘冷却 ck(EW 单日跌 >7% 后 10 天不开新仓)是平台(阈 −6/−8.5%、天 7/14 的 2025 全在 +0.54~+0.67);
#     ck_ts15 = +0.48 / +0.45 为当前 min(2025,2026) 最高且 G2+GR 过。
# 假设:
#   ① 崩盘冷却的推广:崩盘日强平(ckx)、3 日累计崩盘(ck3)、逐币崩盘冷却(ckc)是否更好;
#   ② 假突破规则的叠加(ts + is)、保本止损(be)是否与 ck 正交;
#   ③ 出场载体换成唐奇安 N 日低点(海龟式),验证 SMA45 出场不是唯一可行载体。
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


TG = (0.25, 0.10)


def _p1(d, **kw):
    ent = kw.pop('ent', None)
    ex = kw.pop('ex', None)
    m = kw.pop('m', 45)
    s = kw.pop('s', 0.25)
    n = kw.pop('n', 20)
    return _smx(_px(d), _brk(d, n) if ent is None else ent, _below(d, m) if ex is None else ex, s, **kw)


def _ckb(d, l, th=-0.07, k=10):
    _, ew = _ew(d, l)
    blk = np.nan_to_num(_roll1((ew < th).astype(float), k, 'max')) > 0
    return ~blk[:, None]


def _r(name, doc, ck=(-0.07, 10), **kw):
    def f(d, l):
        a = _ckb(d, l, *ck) if ck is not None else None
        return _p1(d, allow=a, **kw)[0]
    _reg(name, f, doc)


def _ckx(d, l):
    _, ew = _ew(d, l)
    return _p1(d, allow=_ckb(d, l), fexit=(ew < -0.07)[:, None])[0]


_reg('alpha_r6_ckx', _ckx, 'P1 + 崩盘冷却 + 崩盘日收盘强平')


def _ck3(d, l):
    L, _ = _ew(d, l)
    crash = (L - _lag(L, 3)) < np.log1p(-0.12)
    blk = np.nan_to_num(_roll1(crash.astype(float), 10, 'max')) > 0
    return _p1(d, allow=~blk[:, None])[0]


_reg('alpha_r6_ck3', _ck3, 'P1 + EW 3 日累计跌 >12% 后 10 天不开新仓')


def _ckc(d, l):
    blk = np.nan_to_num(_roll((_ret(d) < -0.12).astype(float), 10, 'max')) > 0
    return _p1(d, allow=~blk)[0]


_reg('alpha_r6_ckc', _ckc, 'P1 + 该币单日跌 >12% 后 10 天不开该币新仓')


def _ckb2(d, l):
    """市场级 + 逐币崩盘冷却并用"""
    blk = np.nan_to_num(_roll((_ret(d) < -0.12).astype(float), 10, 'max')) > 0
    return _p1(d, allow=_ckb(d, l) & ~blk)[0]


_reg('alpha_r6_ck_ckc', _ckb2, 'P1 + 市场崩盘冷却 + 逐币崩盘冷却')
_r('alpha_r6_ck_ts15_is10', 'P1 + ck + 时间止损 15/8% + 初始止损 10%', tstop=(15, 0.08), istop=(0.10, 0.10))
_r('alpha_r6_ck_ts12', 'P1 + ck + 时间止损 12 天/6%', tstop=(12, 0.06))
_r('alpha_r6_ck_is10', 'P1 + ck + 初始止损 10%', istop=(0.10, 0.10))
_r('alpha_r6_ck_cd5_ts15', 'P1 + ck + 冷却 5 天 + 时间止损 15/8%', cool=5, tstop=(15, 0.08))
_r('alpha_r6_ck_be15', 'P1 + ck + 峰值浮盈 ≥15% 后保本', be=0.15)
_r('alpha_r6_ck_be15_ts15', 'P1 + ck + 保本 15% + 时间止损 15/8%', be=0.15, tstop=(15, 0.08))
_r('alpha_r6_tg_ck_is10', 'P1_tg + ck + 初始止损 10%', tight=TG, istop=(0.10, 0.10))


def _dl(w):
    return lambda d: _px(d) < _lag(_roll(_px(d), w, 'min'))


_reg('alpha_r6_ck_dl10', lambda d, l: _p1(d, ex=_dl(10)(d), allow=_ckb(d, l))[0],
     '海龟式:20 日突破入场 + 10 日低点出场 + 25% 追踪 + ck')
_reg('alpha_r6_dl10', lambda d, l: _p1(d, ex=_dl(10)(d))[0], '海龟式:20 日突破入场 + 10 日低点出场 + 25% 追踪')
_reg('alpha_r6_dl20', lambda d, l: _p1(d, ex=_dl(20)(d))[0], '20 日突破入场 + 20 日低点出场 + 25% 追踪')
_reg('alpha_r6_dl20_ck', lambda d, l: _p1(d, ex=_dl(20)(d), allow=_ckb(d, l))[0],
     '20 日突破入场 + 20 日低点出场 + 25% 追踪 + ck')
_reg('alpha_r6_dl15_ck_ts15', lambda d, l: _p1(d, ex=_dl(15)(d), allow=_ckb(d, l), tstop=(15, 0.08))[0],
     '20 日突破入场 + 15 日低点出场 + 25% 追踪 + ck + 时间止损')
