from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r4:P1 族(20 日突破 + 25% 追踪止损 + SMA45 出场)整片过 G2+GR,min(2025,2026) 卡在 0.1-0.35。
# 月度敞口:亏损集中在「突破后立刻失败」段(2026-01 敞口 0.51 / EW −7.6%;2025-06、2025-12)。
# 假设:假突破可以用路径依赖规则识别并提前离场 / 不入场:
#   时间止损(持有 k 天峰值浮盈仍 <x ⇒ 走)、初始紧止损(浮盈确认前止损更紧)、出场后冷却期、
#   放量确认突破、突破幅度门槛、相对强度(币 20 日收益 > EW 20 日)入场、
#   系统性止损潮(影子 P1 近 5 日成批出场 ⇒ 暂停开仓直到 EW 创 20 日新高)、EW 崩盘日后 10 天不开仓;
#   以及 P1_tg(浮盈收紧止损,r3 IS 1.80)与 2c / n25 / s30 / 分批减仓 的组合。
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
TG = (0.25, 0.10)


def _p1(d, **kw):
    """P1 状态机,可叠加 _smx 的全部路径依赖选项;ent 可替换。"""
    ent = kw.pop('ent', None)
    m = kw.pop('m', 45)
    s = kw.pop('s', 0.25)
    n = kw.pop('n', 20)
    return _smx(_px(d), _brk(d, n) if ent is None else ent, _below(d, m), s, **kw)


def _r(name, doc, **kw):
    _reg(name, lambda d, l: _p1(d, **kw)[0], doc)


_r('alpha_r4_P1_ts10', 'P1 + 时间止损:持有 10 天峰值浮盈 <5% ⇒ 出场', tstop=(10, 0.05))
_r('alpha_r4_P1_ts15', 'P1 + 时间止损:持有 15 天峰值浮盈 <8% ⇒ 出场', tstop=(15, 0.08))
_r('alpha_r4_P1_is12', 'P1 + 初始紧止损:峰值浮盈 <10% 时止损 12%', istop=(0.12, 0.10))
_r('alpha_r4_P1_is15', 'P1 + 初始紧止损:峰值浮盈 <10% 时止损 15%', istop=(0.15, 0.10))
_r('alpha_r4_P1_cd5', 'P1 + 出场后 5 天冷却', cool=5)
_r('alpha_r4_P1_cd10', 'P1 + 出场后 10 天冷却', cool=10)


def _vc(d, k):
    v = np.where(d['volume'] > 0, d['volume'], np.nan)
    return _brk(d, 20) & (v > k * _lag(_roll(v, 20, 'mean')))


_reg('alpha_r4_P1_vc15', lambda d, l: _p1(d, ent=_vc(d, 1.5))[0], 'P1,突破须放量(量 > 1.5×前 20 日均量)')
_reg('alpha_r4_P1_vc12', lambda d, l: _p1(d, ent=_vc(d, 1.2))[0], 'P1,突破须放量(量 > 1.2×前 20 日均量)')
_reg('alpha_r4_P1_bm3', lambda d, l: _p1(d, ent=_px(d) > 1.03 * _lag(_roll(_px(d), 20, 'max')))[0],
     'P1,突破须超前 20 日高 3%')


def _rs(d, l):
    L, _ = _ew(d, l)
    c = _px(d)
    rs = np.log(c / _lag(c, 20)) > (L - _lag(L, 20))[:, None]
    return _p1(d, ent=_brk(d, 20) & rs)[0]


_reg('alpha_r4_P1_rs', _rs, 'P1,只在币 20 日收益跑赢 EW 时入场')


def _sys(d, l):
    st = _p1(d)[0]
    prev = _lag(st)
    exits = np.nan_to_num(prev) * (1 - st)
    ex5 = _roll1(exits.sum(1), 5, 'sum')
    held = _roll1(np.nan_to_num(prev).sum(1), 5, 'max')
    mass = np.nan_to_num(ex5) >= np.maximum(3, 0.5 * np.nan_to_num(held))
    L, _ = _ew(d, l)
    clear = L > _lag(_roll1(L, 20, 'max'))
    off = _hyst(mass, clear)
    return _p1(d, allow=~off[:, None])[0]


_reg('alpha_r4_P1_sys', _sys, 'P1 + 系统性止损潮:影子 P1 近 5 日 ≥半数持仓出场 ⇒ 暂停开仓至 EW 创 20 日新高')


def _ck(d, l):
    _, ew = _ew(d, l)
    blk = np.nan_to_num(_roll1((ew < -0.07).astype(float), 10, 'max')) > 0
    return _p1(d, allow=~blk[:, None])[0]


_reg('alpha_r4_P1_ck', _ck, 'P1 + EW 单日跌 >7% 后 10 天不开新仓(不强平)')

_r('alpha_r4_tg_ts10', 'P1_tg + 时间止损 10 天 / 5%', tight=TG, tstop=(10, 0.05))
_r('alpha_r4_tg_is12', 'P1_tg + 初始紧止损 12%', tight=TG, istop=(0.12, 0.10))
_r('alpha_r4_tg_n25', 'P1_tg,n=25', tight=TG, n=25)
_r('alpha_r4_tg_s30', 'P1_tg,s=0.30', tight=TG, s=0.30)


def _tg2c(d, l):
    b = _below(d, 45)
    return _smx(_px(d), _brk(d, 20), b & (_lag(b.astype(float)) > 0), 0.25, tight=TG)[0]


_reg('alpha_r4_tg_2c', _tg2c, 'P1_tg,SMA45 出场需连续 2 日')


def _tgso(d, l):
    st, dd, _ = _p1(d, tight=TG)
    return st * np.where(dd >= 0.10, 0.5, 1.0)


_reg('alpha_r4_tg_so', _tgso, 'P1_tg + 回撤 ≥10% 减半')
