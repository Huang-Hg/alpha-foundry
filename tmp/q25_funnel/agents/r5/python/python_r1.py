from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r1:铺结构地形 —— FastExpr 写不出的路径依赖 / 状态机结构,每类 1-3 个参数点
#   B  基线:前沿 #1 逐币趋势闸 C0 = and(ma_dist_20>0, ts_delta(ma_dist_20,30)>0)(用于校准)
#   A  逐币追踪止损状态机:突破入场 + 自入场以来峰值回撤 s 出场(再入场须新突破)
#   K  ATR 吊灯止损(止损距离随逐币波动伸缩)
#   D  持仓以来回撤连续减仓:回撤 d0 内满仓,d1 清仓,之间线性
#   M  市场级熊市状态机(EW 指数:进入 = 跌破 SMA100 且距 365 日高回撤 >25%;退出 = 站上 SMA50 且 SMA50 上行)
#      牛市状态 → 满仓 / 保底仓位;熊市状态 → 逐币趋势闸
#   H  市场级双阈值滞回(EW 指数相对 SMA150 ±3%)× 保底
#   E  权益曲线过滤:逐币 / 组合层影子书近期亏损则降仓("趋势跟随最近是否奏效")
#   W  宽度滞回 × 状态依赖选币:牛市持全部,熊市只持截面 chanpos150 前 3 且绝对位 >0.5
#   F  逐币熊市状态机:币自身 365 日回撤 >50% 且跌破 SMA100 进熊,站上上行 SMA100 出熊
#   G  崩盘冷却:EW 单日跌幅 >8% 后 5 天空仓
# 因果:rolling 右对齐 min_periods=w;状态机逐行前推只读 ≤t;截面只用当日行。
import cupy as np

_C = {}


def _cache(key, fn):
    if key not in _C:
        _C[key] = fn()
    return _C[key]


def _px(d):
    return _cache(('px', d['close'].shape[0]), lambda: np.where(d['close'] > 0, d['close'], np.nan))


def _roll(x, w, how):
    return _cuda_rolling(x, w, how)


def _roll1(x, w, how):
    return _cuda_rolling(x, w, how)


def _lag(x, k=1):
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[k:] = x[:-k]
    return out


def _ret(d):
    def f():
        c = _px(d)
        return c / _lag(c) - 1.0
    return _cache(('ret', d['close'].shape[0]), f)


def _ew(d, liq):
    """(T,) 等权指数对数累计:t 日收益取 t-1 日 liquid 且收益有限的币。"""
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
    return _cache(('c0', m_shape(d)), f)


def m_shape(d):
    return d['close'].shape[0]


def _brk(d, n):
    """n 日突破事件:close > 前 n 日最高收盘。"""
    def f():
        c = _px(d)
        return c > _lag(_roll(c, n, 'max'))
    return _cache(('brk', n, m_shape(d)), f)


def _atrf(d, w):
    """ATR_w / close(真实波幅用前收)。"""
    def f():
        c = _px(d)
        pc = _lag(c)
        tr = np.fmax(d['high'] - d['low'], np.fmax(np.abs(d['high'] - pc), np.abs(d['low'] - pc)))
        return _roll(tr, w, 'mean') / c
    return _cache(('atr', w, m_shape(d)), f)


from alpha_cuda import simple_state as _sm


from backtest.ops_cuda.q25 import hysteresis as _hyst


def _none(d):
    return np.zeros(d['close'].shape, bool)


# ---------------------------------------------------------------- 市场状态
def _mbear(d, liq, sma_in=100, dd_in=0.25, sma_out=50, rise=10):
    """市场级熊市状态机:进入 = EW < SMA_in 且距 365 日高回撤 > dd_in;退出 = EW > SMA_out 且 SMA_out 较 rise 天前上行。"""
    def f():
        L, _ = _ew(d, liq)
        s_in = _roll1(L, sma_in, 'mean')
        s_out = _roll1(L, sma_out, 'mean')
        ddm = L - _roll1(L, 365, 'max')
        ent = (L < s_in) & (ddm < np.log1p(-dd_in))
        ext = (L > s_out) & (s_out > _lag(s_out, rise))
        return _hyst(ent, ext)
    return _cache(('mbear', sma_in, dd_in, sma_out, rise, liq.shape[0]), f)


def _mhyst(d, liq, w=150, band=0.03):
    def f():
        L, _ = _ew(d, liq)
        s = _roll1(L, w, 'mean')
        return _hyst(L > s + band, L < s - band)
    return _cache(('mhyst', w, band, liq.shape[0]), f)


def _breadth(d, liq, w):
    def f():
        c = _px(d)
        s = _roll(c, w, 'mean')
        ok = liq & np.isfinite(s) & np.isfinite(c)
        return np.where(ok.sum(1) > 0, (ok & (c > s)).sum(1) / np.maximum(ok.sum(1), 1), np.nan)
    return _cache(('br', w, liq.shape[0]), f)


def _chanpos(d, w):
    def f():
        c = _px(d)
        hi, lo = _roll(c, w, 'max'), _roll(c, w, 'min')
        return np.where(hi > lo, (c - lo) / np.where(hi > lo, hi - lo, 1.0), np.nan)
    return _cache(('cp', w, m_shape(d)), f)


def _topk(score, liq, k):
    x = np.where(liq & np.isfinite(score), score, -np.inf)
    order = np.argsort(-x, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(x.shape[1])[None, :], axis=1)
    return (rank < k) & np.isfinite(score) & liq


# ================================================================ B 基线
def alpha_b_c0(data, liquid):
    """前沿 #1 逐币趋势闸(校准用)"""
    return _c0(data)


# ================================================================ A 逐币追踪止损
def _a(d, n, s, sma_exit=None):
    c = _px(d)
    ex = _none(d) if sma_exit is None else (c < _roll(c, sma_exit, 'mean'))
    return _sm(c, _brk(d, n), ex, s)[0]


def alpha_a_dc20_s20(data, liquid):
    """20 日突破入场;自入场峰值回撤 20% 出场"""
    return _a(data, 20, 0.20)


def alpha_a_dc20_s30(data, liquid):
    return _a(data, 20, 0.30)


def alpha_a_dc20_s25_m60(data, liquid):
    """20 日突破入场;峰值回撤 25% 或跌破 SMA60 出场"""
    return _a(data, 20, 0.25, 60)


def alpha_a_dc50_s25(data, liquid):
    return _a(data, 50, 0.25)


def alpha_a_x20_s25(data, liquid):
    """上穿 SMA20 且 SMA20 5 日上行入场;峰值回撤 25% 出场"""
    c = _px(data)
    s = _roll(c, 20, 'mean')
    en = (c > s) & (_lag(c) <= _lag(s)) & (s > _lag(s, 5))
    return _sm(c, en, _none(data), 0.25)[0]


# ================================================================ K ATR 吊灯
def alpha_k_ch3(data, liquid):
    """20 日突破入场;峰值 − 3×ATR20 吊灯止损"""
    return _sm(_px(data), _brk(data, 20), _none(data), np.nan_to_num(3 * _atrf(data, 20), nan=0.3))[0]


def alpha_k_ch5(data, liquid):
    return _sm(_px(data), _brk(data, 20), _none(data), np.nan_to_num(5 * _atrf(data, 20), nan=0.3))[0]


# ================================================================ D 持仓以来回撤连续减仓
def _d(d, n, d0, d1):
    st, dd = _sm(_px(d), _brk(d, n), _none(d), d1)
    return st * np.clip((d1 - dd) / (d1 - d0), 0.0, 1.0)


def alpha_d_10_30(data, liquid):
    """20 日突破入场;持仓回撤 <10% 满仓,10-30% 线性减,30% 清仓"""
    return _d(data, 20, 0.10, 0.30)


def alpha_d_05_25(data, liquid):
    return _d(data, 20, 0.05, 0.25)


# ================================================================ M 市场级熊市状态机 × 保底
def alpha_m_full_c0(data, liquid):
    """市场非熊 → 全部 liquid 满仓;熊 → 逐币 C0"""
    b = _mbear(data, liquid)[:, None]
    return (1 - b) * 1.0 + b * _c0(data)


def alpha_m_f50_c0(data, liquid):
    """市场非熊 → max(0.5, C0);熊 → C0"""
    b = _mbear(data, liquid)[:, None]
    c0 = _c0(data)
    return (1 - b) * np.maximum(0.5, c0) + b * c0


def alpha_m_f50_ts(data, liquid):
    """市场非熊 → max(0.5, 追踪止损书 dc20_s25);熊 → 追踪止损书"""
    b = _mbear(data, liquid)[:, None]
    a = _a(data, 20, 0.25)
    return (1 - b) * np.maximum(0.5, a) + b * a


def alpha_m_f50_c0_dd35(data, liquid):
    """同 m_f50_c0,熊市进入回撤门 35%"""
    b = _mbear(data, liquid, dd_in=0.35)[:, None]
    c0 = _c0(data)
    return (1 - b) * np.maximum(0.5, c0) + b * c0


# ================================================================ H 市场双阈值滞回 × 保底
def alpha_h_f50_c0(data, liquid):
    """EW 指数 > SMA150+3% 进牛,< SMA150−3% 出牛;牛 → max(0.5,C0),非牛 → C0"""
    g = _mhyst(data, liquid)[:, None]
    c0 = _c0(data)
    return g * np.maximum(0.5, c0) + (1 - g) * c0


def alpha_h_full_c0(data, liquid):
    g = _mhyst(data, liquid)[:, None]
    return g * 1.0 + (1 - g) * _c0(data)


# ================================================================ E 权益曲线过滤
def _shadow_coin(d, e, w):
    """逐币影子书近 w 日累计收益(用 t-1 敞口 × t 日收益,t 时可知)。"""
    r = np.nan_to_num(_ret(d))
    pnl = _lag(e) * r
    return _roll(np.nan_to_num(pnl), w, 'sum')


def alpha_e_coin60(data, liquid):
    """C0,但该币上 C0 规则近 60 日影子收益 <0 则不持"""
    c0 = _c0(data)
    return c0 * (np.nan_to_num(_shadow_coin(data, c0, 60), nan=0.0) >= 0)


def alpha_e_port30(data, liquid):
    """C0,组合层影子书权益 < 其 30 日均线时减半"""
    c0 = _c0(data)
    r = np.nan_to_num(_ret(data))
    lp = np.zeros_like(liquid)
    lp[1:] = liquid[:-1]
    n = np.maximum(lp.sum(1), 1)
    pr = (np.nan_to_num(_lag(c0)) * r * lp).sum(1) / n
    eq = np.cumsum(np.log1p(pr))
    on = eq >= np.nan_to_num(_roll1(eq, 30, 'mean'), nan=-np.inf)
    return c0 * np.where(on, 1.0, 0.5)[:, None]


# ================================================================ W 宽度滞回 × 状态依赖选币
def alpha_w_br_sel(data, liquid):
    """宽度(close>SMA50 占比)>0.6 进牛 <0.4 出牛;牛 → 全部满仓;非牛 → chanpos150 前 3 且 >0.5"""
    br = _breadth(data, liquid, 50)
    g = _hyst(br > 0.6, br < 0.4)[:, None]
    cp = _chanpos(data, 150)
    sel = (_topk(cp, liquid, 3) & (cp > 0.5)).astype(np.float64)
    return g * 1.0 + (1 - g) * sel


def alpha_w_br_c0sel(data, liquid):
    """牛 → max(0.5, C0);非牛 → C0 ∧ chanpos150 前 3"""
    br = _breadth(data, liquid, 50)
    g = _hyst(br > 0.6, br < 0.4)[:, None]
    cp = _chanpos(data, 150)
    c0 = _c0(data)
    return g * np.maximum(0.5, c0) + (1 - g) * c0 * _topk(cp, liquid, 3)


# ================================================================ F 逐币熊市状态机
from alpha_cuda import coinbear as alpha_f_coinbear


# ================================================================ G 崩盘冷却
def alpha_g_cool(data, liquid):
    """C0,EW 单日跌 >8% 后 5 天空仓"""
    _, ew = _ew(data, liquid)
    crash = (ew < -0.08).astype(np.float64)
    cool = _roll1(crash, 5, 'max')
    return _c0(data) * (1 - np.nan_to_num(cool))[:, None]
