from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r2:围绕 r1 最佳骨架 A0(20 日突破入场 + 自入场峰值回撤 25% 追踪止损 + 跌破 SMA60 出场)
# 假设:追踪止损/突破族在 2025 已接近 0(C0 是 −1.16),短板是 IS≈1.60 与熊年亏损(2018 −19% / 2022 −37%)。
#   1) 邻居:n 15/25、s 0.20/0.30、m 45/75
#   2) 市场级门只卡新开仓(不强平):熊市状态机 / EW>SMA100 / EW>SMA50 / EW 滞回
#   3) 逐币入场过滤:SMA60 上行 / 价在 SMA120 上
#   4) 路径依赖:上一笔亏损后改用 60/40 日突破;浮盈收紧止损;状态依赖止损(熊市 15% + 40 日突破)
#   5) 权益曲线过滤(组合层 / 逐币);牛市保底;与 C0 并集 / 平均;SMA 出场需连续 2 日;ATR 止损
# ==== BODY ====
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
