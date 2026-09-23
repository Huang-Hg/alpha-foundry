from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r3:新骨架 P1 = 20 日突破入场 + 自入场峰值回撤 25% + 跌破 SMA45 出场
# (r2:A0_m45 IS 1.67 / 2025 +0.35 / 2026 +0.22 / 牛年最小跟随 0.62,G2+GR 同过)
# 假设:① P1 的邻居是否整片过门;② 路径依赖的仓位管理能否在不损牛年跟随的前提下抬 IS:
#   浮盈收紧止损、SMA 出场需 2 日确认、熊市状态切换逐币规则、
#   分批加仓(突破先半仓、浮盈 x 确认后满仓 —— 假突破只亏半仓)、分批减仓(回撤过半止损距离先减半)、
#   SMA 收复再入场(牛市里被洗出后不必等新突破)
# ==== BODY ====
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
