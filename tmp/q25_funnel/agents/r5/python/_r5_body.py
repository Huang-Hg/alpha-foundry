from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r5:组合 r4 的正贡献部件 + 邻居 + 修 sys 的解除条件
# r4:P1_ck(EW 单日跌 >7% 后 10 天不开新仓)2025 +0.54 GR✓;P1_sys(影子书成批止损 ⇒ 暂停开仓)2025 +0.64 但
#     2020/2023 跟随比 0.499/0.47 掉线;P1_ts15 / P1_is12 / cd5 小幅正贡献;P1_tg 系 IS 最高(1.80-1.86)。
# 假设:① 崩盘冷却与假突破规则作用在不同段落(可叠加);② 崩盘冷却的阈值/天数是平台而非尖峰;
#   ③ sys 的问题在解除条件太慢(EW 20 日新高),改 10 日新高或限时 20 天应能保住牛年跟随;
#   ④「波动冲击」(EW / 逐币 10 日波动 > 2×120 日波动)作开仓禁令是崩盘冷却的连续推广。
# ==== BODY ====
TG = (0.25, 0.10)


def _p1(d, **kw):
    ent = kw.pop('ent', None)
    m = kw.pop('m', 45)
    s = kw.pop('s', 0.25)
    n = kw.pop('n', 20)
    return _smx(_px(d), _brk(d, n) if ent is None else ent, _below(d, m), s, **kw)


def _ckb(d, l, th=-0.07, k=10):
    """EW 单日跌幅 < th 后 k 天(含当日)禁止开新仓 → (T,1) 许可。"""
    _, ew = _ew(d, l)
    blk = np.nan_to_num(_roll1((ew < th).astype(float), k, 'max')) > 0
    return ~blk[:, None]


def _r(name, doc, ck=None, **kw):
    def f(d, l):
        a = _ckb(d, l, *ck) if ck is not None else None
        return _p1(d, allow=a, **kw)[0]
    _reg(name, f, doc)


CK = (-0.07, 10)
_r('alpha_r5_ck_ts15', 'P1 + 崩盘冷却 + 时间止损 15 天/8%', ck=CK, tstop=(15, 0.08))
_r('alpha_r5_ck_is12', 'P1 + 崩盘冷却 + 初始紧止损 12%', ck=CK, istop=(0.12, 0.10))
_r('alpha_r5_ck_cd5', 'P1 + 崩盘冷却 + 出场后 5 天冷却', ck=CK, cool=5)
_r('alpha_r5_tg_ck', 'P1_tg + 崩盘冷却', ck=CK, tight=TG)
_r('alpha_r5_tg_ck_is12', 'P1_tg + 崩盘冷却 + 初始紧止损 12%', ck=CK, tight=TG, istop=(0.12, 0.10))
_r('alpha_r5_tg_ck_ts15', 'P1_tg + 崩盘冷却 + 时间止损 15 天/8%', ck=CK, tight=TG, tstop=(15, 0.08))
_r('alpha_r5_ck_th06', 'P1 + 崩盘冷却(阈 −6%)', ck=(-0.06, 10))
_r('alpha_r5_ck_th085', 'P1 + 崩盘冷却(阈 −8.5%)', ck=(-0.085, 10))
_r('alpha_r5_ck_k7', 'P1 + 崩盘冷却(7 天)', ck=(-0.07, 7))
_r('alpha_r5_ck_k14', 'P1 + 崩盘冷却(14 天)', ck=(-0.07, 14))
_r('alpha_r5_ck_n25', 'P1(n=25)+ 崩盘冷却', ck=CK, n=25)
_r('alpha_r5_tg_ck_n25', 'P1_tg(n=25)+ 崩盘冷却', ck=CK, tight=TG, n=25)


def _sys(d, l, clear_w=20, maxd=None, frac=0.5, mn=3):
    st = _p1(d)[0]
    prev = np.nan_to_num(_lag(st))
    exits = prev * (1 - st)
    ex5 = np.nan_to_num(_roll1(exits.sum(1), 5, 'sum'))
    held = np.nan_to_num(_roll1(prev.sum(1), 5, 'max'))
    mass = ex5 >= np.maximum(mn, frac * held)
    L, _ = _ew(d, l)
    clear = L > _lag(_roll1(L, clear_w, 'max'))
    if maxd is not None:          # 限时:距最近一次成批出场满 maxd 天自动解除
        clear |= np.nan_to_num(_roll1(mass.astype(float), maxd, 'max')) == 0
    off = _hyst(mass, clear)
    return _p1(d, allow=~off[:, None])[0]


_reg('alpha_r5_sys_c10', lambda d, l: _sys(d, l, clear_w=10), 'P1 + 系统性止损潮,EW 创 10 日新高解除')
_reg('alpha_r5_sys_max20', lambda d, l: _sys(d, l, maxd=20), 'P1 + 系统性止损潮,EW 20 日新高或满 20 天解除')
_reg('alpha_r5_sys_strict', lambda d, l: _sys(d, l, frac=0.6, mn=4), 'P1 + 系统性止损潮(≥60% 且 ≥4 笔)')


def _vs(d, l):
    _, ew = _ew(d, l)
    shock = _roll1(ew, 10, 'std') > 2.0 * _roll1(ew, 120, 'std')
    return _p1(d, allow=~shock[:, None])[0]


_reg('alpha_r5_vs_ew', _vs, 'P1 + EW 10 日波动 > 2×120 日波动时不开新仓')


def _vsc(d, l):
    r = _ret(d)
    shock = _roll(r, 10, 'std') > 2.0 * _roll(r, 120, 'std')
    return _p1(d, allow=~shock)[0]


_reg('alpha_r5_vs_coin', _vsc, 'P1 + 该币 10 日波动 > 2×120 日波动时不开新仓')
_r('alpha_r5_ts12', 'P1 + 时间止损 12 天/6%', tstop=(12, 0.06))
_r('alpha_r5_ts19', 'P1 + 时间止损 19 天/10%', tstop=(19, 0.10))
_r('alpha_r5_is10', 'P1 + 初始紧止损 10%(浮盈 <10%)', istop=(0.10, 0.10))
_r('alpha_r5_is12x15', 'P1 + 初始紧止损 12%(浮盈 <15%)', istop=(0.12, 0.15))
