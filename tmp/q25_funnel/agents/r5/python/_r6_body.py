from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r6:最后一轮结构探索(之后进终选 + 邻域电池)
# r5:崩盘冷却 ck(EW 单日跌 >7% 后 10 天不开新仓)是平台(阈 −6/−8.5%、天 7/14 的 2025 全在 +0.54~+0.67);
#     ck_ts15 = +0.48 / +0.45 为当前 min(2025,2026) 最高且 G2+GR 过。
# 假设:
#   ① 崩盘冷却的推广:崩盘日强平(ckx)、3 日累计崩盘(ck3)、逐币崩盘冷却(ckc)是否更好;
#   ② 假突破规则的叠加(ts + is)、保本止损(be)是否与 ck 正交;
#   ③ 出场载体换成唐奇安 N 日低点(海龟式),验证 SMA45 出场不是唯一可行载体。
# ==== BODY ====
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
