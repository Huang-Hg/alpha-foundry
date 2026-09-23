from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · 终选 13 条(同一机制族的 13 个实现点,全部 G1+G2+GR+G3 过;按 min(2025,2026) 降序)
#
# 骨架 P1(逐币持仓状态机,FastExpr 写不出:入场价 / 自入场峰值 / 持有天数都是路径依赖状态):
#   入场 = 收盘价突破前 20 日最高收盘;出场 = 自入场峰值回撤 25%(追踪止损)或 收盘跌破 SMA45。
#   经济含义:默认空仓、趋势确立才进,但一旦进场就「坏了才走」—— 牛市里持有到趋势破位(牛年跟随比 0.53-0.65),
#   熊市里突破稀少且追踪止损截断亏损(2018 −7~−10% / 2022 −31~−34%,基准 −86% / −75%)。
# 附加部件(全部只改「何时进 / 何时走」,不做连续仓位):
#   ck   市场崩盘冷却:EW 等权指数单日跌 >7% 后 10 天内不开新仓(不强平)。强平后反弹 / 清算瀑布后的
#        死猫跳突破不可靠;r5 邻居(阈 −6/−8.5%、天 7/14)2025 全在 +0.54~+0.67 = 平台。
#   ck3  EW 3 日累计跌 >12% 后 10 天不开新仓(多日式抛售,如 2025-02、2026-06)。
#   ts   时间止损:持有 ≥15 天峰值浮盈仍 <8% ⇒ 出场(突破没走出来 = 假突破)。
#   is   初始紧止损:峰值浮盈 <10% 时止损距离 10%(浮盈确认前不给 25% 的空间)。
#   be   保本:峰值浮盈 ≥15% 后止损位不低于入场价。
#   tg   止损随浮盈收紧 s = max(0.10, 0.25 − 0.25·gain)。
#   2stage 两段出场:半仓按 SMA45 出、半仓按 15 日收盘低点出。
# 邻域电池(每条 8 邻居:n 15/25、s 0.20/0.30、SMA 34/56 或唐奇安 11/19、主附加参数 ±20%),读数在
# candidates/r5/python_r3.json(P1)/ python_r7.json(F*)/ python_r8.json(X*)。
# ==== BODY ====
def _ckb(d, l, th=-0.07, k=10):
    _, ew = _ew(d, l)
    return ~(np.nan_to_num(_roll1((ew < th).astype(float), k, 'max')) > 0)[:, None]


def _ck3b(d, l, th=-0.12, k=10):
    L, _ = _ew(d, l)
    crash = (L - _lag(L, 3)) < np.log1p(th)
    return ~(np.nan_to_num(_roll1(crash.astype(float), k, 'max')) > 0)[:, None]


def _dlx(d, w):
    return _cache(('dl', w, _T(d)), lambda: _px(d) < _lag(_roll(_px(d), w, 'min')))


def _bk(d, l, n=20, s=0.25, m=45, dl=None, ck=None, ck3=None, ts=None, is_=None, be=None, tg=None, cd=0):
    allow = None
    if ck is not None:
        allow = _ckb(d, l, *ck)
    if ck3 is not None:
        a3 = _ck3b(d, l, *ck3)
        allow = a3 if allow is None else allow & a3
    ex = _below(d, m) if dl is None else _dlx(d, dl)
    return _smx(_px(d), _brk(d, n), ex, s, allow=allow, tstop=ts, istop=is_, be=be, tight=tg, cool=cd)[0]


CK = (-0.07, 10)
C3 = (-0.12, 10)
TS = (15, 0.08)


def _one(**cfg):
    return lambda d, l: _bk(d, l, **cfg)


def _two(dl2=15, **cfg):
    return lambda d, l: 0.5 * (_bk(d, l, **cfg) + _bk(d, l, **{**cfg, 'dl': dl2}))


_reg('alpha_x5_ck2_be_ts_is', _one(ck=CK, ck3=C3, be=0.15, ts=TS, is_=(0.10, 0.10)),
     'P1 + 崩盘冷却(单日 ∪ 3 日)+ 保本 15% + 时间止损 15 天/8% + 初始紧止损 10%')
_reg('alpha_x4_2stage_ck2_be_ts', _two(ck=CK, ck3=C3, be=0.15, ts=TS),
     '两段出场(半 SMA45 / 半 15 日低点)+ 崩盘冷却(单日 ∪ 3 日)+ 保本 + 时间止损')
_reg('alpha_x1_ck2_be_ts', _one(ck=CK, ck3=C3, be=0.15, ts=TS), 'P1 + 崩盘冷却(单日 ∪ 3 日)+ 保本 + 时间止损')
_reg('alpha_f01_ck_be_ts', _one(ck=CK, be=0.15, ts=TS), 'P1 + 崩盘冷却 + 保本 + 时间止损')
_reg('alpha_f02_ck_ts', _one(ck=CK, ts=TS), 'P1 + 崩盘冷却 + 时间止损')
_reg('alpha_f03_ck_ts_is', _one(ck=CK, ts=TS, is_=(0.10, 0.10)), 'P1 + 崩盘冷却 + 时间止损 + 初始紧止损')
_reg('alpha_x3_2stage_ck_be_ts', _two(ck=CK, be=0.15, ts=TS), '两段出场 + 崩盘冷却 + 保本 + 时间止损')
_reg('alpha_f05_ck_is', _one(ck=CK, is_=(0.10, 0.10)), 'P1 + 崩盘冷却 + 初始紧止损')
_reg('alpha_f04_ck_be', _one(ck=CK, be=0.15), 'P1 + 崩盘冷却 + 保本')
_reg('alpha_f11_tg_ck_is', _one(ck=CK, tg=(0.25, 0.10), is_=(0.12, 0.10)), 'P1 + 浮盈收紧止损 + 崩盘冷却 + 初始紧止损')
_reg('alpha_f07_ck', _one(ck=CK), 'P1 + 崩盘冷却')
_reg('alpha_f08_ck3', _one(ck3=C3), 'P1 + 3 日崩盘冷却')
_reg('alpha_p1', _one(), 'P1 骨架:20 日突破入场 + 25% 追踪止损 + 跌破 SMA45 出场')
