from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r4:P1 族(20 日突破 + 25% 追踪止损 + SMA45 出场)整片过 G2+GR,min(2025,2026) 卡在 0.1-0.35。
# 月度敞口:亏损集中在「突破后立刻失败」段(2026-01 敞口 0.51 / EW −7.6%;2025-06、2025-12)。
# 假设:假突破可以用路径依赖规则识别并提前离场 / 不入场:
#   时间止损(持有 k 天峰值浮盈仍 <x ⇒ 走)、初始紧止损(浮盈确认前止损更紧)、出场后冷却期、
#   放量确认突破、突破幅度门槛、相对强度(币 20 日收益 > EW 20 日)入场、
#   系统性止损潮(影子 P1 近 5 日成批出场 ⇒ 暂停开仓直到 EW 创 20 日新高)、EW 崩盘日后 10 天不开仓;
#   以及 P1_tg(浮盈收紧止损,r3 IS 1.80)与 2c / n25 / s30 / 分批减仓 的组合。
# ==== BODY ====
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
