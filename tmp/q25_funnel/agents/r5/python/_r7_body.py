from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r7:终选候选中心 + 邻域电池(每条 8 个邻居:突破窗 n ±25%、追踪止损 s ±20%、
# 出场 SMA/唐奇安窗 ±25%、该条主附加参数 ±20%)+ 3 条探索(崩盘冷却并集、两段出场平均)。
# 骨架 P1 = 20 日收盘突破入场;自入场峰值回撤 25% 追踪止损;跌破 SMA45 出场。附加部件(全是路径依赖 / 状态机):
#   ck   市场崩盘冷却:EW 单日跌 >7% 后 10 天不开新仓(不强平)
#   ck3  EW 3 日累计跌 >12% 后 10 天不开新仓
#   ts   时间止损:持有 ≥15 天峰值浮盈仍 <8% ⇒ 出场
#   is   初始紧止损:峰值浮盈 <10% 时止损距离 10%
#   be   保本:峰值浮盈 ≥15% 后止损位不低于入场价
#   tg   止损随浮盈收紧 s = max(0.10, 0.25 − 0.25·gain)
#   cd   出场后 5 天冷却
#   dl   出场载体改唐奇安 N 日收盘低点(替代 SMA45)
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
TS = (15, 0.08)
FIN = {
    'F01_ck_be_ts': (dict(ck=CK, be=0.15, ts=TS), [('be12', dict(be=0.12)), ('be18', dict(be=0.18))]),
    'F02_ck_ts': (dict(ck=CK, ts=TS), [('ts11', dict(ts=(11, 0.08))), ('ts19', dict(ts=(19, 0.08)))]),
    'F03_ck_ts_is': (dict(ck=CK, ts=TS, is_=(0.10, 0.10)), [('is08', dict(is_=(0.08, 0.10))),
                                                             ('is12', dict(is_=(0.12, 0.10)))]),
    'F04_ck_be': (dict(ck=CK, be=0.15), [('be12', dict(be=0.12)), ('be18', dict(be=0.18))]),
    'F05_ck_is': (dict(ck=CK, is_=(0.10, 0.10)), [('is08', dict(is_=(0.08, 0.10))), ('is12', dict(is_=(0.12, 0.10)))]),
    'F06_ck_dl10': (dict(ck=CK, dl=10), [('ck056', dict(ck=(-0.056, 10))), ('ck084', dict(ck=(-0.084, 10)))]),
    'F07_ck': (dict(ck=CK), [('ck056', dict(ck=(-0.056, 10))), ('ck084', dict(ck=(-0.084, 10)))]),
    'F08_ck3': (dict(ck3=(-0.12, 10)), [('c096', dict(ck3=(-0.096, 10))), ('c144', dict(ck3=(-0.144, 10)))]),
    'F09_ck_cd': (dict(ck=CK, cd=5), [('cd4', dict(cd=4)), ('cd6', dict(cd=6))]),
    'F11_tg_ck_is': (dict(ck=CK, tg=(0.25, 0.10), is_=(0.12, 0.10)), [('tg20', dict(tg=(0.20, 0.10))),
                                                                      ('tg30', dict(tg=(0.30, 0.10)))]),
    'F12_dl15_ck_ts': (dict(ck=CK, dl=15, ts=TS), [('ts11', dict(ts=(11, 0.08))), ('ts19', dict(ts=(19, 0.08)))]),
}


def _mk(cfg):
    return lambda d, l: _bk(d, l, **cfg)


for name, (cfg, extra) in FIN.items():
    _reg(f'alpha_r7_{name}', _mk(cfg), f'{name} 中心 {cfg}')
    nb = [('n15', dict(n=15)), ('n25', dict(n=25)), ('s20', dict(s=0.20)), ('s30', dict(s=0.30))]
    if cfg.get('dl') is None:
        nb += [('m34', dict(m=34)), ('m56', dict(m=56))]
    else:
        w = cfg['dl']
        nb += [(f'dl{round(w * 0.75)}', dict(dl=round(w * 0.75))), (f'dl{round(w * 1.25)}', dict(dl=round(w * 1.25)))]
    for tag, kw in nb + extra:
        _reg(f'alpha_r7_{name}__{tag}', _mk({**cfg, **kw}), f'{name} 邻居 {kw}')

_reg('alpha_r7_X1_ckU3_be_ts', _mk(dict(ck=CK, ck3=(-0.12, 10), be=0.15, ts=TS)), '探索:ck ∪ ck3 + 保本 + 时间止损')
_reg('alpha_r7_X2_2stage', lambda d, l: 0.5 * (_bk(d, l, ck=CK, ts=TS) + _bk(d, l, ck=CK, ts=TS, dl=15)),
     '探索:两段出场 —— 半仓 SMA45 出场、半仓 15 日低点出场(ck + ts)')
_reg('alpha_r7_X3_2stage_be', lambda d, l: 0.5 * (_bk(d, l, ck=CK, ts=TS, be=0.15)
                                                  + _bk(d, l, ck=CK, ts=TS, be=0.15, dl=15)),
     '探索:两段出场 + 保本(ck + ts + be)')
