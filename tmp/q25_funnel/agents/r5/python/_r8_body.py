from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 R5 · r8(最后一轮):补 X1 / X3 的邻域电池 + 两条新组合(各带 8 邻居)
# r7:X1 = ck ∪ ck3 + 保本 15% + 时间止损 15/8%(+0.67 / +0.51);X3 = 两段出场(半仓 SMA45、半仓 15 日低点)
#     + ck + 保本 + 时间止损(IS 1.79 / +0.42 / +0.63 / 牛年最小跟随 0.61)。
# 假设:① X1 / X3 邻域整片过 G2+GR 且两年为正;② X4 = 两段出场 × 双崩盘冷却,兼得 X1 的 2025 与 X3 的 2026;
#   ③ X5 = X1 + 初始紧止损 10%(F03 显示 is 抬 2026)。
# 邻居:突破窗 n 15/25、追踪止损 s 0.20/0.30、SMA 窗 34/56、唐奇安窗 11/19、ck3 阈 −9.6%/−14.4%、初始止损 8%/12%。
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


def _two(cfg):
    """两段出场:半仓按 SMA_m 出场,半仓按唐奇安 dl 日低点出场(其余规则相同)。"""
    c = dict(cfg)
    dl = c.pop('dl2', 15)
    return lambda d, l: 0.5 * (_bk(d, l, **c) + _bk(d, l, **{**c, 'dl': dl}))


def _one(cfg):
    return lambda d, l: _bk(d, l, **cfg)


FIN = {
    'X1_ckU3_be_ts': (_one, dict(ck=CK, ck3=C3, be=0.15, ts=TS),
                      [('c096', dict(ck3=(-0.096, 10))), ('c144', dict(ck3=(-0.144, 10)))]),
    'X3_2stage_be': (_two, dict(ck=CK, ts=TS, be=0.15, dl2=15), [('dl11', dict(dl2=11)), ('dl19', dict(dl2=19))]),
    'X4_2stage_ckU3_be': (_two, dict(ck=CK, ck3=C3, ts=TS, be=0.15, dl2=15),
                          [('dl11', dict(dl2=11)), ('dl19', dict(dl2=19))]),
    'X5_ckU3_be_ts_is': (_one, dict(ck=CK, ck3=C3, be=0.15, ts=TS, is_=(0.10, 0.10)),
                         [('is08', dict(is_=(0.08, 0.10))), ('is12', dict(is_=(0.12, 0.10)))]),
}
NB = [('n15', dict(n=15)), ('n25', dict(n=25)), ('s20', dict(s=0.20)), ('s30', dict(s=0.30)),
      ('m34', dict(m=34)), ('m56', dict(m=56))]
for name, (mk, cfg, extra) in FIN.items():
    _reg(f'alpha_r8_{name}', mk(cfg), f'{name} 中心 {cfg}')
    for tag, kw in NB + extra:
        _reg(f'alpha_r8_{name}__{tag}', mk({**cfg, **kw}), f'{name} 邻居 {kw}')
