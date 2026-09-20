from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from alpha_cuda import thresholds as _cuda_thresholds
from contextlib import nullcontext as _nullcontext
# Python 车道 r4:终选候选 —— 核心族 +「电池平均书」+ 门形态替换 + 资本效率档
#   结构:宽度门(市场级截面宽度)× 通道位置截面 top-k 选币 × 逐币逆波动定量
#   r3 官方读数已确认 21/44 全过 G1-G5;r4 补三件事:
#   ① 电池平均书(把 12 / 162 个邻域变体的敞口取等权平均 → 抗单点参数拟合的交付形态)
#   ② 门形态:二值阈值 / 双阈值滞回 / 线性斜坡 三种等价性
#   ③ 资本效率:t=0.25 时毛敞口仅 5%(年化 +9%),t=0.40 抬到 8%(年化 +14%)是否仍过门
# 全部因果:pandas rolling 右对齐 min_periods=w;截面 rank 只用当日;滞回用 cummax("最近触发者胜")。
import cupy as np

_C = {}


def _rmean(x, w):
    return _cuda_rolling(x, w, 'mean')


def _rstd(x, w):
    return _cuda_rolling(x, w, 'std')


def _rets(close):
    with _nullcontext():
        r = close[1:] / close[:-1] - 1.0
    return np.vstack([np.full((1, close.shape[1]), np.nan), np.where(np.isfinite(r), r, np.nan)])


def _chanpos(data, w):
    key = ('cp', w, data['close'].shape[0])
    if key not in _C:
        c = data['close']
        hi = _cuda_rolling(c, w, 'max')
        lo = _cuda_rolling(c, w, 'min')
        with _nullcontext():
            _C[key] = np.where(hi > lo, (c - lo) / (hi - lo), np.nan)
    return _C[key]


def _breadth(data, liquid, w):
    key = ('br', w, liquid.shape[0])
    if key not in _C:
        c = data['close']
        sma = _rmean(c, w)
        with _nullcontext():
            md = np.where(sma > 0, c / sma - 1.0, np.nan)
        valid = np.isfinite(md) & liquid
        den = valid.sum(1)
        _C[key] = np.where(den > 0, ((md > 0) & valid).sum(1) / np.maximum(den, 1), np.nan)
    return _C[key]


def _topk(data, liquid, w, k):
    key = ('tk', w, k, liquid.shape[0])
    if key not in _C:
        s = _chanpos(data, w)
        valid = liquid & np.isfinite(s)
        x = np.where(valid, s, -np.inf)
        order = np.argsort(-x, axis=1, kind='stable')
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(x.shape[1])[None, :], axis=1)
        _C[key] = ((rank < k) & valid).astype(np.float64)
    return _C[key]


def _vol(data, kind, w):
    """年化逐币波动;四个独立估计量族。"""
    key = ('v', kind, w, data['close'].shape[0])
    if key not in _C:
        c = data['close']
        if kind == 'cc':
            v = _rstd(_rets(c), w)
        elif kind == 'gk':
            v = np.sqrt(np.maximum(_rmean(data['gkvol'] ** 2, w), 0.0))
        elif kind == 'pk':
            with _nullcontext():
                hl = np.log(np.where(data['high'] > 0, data['high'], np.nan) / np.where(data['low'] > 0, data['low'], np.nan))
            v = np.sqrt(np.maximum(_rmean(hl ** 2, w) / (4.0 * np.log(2.0)), 0.0))
        elif kind == 'atr':
            with _nullcontext():
                v = _rmean((data['high'] - data['low']) / np.where(c > 0, c, np.nan), w)
        else:
            raise ValueError(kind)
        _C[key] = v * np.sqrt(365.0)
    return _C[key]


def _iv(data, kind, w, t):
    v = _vol(data, kind, w)
    with _nullcontext():
        return np.where(v > 0, np.minimum(1.0, t / v), 0.0)


def _gate(data, liquid, gw, gthr=None, ghyst=None, ramp=None):
    br = _breadth(data, liquid, gw)
    if ghyst:
        return _cuda_thresholds(br, *ghyst)
    if ramp:
        return np.clip((np.nan_to_num(br) - ramp[0]) / (ramp[1] - ramp[0]), 0.0, 1.0)
    return np.where(np.isfinite(br), (br > gthr).astype(float), 0.0)


def _one(data, liquid, gw=60, gthr=0.45, ghyst=None, ramp=None, wc=150, k=3, hi=0.35,
         vk='cc', vw=60, t=0.25):
    e = _topk(data, liquid, wc, k) * (_chanpos(data, wc) > hi) * _iv(data, vk, vw, t)
    return _gate(data, liquid, gw, gthr, ghyst, ramp)[:, None] * e


def _mix(data, liquid, specs):
    return np.mean(np.stack([_one(data, liquid, **s) for s in specs]), axis=0)


def _reg(name, doc, **kw):
    def f(data, liquid):
        return _one(data, liquid, **kw)
    f.__name__ = name
    f.__doc__ = doc
    globals()[name] = f


_D = ('宽度门 br{gw}>{gthr} × chanpos{wc} 截面前{k}且>{hi} × 逆波动 min(1,{t}/{vk}{vw})')

_reg('alpha_f_core', _D.format(gw=60, gthr=0.45, wc=150, k=3, hi=0.35, t=0.25, vk='cc', vw=60))
_reg('alpha_f_gk45', _D.format(gw=60, gthr=0.45, wc=150, k=3, hi=0.35, t=0.25, vk='gk', vw=45),
     vk='gk', vw=45)
_reg('alpha_f_pk45', _D.format(gw=60, gthr=0.45, wc=150, k=3, hi=0.35, t=0.25, vk='pk', vw=45),
     vk='pk', vw=45)
_reg('alpha_f_atr45', _D.format(gw=60, gthr=0.45, wc=150, k=3, hi=0.35, t=0.25, vk='atr', vw=45),
     vk='atr', vw=45)
_reg('alpha_f_gw50', _D.format(gw=50, gthr=0.40, wc=150, k=3, hi=0.35, t=0.25, vk='cc', vw=60),
     gw=50, gthr=0.40)
_reg('alpha_f_gw55', _D.format(gw=55, gthr=0.45, wc=150, k=3, hi=0.35, t=0.25, vk='cc', vw=60),
     gw=55)
_reg('alpha_f_gw45', _D.format(gw=45, gthr=0.45, wc=150, k=3, hi=0.35, t=0.25, vk='cc', vw=60),
     gw=45)
_reg('alpha_f_hyst', '门改双阈值滞回(宽度 br60 >0.60 进 <0.45 出)× chanpos150 前3且>0.35 '
                     '× 逆波动 min(1,0.25/cc60)—— 换手 0.016 更低', ghyst=(0.60, 0.45))
_reg('alpha_f_ramp', '门改线性斜坡(宽度 br60 从 0.35 线性升到 0.55 满仓)× chanpos150 前3且>0.35 '
                     '× 逆波动 —— 检验门的二值/连续形态等价性', gthr=None, ramp=(0.35, 0.55))
_reg('alpha_f_hi45', _D.format(gw=60, gthr=0.45, wc=150, k=3, hi=0.45, t=0.25, vk='cc', vw=60),
     hi=0.45)
_reg('alpha_f_hi50', _D.format(gw=60, gthr=0.45, wc=150, k=3, hi=0.50, t=0.25, vk='cc', vw=60),
     hi=0.50)
_reg('alpha_f_wc135', _D.format(gw=60, gthr=0.45, wc=135, k=3, hi=0.35, t=0.25, vk='cc', vw=60),
     wc=135)
_reg('alpha_f_wc120', _D.format(gw=60, gthr=0.45, wc=120, k=3, hi=0.35, t=0.25, vk='cc', vw=60),
     wc=120)
_reg('alpha_f_t40', _D.format(gw=60, gthr=0.45, wc=150, k=3, hi=0.35, t=0.40, vk='cc', vw=60)
     + '(资本效率档:毛敞口 8%)', t=0.40)
_reg('alpha_f_t40_gk45', _D.format(gw=60, gthr=0.45, wc=150, k=3, hi=0.35, t=0.40,
                                   vk='gk', vw=45) + '(资本效率档)', t=0.40, vk='gk', vw=45)
_reg('alpha_f_gw50_gk45', _D.format(gw=50, gthr=0.40, wc=150, k=3, hi=0.35, t=0.25,
                                    vk='gk', vw=45), gw=50, gthr=0.40, vk='gk', vw=45)

_VOL12 = [dict(vk=a, vw=b) for a in ('cc', 'gk', 'pk', 'atr') for b in (45, 60, 90)]
_FULL = [dict(gw=g, gthr=th, wc=w, k=kk, hi=h, vk=v, vw=vw)
         for g, th in ((50, 0.40), (55, 0.45), (60, 0.45))
         for w in (135, 150, 165) for kk in (3, 4) for h in (0.30, 0.35, 0.45)
         for v, vw in (('cc', 60), ('gk', 45), ('atr', 45))]


def alpha_f_battvol(data, liquid):
    """电池平均书:核心结构 × 4 个波动估计量族 × 3 个窗 共 12 个变体的敞口等权平均。"""
    return _mix(data, liquid, _VOL12)


def alpha_f_battvol_t40(data, liquid):
    """同 battvol,但风险目标 t=0.40(毛敞口约 8%,资本效率档)。"""
    return _mix(data, liquid, [dict(t=0.40, **s) for s in _VOL12])


def alpha_f_battfull(data, liquid):
    """全域电池平均书:门/通道窗/持仓数/位置阈值/波动估计量 五个旋钮的 162 个邻域变体等权平均
    —— 族中心的诚实读数(不挑参数)。"""
    return _mix(data, liquid, _FULL)


def alpha_f_battfull_t40(data, liquid):
    """同 battfull,风险目标 t=0.40。"""
    return _mix(data, liquid, [dict(t=0.40, **s) for s in _FULL])
