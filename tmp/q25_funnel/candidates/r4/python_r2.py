from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 r2:核心族 = 「截面宽度门 × 通道位置选币 × 逐币波动目标化」
#   门(市场级,FastExpr 表达不了):当日 liquid 币中 close>SMA_w 的比例 > thr 才在场
#   选币(截面 rank,FastExpr 表达不了):当日 liquid 内 chanpos(close, wc) 排名前 k 且 >0.5
#   定量(逐币风险平价):敞口 × min(1, target / 年化已实现波动_w)
# 本文件 = 该族的核心 + 单因素消融(去门 / 去选币 / 去波动目标)+ 一档邻域,用于确认
# 三个部件各自的边际贡献。全部因果:pandas rolling 右对齐 min_periods=window;
# 截面 rank 只用当日;滞回用"最近触发者胜"的 cummax 实现(不看未来)。
import cupy as np


# ---------------------------------------------------------------- 因果原语
def _rmean(x, w):
    return _cuda_rolling(x, w, 'mean')


def _rstd(x, w):
    return _cuda_rolling(x, w, 'std')


def _rmax(x, w):
    return _cuda_rolling(x, w, 'max')


def _rmin(x, w):
    return _cuda_rolling(x, w, 'min')


def _rets(close):
    with _nullcontext():
        r = close[1:] / close[:-1] - 1.0
    return np.vstack([np.full((1, close.shape[1]), np.nan), np.where(np.isfinite(r), r, np.nan)])


def _chanpos(close, w):
    hi, lo = (_rmax(close, w), _rmin(close, w))
    with _nullcontext():
        return np.where(hi > lo, (close - lo) / (hi - lo), np.nan)


def _madist(close, w):
    sma = _rmean(close, w)
    with _nullcontext():
        return np.where(sma > 0, close / sma - 1.0, np.nan)


def _breadth(close, liquid, w):
    """(T,) 当日 liquid 且数据有效的币中 close>SMA_w 的比例;无有效币 → NaN。"""
    md = _madist(close, w)
    valid = np.isfinite(md) & liquid
    den = valid.sum(1)
    return np.where(den > 0, ((md > 0) & valid).sum(1) / np.maximum(den, 1), np.nan)


def _top_k(score, liquid, k):
    valid = liquid & np.isfinite(score)
    s = np.where(valid, score, -np.inf)
    order = np.argsort(-s, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(s.shape[1])[None, :], axis=1)
    return ((rank < k) & valid).astype(np.float64)


from alpha_cuda import thresholds as _hyst


def _invvol(close, w, target):
    """逐币波动目标化系数 min(1, target / 年化 close-to-close 波动_w)。"""
    v = _rstd(_rets(close), w) * np.sqrt(365.0)
    with _nullcontext():
        return np.where(v > 0, np.minimum(1.0, target / v), 0.0)


def _core(data, liquid, gw=40, gthr=0.35, ghyst=None, wc=150, k=3, hi=0.5,
          ivw=60, ivt=0.5, use_gate=True, use_sel=True, use_iv=True):
    c = data['close']
    e = np.where(liquid, 1.0, 0.0)
    if use_sel:
        cp = _chanpos(c, wc)
        e = _top_k(cp, liquid, k) * (cp > hi)
    if use_iv:
        e = e * _invvol(c, ivw, ivt)
    if use_gate:
        br = _breadth(c, liquid, gw)
        g = _hyst(br, *ghyst) if ghyst else np.where(np.isfinite(br), (br > gthr).astype(float), 0.0)
        e = g[:, None] * e
    return e


def _reg(name, doc, **kw):
    def f(data, liquid):
        return _core(data, liquid, **kw)
    f.__name__ = name
    f.__doc__ = doc
    globals()[name] = f


_M = ('宽度门(liquid 内 close>SMA{gw} 比例>{gthr})× 通道位置 chanpos{wc} 前{k}且>{hi} '
      '× 逐币波动目标 min(1,{ivt}/rv{ivw})')

# ---- 核心 + 门形态
_reg('alpha_bcv_core', _M.format(gw=40, gthr=0.35, wc=150, k=3, hi=0.5, ivt=0.5, ivw=60))
_reg('alpha_bcv_hyst', '同核心,但宽度门改双阈值滞回(>0.50 进 <0.35 出)—— 压换手 + 抗抖动',
     ghyst=(0.50, 0.35))
_reg('alpha_bcv_hyst2', '同核心,宽度门滞回 >0.55 进 <0.40 出', ghyst=(0.55, 0.40))
_reg('alpha_bcv_g60', '同核心,宽度窗改 60 日、阈值 0.45(中期参与度)', gw=60, gthr=0.45)
_reg('alpha_bcv_g90', '同核心,宽度窗改 90 日、阈值 0.45(长期参与度)', gw=90, gthr=0.45)

# ---- 消融:三个部件各去一个
_reg('alpha_abl_nogate', '消融:去掉宽度门,只留通道位置选币 × 波动目标化', use_gate=False)
_reg('alpha_abl_nosel', '消融:去掉选币,宽度门 × 全 liquid × 波动目标化(纯择时+定量)',
     use_sel=False)
_reg('alpha_abl_noiv', '消融:去掉波动目标化,只留宽度门 × 通道位置选币', use_iv=False)
_reg('alpha_abl_gateonly', '消融:只留宽度门(纯择时,满仓载体)', use_sel=False, use_iv=False)

# ---- 邻域:通道窗 / k / 波动窗 / 目标 / hi 阈值
for _wc in (100, 120, 180, 250):
    _reg(f'alpha_nb_wc{_wc}', _M.format(gw=40, gthr=0.35, wc=_wc, k=3, hi=0.5, ivt=0.5, ivw=60),
         wc=_wc)
for _k in (2, 4, 5):
    _reg(f'alpha_nb_k{_k}', _M.format(gw=40, gthr=0.35, wc=150, k=_k, hi=0.5, ivt=0.5, ivw=60),
         k=_k)
for _ivw in (30, 90, 120):
    _reg(f'alpha_nb_ivw{_ivw}', _M.format(gw=40, gthr=0.35, wc=150, k=3, hi=0.5, ivt=0.5, ivw=_ivw),
         ivw=_ivw)
for _ivt in (0.35, 0.65, 0.9):
    _reg(f'alpha_nb_ivt{int(_ivt*100)}',
         _M.format(gw=40, gthr=0.35, wc=150, k=3, hi=0.5, ivt=_ivt, ivw=60), ivt=_ivt)
for _hi in (0.0, 0.35, 0.65):
    _reg(f'alpha_nb_hi{int(_hi*100)}',
         _M.format(gw=40, gthr=0.35, wc=150, k=3, hi=_hi, ivt=0.5, ivw=60), hi=_hi)
for _gt in (0.25, 0.45, 0.55):
    _reg(f'alpha_nb_gthr{int(_gt*100)}',
         _M.format(gw=40, gthr=_gt, wc=150, k=3, hi=0.5, ivt=0.5, ivw=60), gthr=_gt)
for _gw in (20, 30, 120):
    _reg(f'alpha_nb_gw{_gw}', _M.format(gw=_gw, gthr=0.35, wc=150, k=3, hi=0.5, ivt=0.5, ivw=60),
         gw=_gw)
