from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 r3:核心族收口 —— 「宽度门 × 通道位置选币 × 逐币逆波动定量」
#   1) 门(市场级,截面宽度):当日 liquid 币中 close>SMA_gw 的比例 > gthr(或双阈值滞回)才在场
#   2) 选币(当日截面 rank):chanpos(close, wc) 排名前 k 且绝对值 > hi
#   3) 定量(逐币风险平价):敞口 = min(1, t / 年化波动_w),t<=0.3 时 cap 基本不生效 ⇒ 纯逆波动加权
# r2 已证三个部件各自有边际贡献(消融:gate-only 25 +0.41/26 -0.31;no-sel 0.74/-0.29;
# no-iv 1.05/1.46;full 1.39/1.47)。r3 收口两处:hi 从 0.5 降到 0.35、宽度窗 40→50/60,
# 并把波动估计量换成 5 个独立族(close-to-close / Garman-Klass / Parkinson / ATR / 下行半方差)
# 做"同一结构、不同估计量"的稳健性电池。全部因果(rolling 右对齐 min_periods=w;截面 rank 只用当日)。
import cupy as np


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


def _vol(data, kind, w):
    """年化逐币波动,5 个独立估计量族(全部右对齐滚动,因果)。"""
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
    elif kind == 'dn':
        r = _rets(c)
        d = np.where(np.isfinite(r), np.where(r < 0, r, 0.0), np.nan)
        v = np.sqrt(np.maximum(_rmean(d ** 2, w), 0.0))
    else:
        raise ValueError(kind)
    return v * np.sqrt(365.0)


def _core(data, liquid, gw=60, gthr=0.45, ghyst=None, wc=150, k=3, hi=0.35, vk='cc', vw=60, t=0.25, gate_kind='breadth'):
    c = data['close']
    cp = _chanpos(c, wc)
    e = _top_k(cp, liquid, k) * (cp > hi)
    v = _vol(data, vk, vw)
    with _nullcontext():
        e = e * np.where(v > 0, np.minimum(1.0, t / v), 0.0)
    if gate_kind == 'breadth':
        br = _breadth(c, liquid, gw)
        g = _hyst(br, *ghyst) if ghyst else np.where(np.isfinite(br), (br > gthr).astype(float), 0.0)
    elif gate_kind == 'ewtrend':
        r = _rets(c)
        lp = np.vstack([np.zeros((1, liquid.shape[1]), bool), liquid[:-1]])
        ok = lp & np.isfinite(r)
        ew = np.where(ok, r, 0.0).sum(1) / np.maximum(ok.sum(1), 1)
        idx = np.cumsum(np.log1p(ew))
        sma = _cuda_rolling(idx, gw, 'mean')
        g = np.where(np.isfinite(sma), (idx > sma).astype(float), 0.0)
    else:
        raise ValueError(gate_kind)
    return g[:, None] * e


def _reg(name, doc, **kw):
    def f(data, liquid):
        return _core(data, liquid, **kw)
    f.__name__ = name
    f.__doc__ = doc
    globals()[name] = f


def _doc(**kw):
    a = dict(gw=60, gthr=0.45, wc=150, k=3, hi=0.35, vk='cc', vw=60, t=0.25)
    a.update({k: v for k, v in kw.items() if v is not None})
    g = (f'宽度滞回 br{a["gw"]} {kw["ghyst"][0]}进/{kw["ghyst"][1]}出' if kw.get('ghyst')
         else f'宽度门 br{a["gw"]}>{a["gthr"]}')
    return (f'{g} × chanpos{a["wc"]} 前{a["k"]}且>{a["hi"]} × 逆波动 min(1,{a["t"]}/'
            f'{a["vk"]}{a["vw"]})')


# ---- 波动估计量电池(同结构 × 5 族 × 3 窗)
for _vk in ('cc', 'gk', 'pk', 'atr', 'dn'):
    for _vw in (45, 60, 90):
        _reg(f'alpha_v_{_vk}{_vw}', _doc(vk=_vk, vw=_vw), vk=_vk, vw=_vw)

# ---- 门电池
for _gw in (40, 50, 70, 90):
    _reg(f'alpha_g_w{_gw}', _doc(gw=_gw), gw=_gw)
for _gt in (0.35, 0.40, 0.50):
    _reg(f'alpha_g_t{int(_gt*100)}', _doc(gthr=_gt), gthr=_gt)
for _e, _x in ((0.60, 0.45), (0.55, 0.40), (0.50, 0.35)):
    _reg(f'alpha_g_h{int(_e*100)}x{int(_x*100)}', _doc(ghyst=(_e, _x)), ghyst=(_e, _x))
_reg('alpha_g_ew120', '门换成 EW 指数 > 自身 120 日均线(替换宽度门,检验门可替换性)'
                      ' × chanpos150 前3且>0.35 × 逆波动 min(1,0.25/cc60)',
     gate_kind='ewtrend', gw=120)
_reg('alpha_g_ew180', '门换成 EW 指数 > 自身 180 日均线 × chanpos150 前3且>0.35 × 逆波动',
     gate_kind='ewtrend', gw=180)

# ---- 选币电池
for _wc in (120, 135, 165, 180, 250):
    _reg(f'alpha_s_wc{_wc}', _doc(wc=_wc), wc=_wc)
for _k in (2, 4, 5):
    _reg(f'alpha_s_k{_k}', _doc(k=_k), k=_k)
for _hi in (0.20, 0.30, 0.45, 0.60):
    _reg(f'alpha_s_hi{int(_hi*100)}', _doc(hi=_hi), hi=_hi)

# ---- 风险目标电池(t 平台性)
for _t in (0.15, 0.20, 0.30, 0.40):
    _reg(f'alpha_t_{int(_t*100)}', _doc(t=_t), t=_t)

# ---- 核心
_reg('alpha_core', _doc())
