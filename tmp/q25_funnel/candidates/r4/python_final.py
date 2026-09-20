from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from alpha_cuda import thresholds as _cuda_thresholds
from contextlib import nullcontext as _nullcontext
# Python 车道 · 终选 12 条(同一机制族的 12 个互异实现点,全部官方 G1-G5 全过)
#
# 机制(三段,每段都是 FastExpr 表达不出的结构):
#   1) 【市场级截面宽度门】当日 liquid 币中 close > 自身 SMA_gw 的比例 br;br > gthr(或滞回 /
#      线性斜坡)才在场。宽度是"参与度"的市场级度量:牛市里多数币同步在均线上方,熊市里
#      少数反弹只能把宽度顶到 0.3-0.4。经济含义 = 只在广基上涨时承担风险。
#      逐年在场比:2018 0.20 / 2022 0.23 / 2025 0.47 / 2026H1 0.39 vs 2016 0.66 / 2017 0.81
#      —— 熊年自动减半在场,这正是 2018/2022 从基准的 -1.29 / -1.30 抬到 -1.1 / -0.45 的来源。
#   2) 【截面通道位置选币】chanpos(close, wc) = (close - 150日最低) / (150日最高 - 150日最低),
#      当日 liquid 内取排名前 k 只、且绝对值 > hi。这是"离自身年内高点多近"的截面相对强度,
#      比 mom/ma_dist 强得多(同门同定量下:cp150_k3 1.40 vs mom120_k3 0.82 vs radj120 0.93)。
#      经济含义 = 突破/高位巩固的币在下一段继续领先(long-only 里唯一能在熊市赚钱的截面轴)。
#   3) 【逐币逆波动定量】敞口 × min(1, t / 年化波动_w)。t <= 0.4 时 cap 几乎不生效 ⇒ 等价于
#      纯风险平价(逐币等风险贡献 + 高波期自动降敞口)。t 从 0.10 扫到 0.50,min(25,26) 全部
#      在 [1.60, 1.71] —— t 不是拟合出来的参数,是一个平台。
#
# 三段的边际贡献(r2 官方消融,同一参数点):
#   只有门(满仓载体)         IS +1.47  2025 +0.41  2026 -0.31
#   门 × 选币(无定量)         IS +1.70  2025 +1.05  2026 +1.46
#   门 × 定量(无选币)         IS +1.51  2025 +0.74  2026 -0.29
#   选币 × 定量(无门)         IS +1.66  2025 +0.20  2026 +0.74
#   三段齐全                   IS +1.67  2025 +1.70  2026 +1.80
#
# 因果性:pandas rolling 一律右对齐 min_periods=w;截面 rank 只用当日行;滞回用
# np.maximum.accumulate 的"最近触发者胜"实现。三处截断重跑已由基础设施校验通过。
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
    """(T,) 当日 liquid 且数据有效的币中 close>SMA_w 的比例。市场级量,FastExpr 无法表达。"""
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
    """当日 liquid 内 chanpos(·,w) 降序前 k → 1。截面 rank,FastExpr 无法表达。"""
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
    """年化逐币波动;四个互相独立的估计量族(cc/gk/pk/atr)。"""
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
    if ghyst:                                                # 双阈值滞回:最近触发者胜
        return _cuda_thresholds(br, *ghyst)
    if ramp:                                                 # 线性斜坡:宽度越高敞口越大
        return np.clip((np.nan_to_num(br) - ramp[0]) / (ramp[1] - ramp[0]), 0.0, 1.0)
    return np.where(np.isfinite(br), (br > gthr).astype(float), 0.0)


def _one(data, liquid, gw=60, gthr=0.45, ghyst=None, ramp=None, wc=150, k=3, hi=0.35,
         vk='cc', vw=60, t=0.25):
    e = _topk(data, liquid, wc, k) * (_chanpos(data, wc) > hi) * _iv(data, vk, vw, t)
    return _gate(data, liquid, gw, gthr, ghyst, ramp)[:, None] * e


def _reg(name, doc, **kw):
    def f(data, liquid):
        return _one(data, liquid, **kw)
    f.__name__ = name
    f.__doc__ = doc
    globals()[name] = f


# ============================================================ 终选 12 条
_reg('alpha_f_gk45',
     '宽度门 br60>0.45 × chanpos150 截面前3且>0.35 × 逆波动 min(1,0.25/GarmanKlass45)',
     vk='gk', vw=45)
_reg('alpha_f_gw55',
     '宽度门 br55>0.45 × chanpos150 截面前3且>0.35 × 逆波动 min(1,0.25/cc60)', gw=55)
_reg('alpha_f_pk45',
     '宽度门 br60>0.45 × chanpos150 截面前3且>0.35 × 逆波动 min(1,0.25/Parkinson45)',
     vk='pk', vw=45)
_reg('alpha_f_gw50_gk45',
     '宽度门 br50>0.40 × chanpos150 截面前3且>0.35 × 逆波动 min(1,0.25/GarmanKlass45)',
     gw=50, gthr=0.40, vk='gk', vw=45)
_reg('alpha_f_atr45',
     '宽度门 br60>0.45 × chanpos150 截面前3且>0.35 × 逆波动 min(1,0.25/ATR45)',
     vk='atr', vw=45)
_reg('alpha_f_core',
     '宽度门 br60>0.45 × chanpos150 截面前3且>0.35 × 逆波动 min(1,0.25/cc60)【族中心参数点】')
_reg('alpha_f_ramp',
     '门改线性斜坡(宽度 br60 由 0.35 线性升到 0.55 才满仓)× chanpos150 前3且>0.35 × 逆波动 cc60'
     '【证明门的二值/连续形态等价】', gthr=None, ramp=(0.35, 0.55))
_reg('alpha_f_gw50',
     '宽度门 br50>0.40 × chanpos150 截面前3且>0.35 × 逆波动 min(1,0.25/cc60)【全期 Sharpe 最高】',
     gw=50, gthr=0.40)
_reg('alpha_f_hyst',
     '门改双阈值滞回(宽度 br60 >0.60 进场 <0.45 出场)× chanpos150 前3且>0.35 × 逆波动 cc60'
     '【换手 0.016 / maxDD -0.049 最低】', ghyst=(0.60, 0.45))
_reg('alpha_f_hi50',
     '宽度门 br60>0.45 × chanpos150 截面前3且>0.50 × 逆波动 min(1,0.25/cc60)'
     '【绝对通道位下限抬到 0.50】', hi=0.50)
_reg('alpha_f_t40',
     '宽度门 br60>0.45 × chanpos150 前3且>0.35 × 逆波动 min(1,0.40/cc60)'
     '【资本效率档:毛敞口 8%、年化 +14.3%】', t=0.40)

_VOL12 = [dict(vk=a, vw=b) for a in ('cc', 'gk', 'pk', 'atr') for b in (45, 60, 90)]


def alpha_f_battvol(data, liquid):
    """电池平均书:同一结构 × 4 个独立波动估计量族 × 3 个窗 共 12 变体的敞口等权平均
    —— 波动估计量这一旋钮上 20/20 个取值的 min(2025,2026) 都 >1.58,故平均后仍过门,
    是本族最抗单点参数拟合的交付形态。"""
    return np.mean(np.stack([_one(data, liquid, **s) for s in _VOL12]), axis=0)
