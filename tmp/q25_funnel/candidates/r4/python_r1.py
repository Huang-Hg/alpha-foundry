from alpha_cuda import rolling as _cuda_rolling, rank_pct as _cuda_rank_pct
from contextlib import nullcontext as _nullcontext
# Python 车道 r1:铺地形 —— 五类 FastExpr 写不出的结构各一片
#   A 截面宽度择时(breadth = liquid 币中 ma_dist_w>0 的比例,低于阈值全空仓)
#   B 截面排名选币(当日 liquid 币内按动量排名取前 k)
#   C 波动目标化敞口(min(1, target/realized_vol))
#   D 双阈值滞回状态机(进/出阈值不同 + 最短持有)
#   E 市场级 regime 分类(btc_rv/alt_rv 滚动分位三档;EW 指数滚动回归斜率×R²)
# 全部因果:rolling 右对齐、min_periods=window(面板 2015-01-01 起,IS 前有 365 行 warmup)。
import cupy as np


# ---------------------------------------------------------------- 因果原语
def _rmean(x, w):
    return _cuda_rolling(x, w, 'mean')


def _rstd(x, w):
    return _cuda_rolling(x, w, 'std')


def _rets(close):
    with _nullcontext():
        r = close[1:] / close[:-1] - 1.0
    return np.vstack([np.full((1, close.shape[1]), np.nan), r])


def _breadth(cond, valid, liquid):
    """liquid 且 valid 的币中满足 cond 的比例;无有效币 → NaN。(T,)"""
    den = (valid & liquid).sum(1)
    num = (cond & valid & liquid).sum(1)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


from backtest.ops_cuda.q25 import hold as _hyst


def _ew_index(data, liquid):
    """liquid 币等权日收益(用 t-1 的 liquid 集)→ 累计对数指数。(T,)"""
    r = _rets(data['close'])
    lp = np.vstack([np.zeros((1, liquid.shape[1]), bool), liquid[:-1]])
    ok = lp & np.isfinite(r)
    ew = np.where(ok, r, 0.0).sum(1) / np.maximum(ok.sum(1), 1)
    return np.cumsum(np.log1p(ew))


def _rank_pct(x, w):
    """(T,) 滚动分位:当前值在过去 w 天(含当天)中的百分位。"""
    return _cuda_rank_pct(x, w)


def _top_k(score, liquid, k):
    """当日 liquid 币内 score 降序前 k 置 1,其余 0;NaN 分数排最后。(T,S)"""
    s = np.where(liquid & np.isfinite(score), score, -np.inf)
    order = np.argsort(-s, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(s.shape[1])[None, :], axis=1)
    return ((rank < k) & np.isfinite(score) & liquid).astype(np.float64)


def _carrier(data, liquid, kind, w=60):
    md = data[f'ma_dist_{w}']
    if kind == 'all':
        return np.where(liquid, 1.0, 0.0)
    if kind == 'bin':
        return np.where(md > 0, 1.0, 0.0)
    if kind == 'soft':
        return np.clip(md / 0.10, 0.0, 1.0)
    raise ValueError(kind)


def _reg(name, doc, fn):
    fn.__name__ = name
    fn.__doc__ = doc
    globals()[name] = fn


# ---------------------------------------------------------------- A 宽度择时
def _mk_breadth(w, thr, carrier):
    def f(data, liquid):
        md = data[f'ma_dist_{w}']
        br = _breadth(md > 0, np.isfinite(md), liquid)
        gate = np.where(br > thr, 1.0, 0.0)
        return gate[:, None] * _carrier(data, liquid, carrier, w)
    return f


for _w in (20, 60, 120):
    for _thr in (0.3, 0.5, 0.7):
        for _c in ('all', 'bin'):
            _reg(f'alpha_br_w{_w}_t{int(_thr*100)}_{_c}',
                 f'宽度择时:liquid 币中 close>SMA{_w} 比例>{_thr} 才在场,载体={_c}',
                 _mk_breadth(_w, _thr, _c))


# ---------------------------------------------------------------- D 滞回(宽度载体)
def _mk_breadth_hyst(w, enter, exit_, carrier, min_hold=0):
    def f(data, liquid):
        md = data[f'ma_dist_{w}']
        br = _breadth(md > 0, np.isfinite(md), liquid)
        gate = _hyst(br, enter, exit_, min_hold)
        return gate[:, None] * _carrier(data, liquid, carrier, w)
    return f


for _w in (20, 60, 120):
    _reg(f'alpha_brh_w{_w}_e60x40_bin', f'宽度滞回:宽度{_w}>0.6 进 <0.4 出,载体=bin',
         _mk_breadth_hyst(_w, 0.6, 0.4, 'bin'))
for _e, _x in ((0.7, 0.5), (0.5, 0.3)):
    _reg(f'alpha_brh_w60_e{int(_e*100)}x{int(_x*100)}_bin',
         f'宽度滞回:宽度60>{_e} 进 <{_x} 出,载体=bin', _mk_breadth_hyst(60, _e, _x, 'bin'))
_reg('alpha_brh_w60_e60x40_bin_h10', '宽度滞回 60>0.6/<0.4 + 最短持有 10 天,载体=bin',
     _mk_breadth_hyst(60, 0.6, 0.4, 'bin', 10))
_reg('alpha_brh_w60_e60x40_all', '宽度滞回 60>0.6/<0.4,载体=全体 liquid(纯择时)',
     _mk_breadth_hyst(60, 0.6, 0.4, 'all'))


# ---------------------------------------------------------------- B 截面排名选币
def _mk_topk(field, k, gate_thr):
    def f(data, liquid):
        sel = _top_k(data[field], liquid, k)
        if gate_thr is None:
            return sel
        md = data['ma_dist_60']
        br = _breadth(md > 0, np.isfinite(md), liquid)
        return np.where(br > gate_thr, 1.0, 0.0)[:, None] * sel
    return f


for _f in ('ma_dist_60', 'ret_60', 'ma_dist_120'):
    for _k in (3, 5):
        for _g in (None, 0.5):
            _reg(f'alpha_top{_k}_{_f}' + ('' if _g is None else '_br50'),
                 f'截面选币:liquid 内 {_f} 排名前 {_k}' + ('' if _g is None else ' × 宽度60>0.5 门'),
                 _mk_topk(_f, _k, _g))


# ---------------------------------------------------------------- C 波动目标化
def _mk_voltarget(w, target, gate):

    def f(data, liquid):
        rv = _rstd(_rets(data['close']), w) * np.sqrt(365.0)
        with _nullcontext():
            e = np.minimum(1.0, target / rv)
        if gate == 'bin':
            e = e * np.where(data['ma_dist_60'] > 0, 1.0, 0.0)
        return e
    return f


for _w in (20, 60):
    for _t in (0.4, 0.6):
        for _g in ('none', 'bin'):
            _reg(f'alpha_vt_w{_w}_t{int(_t*100)}_{_g}',
                 f'波动目标:min(1,{_t}/rv{_w}) ' + ('× ma_dist_60>0 门' if _g == 'bin' else '裸'),
                 _mk_voltarget(_w, _t, _g))


# ---------------------------------------------------------------- E 市场级 regime
def _mk_vol_regime(field, w, tiers, carrier):
    def f(data, liquid):
        p = _rank_pct(data[field][:, 0], w)
        e = np.where(p < 1 / 3, tiers[0], np.where(p < 2 / 3, tiers[1], tiers[2]))
        e = np.where(np.isfinite(p), e, 0.0)
        return e[:, None] * _carrier(data, liquid, carrier)
    return f


for _f in ('btc_rv', 'alt_rv'):
    _reg(f'alpha_vr_{_f}_lowfull', f'{_f} 365 日滚动分位三档:低波满仓/中半仓/高波空仓,载体=bin60',
         _mk_vol_regime(_f, 365, (1.0, 0.5, 0.0), 'bin'))
    _reg(f'alpha_vr_{_f}_highfull', f'{_f} 365 日滚动分位三档:高波满仓/中半仓/低波空仓,载体=bin60',
         _mk_vol_regime(_f, 365, (0.0, 0.5, 1.0), 'bin'))


def _mk_slope_r2(w, r2_thr, carrier):
    def f(data, liquid):
        from backtest import ops_cuda as oc
        y = _ew_index(data, liquid)[:, None]
        t = np.arange(len(y), dtype=np.float64)[:, None]
        slope = (oc.ts_cov(y,t,w) / (oc.ts_std(t,w)**2))[:,0]
        r2 = oc.ts_corr(y,t,w)[:,0]**2
        e = np.where(slope>0,np.where(r2>r2_thr,1.0,0.5),0.0)
        e = np.where(np.isfinite(slope)&np.isfinite(r2),e,0.0)
        return e[:,None]*_carrier(data,liquid,carrier)
    return f


for _w in (60, 120):
    for _c in ('all', 'bin'):
        _reg(f'alpha_sr_w{_w}_{_c}', f'EW 指数 {_w} 日回归:斜率>0 且 R²>0.5 满仓/斜率>0 半仓/否则空仓,载体={_c}',
             _mk_slope_r2(_w, 0.5, _c))
