"""组合权重搜索的 GPU 批量回测。因子、持仓、静态价格与指标均为 CUDA f64。

候选持仓保留原算子口径;GPU 一次计算全部权重下的真实合成持仓收益。
GPU f64 结果直接用于准入,生产路径不再做 CPU 回测。CPU 黄金参考只在独立测试脚本使用。
"""
import numpy as np
import cupy as cp
from backtest.ops_cuda import _MOD

KERNEL = _MOD.get_function('k_qnt_bt_f64')
COMBO_KERNEL = _MOD.get_function('k_qnt_combo_f64')
_STATIC_WIN = None
_STATIC_DEVICE = None
_WINDOWS = {}


def prepare_static(o,h,l,c,fract=.04):
    from backtest.ops_cuda.q25 import backfill
    T,S=o.shape
    prev=cp.concatenate((cp.full((1,S),cp.nan),c[:-1]),axis=0)
    d1=h-l;d2=cp.abs(h-prev);d3=cp.abs(prev-l)
    tr=cp.where(cp.isnan(d1)|cp.isnan(d2)|cp.isnan(d3),cp.nan,cp.fmax(d1,cp.fmax(d2,d3)))
    w=min(T,14)
    atr=cp.full_like(tr,cp.nan)
    atr[w-1:]=cp.lib.stride_tricks.sliding_window_view(tr,w,axis=0).mean(2)
    # qnt forward-fill considers finite values; leading missing values stay missing.
    def fill(x):return backfill(cp.where(cp.isfinite(x),x,cp.nan),T+1)
    slip=fill(atr)*fract
    of=fill(o);cf=fill(c)
    of=cp.where(cp.isnan(of),0,of);cf=cp.where(cp.isnan(cf),0,cf)
    unlocked=cp.isfinite(o)&cp.isfinite(c)&cp.isfinite(slip)&(of>1e-7)
    return dict(open_raw=o,prev_close=fill(prev),open_ff0=of,close_ff0=cf,slippage=slip,
                unlocked=unlocked.astype(cp.uint8))


def device_window(win):
    """Only host metadata is prepared here; all numerical arrays stay on CUDA."""
    key = id(win)
    if key not in _WINDOWS:
        from gpu import GpuBatch
        g = GpuBatch(win['panels'],win['liquid'],win['static'],win['row0'],win['dates'])
        if g.dp[next(iter(g.dp))].dtype != cp.float64:
            raise RuntimeError('Q25 requires CUDA float64 factor inputs')
        _WINDOWS[key] = (win,g)
    return _WINDOWS[key][1]


def stack_books(books):
    return cp.ascontiguousarray(books if isinstance(books,cp.ndarray) else
                                cp.stack([cp.asarray(w,dtype=cp.float64) for w in books]))


def metrics_many(books,win,result=None):
    """CUDA batch reductions; the host receives report scalars, never daily PnL/books."""
    g = device_window(win)
    W = stack_books(books)
    W[:,:win['row0']] = 0
    rr,st = launch(W,static_device(win),win['row0']) if result is None else result
    rt = rr[:,win['row0']:]
    wt = W[:,win['row0']:]
    lg = cp.log1p(rt)
    gross = cp.abs(wt).sum(2)
    attend = gross > 1e-9
    turnover = cp.abs(cp.diff(wt,axis=1)).sum(2).mean(1)
    ann = rt.mean(1)*365
    vol = rt.std(1)
    sh = cp.where(vol>0,rt.mean(1)/cp.where(vol>0,vol,1)*365**0.5,0)
    total = rt.sum(1)
    concentration = cp.where(total>0,cp.sort(rt,axis=1)[:,-g.k_top:].sum(1)/cp.where(total>0,total,1),cp.inf)
    vals = cp.stack((sh,ann,st[:,3],cp.expm1(lg.mean(1)*365),st[:,4],turnover,gross.mean(1),
                     ann/cp.maximum(turnover*365,1e-12)*1e4,cp.abs(wt).max(axis=(1,2)),concentration),axis=1)
    ysh,yatt,ylg = g.yearly(rt,attend,lg)
    y = cp.stack((ysh,yatt,ylg),axis=2)
    valid = cp.isfinite(rt).all(1)&(rt>-1).all(1)
    host = cp.asnumpy(cp.concatenate((vals,y.reshape(len(W),-1),valid[:,None]),axis=1)).tolist()
    names=('sharpe','ann_ret','maxdd','cagr','total_ret','turnover','exposure','margin_bps','max_single_weight','concentration')
    dates=win['dates'][win['row0']:]
    result=[]
    for i,v in enumerate(host):
        years=[dict(year=int(year),days=int(days),sharpe=round(v[10+j*3],3),
                    attend=round(v[11+j*3],3),lg=round(v[12+j*3],4))
               for j,(year,days) in enumerate(zip(g.years,g.days))]
        result.append(dict(zip(names,v[:10]),yearly=years,span=f'{dates[0]}..{dates[-1]}',
                           n_days=len(dates),rr=rt[i],W=W[i],_valid_rr=bool(v[-1])))
    return result


def simulate_many(exprs,window='is'):
    import engine
    import gates
    from fastexpr import validate
    from gpu import eval_node,_book,can_gpu,unsupported_ops
    win=engine.load()['windows'][window]
    g=device_window(win)
    ready=cp.cuda.Event()
    ready.record()
    for stream in g._streams:stream.wait_event(ready)
    rows=[]; ids=[]; values=[]
    caches=[{} for _ in g._streams]
    for expr in exprs:
        s=expr.pretty() if hasattr(expr,'pretty') else str(expr).strip()
        v=validate(s,engine.operands())
        why=v.get('error') if not v['ok'] else gates.g1_causality(s)
        row=dict(ok=not why,error=why,expr=s,window=window)
        rows.append(row)
        if why: continue
        tree=v['tree']
        if not can_gpu(tree):
            raise RuntimeError(f'CUDA operator implementation required: {unsupported_ops(tree)}: {s}')
        k=len(ids)%len(g._streams)
        with g._streams[k]:
            values.append(eval_node(tree.root,g.dp,caches[k]))
        row['tree']=tree
        ids.append(len(rows)-1)
    for stream in g._streams: stream.synchronize()
    if ids:
        values=cp.stack(values)
        valid=cp.asnumpy((~cp.isinf(values).any(axis=(1,2)))&cp.isfinite(values).any(axis=(1,2))).tolist()
        W=_book(values,g.dliq,g.nliq)
        metrics=metrics_many(W,win)
        for i,m,ok in zip(ids,metrics,valid):
            rows[i].update(ok=ok,error=None if ok else 'G1 数值无效:含无穷值或全为缺失',
                           **{k:v for k,v in m.items() if k not in ('rr','W')},_rr=m['rr'],_W=m['W'])
    return rows


def simulate_python(fn,window,check_lookahead,cuts):
    import engine
    win=engine.load()['windows'][window];g=device_window(win)
    exp=fn(g.dp,g.dliq)
    if not isinstance(exp,cp.ndarray):
        raise TypeError('CUDA alpha must return a device array; implicit CPU execution is not allowed')
    exp=cp.asarray(exp,dtype=cp.float64)
    name=getattr(fn,'__name__','python_alpha')
    if exp.shape!=g.dliq.shape:
        return dict(ok=False,expr=name,error=f'返回形状 {exp.shape} != 面板形状 {g.dliq.shape}',window=window)
    bad=None
    if check_lookahead:
        for frac in cuts:
            n=int(len(exp)*frac)
            e2=cp.asarray(fn({k:v[:n] for k,v in g.dp.items()},g.dliq[:n]),dtype=cp.float64)
            if e2.shape!=exp[:n].shape:
                bad=f'截断 {n} 行后的返回形状错误';break
            a=exp[:n]
            diff=~((cp.isfinite(a)==cp.isfinite(e2))&(cp.isclose(a,e2,rtol=1e-9,atol=1e-12)|~cp.isfinite(a)))
            count=int(diff.sum())
            if count:
                bad=f'前视泄漏:截断到 {n} 行重跑后有 {count} 个单元改变';break
    valid=bool((~cp.isinf(exp).any())&cp.isfinite(exp).any())
    if not valid:bad=bad or 'G1 数值无效:含无穷值或全为缺失'
    from gpu import _book
    W=_book(exp,g.dliq,g.nliq)
    m=metrics_many([W],win)[0]
    return dict(ok=bad is None,error=bad,expr=name,lane='python',window=window,
                **{k:v for k,v in m.items() if k not in ('rr','W')},_rr=m['rr'],_W=m['W'])


def static_device(win):
    global _STATIC_WIN, _STATIC_DEVICE
    if _STATIC_WIN is not win:
        _STATIC_WIN = win
        _STATIC_DEVICE = {k:cp.asarray(v,dtype=cp.uint8 if k=='unlocked' else cp.float64)
                          for k,v in win['static'].items()}
    return _STATIC_DEVICE


def launch(mixed,st,row0):
    B,T,S = mixed.shape
    assert S <= 128
    rr = cp.empty((B,T),dtype=cp.float64)
    stats = cp.empty((B,5),dtype=cp.float64)
    KERNEL(((B+3)//4,), (128,),
           (mixed,st['open_raw'],st['prev_close'],st['open_ff0'],st['close_ff0'],st['slippage'],st['unlocked'],
            np.int32(T),np.int32(S),np.int32(B),np.int64(row0),np.float64(365),rr,stats))
    return rr,stats


def proposal_many(rows,ref):
    """Weight proposal, net-cost trials, constraints and winner selection on CUDA."""
    from gates import CFG
    from engine import CUDA
    W=stack_books([r['_W'] for r in rows])
    R=cp.stack([r['_rr'] for r in rows])
    q=cp.asarray(ref['rr']); p=cp.asarray(ref['p'])
    ma=cp.sum(R*p,axis=1); mq=cp.sum(q*p)
    va=cp.sum((R-ma[:,None])**2*p,axis=1); vq=cp.sum((q-mq)**2*p)
    cov=cp.sum((R-ma[:,None])*(q-mq)*p,axis=1)
    v_a=ma*vq-mq*cov; v_p=mq*va-ma*cov
    both=(v_a>0)&(v_p>0)
    suggestion=cp.where(v_a<=0,0,cp.where(v_p<=0,1,v_a/cp.where(both,v_a+v_p,1)))
    suggestion=cp.where((suggestion>0)&(suggestion<=1),suggestion,1)
    B=len(rows); K=CFG.weight_steps+1
    grid=cp.broadcast_to(cp.arange(1,K,dtype=cp.float64)/(K-1),(B,K-1))
    weights=cp.sort(cp.concatenate((grid,suggestion[:,None]),axis=1),axis=1)
    unique=cp.concatenate((cp.ones((B,1),dtype=cp.bool_),weights[:,1:]!=weights[:,:-1]),axis=1)
    owners=cp.repeat(cp.arange(B,dtype=cp.int32),K)
    flat=weights.ravel()
    st=static_device(ref['win']); T,S=W.shape[1:]
    qW=cp.asarray(ref['W'],dtype=cp.float64)
    returns=[]; summaries=[]
    for lo in range(0,B*K,CUDA.trial_batch):
        n=min(CUDA.trial_batch,B*K-lo)
        rr=cp.empty((n,T),dtype=cp.float64); stats=cp.empty((n,5),dtype=cp.float64)
        COMBO_KERNEL(((n+3)//4,),(128,),
            (W,qW,owners[lo:lo+n],flat[lo:lo+n],st['open_raw'],st['prev_close'],
             st['open_ff0'],st['close_ff0'],st['slippage'],st['unlocked'],
             np.int32(T),np.int32(S),np.int32(n),np.int64(ref['win']['row0']),np.float64(365),rr,stats))
        returns.append(rr[:,ref['win']['row0']:]); summaries.append(stats)
    rr=cp.concatenate(returns)
    stats=cp.concatenate(summaries)
    valid=cp.isfinite(rr).all(1)&(rr>-1).all(1)
    lg=cp.log1p(rr)
    mw=cp.sum(rr*p,axis=1); vw=cp.sum((rr-mw[:,None])**2*p,axis=1)
    ws=cp.where(vw>0,mw/cp.sqrt(cp.where(vw>0,vw,1))*365**0.5,0).reshape(B,K)
    sd=rr.std(1)
    sh=cp.where(sd>0,rr.mean(1)/cp.where(sd>0,sd,1)*365**0.5,0).reshape(B,K)
    cagr=cp.expm1(lg.mean(1)*365).reshape(B,K)
    growth=(cp.sum(lg*p,axis=1)*365).reshape(B,K)
    dd=stats[:,3].reshape(B,K)
    delta=cp.stack((ws-ref['wsharpe'],sh-ref['sharpe'],cagr-ref['cagr'],growth-ref['recent_growth']),axis=2)
    checks=delta>=cp.asarray([CFG.min_sharpe_gain-1e-10,-1e-10,-1e-10,-1e-10])
    valid=valid.reshape(B,K)&unique
    new=cp.asarray([(r['expr'],r.get('lane','fastexpr')) not in ref.get('member_keys',set()) for r in rows])
    capacity=~((ref['n']>=CFG.max_members)&new[:,None]&(weights<1))
    feasible=checks.all(2)&valid&capacity
    passed=feasible.any(1)
    score=cp.where(feasible,ws,-cp.inf)
    best_ws=score.max(1)
    tie=feasible&(ws==best_ws[:,None])
    best_cagr=cp.where(tie,cagr,-cp.inf).max(1)
    best=cp.argmax(tie&(cagr==best_cagr[:,None]),axis=1)
    near=cp.argmax(cp.where(valid,ws,-cp.inf),axis=1)
    chosen=cp.where(passed,best,near); ix=cp.arange(B)
    mine=cp.where(va>0,ma/cp.sqrt(cp.where(va>0,va,1))*365**0.5,0)
    rho=cp.where((va>0)&(vq>0),cov/cp.sqrt(cp.where((va>0)&(vq>0),va*vq,1)),0)
    need=rho*ref['wsharpe']+0.6*cp.sqrt(cp.maximum(1-rho*rho,0))
    report=cp.stack((passed,valid.any(1),weights[ix,chosen],ws[ix,chosen],sh[ix,chosen],
                     cagr[ix,chosen],growth[ix,chosen],dd[ix,chosen],unique.sum(1),capacity[ix,chosen],
                     mine,rho,need),axis=1)
    report=cp.concatenate((report,delta[ix,chosen],checks[ix,chosen]),axis=1)
    host=cp.asnumpy(report).tolist()
    result=[]
    keys=('sharpe','full_sharpe','cagr','recent_growth')
    labels=('近期加权 Sharpe 增量不足','全期 Sharpe 下降','全期 CAGR 下降','近期加权对数增长下降')
    for row,v in zip(rows,host):
        row['_gpu_view']=(v[10],v[12],v[11])
        if not v[1]:
            result.append(dict(passed=False,checks={},vetoes=['所有试探组合均产生非有限收益或破产'],weight=0.,
                               trials=int(v[8]),cpu_trials=0,backend='cuda-f64',_ref=ref))
            continue
        ck={key:(bool(v[17+j]),v[13+j]) for j,key in enumerate(keys)}
        if not v[9]: ck['capacity']=(False,float(CFG.max_members-ref['n']-1))
        vetoes=[f'{label} ({v[13+j]:+.6f})' for j,label in enumerate(labels) if not v[17+j]]
        if not v[9]: vetoes.append('组合成员数超过上限')
        if not v[0]: vetoes.insert(0,f'{int(v[8])} 个权重点未找到可行组合;以下为最高 Sharpe 试探点')
        result.append(dict(passed=bool(v[0]),checks=ck,vetoes=vetoes,weight=v[2],wsharpe=v[3],sharpe=v[4],
                           cagr=v[5],recent_growth=v[6],maxdd=v[7],trials=int(v[8]),cpu_trials=0,backend='cuda-f64',_ref=ref))
    return result
