"""Q25 CUDA 组合准入；Python 只编排、语法检查和序列化。"""
from __future__ import annotations
import math,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from config.config import Q25ComboConfig
import engine
CFG=Q25ComboConfig()
DEVICE='cuda'
PROTOCOL='combo-cuda-v3'

def g1_causality(expr):
    if not expr:return None
    windows=[]
    for match in re.finditer(r'\bts_rank\s*\(',expr):
        depth=1
        for end in range(match.end(),len(expr)):
            depth+=(expr[end]=='(')-(expr[end]==')')
            if depth==0:
                windows.append(int(expr[match.end():end].rsplit(',',1)[1].strip()))
                break
    bad=[w for w in windows if w>31]
    if bad:return f'非因果算子:ts_rank 窗口 {bad}>31，历史 bucket 实现使用全样本桶边界'
    return None

def make_ref(W,win,n,seed=False):
    import cupy as cp
    m=engine._metrics(W.copy(),win)
    dates=cp.asarray(win['dates'][win['row0']:].astype('int64'))
    p=cp.power(0.5,(dates[-1]-dates)/CFG.recent_halflife_days);p/=p.sum()
    rr=m['rr'];mu=cp.sum(rr*p);var=cp.sum((rr-mu)**2*p)
    ws=cp.where(var>0,mu/cp.sqrt(cp.where(var>0,var,1))*365**0.5,0)
    vals=cp.asnumpy(cp.stack((ws,cp.sum(cp.log1p(rr)*p)*365))).tolist()
    return dict(**m,p=p,wsharpe=vals[0],n=n,seed=seed,win=win,recent_growth=vals[1])

def seed_ref(window):
    from combo_gpu import device_window
    win=engine.load()['windows'][window];g=device_window(win)
    return make_ref(g.dliq/g.nliq,win,1,seed=True)

def combo_row(r,ref,p):
    mine,_,rho=r['_gpu_view']
    out=dict(n=ref['n'],seed=ref['seed'],sharpe=ref['wsharpe'],full_sharpe=ref['sharpe'],
             cagr=ref['cagr'],maxdd=ref['maxdd'],mine=mine,rho=rho,w=p['weight'],
             trials=p['trials'],cpu_trials=0,backend='cuda-f64',feasible=p['passed'])
    if 'wsharpe' in p:
        out.update(sharpe_after=p['wsharpe'],full_sharpe_after=p['sharpe'],
                   cagr_after=p['cagr'],maxdd_after=p['maxdd'],
                   delta_sharpe=p['checks']['sharpe'][1],delta_cagr=p['checks']['cagr'][1],
                   checks={k:dict(passed=bool(v[0]),delta=v[1]) for k,v in p['checks'].items()},
                   submission_ready=p['passed'] and p['sharpe']>=CFG.submission_sharpe)
    return out

def run(r,ref=None):
    return run_batch([r],seed_ref(r['window']) if ref is None else ref)[0]

def run_batch(rows,ref):
    from combo_gpu import proposal_many
    results=[]
    for lo in range(0,len(rows),engine.CUDA.candidate_batch):
        part=rows[lo:lo+engine.CUDA.candidate_batch]
        why=[r.get('error') if not r['ok'] else
             (g1_causality(r.get('expr')) or (None if r.get('_valid_rr',False) else '无效收益序列:非有限值或破产'))
             for r in part]
        idx=[i for i,r in enumerate(part) if r['ok'] and not why[i]]
        proposals=proposal_many([part[i] for i in idx],ref) if idx else []
        cache=dict(zip(idx,proposals))
        for i,r in enumerate(part):
            if i not in cache:
                results.append(dict(passed=False,stopped_at='G1',layers={'G1_structure':dict(passed=False,vetoes=[why[i]])},
                                    expr=r.get('expr'),protocol=PROTOCOL))
                continue
            p=cache[i]
            results.append(dict(passed=p['passed'],stopped_at=None if p['passed'] else 'G2',
                layers=dict(G1_structure=dict(passed=True,vetoes=[]),G2_performance=p),
                expr=r['expr'],protocol=PROTOCOL,combo=combo_row(r,ref,p)))
    return results

def explain(g):
    if g['passed']:return '结构有效；加入后组合 Sharpe 提升且全期/近期收益不下降'
    return '; '.join(str(v) for layer in g['layers'].values() for v in layer.get('vetoes',[]))

def reject_row(expr,error,stop='G1'):
    return dict(protocol=PROTOCOL,expr=expr,lane='fastexpr',ok=False,error=error,
                passed=False,stopped_at=stop,explain=str(error),accepted=False,backend='cuda-f64')

def rowify(r,g):
    def js(x):
        if isinstance(x,dict):return {k:js(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [js(v) for v in x]
        return x if not isinstance(x,float) or math.isfinite(x) else None
    row=dict(protocol=PROTOCOL,expr=r.get('expr'),lane=r.get('lane','fastexpr'),ok=bool(r['ok']),
             error=r.get('error'),passed=bool(g['passed']),stopped_at=g['stopped_at'],
             explain=explain(g),accepted=False,backend='cuda-f64')
    for k in ('sharpe','cagr','ann_ret','maxdd','total_ret','turnover','exposure','margin_bps',
              'max_single_weight','concentration','span','n_days','yearly'):
        if k in r:row[k]=js(r[k])
    if 'combo' in g:row['combo']=g['combo']
    return js(row)
