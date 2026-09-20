"""Q25 GPU-only engine. Python handles parquet I/O, metadata and CUDA dispatch."""
from __future__ import annotations
import os,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
os.environ['Q25_CUDA_FP64']='1'
import numpy as np  # parquet host buffers and calendar metadata only
import cupy as cp
from config.config import FactorMiningConfig, Q25CudaConfig
from backtest import ops_cuda as oc
PPY=365.0
CUDA=Q25CudaConfig()
_D=None
_BM={}

def load(force=False):
    global _D
    if _D is not None and not force:return _D
    import pyarrow.parquet as pq
    from combo_gpu import prepare_static
    fm=FactorMiningConfig()
    dd=Path(fm.qnt_data_dir)
    if not dd.is_absolute():dd=REPO/dd
    ptbl,mtbl,ftbl=(pq.read_table(dd/f'{n}.parquet') for n in ('panel','market','features'))
    dates0=ptbl.column('date').to_pylist()
    assert dates0==mtbl.column('date').to_pylist()==ftbl.column('date').to_pylist()
    dates=np.array(dates0,dtype='datetime64[D]')
    assets=[n.split('__',1)[1] for n in ptbl.column_names if n.startswith('close__')]
    def grid(field,tbl=ptbl):
        host=np.column_stack([tbl.column(f'{field}__{a}').to_numpy() for a in assets])
        return cp.asarray(host,dtype=cp.float64)
    o,h,l,c,v=(grid(k) for k in ('open','high','low','close','vol'))
    liquid=grid('liquid')>0
    T,S=c.shape
    feat=list(dict.fromkeys(n.split('__',1)[0] for n in ftbl.column_names if '__' in n))
    def fgrid(k):
        a=grid(k,ftbl)
        return cp.log(cp.where(a>0,a,cp.nan)) if k=='amihud' else a
    price=cp.where(c>0,c,cp.nan)
    trend={}
    for w in (20,60,120):
        trend[f'ma_dist_{w}']=price/oc.ts_mean(price,w)-1
        trend[f'ret_{w}']=price/oc.ts_ref(price,w)-1
    market={n:cp.ascontiguousarray(cp.broadcast_to(cp.asarray(mtbl.column(n).to_numpy())[:,None],(T,S)))
            for n in mtbl.column_names if n!='date'}
    panels=dict(open=o,high=h,low=l,close=c,volume=v,**{k:fgrid(k) for k in feat},**trend,**market)
    vocab=sorted(k for k in panels if k not in ('open','high','low','close','volume'))
    valid=cp.asnumpy(cp.stack([~cp.isinf(panels[k]).any() for k in vocab])).tolist()
    assert all(valid), 'dataset features contain infinity'
    row0=int(np.searchsorted(dates,np.datetime64(fm.train_start)))
    train_end=int(np.searchsorted(dates,np.datetime64(fm.train_end),side='right'))
    val_end=int(np.searchsorted(dates,np.datetime64(fm.val_end),side='right'))
    val_start=int(np.searchsorted(dates,np.datetime64(fm.val_start)))
    static=prepare_static(o,h,l,c)
    def window(end,start,span):
        return dict(panels={k:cp.ascontiguousarray(p[:end]) for k,p in panels.items()},
                    liquid=liquid[:end],static={k:cp.ascontiguousarray(p[:end]) for k,p in static.items()},
                    row0=start,dates=dates[:end],span=span)
    _D=dict(dates=dates,assets=assets,liquid=liquid,vocab=vocab,windows={
        'is':window(val_end,row0,(fm.train_start,fm.val_end)),
        'train':window(train_end,row0,(fm.train_start,fm.train_end)),
        'val':window(val_end,val_start,(fm.val_start,fm.val_end))})
    # Bound the configured maxima by available device memory. This estimates
    # live books, AST intermediates and trial reductions, leaving 35% headroom.
    free,_=cp.cuda.runtime.memGetInfo()
    limit=max(1,int(free*.65/(T*S*8*20)))
    CUDA.candidate_batch=min(CUDA.candidate_batch,1<<(limit.bit_length()-1))
    CUDA.trial_batch=min(CUDA.trial_batch,max(1,int(free*.10/(T*8*10))))
    return _D

def operands():return load()['vocab']
def field_panel(name,window='is'):return load()['windows'][window]['panels'][name]

def _book(exposure,liquid):
    from gpu import _book as device_book
    liq=cp.asarray(liquid)
    return device_book(cp.asarray(exposure,dtype=cp.float64),liq,cp.maximum(liq.sum(1,keepdims=True),1).astype(cp.float64))

def _metrics(W,win,result=None):
    from combo_gpu import metrics_many
    return metrics_many([W],win,result)[0]

def official_sharpe(rr):
    rr=cp.asarray(rr);sd=rr.std()
    return float(cp.where(sd>0,rr.mean()/cp.where(sd>0,sd,1)*PPY**0.5,0))

def benchmark(window='is'):
    if window not in _BM:
        from combo_gpu import device_window
        win=load()['windows'][window];g=device_window(win)
        _BM[window]=_metrics(g.dliq/g.nliq,win)
    return _BM[window]

def simulate_many(exprs,window='is'):
    from combo_gpu import simulate_many as device_simulate
    return device_simulate(exprs,window)

def simulate(expr,window='is'):return simulate_many([expr],window)[0]

def simulate_python(fn,window='is',check_lookahead=True,cuts=(.4,.6,.85)):
    from combo_gpu import simulate_python as device_python
    return device_python(fn,window,check_lookahead,cuts)
