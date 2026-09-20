"""Q25 alpha glue: CUDA primitives and source registration, no CPU numerical path."""
import ast
import hashlib
from collections import OrderedDict
import cupy as cp
from backtest import ops_cuda as oc
from backtest.ops_cuda import q25 as native

rank_pct=native.rank_pct

def rolling(x,w,how):
    shape=x.shape
    x=cp.ascontiguousarray(cp.where(cp.isfinite(x),x,cp.nan).reshape(len(x),-1))
    fn={'mean':native.rolling_mean,'std':oc.ts_std,'max':oc.ts_max,'min':oc.ts_min,'sum':oc.ts_sum}[how]
    return fn(x,w).reshape(shape)

def state2(*a,**kw):return native.state(*a,**kw)[:2]
def simple_state(c,enter,exit_,stop):return native.state(c,enter,exit_,stop)[:2]
def thresholds(x,enter,exit_):
    return native.hysteresis(cp.isfinite(x)&(x>enter),cp.isfinite(x)&(x<exit_)).astype(cp.float64)

def coinbear(data,liquid):
    c=cp.where(data['close']>0,data['close'],cp.nan)
    s100=oc.ts_mean(c,100)
    dd=c/oc.ts_max(c,365)-1
    ent=(dd<-.5)&(c<s100)
    ext=(c>s100)&(s100>oc.ts_ref(s100,20))
    bear=native.hysteresis(ent,ext)
    return (1-bear.astype(cp.float64))*cp.where(c>oc.ts_mean(c,20),1.,.5)

_SOURCES=OrderedDict()

def load_source(source,path):
    key=hashlib.sha256(source.encode()).hexdigest()
    if key in _SOURCES:
        _SOURCES.move_to_end(key)
        return _SOURCES[key]
    tree=ast.parse(source,filename=path)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            names=[x.name for x in node.names]
        elif isinstance(node,ast.ImportFrom):names=[node.module or '']
        else:names=[]
        if any(x.split('.')[0] in ('numpy','pandas') for x in names):
            raise ValueError(f'{path}: NumPy/Pandas alpha execution was removed; use CUDA primitives')
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(isinstance(x,(ast.For,ast.While)) for x in ast.walk(node)):
            raise ValueError(f'{path}:{node.lineno}: numerical loops belong in C++/CUDA')
    ns={}
    exec(compile(tree,path,'exec'),ns)
    _SOURCES[key]=ns
    while len(_SOURCES)>2:_SOURCES.popitem(last=False)
    return ns
