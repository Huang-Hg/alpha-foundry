"""Bindings for Q25 CUDA C++ primitives; arrays never leave the device."""
from pathlib import Path
import cupy as cp
import numpy as np  # C ABI scalar packing only

_MOD=cp.RawModule(code=Path(__file__).with_name('q25_native.cu').read_text(encoding='utf-8'),
                  options=('--std=c++14','--fmad=false'))

def _panel(x):
    a=cp.asarray(x,dtype=cp.float64)
    return cp.ascontiguousarray(a[:,None] if a.ndim==1 else a)

def recursive(x,y=None,z=None,mode=0,window=0,k=0):
    shape=x.shape;a=_panel(x);T,S=a.shape
    b=a if y is None else _panel(y);c=a if z is None else _panel(z)
    out=cp.empty_like(a)
    _MOD.get_function('q25_recursive')(((S+127)//128,),(128,),
        (a,b,c,out,np.int32(T),np.int32(S),np.int32(mode),np.int32(window),np.float64(k)))
    return out.reshape(shape)

def trade_when(e,a,x): return recursive(e,a,x)
def hump(x,k): return recursive(x,mode=1,k=k)
def backfill(x,w): return recursive(x,mode=2,window=w)
def cummax(x): return recursive(x,mode=3)

def decay(x,w,factor):
    a=_panel(x);T,S=a.shape;out=cp.empty_like(a)
    _MOD.get_function('q25_decay')(((a.size+255)//256,),(256,),
        (a,out,np.int32(T),np.int32(S),np.int32(w),np.float64(factor)))
    return out.reshape(x.shape)

def median(x,w):
    a=_panel(x);T,S=a.shape;out=cp.full_like(a,cp.nan)
    if T<w:return out.reshape(x.shape)
    view=cp.lib.stride_tricks.sliding_window_view(a,w,axis=0)
    # Bounded scratch; each chunk's sorting and validity checks are CUDA kernels.
    for lo in range(0,T-w+1,128):
        v=view[lo:lo+128];s=cp.sort(v,axis=2)
        mid=s[:,:,w//2] if w%2 else (s[:,:,w//2]+s[:,:,w//2-1])*0.5
        out[lo+w-1:lo+w-1+len(v)]=cp.where(cp.isnan(v).any(2),cp.nan,mid)
    return out.reshape(x.shape)

def rank_pct(x,w):
    a=_panel(x);T,S=a.shape;out=cp.empty_like(a)
    _MOD.get_function('q25_rank_pct')(((a.size+255)//256,),(256,),
        (a,out,np.int32(T),np.int32(S),np.int32(w)))
    return out.reshape(x.shape)

def rolling_mean(x,w):
    a=_panel(x);T,S=a.shape;out=cp.empty_like(a)
    _MOD.get_function('q25_rolling_mean')(((S+127)//128,),(128,),
        (a,out,np.int32(T),np.int32(S),np.int32(w)))
    return out.reshape(x.shape)

def hysteresis(enter,exit):
    shape=enter.shape
    e=cp.ascontiguousarray(enter.reshape(len(enter),-1),dtype=cp.uint8)
    x=cp.ascontiguousarray(exit.reshape(e.shape),dtype=cp.uint8)
    T,S=e.shape;out=cp.empty_like(e)
    _MOD.get_function('q25_hysteresis')(((S+127)//128,),(128,),
        (e,x,out,np.int32(T),np.int32(S)))
    return out.reshape(shape).astype(cp.bool_)

def hold(x,enter,exit_,min_hold=0):
    a=cp.ascontiguousarray(x,dtype=cp.float64);out=cp.empty_like(a)
    _MOD.get_function('q25_hold')((1,),(1,),
        (a,out,np.int32(len(a)),np.float64(enter),np.float64(exit_),np.int32(min_hold)))
    return out

def state(c,ent,exit_,stop,allow=None,ent_loss=None,tight=None,tstop=None,istop=None,cool=0,be=None,fexit=None):
    c=_panel(c);T,S=c.shape
    def broad(x,dtype):return cp.ascontiguousarray(cp.broadcast_to(cp.asarray(x,dtype=dtype),c.shape))
    ent=broad(ent,cp.uint8);ext=broad(exit_,cp.uint8)
    stop=broad(cp.inf if stop is None else stop,cp.float64)
    alw=broad(True if allow is None else allow,cp.uint8)
    loss=ent if ent_loss is None else broad(ent_loss,cp.uint8)
    force=ent if fexit is None else broad(fexit,cp.uint8)
    flags=(1 if tight is not None else 0)|(2 if istop is not None else 0)|(4 if be is not None else 0)|\
          (8 if tstop is not None else 0)|(16 if fexit is not None else 0)|(32 if ent_loss is not None else 0)
    tk,tm=tight or (0.,0.);td,tg=tstop or (0.,0.);is_,ig=istop or (0.,0.)
    out=cp.empty_like(c);dd=cp.empty_like(c);gain=cp.empty_like(c)
    _MOD.get_function('q25_state')(((S+127)//128,),(128,),
        (c,ent,ext,stop,alw,loss,force,out,dd,gain,np.int32(T),np.int32(S),np.int32(flags),np.int32(cool),
         np.float64(tk),np.float64(tm),np.float64(td),np.float64(tg),np.float64(is_),np.float64(ig),np.float64(be or 0.)))
    return out,dd,gain
