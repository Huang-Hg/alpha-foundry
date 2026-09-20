"""Offline CUDA verification; the C++ oracle is never called in production."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'tmp/q25_funnel'), str(REPO)]
import engine  # selects FP64 before CUDA module initialization
import numpy as np
import cupy as cp
import gates
from combo_gpu import launch, static_device, COMBO_KERNEL
from combolib import ComboLib
from rl.backtest_reward import prepare_qnt_static, run_bt


def main():
    win = engine.load()['windows']['is']
    P = win['panels']; row0 = win['row0']
    static = prepare_qnt_static(*(cp.asnumpy(P[k]) for k in ('open','high','low','close')))
    for k,v in static.items():
        np.testing.assert_allclose(cp.asnumpy(win['static'][k]), v, rtol=1e-13, atol=1e-12)
    lib=ComboLib.load(); ref=lib.ref()
    names=[m['expr'] for m in lib.members if m['lane']!='python']+[
        'gt(ma_dist_20,0)','lt(ma_dist_20,0)','gt(ret_120,0)','ts_mean(candlebody,30)']
    rows=engine.simulate_many(names)
    assert all(r['ok'] for r in rows)
    # Include all-cash, abrupt full holdings, and an illiquid-only target.
    W=cp.stack([r['_W'] for r in rows]+[cp.zeros_like(ref['W']),
        cp.where(win['liquid'],0.,0.01)])
    W[:,:row0]=0
    rr,_=launch(W,static_device(win),row0)
    bcfg=SimpleNamespace(bars_per_year=365.)
    error=0.
    for i,w in enumerate(cp.asnumpy(W)):
        expected=run_bt(w,static,row0,bcfg,True)['rr']
        error=max(error,float(np.max(np.abs(cp.asnumpy(rr[i])-expected))))
    assert error<1e-10,error
    # Check fused mix, independent C++ PnL, and independently selected gate winner.
    qs=cp.asnumpy(ref['rr']); p=cp.asnumpy(ref['p']); qW=cp.asnumpy(ref['W'])
    verdicts=gates.run_batch(rows,ref)
    for row,verdict in zip(rows,verdicts):
        r=cp.asnumpy(row['_rr']); w=cp.asnumpy(row['_W'])
        ma=r@p; mq=qs@p; va=((r-ma)**2)@p; vq=((qs-mq)**2)@p
        cov=((r-ma)*(qs-mq))@p; a=ma*vq-mq*cov; b=mq*va-ma*cov
        suggestion=0 if a<=0 else 1 if b<=0 else a/(a+b)
        weights=sorted(set([i/20 for i in range(1,21)]+([suggestion] if 0<suggestion<=1 else [1.])))
        dw=cp.asarray(weights); n=len(weights); T,S=w.shape
        fused=cp.empty((n,T)); stats=cp.empty((n,5))
        st=static_device(win)
        COMBO_KERNEL(((n+3)//4,),(128,),(row['_W'][None],ref['W'],cp.zeros(n,dtype=cp.int32),dw,
            st['open_raw'],st['prev_close'],st['open_ff0'],st['close_ff0'],st['slippage'],st['unlocked'],
            np.int32(T),np.int32(S),np.int32(n),np.int64(row0),np.float64(365),fused,stats))
        feasible=[]
        for i,weight in enumerate(weights):
            expected=run_bt((1-weight)*qW+weight*w,static,row0,bcfg,True)['rr']
            e=float(np.max(np.abs(cp.asnumpy(fused[i])-expected)));error=max(error,e)
            assert e<1e-10,e
            r=expected[row0:];mu=r@p;var=((r-mu)**2)@p
            ws=mu/np.sqrt(var)*np.sqrt(365) if var>0 else 0.
            sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0.
            lg=np.log1p(r);cagr=np.expm1(lg.mean()*365);growth=lg@p*365
            ok=(ws-ref['wsharpe']>=gates.CFG.min_sharpe_gain-1e-10 and sh>=ref['sharpe']-1e-10
                and cagr>=ref['cagr']-1e-10 and growth>=ref['recent_growth']-1e-10)
            if ok:feasible.append((ws,cagr,-weight))
        assert verdict['passed']==bool(feasible),(row['expr'],verdict)
        if feasible:
            best=max(feasible)
            assert abs(verdict['combo']['w']+best[2])<1e-8
            assert abs(verdict['combo']['sharpe_after']-best[0])<1e-10
    report=dict(protocol=gates.PROTOCOL,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        max_daily_return_error=error,static_match=True,fused_mix_match=True,gate_winners_match=True,
        tested_books=len(W),tested_candidates=len(rows))
    out=REPO/'tmp/q25_cuda/verification.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


def verify_sources():
    from alpha_cuda import load_source
    from fastexpr import validate
    from gpu import can_gpu,unsupported_ops
    pool=json.loads((REPO/'tmp/q25_funnel/candidates/r6/full_pool.json').read_text())
    rejected=[];failures=[];unsupported=[];tested=0
    for item in pool:
        if item['lane']!='python':
            v=validate(item['expr'],engine.operands())
            if not v['ok']:rejected.append((item['expr'],v['error']))
            elif not can_gpu(v['tree']):unsupported.append((item['expr'],unsupported_ops(v['tree'])))
            continue
        rel,name=item['expr'].rsplit(':',1);path=REPO/'tmp/q25_funnel'/rel
        ns=load_source(path.read_text(),rel)
        r=engine.simulate_python(ns[name])
        tested+=1
        if not r['ok']:failures.append((item['expr'],r['error']))
    report=dict(total=len(pool),python_tested=tested,structural_rejections=len(rejected),
                unsupported=unsupported,python_failures=failures)
    (REPO/'tmp/q25_cuda/source_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    assert not unsupported and not failures


def benchmark():
    import fusion
    import statistics
    pool=json.loads((REPO/'tmp/q25_funnel/candidates/r6/full_pool.json').read_text())
    fast=[x['expr'] for x in pool if x['lane']!='python']
    names=[fast[i*len(fast)//1024] for i in range(1024)]
    ref=ComboLib.load().ref()
    previous=fusion.ENABLED;reports=[]
    try:
        for enabled in ((False,True) if '--ablation' in sys.argv else (True,)):
            fusion.ENABLED=enabled;samples=[]
            for pass_id in range(6):
                cp.cuda.get_current_stream().synchronize()
                t=time.perf_counter();valid=0;batch=engine.CUDA.candidate_batch
                for lo in range(0,len(names),batch):
                    rows=engine.simulate_many(names[lo:lo+batch])
                    gates.run_batch(rows,ref);valid+=sum(r['ok'] for r in rows)
                cp.cuda.get_current_stream().synchronize()
                seconds=time.perf_counter()-t
                if pass_id:samples.append(seconds)
            report=dict(fusion=enabled,candidates=len(names),valid=valid,batch=batch,
                        seconds=statistics.mean(samples),samples_seconds=samples,
                        candidates_per_second=len(names)/statistics.mean(samples))
            reports.append(report);print(json.dumps(report),flush=True)
        (REPO/'tmp/q25_cuda/benchmark.json').write_text(json.dumps(reports,indent=2))
    finally:fusion.ENABLED=previous

def batch_sweep():
    import gc
    from combo_gpu import proposal_many
    pool=json.loads((REPO/'tmp/q25_funnel/candidates/r6/full_pool.json').read_text())
    fast=[x['expr'] for x in pool if x['lane']!='python']
    names=[fast[i*len(fast)//2048] for i in range(2048)]
    ref=ComboLib.load().ref();results=[]
    poolmem=cp.get_default_memory_pool()
    # Warm the same workload before testing batch geometry.
    for lo in range(0,len(names),128):
        rows=engine.simulate_many(names[lo:lo+128]);gates.run_batch(rows,ref)
    del rows
    for batch in (128,256,512,1024):
        engine.CUDA.candidate_batch=batch
        for trial_batch in (4096,16384):
            engine.CUDA.trial_batch=trial_batch
            gc.collect();poolmem.free_all_blocks()
            t=time.perf_counter();signatures=[];peak=0
            try:
                for lo in range(0,len(names),batch):
                    rows=engine.simulate_many(names[lo:lo+batch])
                    gs=gates.run_batch(rows,ref)
                    signatures.extend((g['passed'],round(g.get('combo',{}).get('w',0),8)) for g in gs)
                    peak=max(peak,poolmem.total_bytes())
                    del rows,gs
                cp.cuda.get_current_stream().synchronize()
                seconds=time.perf_counter()-t
                if not results:baseline=signatures
                assert signatures==baseline,'Batch size changed gate decisions'
                record=dict(batch=batch,trial_batch=trial_batch,seconds=seconds,
                    candidates_per_second=len(names)/seconds,pool_peak_gib=peak/2**30,verdicts_match=True)
            except cp.cuda.memory.OutOfMemoryError:
                record=dict(batch=batch,trial_batch=trial_batch,error='CUDA OOM')
            results.append(record);print(json.dumps(record),flush=True)
    (REPO/'tmp/q25_cuda/batch_sweep.json').write_text(json.dumps(results,indent=2))

def verify_composition(all_candidates=False):
    import fusion
    pool=json.loads((REPO/'tmp/q25_funnel/candidates/r6/full_pool.json').read_text())
    fast=[x['expr'] for x in pool if x['lane']!='python']
    names=fast if all_candidates else [fast[i*len(fast)//2048] for i in range(2048)]
    ref=ComboLib.load().ref();maxerr=0.;valid=0;t=time.perf_counter()
    previous=fusion.ENABLED
    try:
        for lo in range(0,len(names),engine.CUDA.candidate_batch):
            batch=names[lo:lo+engine.CUDA.candidate_batch]
            fusion.ENABLED=False
            baseline=engine.simulate_many(batch);bg=gates.run_batch(baseline,ref)
            fusion.ENABLED=True
            composed=engine.simulate_many(batch);cg=gates.run_batch(composed,ref)
            for a,b,ga,gb in zip(baseline,composed,bg,cg):
                assert a['ok']==b['ok'],a['expr']
                assert ga['passed']==gb['passed'],a['expr']
                if a['ok']:
                    valid+=1
                    err=float(cp.max(cp.abs(a['_W']-b['_W'])));maxerr=max(maxerr,err)
                    cp.testing.assert_allclose(a['_W'],b['_W'],atol=1e-12,rtol=1e-12)
                    cp.testing.assert_allclose(a['_rr'],b['_rr'],atol=1e-12,rtol=1e-12)
                    assert abs(ga.get('combo',{}).get('w',0)-gb.get('combo',{}).get('w',0))<1e-8
            del baseline,composed,bg,cg
            if lo%4096==0:print('composition',lo+len(batch),'/',len(names),'patterns',len(fusion._KERNELS),flush=True)
        report=dict(candidates=len(names),valid=valid,patterns=len(fusion._KERNELS),
                    max_book_error=maxerr,gate_verdicts_match=True,seconds=time.perf_counter()-t)
        (REPO/'tmp/q25_cuda/composition_verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:fusion.ENABLED=previous


if __name__=='__main__':
    main()
    if '--sources' in sys.argv:verify_sources()
    if '--benchmark' in sys.argv:benchmark()
    if '--batch-sweep' in sys.argv:batch_sweep()
    if '--composition' in sys.argv or '--all-composition' in sys.argv:
        verify_composition('--all-composition' in sys.argv)
