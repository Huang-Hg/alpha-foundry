"""All registry operators: one measured call per real-input condition on the authorized GPU."""
import json,time,csv,sys
from pathlib import Path
import numpy as np
import cupy as cp
from foundry.config import load_config
from foundry.contracts import Study
from foundry.runtime import load_study,prepare_backend
from foundry.ops.registry import OPS
from test_portable import PortableParity,FIELDS
from precision_probe.cuda_single import prepare as prepare_single
from precision_probe.cuda_paired import prepare as prepare_pair

out=Path('output/operator66');out.mkdir(parents=True,exist_ok=True)
tag=sys.argv[1]
selected=tuple(sys.argv[2:]) or tuple(OPS)
cfg=load_config();cpu=prepare_backend('cpu');gpu=prepare_backend('cuda');single=prepare_single()
paired=prepare_pair()
loaded=load_study(cfg,Study('us_equity',cfg.us_equity.start,cfg.us_equity.end),cpu)
cases=PortableParity();cases.x={k:np.ascontiguousarray(loaded.panel[k]) for k in FIELDS}
records=[]
for name,description,args in cases.cases():
    if name not in selected:continue
    start=time.perf_counter();expected=cpu.impl[name](*args);cpu_ms=(time.perf_counter()-start)*1000
    row=dict(operator=name,case=description,shape=expected.shape,cpu_ms=cpu_ms,modes={})
    for mode,implementation,dtype in [('production',gpu.impl,cp.float64),('single',single.IMPL,cp.float32),('paired_trial',paired.IMPL,cp.float64)]:
        values=tuple(cp.asarray(v,dtype=dtype) if isinstance(v,np.ndarray) else v for v in args)
        implementation[name](*values);cp.cuda.get_current_stream().synchronize()
        begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record();started=time.perf_counter()
        actual=implementation[name](*values);end.record();end.synchronize()
        wall_ms=(time.perf_counter()-started)*1000;device_ms=cp.cuda.get_elapsed_time(begin,end)
        host=cp.asnumpy(actual);finite=np.isfinite(host)&np.isfinite(expected)
        error=np.abs(host[finite]-expected[finite]);scale=np.maximum(1.,np.abs(expected[finite]))
        row['modes'][mode]=dict(wall_ms=wall_ms,device_ms=device_ms,
            nan_mismatch=int(np.count_nonzero(np.isnan(host)!=np.isnan(expected))),
            inf_mismatch=int(np.count_nonzero(np.isinf(host)!=np.isinf(expected)))+
                         int(np.count_nonzero(np.isinf(host)&np.isinf(expected)&(np.signbit(host)!=np.signbit(expected)))),
            max_abs=float(error.max(initial=0.)),max_scaled=float((error/scale).max(initial=0.)),
            predicate_mismatch=int(np.count_nonzero((host>0)!=(expected>0))),cells=host.size)
    records.append(row)
summary=[]
for name in selected:
    rows=[r for r in records if r['operator']==name]
    item=dict(operator=name,cases=len(rows),cpu_ms=sum(r['cpu_ms'] for r in rows))
    for mode in ('production','single','paired_trial'):
        for key in ('wall_ms','device_ms','nan_mismatch','inf_mismatch','predicate_mismatch'):
            item[mode+'_'+key]=sum(r['modes'][mode][key] for r in rows)
        for key in ('max_abs','max_scaled'):
            item[mode+'_'+key]=max(r['modes'][mode][key] for r in rows)
    summary.append(item)
    print(json.dumps(item),flush=True)
(out/f'{tag}.json').write_text(json.dumps(dict(summary=summary,cases=records),indent=2))
with (out/f'{tag}.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
assert len(summary)==len(selected)
single.close();gpu.close();paired.close();cpu.close()
print('OPERATOR AUDIT COMPLETE',len(summary),flush=True)
