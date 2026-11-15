"""Real 4096-formula research workload, covering every registry operator."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json,random,time,sys
import numpy as np
import cupy as cp
import foundry.runtime as assembly
from foundry.config import load_config
from foundry.contracts import Study
from foundry.ops.registry import OPS
from foundry.expr.parse import parse,validate
from foundry.candidates import Formula
from foundry.research_types import Candidate
from foundry.research import Research
from foundry.portfolio.library import Library
from foundry.adapters.library import MemoryLibraryStore
from foundry.adapters.arrays import MemoryArrayStore
from foundry.adapters.serialization import evaluation_result,json_value
from precision_probe.cuda_single import prepare as single_ops

out=Path('output/operator66')/sys.argv[1];out.mkdir(parents=True,exist_ok=True)
cfg=load_config();study=Study('us_equity',cfg.us_equity.start,cfg.us_equity.end)
with assembly.open_runtime(replace(cfg,execution=replace(cfg.execution,backend='cpu')),study) as definition:
    fields=tuple(definition.panel.vocabulary)
rng=random.Random(25);windows=(2,3,5,7,10,14,20,30,45,60,90,120)
def panel():
    x,y=rng.sample(fields,2);w=rng.choice(windows);k=rng.choice((-.5,-.1,.05,.2,.5))
    return rng.choice((x,f'abs({x})',f'neg({x})',f'ts_mean({x},{w})',f'ts_delta({x},{w})',
        f'ts_rank({x},{w})',f'add_const({x},{k})',f's_log_1p({x})',
        f'div({x},add_const(abs({y}),0.1))',f'mul({x},ts_ref({y},{w}))'))
expressions={};i=0
while len(expressions)<4096:
    name=list(OPS)[i%len(OPS)];i+=1;kind=OPS[name].kind
    x,y,z=panel(),panel(),panel();w=rng.choice(windows)
    args=[x]
    if kind in ('binary','pair','pair_param'):args.append(y)
    if kind=='ternary':args.extend((y,z))
    if kind in ('ts','pair','ts_param','pair_param'):args.append(str(w))
    if kind=='binary_const':args.append(str(rng.choice((.01,.05,.2)) if name=='hump' else rng.choice((-2.,-.5,.05,.5,1.7))))
    if kind=='ts_param':args.append(str(rng.choice((0,1,2)) if name=='ts_quantile' else rng.choice((.5,.9,1.)) if name=='ts_decay_exp_window' else rng.choice((-.5,0,.25))))
    if kind=='pair_param':args.append(str(rng.randrange(5)))
    expr=f'{name}('+','.join(args)+')'
    checked=validate(expr,fields,cfg.expr.max_cost)
    if checked['ok']:expressions.setdefault(checked['tree'].hash,expr)
exprs=list(expressions.values());covered={n.op for e in exprs for n in parse(e,fields).root.walk() if not n.leaf}
assert covered==set(OPS),(covered,set(OPS)-covered)
(out/'formulas.json').write_text(json.dumps(exprs,indent=2))
candidates=tuple(Candidate(f'f-{i}',Formula(e)) for i,e in enumerate(exprs))
native=assembly.prepare_backend
def probe(name):
    backend=native(name);ops=single_ops()
    def cast(fn):return lambda *args:fn(*[a.astype(cp.float32,copy=False) if isinstance(a,cp.ndarray) else a for a in args])
    return replace(backend,impl={k:cast(v) for k,v in ops.IMPL.items()},close=lambda:(ops.close(),backend.close()))
for mode in sys.argv[2:]:
    settings=replace(cfg,execution=replace(cfg.execution,backend='cpu' if mode=='cpu_reference' else 'cuda'),
                     cuda=replace(cfg.cuda,precision='mixed' if mode=='single' else 'paired'))
    prepare_start=time.perf_counter()
    with patch.object(assembly,'prepare_backend',probe if mode=='single' else native):
        with assembly.open_runtime(settings,study) as runtime:
            research=Research(runtime.engine,MemoryLibraryStore(Library((),study)),MemoryArrayStore())
            e=runtime.engine;e.resources.synchronize();prepare=time.perf_counter()-prepare_start
            assert all(validate(expr,e.vocabulary,e.expr_cfg.max_cost)['ok'] for expr in exprs)
            # Warm all operator signatures and account paths, with no portfolio writes.
            start=time.perf_counter();research.evaluate(candidates,research.context.id);e.resources.synchronize()
            warmup=time.perf_counter()-start;e._reports.clear()
            print('BEGIN',mode,'4096 formulas; 66 operators',flush=True)
            start=time.perf_counter();query=research.evaluate(candidates,research.context.id);e.resources.synchronize()
            qt=time.perf_counter()-start
            print('QUERY',mode,qt,flush=True)
            start=time.perf_counter();commit=research.commit(candidates,research.context.id);e.resources.synchronize()
            ct=time.perf_counter()-start
            row=dict(mode=mode,market='us_equity',shape=e.panel.shape,candidates=len(candidates),operator_count=len(covered),
                prepare=prepare,warmup=warmup,query=qt,commit=ct,query_commit=qt+ct,
                accepted=sum(v.accepted for v in commit.rows),valid=sum(v.row.ok for v in query.rows))
            if mode!='cpu_reference':row['device_pool_reserved_mib']=e.resources.pool.total_bytes()/1048576
            data=json_value(dict(timings=row,query=evaluation_result(query),commit=evaluation_result(commit)))
            (out/f'{mode}.json').write_text(json.dumps(data,allow_nan=False))
            print(json.dumps(row),flush=True)
print('4096 FORMULA FLOW COMPLETE',flush=True)
