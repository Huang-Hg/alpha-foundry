"""Inspect every formula book/return on Colab; separate from timing."""
from pathlib import Path
from dataclasses import replace
import json,sys
import numpy as np
import cupy as cp
from foundry.config import load_config
from foundry.contracts import Study
from foundry.runtime import open_runtime
cfg=load_config();study=Study('us_equity',cfg.us_equity.start,cfg.us_equity.end)
expressions=json.loads(Path(sys.argv[1]).read_text())
result=dict(formulas=0,validity_mismatch=[],position_sign_mismatch=0,book_max=0.,return_max=0.,details=[])
with open_runtime(replace(cfg,execution=replace(cfg.execution,backend='cpu')),study) as host,open_runtime(cfg,study) as gpu:
    for begin in range(0,len(expressions),64):
        source=expressions[begin:begin+64]
        a,b=host.engine.evaluate(source),gpu.engine.evaluate(source)
        for offset,(h,g) in enumerate(zip(a,b)):
            index=begin+offset;result['formulas']+=1
            assert h.tree is not None and g.tree is not None,(index,h.error,g.error)
            if (h.ok,h.valid_returns)!=(g.ok,g.valid_returns):result['validity_mismatch'].append(index)
            book=cp.asnumpy(g.book);returns=cp.asnumpy(g.returns)
            signs=int(np.count_nonzero(np.sign(h.book)!=np.sign(book)))
            err=float(np.max(np.abs(book-h.book)));ret=float(np.max(np.abs(returns-h.returns)))
            result['position_sign_mismatch']+=signs
            result['book_max']=max(result['book_max'],err);result['return_max']=max(result['return_max'],ret)
            if signs or err>1e-10 or ret>cfg.cuda.return_atol:
                result['details'].append(dict(id=index,expression=source[offset],signs=signs,book_max=err,return_max=ret))
Path(sys.argv[2]).write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
assert not result['validity_mismatch']
assert result['position_sign_mismatch']==0
assert result['return_max']<=cfg.cuda.return_atol
