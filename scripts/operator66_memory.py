"""Separate memory-profile condition; never used as a timing denominator."""
from pathlib import Path
import json,subprocess,sys
out=Path('output/operator66/memory');out.mkdir(parents=True,exist_ok=True)
with (out/'device-memory-mib.csv').open('w') as log:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--loop-ms=50'],stdout=log)
    status=subprocess.run([sys.executable,'scripts/operator66_flow.py','memory','production']).returncode
    monitor.terminate();monitor.wait()
assert status==0
values=[int(v) for v in (out/'device-memory-mib.csv').read_text().splitlines()]
result=dict(sample_period_ms=50,samples=len(values),sampled_peak_device_mib=max(values),scope='prepare, warmup, query and ordered commit; entire device')
(out/'memory.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
