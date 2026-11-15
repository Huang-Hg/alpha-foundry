"""Run the community validation tools and publish one coherent result bundle."""
import json
from pathlib import Path
import subprocess
import sys

from foundry.parameters import ROOT, configure
from foundry.adapters.serialization import parameter_values
from foundry.benchmark_report import render


def run(output, *, sizes=(1, 64, 4096), **parameters):
    """Compare CPU and CUDA once per workload, retaining all validation evidence."""
    if 'backend' in parameters:
        raise ValueError('benchmark compares both cpu and cuda; omit backend')
    if not sizes or any(type(size) is not int or not 1 <= size <= 4096 for size in sizes):
        raise ValueError('sizes must contain candidate counts between 1 and 4096')
    settings = configure(**parameters)
    target = Path(output).resolve()
    target.mkdir(parents=True, exist_ok=False)
    plan = dict(parameters=parameter_values(settings), sizes=list(sizes),
                measured_rounds=1, warmup_rounds=1, initial_portfolio='equal_weight',
                storage='memory', transport='in_process')
    (target / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    command = [sys.executable, '-m', 'pytest', str(ROOT / 'validation'), '-q',
               '--foundry-run', str(target), '--benchmark-json', str(target / 'benchmark.json'),
               '--benchmark-columns=mean', '--benchmark-sort=name',
               '--junitxml', str(target / 'junit.xml'), '--html', str(target / 'checks.html'),
               '--self-contained-html']
    with (target / 'run.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            if 'validation/' in line or 'failed' in line or 'passed' in line or 'ERROR' in line:
                print(line.rstrip(), flush=True)
        code = process.wait()
    passed = render(target, exit_code=code)
    if code or not passed:
        raise RuntimeError(f'Benchmark failed; inspect {target / "report.zh-CN.md"}')
    return target
