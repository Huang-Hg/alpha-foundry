"""Generate isolated comparison sources locally; never edits production files."""
from pathlib import Path
import re

root=Path('scripts/precision_probe')
literal=re.compile(r'(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])')
def single(source):
    source=source.replace('__longlong_as_double((long long)0x7ff8000000000000ULL)', '__int_as_float(0x7fc00000)')
    source=re.sub(r'\bdouble\b','float',source)
    source=literal.sub(lambda m:m[1]+'f',source)
    for name in ('fabs','sqrt','log','log1p','exp','expm1','tanh','pow','fma','erf','erfc','tan','floor','ceil','nearbyint','frexp','ldexp'):
        source=re.sub(r'\b'+name+r'(?=\s*\()',name+'f',source)
        source=source.replace('std::'+name+'f(', 'std::'+name+'(')
    return source

cuda=(Path('foundry/ops/cuda.py').read_text(encoding='utf-8')
      .replace('float64','float32').replace('double','float').replace('_ELEM = 8','_ELEM = 4')
      .replace('__longlong_as_float((long long)0x7ff8000000000000ULL)','__int_as_float(0x7fc00000)'))
(root/'cuda_single.py').write_text(cuda,encoding='utf-8')
(root/'two_point.hpp').write_text(single(Path('foundry/ops/two_point.hpp').read_text()),encoding='utf-8')
for name in ('ops.cu','paths.cu','select.cu'):
    path=root/'kernels'/name;path.parent.mkdir(exist_ok=True)
    path.write_text(single((Path('foundry/ops/kernels')/name).read_text(encoding='utf-8')),encoding='utf-8')
print('Prepared isolated single-precision probe sources; no code was executed.')
