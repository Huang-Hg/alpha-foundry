# 拼装 + 分块提交:_lib.py + _rK_body.py → python_rK.py(全量,供引用)+ 每块 ≤6 个 alpha 的分块文件
# (主控 2026-09-19 通知:每次 --python 调用 ≤6 个函数),逐块串行走 run.py(云端),合并为 candidates/r5/python_rK.json。
# 任一块 run.py 失败 / 云端报错 ⇒ 立即停下(不绕路本机)。
# 用法:python _run.py r3 [只跑这些函数名前缀...]
import json
import re
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
root = here.parents[2]                       # tmp/q25_funnel
k = sys.argv[1]
only = sys.argv[2:]
body = (here / f'_{k}_body.py').read_text(encoding='utf-8')
head, _, rest = body.partition('# ==== BODY ====\n')
full = head + (here / '_lib.py').read_text(encoding='utf-8') + '\n\n' + rest
(here / f'python_{k}.py').write_text(full, encoding='utf-8')

ns = {}
exec(compile(full, f'python_{k}.py', 'exec'), ns)
names = [n for n, v in ns.items() if callable(v) and n.startswith('alpha')]
if only:
    names = [n for n in names if any(n.startswith(o) for o in only)]
print(f'{k}: {len(names)} 个 alpha')
chunks = [names[i:i + 24] for i in range(0, len(names), 24)]
rows = []
outj = root / 'candidates' / 'r5' / f'python_{k}.json'
if outj.exists() and only:
    rows = [r for r in json.loads(outj.read_text(encoding='utf-8')) if r['expr'] not in names]
for i, ch in enumerate(chunks):
    tail = ('\n\n_KEEP = ' + repr(set(ch)) + '\nfor _n in [_n for _n in list(globals()) '
            'if _n.startswith("alpha") and _n not in _KEEP]:\n    del globals()[_n]\n')
    f = here / '_chunks' / f'{k}_{i:02d}.py'
    f.parent.mkdir(exist_ok=True)
    f.write_text(full + tail, encoding='utf-8')
    j = here / '_chunks' / f'{k}_{i:02d}.json'
    j.unlink(missing_ok=True)
    cmd = [sys.executable, '-X', 'utf8', str(root / 'run.py'), '--no-panel', '--quiet',
           '--python', str(f), '--json', str(j)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', cwd=str(root.parents[1]))
    print(f'  块 {i + 1}/{len(chunks)} {ch[0]}..  rc={p.returncode}  {p.stdout.strip().splitlines()[-1:] }')
    if p.returncode != 0 or not j.exists():
        print(p.stdout[-3000:], p.stderr[-3000:])
        sys.exit(f'块 {i} 失败 —— 停下,不绕路本机')
    got = json.loads(j.read_text(encoding='utf-8'))
    bad = [r for r in got if not r.get('ok') and re.search('云端|超时|失败', r.get('error') or '')]
    if bad:
        sys.exit(f'云端作业失败:{bad[0]["error"][:500]}')
    rows += got
outj.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print('wrote', outj, len(rows))
