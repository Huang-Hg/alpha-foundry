# 拼装自包含 alpha 文件:_lib.py(共用原语)+ _rK_body.py(本轮定义)→ python_rK.py
# 用法:python _build.py r2
import sys
from pathlib import Path

here = Path(__file__).parent
k = sys.argv[1]
body = (here / f'_{k}_body.py').read_text(encoding='utf-8')
head, _, rest = body.partition('# ==== BODY ====\n')
(here / f'python_{k}.py').write_text(head + (here / '_lib.py').read_text(encoding='utf-8') + '\n\n' + rest,
                                     encoding='utf-8')
print('wrote', here / f'python_{k}.py')
