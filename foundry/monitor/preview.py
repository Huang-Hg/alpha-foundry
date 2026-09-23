"""HTML design preview, exported from the very frames the shell panel draws.

python -m foundry preview [STATUS ...]  ->  tmp/monitor/preview.html   (no STATUS: the one the dashboard shows)

The first snapshot is shown in every terminal size in a running state; the other snapshots
are shown as recorded. Every cell is laid out on an explicit character grid, so CJK text and
block glyphs line up exactly as they do in the terminal.
"""
import html
from itertools import groupby
import json
from pathlib import Path
import time

from rich.cells import cell_len

from . import cells, view
from .launcher import SOURCE, STATE

OUTPUT = STATE/'preview.html'
# Drawing glyphs are painted by CSS, so the preview does not depend on font coverage.
GLYPHS = {'━': 'h', '╸': 'hl', '─': 'l', '▄': 'lo', '▀': 'up', '▍': 'lb'}
CSS = """
:root{color-scheme:dark}
body{margin:0;background:#06080c;color:#aab6c5;font:14px/1.6 "Segoe UI","Microsoft YaHei",sans-serif}
main{width:max-content;margin:0 auto;padding:40px 32px 64px}
h1{margin:0 0 8px;color:#e8eef6;font-size:22px;font-weight:600}
h2{margin:44px 0 14px;color:#e8eef6;font-size:15px;font-weight:600}
p,li{max-width:86ch;margin:0 0 6px}ul{margin:0;padding-left:20px}code{font:13px "Cascadia Mono",Consolas,monospace;color:#e8eef6}
.tokens{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.token{display:flex;align-items:center;gap:8px;padding:6px 10px 6px 6px;border-radius:6px;background:#121822;font:12px "Cascadia Mono",Consolas,monospace}
.token i{width:18px;height:18px;border-radius:4px;box-shadow:inset 0 0 0 1px #ffffff1f}
figure{margin:0 0 34px}figcaption{margin-bottom:10px;font-size:13px}figcaption strong{color:#e8eef6;font-weight:600;margin-right:10px}
.term{display:inline-block;padding:10px 12px;border-radius:10px;background:%(base)s;box-shadow:0 0 0 1px #1d2633,0 28px 60px -24px #000;
  font:15.36px/20px "Cascadia Mono","Cascadia Code",Consolas,monospace}
.r{display:flex;height:20px;white-space:pre}.r>*{flex:none;height:100%%}
.r b{width:1ch;font-weight:inherit;text-align:center;background-repeat:no-repeat}.r b.w{width:2ch}
.r b.h,.r b.hl,.r b.l,.r b.lo,.r b.up,.r b.lb{background-image:linear-gradient(currentColor,currentColor)}
.r b.h{background-size:100%% 3px;background-position:center}.r b.hl{background-size:50%% 3px;background-position:left center}
.r b.l{background-size:100%% 1px;background-position:center}.r b.lo{background-size:100%% 50%%;background-position:bottom}
.r b.up{background-size:100%% 50%%;background-position:top}.r b.lb{background-size:38%% 100%%;background-position:left}
"""


def load(path):
    path = Path(path)
    return dict(json.loads(path.read_text(encoding='utf-8-sig')),
                _source_mtime=path.stat().st_mtime, _source_name=path.name)


def export(frame, styles):
    """One flex row per terminal row; ASCII runs stay text, every other cell gets its own box."""
    rows = []
    for row in frame:
        out = []
        for seg in row:
            name = styles.setdefault((seg.fg, seg.bg, seg.bold), f's{len(styles)}')
            # Runs of one kind share a box: ASCII as text, a repeated glyph as one painted strip.
            for kind, run in groupby(seg.text, key=lambda c: 'text' if ord(c) < 128 else c):
                run = ''.join(run)
                if kind == 'text':
                    out.append(f'<span class={name}>{html.escape(run)}</span>')
                elif kind in GLYPHS:
                    out.append(f'<b class="{name} {GLYPHS[kind]}" style="width:{len(run)}ch"></b>')
                else:
                    out.extend(f'<b class="{name}{" w" if cell_len(kind) == 2 else ""}">{kind}</b>' for _ in run)
        rows.append(f'<div class=r>{"".join(out)}</div>')
    return '<div class=term>'+''.join(rows)+'</div>'


def running(st):
    """The first snapshot as if it had just been written: a search run mid-way through, the service as it was."""
    if st.get('kind') == 'service':
        return dict(st, _source_mtime=time.time()), f'{st["_source_name"]} · 改写写入时刻，只为演示在线态'
    return (dict(st, phase='evaluating', epoch=(st['epochs']+1)//2, _source_mtime=time.time()),
            '运行中（由首个快照改写 phase / epoch，只为演示运行态）')


def page(paths):
    first = load(paths[0])
    live, note = running(first)
    shots = [('120 × 30', f'Windows Terminal 默认尺寸 · {note}', live, 120, 30),
             ('120 × 36', f'{first["_source_name"]} · 原样', first, 120, 36)]
    shots += [('120 × 30', f'{s["_source_name"]} · 原样', s, 120, 30) for s in map(load, paths[1:])]
    shots += [('120 × 30', '尚无数据源', dict(_source_name='live.json'), 120, 30)]
    shots += [(f'{c} × {r}', '同一运行态 · 窗口缩小时整块让位，不裁字', live, c, r) for c, r in ((100, 30), (80, 24), (60, 15))]
    styles = {}
    figures = ''.join(f'<figure><figcaption><strong>{size}</strong>{html.escape(note)}</figcaption>'
                      f'{export(view.compose(st, cols, rows, tick=2, interval=.5), styles)}</figure>'
                      for size, note, st, cols, rows in shots)
    rules = ''.join(f'.{name}{{color:{fg};background-color:{bg}{";font-weight:700" if bold else ""}}}'
                    for (fg, bg, bold), name in styles.items())
    tokens = ''.join(f'<span class=token><i style="background:{getattr(cells, name)}"></i>{name} {getattr(cells, name)}</span>'
                     for name in cells.PALETTE)
    return f"""<!doctype html><html lang=zh-CN><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>监控面板 · 设计预览</title>
<style>{CSS % dict(base=cells.BASE)}{rules}</style><main>
<h1>监控面板 · 设计预览</h1>
<p>下面每一帧都由 <code>foundry/monitor/view.py · compose()</code> 逐格生成，shell 面板画的是同一帧：预览里看到什么，终端里就是什么。
版面分层：<code>schema.py</code> 生产方写入的状态契约、<code>cells.py</code> 绘制原语、<code>objectives.py</code> 各挖掘目标的漏斗 / 判定 / 候选列、<code>components.py</code> 组件；<code>view.py</code> 按状态选屏（搜索 / 评价服务），每块屏只管窗口大小对应放哪些组件。</p>
<h2>设计规范</h2><ul>
<li>一种强调色（ACCENT）加四种语义色：绿 = 改善 / 产出，琥珀 = 瓶颈 / 数据停滞，红 = 失败 / 恶化，紫 / 青只用于阶段耗时图例；其余全部是灰阶。</li>
<li>用三层底色（窗口 / 卡片 / 条栏）分区，不画框线；卡片上下各留半格呼吸空间。</li>
<li>三级文字：粗体亮白 = 读数，次灰 = 标签，弱灰 = 单位与注释；数字一律右对齐、等宽。</li>
<li>统一节奏：外边距 2 格、卡片内距 2 格、卡片间距 2 格；进度与漏斗统一用细轨。</li>
<li>窗口变小时按“候选表 → 性能卡 → 漏斗”整块让位，核心读数与瓶颈诊断始终保留。</li></ul>
<div class=tokens>{tokens}</div>
<h2>帧</h2>{figures}</main></html>"""


def write(statuses):
    """statuses: list[Path]; empty: the status the dashboard currently shows. -> the preview file."""
    paths = statuses or [Path(json.loads(SOURCE.read_text(encoding='utf-8'))['status_path'])]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page(paths), encoding='utf-8')
    return OUTPUT
