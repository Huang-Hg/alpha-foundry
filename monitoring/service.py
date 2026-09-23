"""The evaluation service screen: status written by tmp/q25_funnel/daemon.py (kind='service').

The search screen follows the epochs of one run; this one follows a queue of jobs that agents submit
(one hypothesis file per round). Same drawing layer and card blocks; a submitter is the directory the
submitted file lives in, so every agent's rounds add up under its own name.
"""
from pathlib import PurePosixPath
import time

from . import objectives
from .cells import (ACCENT, AMBER, BASE, GREEN, INK, INK2, INK3, INK4, RAISED, SPIN, SURFACE, TEAL, VIOLET,
                    Row, Seg, ago, badge, bar, clock, first_fit, fit, num, size)
from .components import STALE_SECONDS, Card, Component, Footer, Kpis, Screen

MAX_JOBS = 12
OPS = {'family': '假设族', 'eval': '表达式', 'python': 'Python'}
VERDICTS = (('成立', GREEN), ('稀疏', AMBER), ('尖峰', VIOLET), ('无效', INK3), ('通过', GREEN), ('未过', INK3))


def since(st, t):
    """Seconds from a service timestamp to this frame: the service clock is not ours, only differences are."""
    return st['now']-t + (time.time()-st['_source_mtime'] if '_source_mtime' in st else 0.)


def who(source):
    """(submitter, file) of a submitted path."""
    path = PurePosixPath((source or '').replace('\\', '/'))
    return path.parent.name or '—', path.name or '—'


def verdicts(counts, bg):
    out = []
    for name, color in VERDICTS:
        if counts.get(name):
            out += [Seg('  '*bool(out), bg=bg), Seg(f'{name} ', INK3, bg), Seg(str(counts[name]), color, bg, True)]
    return out or [Seg('—', INK4, bg)]


class Header(Component):
    def rows(self, screen, cols):
        st = screen.st
        if screen.stale is not None:
            label, color, glyph = f'数据停滞 {ago(screen.stale)}', AMBER, '!'
        elif st.get('phase') == 'warm':
            label, color, glyph = '预热中', ACCENT, None
        elif st.get('cur'):
            label, color, glyph = '计算中', ACCENT, None
        elif st.get('phase') == 'serve':
            label, color, glyph = '空闲', GREEN, '○'
        else:
            label, color, glyph = '等待数据源', INK3, '○'
        right = [badge(f'{glyph or SPIN[screen.tick % len(SPIN)]} {label}', color, RAISED)]
        if cols >= 64:
            right += [Seg('   运行 ', INK3, RAISED), Seg(clock(st.get('elapsed_seconds')), INK2, RAISED)]
        right += [Seg('   '+time.strftime('%H:%M:%S')+' ', INK, RAISED, True)]
        title = [Seg(' ◆ ', ACCENT, RAISED, True), Seg('Q25 评价服务', INK, RAISED, True)]
        device = (st.get('device') or '').replace('Tesla ', '').replace('-PCIE-32GB', ' 32GB')
        left = first_fit(cols-size(right)-1, title + [Seg('   '+device, INK3, RAISED)], title)
        return [fit(left, cols-size(right), RAISED) + right]


class Job(Component):
    """Two rows: the job on the GPU (whose, which stage, how far), or what the service is waiting for."""
    def rows(self, screen, width):
        st, cur = screen.st, screen.st.get('cur')
        if st.get('phase') == 'warm':
            return [Row().add('预热  ', INK3).add(st.get('warm') or '', INK2).cells(width), bar(0, width, ACCENT, BASE)]
        if not cur:
            jobs = st.get('jobs') or []
            line = Row().add('空闲', INK2, True).add(f'  ·  队列 {num(st.get("queue"))}', INK3)
            if jobs:
                line.end(f'上一个作业 {ago(since(st, jobs[-1]["t"]))} 前', INK3)
            return [line.cells(width), bar(0, width, ACCENT, BASE)]
        owner, name = who(cur.get('source'))
        unit = '条公式' if cur['op'] == 'family' else '条'
        line = Row().add(OPS.get(cur['op'], cur['op']), ACCENT, True).add(f'  {owner}', INK, True).add(f' / {name}', INK3) \
            .add(f'  ·  {num(cur.get("n"))} {unit}', INK2)
        line.end(f'已用 {clock(since(st, cur["t0"]))}', INK3)
        done, total = cur.get('done') or 0, cur.get('total') or 0
        right = [Seg(f'  {cur.get("stage") or "排队"}  ', INK2), Seg(f'{num(done)}/{num(total)}', INK3)]
        return [line.cells(width), bar(done/total if total else 0, width-size(right), ACCENT, BASE, TEAL) + right]


class Tiles(Kpis):
    def tiles(self, screen):
        st = screen.st
        p = st.get('portfolio') or {}
        elapsed, busy = st.get('elapsed_seconds'), st.get('busy_seconds')
        return [
            ('组合 Sharpe', num(p.get('sharpe'), '.3f'), INK,
             [[Seg(f"{num(p.get('n'))} 个成员 · CAGR {num(p.get('cagr'), '.1%')} · 回撤 {num(p.get('maxdd'), '.1%')}", INK3, SURFACE)],
              [Seg(f"{num(p.get('n'))} 个成员", INK3, SURFACE)]]),
            ('排队作业', num(st.get('queue')), AMBER if st.get('queue') else INK,
             [[Seg(f"等结果的客户端 {num(st.get('clients'))}", INK3, SURFACE)], [Seg(f"客户端 {num(st.get('clients'))}", INK3, SURFACE)]]),
            ('已评价变体', num(st.get('exprs')), INK,
             [[Seg(f"作业 {num(st.get('served'))} · GPU 批 {num(st.get('batches'))}", INK3, SURFACE)],
              [Seg(f"作业 {num(st.get('served'))}", INK3, SURFACE)]]),
            ('GPU 忙碌', num(busy/elapsed if busy is not None and elapsed else None, '.0%'), INK,
             [[Seg(f"显存 {st.get('vram') or '—'}", INK3, SURFACE)]]),
        ]


class Table(Card):
    """Rows of cells under a heading. `columns` = (name, width, cell(item, bg), min_inner): width 0 is the one
    flexible column, a column shows only when the card is at least min_inner wide."""
    columns = ()

    def __init__(self, items):
        self.items = items

    def height(self, screen):
        return 2+len(self.items)

    def body(self, screen, inner):
        columns = [c for c in self.columns if inner >= c[3]]
        flexible = inner-sum(c[1]+1 for c in columns if c[1])
        widths = [c[1] or flexible for c in columns]
        rows = []
        for item in (None,)+tuple(self.items):
            line = []
            for (name, _, cell, _), w in zip(columns, widths):
                content = [Seg(name, INK4, SURFACE)] if item is None else cell(item, SURFACE)
                line += [Seg(' ', bg=SURFACE)]*bool(line) + fit(content, w, SURFACE)
            rows.append(Row(SURFACE).extend(line))
        return rows


class Submitters(Table):
    """Every submitter's rounds so far, most recent first. Items: (submitter, totals)."""
    title = '提交者'
    columns = (('目录', 0, lambda it, bg: [Seg(it[0], INK, bg, True)], 0),
               ('轮次', 5, lambda it, bg: [Seg(num(it[1]['rounds']), INK2, bg)], 0),
               ('公式', 6, lambda it, bg: [Seg(num(it[1]['formulas']), INK2, bg)], 70),
               ('变体', 9, lambda it, bg: [Seg(num(it[1]['variants']), INK2, bg)], 0),
               ('判决', 30, lambda it, bg: verdicts(it[1]['verdicts'], bg), 0))

    def note(self, screen):
        return '按提交文件所在目录归属 · 服务启动以来累计'


class Jobs(Table):
    """Finished jobs, newest first. Items: (job, seconds since it finished)."""
    title = '最近作业'
    columns = (('完成', 6, lambda it, bg: [Seg(ago(it[1])+' 前', INK3, bg)], 0),
               ('提交', 0, lambda it, bg: [Seg(who(it[0]['source'])[0], INK, bg), Seg(' / '+who(it[0]['source'])[1], INK3, bg)], 0),
               ('类型', 6, lambda it, bg: [Seg(OPS.get(it[0]['op'], it[0]['op']), INK2, bg)], 80),
               ('数量', 6, lambda it, bg: [Seg(num(it[0].get('n')), INK2, bg)], 70),
               ('变体', 8, lambda it, bg: [Seg(num(it[0].get('variants')), INK2, bg)], 0),
               ('耗时', 7, lambda it, bg: [Seg(f"{it[0]['dt']:.1f}s", INK2, bg)], 90),
               ('判决', 30, lambda it, bg: verdicts(it[0].get('verdicts') or {}, bg) if it[0]['ok']
                else [Seg('失败', AMBER, bg)], 0))

    def note(self, screen):
        return '最新在前'


class ServiceDashboard:
    header, footer, job, tiles = Header(), Footer(), Job(), Tiles()

    def compose(self, st, cols, rows=None, tick=0, edges=True, interval=None):
        age = time.time()-st['_source_mtime'] if '_source_mtime' in st else None
        stale = age if age is not None and age > STALE_SECONDS else None
        limit = rows or 10**6
        tight = limit < 12
        screen = Screen(st, objectives.of(st), tick, edges and not tight, tight, age, stale, interval)
        margin = 2 if cols >= 72 else 1
        width = cols-2*margin
        body = self.job.rows(screen, width) + [[]]*(not tight) + self.tiles.rows(screen, width)
        fixed = 2 + (not tight)                  # header, its spacer, footer
        chrome = 3+2*screen.edges                # spacer, card title, column heading
        sources = sorted((st.get('sources') or {}).items(), key=lambda kv: -kv[1]['last'])
        room = limit-fixed-len(body)-chrome
        if sources and width >= 60 and room > 0:
            body += [[]] + Submitters(sources[:room]).rows(screen, width)
        jobs = st.get('jobs') or []
        count = min(len(jobs), MAX_JOBS, limit-fixed-len(body)-chrome)
        if count > 0 and width >= 60:
            body += [[]] + Jobs([(j, since(st, j['t'])) for j in reversed(jobs[-count:])]).rows(screen, width)
        side = [Seg(' '*margin)]
        frame = self.header.rows(screen, cols) + [[]]*(not tight) + [side + (row or [Seg(' '*width)]) + side for row in body]
        frame = frame[:limit-1] if rows else frame
        frame += [[]]*((rows or 0)-len(frame)-1)
        return [row or [Seg(' '*cols)] for row in frame] + self.footer.rows(screen, cols)


SERVICE = ServiceDashboard()
