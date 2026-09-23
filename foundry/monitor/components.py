"""The blocks of the dashboard. Each is an object with rows(ctx, width) -> cell rows exactly `width` wide.

A component reads the status through the contract in schema.py and asks the objective (objectives.py) for
anything that depends on what a search run optimises; it never decides whether it is shown - that is the
layout's job (view.py). Three sections: blocks both screens use, the search blocks, the service blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from os.path import commonprefix
from pathlib import PurePosixPath, PureWindowsPath
import re
import time
from typing import Callable, NamedTuple, Optional

from rich.cells import set_cell_size

from .cells import (ACCENT, AMBER, BASE, GREEN, INK, INK2, INK3, INK4, RAISED, RED, SPIN, SURFACE, TEAL, TRACK, VIOLET,
                    Row, Seg, ago, badge, bar, beside, card, clock, first_fit, fit, mix, num, seconds, size, tone, wrap)
from .objectives import Objective
from .schema import ENDED, Status

WAITING = 'waiting'                          # the phase of a status that has reported none yet, or of no status
FRESH_SECONDS = 5                            # a snapshot this recent counts as "now" in the footer
STALE_SECONDS = 30                           # a live source this quiet is flagged in the header


@dataclass(frozen=True)
class Context:
    """Everything a component may read while one frame is drawn."""
    st: Status
    tick: int
    edges: bool                              # half-block card edges (off on tight or colourless screens)
    tight: bool                              # no spacer rows, no stage rail
    age: Optional[float]                     # seconds since the status file was written
    stale: Optional[float]                   # the same, once a live status has gone quiet for too long
    interval: Optional[float]


@dataclass(frozen=True)
class SearchContext(Context):
    objective: Objective


class Component:
    def rows(self, ctx, width):
        raise NotImplementedError


def gpu_name(device):
    """The header has little room: 'Tesla V100S-PCIE-32GB' -> 'V100S 32GB'."""
    return device.replace('Tesla ', '').replace('-PCIE-32GB', ' 32GB')


# ------------------------------------------------------------------ shared by both screens

class Header(Component):
    """Title (and the market, when the status names one), then as many facts as fit; on the right the state badge,
    how long the producer has been up and the wall clock."""
    title = elapsed = ''

    def state(self, ctx):
        """(label, colour, glyph) of the badge; glyph None spins."""
        raise NotImplementedError

    def facts(self, ctx):
        """Fact tuples beside the title, richest first: a narrower window drops whole facts."""
        raise NotImplementedError

    def rows(self, ctx, cols):
        st = ctx.st
        label, color, glyph = (f'数据停滞 {ago(ctx.stale)}', AMBER, '!') if ctx.stale is not None else self.state(ctx)
        right = [badge(f'{glyph or SPIN[ctx.tick % len(SPIN)]} {label}', color, RAISED)]
        if cols >= 64:
            right += [Seg(f'   {self.elapsed} ', INK3, RAISED), Seg(clock(st.get('elapsed_seconds')), INK2, RAISED)]
        right += [Seg('   '+time.strftime('%H:%M:%S')+' ', INK, RAISED, True)]
        title = [Seg(' ◆ ', ACCENT, RAISED, True), Seg(self.title, INK, RAISED, True)]
        named = title + [Seg(' · ', INK3, RAISED), Seg(st['market'].upper(), INK2, RAISED, True)] if 'market' in st else title
        left = first_fit(cols-size(right)-1, *(named + [Seg('   '+' · '.join(filter(None, parts)), INK3, RAISED)]
                                               for parts in self.facts(ctx)), named, title)
        return [fit(left, cols-size(right), RAISED) + right]


class Footer(Component):
    def rows(self, ctx, cols):
        st, age, stale, interval = ctx.st, ctx.age, ctx.stale, ctx.interval
        live = age is not None and stale is None and st.get('phase', WAITING) not in ENDED
        right = [Seg(f'{interval:g}s 刷新 · ' if interval and cols >= 72 else '', INK3, RAISED), Seg('Ctrl+C 退出 ', INK3, RAISED)]
        dot = [Seg(' ● ', AMBER if stale is not None else GREEN if live else INK4, RAISED)]
        # A search run is named by its directory (the live mirror adds it), anything else by its file.
        name = dot + [Seg(PureWindowsPath(st['run']).name if st.get('run') else st.get('_source_name', '等待数据源'), INK2, RAISED)]
        options = [name]
        if live and age <= FRESH_SECONDS:
            # Snapshots land every couple of seconds: while they keep coming the stamp simply runs with
            # the local clock, instead of jumping from push to push with a "1s ago" flickering behind it.
            options = [name + [Seg('   '+time.strftime('%H:%M:%S'), INK3, RAISED)], name]
        elif age is not None:
            stamp = time.strftime('%H:%M:%S', time.localtime(st['_source_mtime']))
            options = [name + [Seg(f'   {stamp} · {ago(age)} 前', INK3, RAISED)],
                       name + [Seg(f'   {ago(age)} 前', INK3, RAISED)], name]
        return [fit(first_fit(cols-size(right)-1, *options), cols-size(right), RAISED) + right]


class Kpis(Component):
    """Four tiles: (label, reading, colour, context variants from richest to shortest)."""
    def tiles(self, ctx):
        raise NotImplementedError

    def rows(self, ctx, width):
        gutter = 2 if width >= 96 else 1
        pad = 2 if width >= 76 else 1
        widths = [(width-3*gutter)//4 + (i < (width-3*gutter) % 4) for i in range(4)]
        blocks = [card([Row(SURFACE).add(label, INK3), Row(SURFACE).add(value, color, True),
                        Row(SURFACE).extend(first_fit(w-2*pad, *context))], w, ctx.edges, pad)
                  for (label, value, color, context), w in zip(self.tiles(ctx), widths)]
        return beside(blocks, widths, gutter)


class Card(Component):
    """A titled surface. height() counts the title and the body, not the half-block edges."""
    title = ''

    def note(self, ctx):
        return ''

    def body(self, ctx, inner):
        raise NotImplementedError

    def height(self, ctx):
        raise NotImplementedError

    def rows(self, ctx, width, height=0):
        head = Row(SURFACE).add(self.title, INK, True).add('   '+self.note(ctx), INK3)
        return card([head] + self.body(ctx, width-4), width, ctx.edges, 2, height)


class Column(NamedTuple):
    """A table column. Width 0 marks the one flexible column; a card narrower than min_inner drops the column."""
    name: str
    width: int
    cell: Callable                           # (item, width) -> [Seg] on the card surface
    min_inner: int = 0
    right: bool = False                      # numbers: cell and heading align right


class Table(Card):
    """A heading over one row per item; the layout decides how many items fit."""
    def __init__(self, items):
        self.items = items

    def columns(self, ctx):
        raise NotImplementedError

    def height(self, ctx):
        return 2+len(self.items)

    def body(self, ctx, inner):
        columns = [c for c in self.columns(ctx) if inner >= c.min_inner]
        flexible = inner-sum(c.width+1 for c in columns if c.width)

        def line(contents):
            out = []
            for column, content in zip(columns, contents):
                out += [Seg(' ', bg=SURFACE)]*bool(out) + fit(content, column.width or flexible, SURFACE, column.right)
            return Row(SURFACE).extend(out)
        return [line([Seg(c.name, INK4, SURFACE)] for c in columns)] \
            + [line(c.cell(item, c.width or flexible) for c in columns) for item in self.items]


# ------------------------------------------------------------------ search screen

PHASES = {WAITING: ('等待任务', INK3, '○'), 'loading': ('加载数据', ACCENT, None),
          'sampling': ('生成候选', ACCENT, None), 'evaluating': ('GPU 回测', ACCENT, None),
          'training': ('策略学习', ACCENT, None), 'reweighting': ('组合重配', ACCENT, None),
          'checkpoint': ('保存进度', ACCENT, None), 'complete': ('已完成', GREEN, '✓'),
          'failed': ('运行失败', RED, '✕'), 'stopped': ('已暂停', AMBER, '■'),
          'interrupted': ('已中断', AMBER, '■')}
STAGES = [('sampling', '采样'), ('evaluating', '回测'), ('training', '学习'),
          ('reweighting', '重配'), ('checkpoint', '保存')]
TOKEN = re.compile(r'([A-Za-z_]\w*(?=\())|([A-Za-z_]\w*)|(\d+\.?\d*)|(.)')


class SearchHeader(Header):
    title, elapsed = 'ALPHA SEARCH', '用时'

    def state(self, ctx):
        return PHASES[ctx.st.get('phase', WAITING)]

    def facts(self, ctx):
        st = ctx.st
        mode = st.get('mode', '').upper()
        goal = ctx.objective.label + ' · 噪声对照'*bool(st.get('null_shift')) if mode else ''
        return (mode, goal, gpu_name(st.get('device', ''))), (mode, goal), (mode,)


class Progress(Component):
    def rows(self, ctx, width):
        st = ctx.st
        phase, epoch, epochs = st.get('phase', WAITING), st.get('epoch', 0), st.get('epochs', 0)
        done = epoch if phase in ('complete', 'checkpoint') else max(0, epoch-1)
        fraction = done/epochs if epochs else 0
        left = [Seg('轮次 ', INK3), Seg(str(epoch), INK, bold=True), Seg(f' / {epochs}  ', INK3)] if epochs \
            else [Seg('轮次 —  ', INK3)]
        right = [Seg(f'  {fraction:4.0%}', INK2)]
        color, to = {'complete': (GREEN, None), 'failed': (RED, None), 'stopped': (AMBER, None),
                     'interrupted': (AMBER, None)}.get(phase, (ACCENT, TEAL))
        rows = [left + bar(fraction, width-size(left)-size(right), color, BASE, to) + right]
        if ctx.tight:
            return rows
        current = [key for key, _ in STAGES].index(phase) if phase in dict(STAGES) else -1
        steps = Row()
        for i, (_, name) in enumerate(STAGES):
            steps.add(' ── ' if i else '', INK4)
            if phase == 'complete' or (current >= 0 and i < current):
                steps.add('✓ ', GREEN).add(name, INK2)
            elif i == current:
                steps.add(SPIN[ctx.tick % len(SPIN)]+' '+name, ACCENT, True)
            else:
                steps.add('· '+name, INK4)
        elapsed = st.get('elapsed_seconds')
        if current >= 0 and done and elapsed:
            steps.end('预计剩余 ', INK3).end(clock(elapsed/done*(epochs-done)), INK2)
        return rows + [steps.cells(width)]


def change(start, current, percent, bg):
    """Movement since the run started, followed by the starting point when there is room."""
    base = [Seg('起点 '+num(start, '.2%' if percent else '.3f'), INK3, bg)]
    if start is None or current is None:
        return [base]
    value = current-start
    moved = [Seg(('▲ ' if value > 0 else '▼ ' if value < 0 else '■ ')
                 + (f'{value*100:+.2f} pp' if percent else f'{value:+.3f}'), tone(value), bg)]
    return [moved + [Seg('  ', bg=bg)] + base, moved]


class SearchKpis(Kpis):
    def tiles(self, ctx):
        st = ctx.st
        requested, evaluated, accepted = st.get('requested'), st.get('unique_evaluations'), st.get('accepted')
        eligible = f"可接纳 {num(st.get('low_corr'))}"
        asked = f'请求 {num(requested)}'
        return [
            ('组合 Sharpe', num(st.get('portfolio_sharpe'), '.3f'), INK,
             change(st.get('initial_sharpe'), st.get('portfolio_sharpe'), False, SURFACE)),
            ('组合 CAGR', num(st.get('portfolio_cagr'), '.2%'), INK,
             change(st.get('initial_cagr'), st.get('portfolio_cagr'), True, SURFACE)),
            ('新增构件', num(accepted), GREEN if accepted else INK,
             [[Seg(f"{eligible} · {ctx.objective.passed_label} {num(st.get(ctx.objective.gate_key))}", INK3, SURFACE)],
              [Seg(eligible, INK3, SURFACE)]]),
            ('已评价候选', num(evaluated), INK,
             [[Seg(f'{asked} · 缓存命中 {num(None if None in (requested, evaluated) else requested-evaluated)}', INK3, SURFACE)],
              [Seg(asked, INK3, SURFACE)]]),
        ]


class Callout(Component):
    """The objective's diagnosis of the run in at most `lines` rows."""
    def __init__(self, lines):
        self.lines = lines

    def rows(self, ctx, width):
        _, tag, color, message = ctx.objective.diagnosis(ctx.st)
        lead = [Seg('▍', color), badge(tag, color, BASE), Seg('  ')]
        room = width-size(lead)
        chunks = wrap(message, room)
        if len(chunks) > self.lines:
            chunks = chunks[:self.lines-1] + [set_cell_size(''.join(chunks[self.lines-1:]), room-1)+'…']
        return [(lead if i == 0 else [Seg('▍', color), Seg(' '*(size(lead)-1))]) + fit([Seg(chunk, INK2)], room, BASE)
                for i, chunk in enumerate(chunks)]


class Funnel(Component):
    """The objective's stages as bars. Bare on the window background when there is no room for the card."""
    LABEL, COUNT, SHARE = 15, 9, 8           # column widths; COUNT leaves a gap after the rail up to 9,999,999

    def lines(self, ctx, width, bg, heading):
        st = ctx.st
        stages = ctx.objective.stages(st)
        hot = ctx.objective.diagnosis(st)[0]
        top = max(st.get(stage.key) or 0 for stage in stages)
        tag = 6 if width >= 50 else 0
        rail = width-self.LABEL-self.COUNT-self.SHARE-tag
        rows = []
        if heading:
            rows.append(Row(bg).extend(fit([Seg('阶段', INK4, bg)], self.LABEL+rail, bg) + fit([Seg('数量', INK4, bg)], self.COUNT, bg, True)
                                       + fit([Seg('通过率', INK4, bg)], self.SHARE, bg, True)))
        above = None
        for stage in stages:
            reading = st.get(stage.key)
            share = '' if reading is None or not above else f'{reading/above:.1%}'
            row = fit([Seg(stage.label, AMBER if stage.key == hot else INK2, bg, stage.key == hot)], self.LABEL, bg)
            row += bar((reading or 0)/top if top else 0, rail, AMBER if stage.key == hot else stage.color, bg)
            row += fit([Seg(num(reading), INK, bg, True)], self.COUNT, bg, True) + fit([Seg(share, INK3, bg)], self.SHARE, bg, True)
            if stage.key == hot and tag:
                row += fit([badge('瓶颈', AMBER, bg)], tag, bg, True)
            rows.append(Row(bg).extend(row))
            above = reading if reading is not None else above
        return rows

    def height(self, ctx):
        return 1+len(ctx.objective.stages(ctx.st))

    def rows(self, ctx, width):
        return [[]] + [row.cells(width) for row in self.lines(ctx, width, BASE, False)]


class FunnelCard(Card):
    title = '候选漏斗'

    def note(self, ctx):
        return '本次运行累计'

    def height(self, ctx):
        return 5+len(ctx.objective.stages(ctx.st))

    def body(self, ctx, inner):
        notes = ctx.objective.footnotes(ctx.st)
        variants = ([Seg('  ·  '.join(f'{label} {text}' for label, text in notes[:k]), INK3, SURFACE)]
                    for k in range(len(notes), 1, -1))
        return [Row(SURFACE)] + Funnel().lines(ctx, inner, SURFACE, True) \
            + [Row(SURFACE), Row(SURFACE).extend(first_fit(inner, *variants))]


def facts(st, width, bg, gap):
    """As many throughput / memory facts as fit on one line, most important first."""
    memory = st.get('gpu_used_mib')
    out = []
    for label, value in (('显存', num(None if memory is None else memory/1024, '.2f', ' GB')),
                         ('批量', num(st.get('candidate_batch'))), ('权重试探', num(st.get('weight_trials'))),
                         ('重配评价', num(st.get('reweight_evaluations')))):
        item = [Seg(' '*gap*bool(out), bg=bg), Seg(label+' ', INK3, bg), Seg(value, INK2, bg)]
        if size(out)+size(item) > width:
            break
        out += item
    return out


class PerformanceCard(Card):
    title = '运行性能'
    TIMING = (('采样', 'sampling_seconds', ACCENT), ('回测', 'evaluation_seconds', TEAL),
              ('学习', 'training_seconds', VIOLET), ('重配', 'reweight_seconds', AMBER))

    def note(self, ctx):
        return '最近一轮'

    def height(self, ctx):
        return 10

    def body(self, ctx, inner):
        st = ctx.st
        rates = []
        for label, key in (('GPU 评价', 'evaluation_candidates_per_second'), ('全程平均', 'candidates_per_second')):
            rates += fit([Seg(label+'  ', INK3, SURFACE), Seg(num(st.get(key), '.1f'), INK, SURFACE, True),
                          Seg(' 条/s', INK3, SURFACE)], inner//2, SURFACE)
        rows = [Row(SURFACE), Row(SURFACE).extend(rates), Row(SURFACE).extend(facts(st, inner, SURFACE, 3)), Row(SURFACE)]
        spent = [st.get(key) for _, key, _ in self.TIMING]
        total = None if None in spent else sum(spent)
        rows.append(Row(SURFACE).add('阶段耗时', INK3).end(seconds(total)+' s', INK2))
        if total:
            # Largest-remainder split keeps the stacked rail exactly `inner` cells wide.
            exact = [s/total*inner for s in spent]
            cells = [int(x) for x in exact]
            for i in sorted(range(len(cells)), key=lambda i: cells[i]-exact[i])[:inner-sum(cells)]:
                cells[i] += 1
            rows.append(Row(SURFACE).extend([Seg('━'*n, color, SURFACE) for n, (_, _, color) in zip(cells, self.TIMING)]))
        else:
            rows.append(Row(SURFACE).add('─'*inner, TRACK))
        legend = []
        for (name, _, color), value in zip(self.TIMING, spent):
            legend += fit([Seg('● ', color, SURFACE), Seg(name+' ', INK3, SURFACE), Seg(seconds(value), INK2, SURFACE)],
                          inner//len(self.TIMING), SURFACE)
        rows += [Row(SURFACE).extend(legend), Row(SURFACE)]
        before, after = st.get('loss_before'), st.get('loss_after')
        learning = Row(SURFACE).add('策略学习  ', INK3)
        if before is None or after is None:
            learning.add('—', INK3)
        else:
            learning.add(f'loss {before:.1f} → {after:.1f}', INK2).add(f'  {after/before-1:+.1%}', GREEN if after < before else RED)
            learning.end(f"logZ {st.get('logz', 0):+.3f}", INK3)
        return rows + [learning]


class PerformanceStrip(Component):
    """One line of the performance card, for windows too narrow to put it beside the funnel."""
    def rows(self, ctx, width):
        st = ctx.st
        rate = [Seg('GPU 评价 ', INK3), Seg(num(st.get('evaluation_candidates_per_second'), '.1f', ' 条/s'), INK2), Seg('    ')]
        return [fit(rate + facts(st, width-size(rate), BASE, 4), width, BASE)]


def expression(expr, width, shared, bg):
    """Operators recede, fields lead, constants stand out. Sibling candidates repeat one
    skeleton, so a long formula gives up the head it shares with its neighbours first."""
    shades = (mix(ACCENT, INK3, .45), INK, mix(AMBER, INK2, .6), INK4)
    cells = [(char, shades[m.lastindex-1]) for m in TOKEN.finditer(expr) for char in m.group()]
    if len(cells) > width:
        rest = cells[shared:] if shared > 7 else cells
        head = max(6, width-1-len(rest)) if shared > 7 else 0
        lead = cells[:head] + [('…', INK3)] if head else []
        room = width-len(lead)
        cells = lead + (rest if len(rest) <= room else rest[:room-1] + [('…', INK3)])
    return [Seg(''.join(c for c, _ in run), shade, bg) for shade, run in groupby(cells, key=lambda c: c[1])]


class CandidatesCard(Table):
    """Reports newest first: expression, the objective's verdict, then the objective's columns."""
    title = '最近候选'
    VERDICT = 10

    def note(self, ctx):
        return ctx.objective.caption

    def columns(self, ctx):
        st, objective = ctx.st, ctx.objective
        shared = len(commonprefix([report['expr'] for report in self.items]))
        return (Column('候选表达式', 0, lambda report, width: expression(report['expr'], width, shared, SURFACE)),
                Column('判定', self.VERDICT, lambda report, width: [badge(*objective.verdict(report), SURFACE)]),
                *(Column(c.name, c.width, lambda report, width, c=c: c.cell(report, st, SURFACE), c.min_inner, True)
                  for c in objective.columns))


# ------------------------------------------------------------------ evaluation service screen

OPS = {'family': '假设族', 'eval': '表达式'}
VERDICTS = (('成立', GREEN), ('稀疏', AMBER), ('尖峰', VIOLET), ('无效', INK3), ('通过', GREEN), ('未过', INK3))


def since(st, t):
    """Seconds from a service timestamp to this frame: the service clock is not ours, only differences are."""
    return st['now']-t + (time.time()-st['_source_mtime'] if '_source_mtime' in st else 0.)


def who(source):
    """(submitter, file) of a submitted path; a request without a file has neither."""
    path = PurePosixPath((source or '').replace('\\', '/'))
    return path.parent.name or '—', path.name or '—'


def verdicts(counts, bg):
    out = []
    for name, color in VERDICTS:
        if counts.get(name):
            out += [Seg('  '*bool(out), bg=bg), Seg(f'{name} ', INK3, bg), Seg(str(counts[name]), color, bg, True)]
    return out or [Seg('—', INK4, bg)]


class ServiceHeader(Header):
    title, elapsed = '评价服务', '运行'

    def state(self, ctx):
        st = ctx.st
        if st['phase'] == 'warm':
            return '预热中', ACCENT, None
        if st.get('cur'):
            return '计算中', ACCENT, None
        return '空闲', GREEN, '○'

    def facts(self, ctx):
        return (gpu_name(ctx.st.get('device') or ''),),


class Job(Component):
    """Two rows: the job on the GPU (whose, which stage, how far), or what the service is waiting for."""
    def rows(self, ctx, width):
        st, cur = ctx.st, ctx.st.get('cur')
        if st['phase'] == 'warm':
            return [Row().add('预热  ', INK3).add(st.get('warm') or '', INK2).cells(width), bar(0, width, ACCENT, BASE)]
        if not cur:
            jobs = st.get('jobs', [])
            line = Row().add('空闲', INK2, True).add(f'  ·  队列 {num(st.get("queue"))}', INK3)
            if jobs:
                line.end(f'上一个作业 {ago(since(st, jobs[-1]["t"]))} 前', INK3)
            return [line.cells(width), bar(0, width, ACCENT, BASE)]
        owner, name = who(cur.get('source'))
        unit = '条公式' if cur['op'] == 'family' else '条'
        line = Row().add(OPS[cur['op']], ACCENT, True).add(f'  {owner}', INK, True).add(f' / {name}', INK3) \
            .add(f'  ·  {num(cur.get("n"))} {unit}', INK2)
        line.end(f'已用 {clock(since(st, cur["t0"]))}', INK3)
        done, total = cur.get('done') or 0, cur.get('total') or 0
        right = [Seg(f'  {cur.get("stage") or "排队"}  ', INK2), Seg(f'{num(done)}/{num(total)}', INK3)]
        return [line.cells(width), bar(done/total if total else 0, width-size(right), ACCENT, BASE, TEAL) + right]


class ServiceKpis(Kpis):
    def tiles(self, ctx):
        st = ctx.st
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


class Submitters(Table):
    """Every submitter's rounds so far, most recent first. Items: (submitter, totals)."""
    title = '提交者'
    COLUMNS = (Column('目录', 0, lambda it, w: [Seg(it[0], INK, SURFACE, True)]),
               Column('轮次', 5, lambda it, w: [Seg(num(it[1]['rounds']), INK2, SURFACE)]),
               Column('公式', 6, lambda it, w: [Seg(num(it[1]['formulas']), INK2, SURFACE)], 70),
               Column('变体', 9, lambda it, w: [Seg(num(it[1]['variants']), INK2, SURFACE)]),
               Column('判决', 30, lambda it, w: verdicts(it[1]['verdicts'], SURFACE)))

    def note(self, ctx):
        return '按提交文件所在目录归属 · 服务启动以来累计'

    def columns(self, ctx):
        return self.COLUMNS


class Jobs(Table):
    """Finished jobs, newest first."""
    title = '最近作业'

    def note(self, ctx):
        return '最新在前'

    def columns(self, ctx):
        st = ctx.st
        return (Column('完成', 6, lambda job, w: [Seg(ago(since(st, job['t']))+' 前', INK3, SURFACE)]),
                Column('提交', 0, lambda job, w: [Seg(who(job['source'])[0], INK, SURFACE),
                                                   Seg(' / '+who(job['source'])[1], INK3, SURFACE)]),
                Column('类型', 6, lambda job, w: [Seg(OPS[job['op']], INK2, SURFACE)], 80),
                Column('数量', 6, lambda job, w: [Seg(num(job.get('n')), INK2, SURFACE)], 70),
                Column('变体', 8, lambda job, w: [Seg(num(job.get('variants')), INK2, SURFACE)]),
                Column('耗时', 7, lambda job, w: [Seg(f"{job['dt']:.1f}s", INK2, SURFACE)], 90),
                Column('判决', 30, lambda job, w: verdicts(job.get('verdicts') or {}, SURFACE) if job['ok']
                       else [Seg('失败', AMBER, SURFACE)]))
