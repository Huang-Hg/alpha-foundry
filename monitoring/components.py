"""The blocks of the dashboard. Each is an object with rows(screen, width) -> cell rows exactly `width` wide.

A component reads the status and asks the objective (objectives.py) for anything that depends on what
the run optimises; it never decides whether it is shown - that is the layout's job (view.py).
"""
from itertools import groupby
from os.path import commonprefix
from pathlib import PureWindowsPath
import re
import time
from typing import NamedTuple, Optional

from rich.cells import set_cell_size

from .cells import (ACCENT, AMBER, BASE, GREEN, INK, INK2, INK3, INK4, RAISED, RED, SPIN, SURFACE, TEAL, TRACK, VIOLET,
                    Row, Seg, ago, badge, bar, beside, card, clock, first_fit, fit, mix, num, seconds, size, tone, wrap)
from .objectives import Objective

STAGES = [('sampling', '采样'), ('evaluating', '回测'), ('training', '学习'),
          ('reweighting', '重配'), ('checkpoint', '保存')]
PHASES = {'waiting': ('等待任务', INK3, '○'), 'loading': ('加载数据', ACCENT, None),
          'sampling': ('生成候选', ACCENT, None), 'evaluating': ('GPU 回测', ACCENT, None),
          'training': ('策略学习', ACCENT, None), 'reweighting': ('组合重配', ACCENT, None),
          'checkpoint': ('保存进度', ACCENT, None), 'complete': ('已完成', GREEN, '✓'),
          'failed': ('运行失败', RED, '✕'), 'stopped': ('已暂停', AMBER, '■'),
          'interrupted': ('已中断', AMBER, '■')}
ENDED = ('complete', 'failed', 'stopped', 'interrupted')
FRESH_SECONDS = 5                            # a snapshot this recent counts as "now" in the footer
STALE_SECONDS = 30                           # a live source this quiet is flagged in the header
TOKEN = re.compile(r'([A-Za-z_]\w*(?=\())|([A-Za-z_]\w*)|(\d+\.?\d*)|(.)')


class Screen(NamedTuple):
    """Everything a component may read while one frame is drawn."""
    st: dict
    objective: Objective
    tick: int
    edges: bool                              # half-block card edges (off on tight or colourless screens)
    tight: bool                              # no spacer rows, no stage rail
    age: Optional[float]                     # seconds since the status file was written
    stale: Optional[float]                   # the same, once a live run has gone quiet for too long
    interval: Optional[float]


class Component:
    def rows(self, screen, width):
        raise NotImplementedError


class Header(Component):
    def rows(self, screen, cols):
        st = screen.st
        phase = st.get('phase', 'waiting')
        label, color, glyph = PHASES.get(phase, (phase, INK3, '○'))
        if screen.stale is not None:
            label, color, glyph = f'数据停滞 {ago(screen.stale)}', AMBER, '!'
        right = [badge(f'{glyph or SPIN[screen.tick % len(SPIN)]} {label}', color, RAISED)]
        if cols >= 64:
            right += [Seg('   用时 ', INK3, RAISED), Seg(clock(st.get('elapsed_seconds')), INK2, RAISED)]
        right += [Seg('   '+time.strftime('%H:%M:%S')+' ', INK, RAISED, True)]
        title = [Seg(' ◆ ', ACCENT, RAISED, True), Seg('Q25 ALPHA SEARCH', INK, RAISED, True)]
        device = st.get('device', '').replace('Tesla ', '').replace('-PCIE-32GB', ' 32GB')
        mode = st.get('mode', '').upper()
        goal = screen.objective.label + ' · 噪声对照'*bool(st.get('null_shift')) if mode else ''
        left = first_fit(cols-size(right)-1, *(title + [Seg('   '+' · '.join(filter(None, parts)), INK3, RAISED)]
                                              for parts in ((mode, goal, device), (mode, goal), (mode,))), title)
        return [fit(left, cols-size(right), RAISED) + right]


class Footer(Component):
    def rows(self, screen, cols):
        st, age, stale, interval = screen.st, screen.age, screen.stale, screen.interval
        live = age is not None and stale is None and st.get('phase', 'waiting') not in ENDED
        right = [Seg(f'{interval:g}s 刷新 · ' if interval and cols >= 72 else '', INK3, RAISED), Seg('Ctrl+C 退出 ', INK3, RAISED)]
        dot = [Seg(' ● ', AMBER if stale is not None else GREEN if live else INK4, RAISED)]
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


class Progress(Component):
    def rows(self, screen, width):
        st = screen.st
        phase, epoch, epochs = st.get('phase', 'waiting'), st.get('epoch', 0), st.get('epochs', 0)
        done = epoch if phase in ('complete', 'checkpoint') else max(0, epoch-1)
        fraction = done/epochs if epochs else 0
        left = [Seg('轮次 ', INK3), Seg(str(epoch), INK, bold=True), Seg(f' / {epochs}  ', INK3)] if epochs \
            else [Seg('轮次 —  ', INK3)]
        right = [Seg(f'  {fraction:4.0%}', INK2)]
        color, to = {'complete': (GREEN, None), 'failed': (RED, None), 'stopped': (AMBER, None),
                     'interrupted': (AMBER, None)}.get(phase, (ACCENT, TEAL))
        rows = [left + bar(fraction, width-size(left)-size(right), color, BASE, to) + right]
        if screen.tight:
            return rows
        current = [key for key, _ in STAGES].index(phase) if phase in dict(STAGES) else -1
        steps = Row()
        for i, (_, name) in enumerate(STAGES):
            steps.add(' ── ' if i else '', INK4)
            if phase == 'complete' or (current >= 0 and i < current):
                steps.add('✓ ', GREEN).add(name, INK2)
            elif i == current:
                steps.add(SPIN[screen.tick % len(SPIN)]+' '+name, ACCENT, True)
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


class Kpis(Component):
    """Four tiles: (label, reading, colour, context variants from richest to shortest)."""
    def tiles(self, screen):
        st = screen.st
        requested, evaluated, accepted = st.get('requested'), st.get('unique_evaluations'), st.get('accepted')
        eligible = f"可接纳 {num(st.get('low_corr'))}"
        asked = f'请求 {num(requested)}'
        return [
            ('组合 Sharpe', num(st.get('portfolio_sharpe'), '.3f'), INK,
             change(st.get('initial_sharpe'), st.get('portfolio_sharpe'), False, SURFACE)),
            ('组合 CAGR', num(st.get('portfolio_cagr'), '.2%'), INK,
             change(st.get('initial_cagr'), st.get('portfolio_cagr'), True, SURFACE)),
            ('新增构件', num(accepted), GREEN if accepted else INK,
             [[Seg(f"{eligible} · {screen.objective.passed_label} {num(st.get(screen.objective.gate_key))}", INK3, SURFACE)],
              [Seg(eligible, INK3, SURFACE)]]),
            ('已评价候选', num(evaluated), INK,
             [[Seg(f'{asked} · 缓存命中 {num(None if None in (requested, evaluated) else requested-evaluated)}', INK3, SURFACE)],
              [Seg(asked, INK3, SURFACE)]]),
        ]

    def rows(self, screen, width):
        gutter = 2 if width >= 96 else 1
        pad = 2 if width >= 76 else 1
        widths = [(width-3*gutter)//4 + (i < (width-3*gutter) % 4) for i in range(4)]
        blocks = [card([Row(SURFACE).add(label, INK3), Row(SURFACE).add(value, color, True),
                        Row(SURFACE).extend(first_fit(w-2*pad, *context))], w, screen.edges, pad)
                  for (label, value, color, context), w in zip(self.tiles(screen), widths)]
        return beside(blocks, widths, gutter)


class Callout(Component):
    """The objective's diagnosis of the run in at most `lines` rows."""
    def __init__(self, lines):
        self.lines = lines

    def rows(self, screen, width):
        _, tag, color, message = screen.objective.diagnosis(screen.st)
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

    def lines(self, screen, width, bg, heading):
        st = screen.st
        stages = screen.objective.stages(st)
        hot = screen.objective.diagnosis(st)[0]
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

    def height(self, screen):
        return 1+len(screen.objective.stages(screen.st))

    def rows(self, screen, width):
        return [[]] + [row.cells(width) for row in self.lines(screen, width, BASE, False)]


class Card(Component):
    """A titled surface. height() counts the title and the body, not the half-block edges."""
    title = ''

    def note(self, screen):
        return ''

    def body(self, screen, inner):
        raise NotImplementedError

    def height(self, screen):
        raise NotImplementedError

    def rows(self, screen, width, height=0):
        head = Row(SURFACE).add(self.title, INK, True).add('   '+self.note(screen), INK3)
        return card([head] + self.body(screen, width-4), width, screen.edges, 2, height)


class FunnelCard(Card):
    title = '候选漏斗'

    def note(self, screen):
        return '本次运行累计'

    def height(self, screen):
        return 5+len(screen.objective.stages(screen.st))

    def body(self, screen, inner):
        notes = screen.objective.footnotes(screen.st)
        variants = ([Seg('  ·  '.join(f'{label} {text}' for label, text in notes[:k]), INK3, SURFACE)]
                    for k in range(len(notes), 1, -1))
        return [Row(SURFACE)] + Funnel().lines(screen, inner, SURFACE, True) \
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

    def note(self, screen):
        return '最近一轮'

    def height(self, screen):
        return 10

    def body(self, screen, inner):
        st = screen.st
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
    def rows(self, screen, width):
        st = screen.st
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


class CandidatesCard(Card):
    """Newest reports first: expression, the objective's verdict, then the objective's columns."""
    title = '最近候选'
    VERDICT = 10

    def __init__(self, count):
        self.count = count

    def note(self, screen):
        return screen.objective.caption

    def height(self, screen):
        return 2+self.count

    def body(self, screen, inner):
        st, objective = screen.st, screen.objective
        columns = [c for c in objective.columns if inner >= c.min_inner]
        flexible = inner-self.VERDICT-1-sum(c.width+1 for c in columns)
        head = fit([Seg('候选表达式', INK4, SURFACE)], flexible, SURFACE) \
            + [Seg(' ', bg=SURFACE)] + fit([Seg('判定', INK4, SURFACE)], self.VERDICT, SURFACE)
        for column in columns:
            head += [Seg(' ', bg=SURFACE)] + fit([Seg(column.name, INK4, SURFACE)], column.width, SURFACE, True)
        rows = [Row(SURFACE).extend(head)]
        shown = st['recent'][-self.count:]
        shared = len(commonprefix([r.get('expr', '') for r in shown]))
        for report in reversed(shown):
            label, color = objective.verdict(report)
            line = fit(expression(report.get('expr', '—'), flexible, shared, SURFACE), flexible, SURFACE) \
                + [Seg(' ', bg=SURFACE)] + fit([badge(label, color, SURFACE)], self.VERDICT, SURFACE)
            for column in columns:
                line += [Seg(' ', bg=SURFACE)] + fit(column.cell(report, st, SURFACE), column.width, SURFACE, True)
            rows.append(Row(SURFACE).extend(line))
        return rows
