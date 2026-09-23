"""Layout of the dashboard: status -> frame of styled cells.

Layers, each changeable on its own:
  schema.py      what the producers write                          (the input contract)
  cells.py       palette, styled cells, exact-width helpers        (how things are drawn)
  objectives.py  stages, verdicts, candidate columns per objective (what a search run means)
  components.py  header, progress, tiles, cards, tables ...        (the blocks)
This module picks the screen for a status (a search run, or kind='service' for the evaluation service), and a
screen only decides which blocks a window of a given size gets, and in which order.
preview.py exports the frame to HTML (the design is reviewed there first); render() hands the very
same frame to rich, so the shell panel cannot drift away from the approved preview.
"""
import time

from rich.text import Text

from . import objectives
from .cells import Seg, beside
from .components import (STALE_SECONDS, WAITING, Callout, CandidatesCard, Context, Footer, Funnel, FunnelCard, Job, Jobs,
                         PerformanceCard, PerformanceStrip, Progress, SearchContext, SearchHeader, SearchKpis,
                         ServiceHeader, ServiceKpis, Submitters)
from .schema import ENDED

MAX_CANDIDATES = 8
MAX_JOBS = 12


def chrome(ctx):
    """Rows that are not body: header, its spacer, footer."""
    return 2 + (not ctx.tight)


class Screen:
    """One kind of status: its header (a subclass attribute), and the blocks of the body for a window of a given size."""
    footer = Footer()

    def context(self, st, *frame):
        return Context(st, *frame)

    def body(self, ctx, width, limit):
        raise NotImplementedError

    def compose(self, st, cols, rows=None, tick=0, edges=True, interval=None):
        """Whole screen for a `cols` x `rows` terminal; rows=None yields the natural height."""
        age = time.time()-st['_source_mtime'] if '_source_mtime' in st else None
        # Only a status that is still expected to change can go quiet for too long.
        stale = age if age is not None and age > STALE_SECONDS and st.get('phase', WAITING) not in ENDED+(WAITING,) else None
        limit = rows or 10**6
        tight = limit < 12
        ctx = self.context(st, tick, edges and not tight, tight, age, stale, interval)
        margin = 2 if cols >= 72 else 1
        width = cols-2*margin
        side = [Seg(' '*margin)]
        body = [side + (row or [Seg(' '*width)]) + side for row in self.body(ctx, width, limit)]
        frame = (self.header.rows(ctx, cols) + [[]]*(not tight) + body)[:limit-1]
        # The footer is pinned to the last row; spare rows stay painted in the base colour.
        frame += [[]]*((rows or 0)-len(frame)-1)
        return [row or [Seg(' '*cols)] for row in frame] + self.footer.rows(ctx, cols)


class SearchScreen(Screen):
    """A search run: progress, tiles and diagnosis always; then the funnel and performance cards, then the recent
    candidates, as far as the window allows. A smaller window gives up whole blocks: candidates, performance, funnel."""
    header, progress, kpis = SearchHeader(), Progress(), SearchKpis()
    funnel, funnel_card, performance_card, performance_strip = Funnel(), FunnelCard(), PerformanceCard(), PerformanceStrip()

    def context(self, st, *frame):
        return SearchContext(st, *frame, objectives.of(st))

    def body(self, ctx, width, limit):
        st, fixed = ctx.st, chrome(ctx)
        body = self.progress.rows(ctx, width) + [[]]*(not ctx.tight) + self.kpis.rows(ctx, width)
        body += Callout(2 if limit >= 24 else 1).rows(ctx, width)
        room = limit-fixed-len(body)
        cards = (self.funnel_card, self.performance_card)
        tall = max(c.height(ctx) for c in cards)
        if width >= 96 and room >= tall+2*ctx.edges:
            left = (width-2)//2
            widths = [left, width-2-left]
            body += beside([c.rows(ctx, w, tall) for c, w in zip(cards, widths)], widths, 2)
        elif room >= self.funnel_card.height(ctx)+2*ctx.edges:
            body += self.funnel_card.rows(ctx, width)
            # The strip needs a spare row below it, or it would sit on the footer.
            if limit-fixed-len(body) >= 2:
                body += self.performance_strip.rows(ctx, width)
        elif room >= self.funnel.height(ctx):
            body += self.funnel.rows(ctx, width)
        recent = st.get('recent', [])
        count = min(len(recent), MAX_CANDIDATES, limit-fixed-len(body)-2-2*ctx.edges)
        if count > 0 and width >= 60:
            body += CandidatesCard(recent[-count:][::-1]).rows(ctx, width)
        return body


class ServiceScreen(Screen):
    """The evaluation service: the job on the GPU and the tiles always; then the submitters and the recent jobs.
    It follows a queue of jobs that agents submit (one hypothesis file per round); a submitter is the directory
    the submitted file lives in, so every agent's rounds add up under its own name."""
    header, job, kpis = ServiceHeader(), Job(), ServiceKpis()

    def body(self, ctx, width, limit):
        st, fixed = ctx.st, chrome(ctx)
        body = self.job.rows(ctx, width) + [[]]*(not ctx.tight) + self.kpis.rows(ctx, width)
        table = 3+2*ctx.edges                # spacer, card title, column heading
        sources = sorted(st.get('sources', {}).items(), key=lambda kv: -kv[1]['last'])
        room = limit-fixed-len(body)-table
        if sources and width >= 60 and room > 0:
            body += [[]] + Submitters(sources[:room]).rows(ctx, width)
        jobs = st.get('jobs', [])
        count = min(len(jobs), MAX_JOBS, limit-fixed-len(body)-table)
        if count > 0 and width >= 60:
            body += [[]] + Jobs(jobs[-count:][::-1]).rows(ctx, width)
        return body


# A search run does not write `kind`.
SCREENS = {'search': SearchScreen(), 'service': ServiceScreen()}


def compose(st, cols, rows=None, tick=0, edges=True, interval=None):
    return SCREENS[st.get('kind', 'search')].compose(st, cols, rows, tick, edges, interval)


def render(st, rows=None, width=120, tick=0, edges=True, interval=None):
    text = Text(no_wrap=True, overflow='crop')
    for i, row in enumerate(compose(st, width, rows, tick, edges, interval)):
        text.append('\n'*bool(i))
        for seg in row:
            text.append(seg.text, f"{'bold '*seg.bold}{seg.fg} on {seg.bg}")
    return text
