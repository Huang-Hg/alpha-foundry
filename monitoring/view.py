"""Layout of the Q25 dashboard: status -> frame of styled cells.

Three layers, each changeable on its own:
  cells.py       palette, styled cells, exact-width helpers        (how things are drawn)
  objectives.py  stages, verdicts, candidate columns per objective (what a run means)
  components.py  header, progress, tiles, cards ...                (the blocks)
This module only decides which blocks a window of a given size gets, and in which order. A status with
kind='service' (the evaluation service, tmp/q25_funnel/daemon.py) is laid out by service.py instead.
preview.py exports the frame to HTML (the design is reviewed there first); render() hands the very
same frame to rich, so the shell panel cannot drift away from the approved preview.
"""
import time

from rich.text import Text

from . import objectives, service
from .cells import Seg, beside
from .components import (ENDED, STALE_SECONDS, Callout, CandidatesCard, Footer, Funnel, FunnelCard, Header, Kpis,
                         PerformanceCard, PerformanceStrip, Progress, Screen)

MAX_CANDIDATES = 8


class Dashboard:
    header, footer, progress, kpis = Header(), Footer(), Progress(), Kpis()
    funnel, funnel_card, performance_card, performance_strip = Funnel(), FunnelCard(), PerformanceCard(), PerformanceStrip()

    def compose(self, st, cols, rows=None, tick=0, edges=True, interval=None):
        """Whole screen for a `cols` x `rows` terminal; rows=None yields the natural height."""
        age = time.time()-st['_source_mtime'] if '_source_mtime' in st else None
        stale = age if age is not None and age > STALE_SECONDS and st.get('phase', 'waiting') not in ENDED+('waiting',) else None
        limit = rows or 10**6
        tight = limit < 12
        screen = Screen(st, objectives.of(st), tick, edges and not tight, tight, age, stale, interval)
        margin = 2 if cols >= 72 else 1
        width = cols-2*margin
        body = self.progress.rows(screen, width) + [[]]*(not tight) + self.kpis.rows(screen, width)
        body += Callout(2 if limit >= 24 else 1).rows(screen, width)
        fixed = 2 + (not tight)                  # header, its spacer, footer
        room = limit-fixed-len(body)
        cards = (self.funnel_card, self.performance_card)
        tall = max(c.height(screen) for c in cards)
        if width >= 96 and room >= tall+2*screen.edges:
            left = (width-2)//2
            widths = [left, width-2-left]
            body += beside([c.rows(screen, w, tall) for c, w in zip(cards, widths)], widths, 2)
        elif room >= self.funnel_card.height(screen)+2*screen.edges:
            body += self.funnel_card.rows(screen, width)
            # The strip needs a spare row below it, or it would sit on the footer.
            if limit-fixed-len(body) >= 2:
                body += self.performance_strip.rows(screen, width)
        elif room >= self.funnel.height(screen):
            body += self.funnel.rows(screen, width)
        count = min(len(st.get('recent') or []), MAX_CANDIDATES, limit-fixed-len(body)-2-2*screen.edges)
        if count > 0 and width >= 60:
            body += CandidatesCard(count).rows(screen, width)
        side = [Seg(' '*margin)]
        frame = self.header.rows(screen, cols) + [[]]*(not tight) + [side + (row or [Seg(' '*width)]) + side for row in body]
        # The footer is pinned to the last row; spare rows stay painted in the base colour.
        frame += [[]]*((rows or 0)-len(frame)-1)
        return [row or [Seg(' '*cols)] for row in frame] + self.footer.rows(screen, cols)


DASHBOARD = Dashboard()


def compose(st, cols, rows=None, tick=0, edges=True, interval=None):
    screen = service.SERVICE if st.get('kind') == 'service' else DASHBOARD
    return screen.compose(st, cols, rows, tick, edges, interval)


def render(st, rows=None, width=120, tick=0, edges=True, interval=None):
    text = Text(no_wrap=True, overflow='crop')
    for i, row in enumerate(compose(st, width, rows, tick, edges, interval)):
        text.append('\n'*bool(i))
        for seg in row:
            text.append(seg.text, f"{'bold '*seg.bold}{seg.fg} on {seg.bg}")
    return text
