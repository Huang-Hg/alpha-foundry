"""Drawing primitives of the dashboard: palette, styled cells, exact-width layout helpers, number formats.

A row is a list of Seg whose cell widths add up to the width it was asked for. Nothing here knows
what a search status is. Only characters that occupy one cell in Windows Terminal are used for
drawing; CJK text is measured with rich's cell table.
"""
from typing import NamedTuple

from rich.cells import cell_len, set_cell_size

# Surfaces, from the window background up to raised bars.
BASE, SURFACE, RAISED, TRACK = '#0b0e14', '#121822', '#1a2230', '#273244'
# Text, from primary to faint.
INK, INK2, INK3, INK4 = '#e8eef6', '#aab6c5', '#7a889c', '#4a5669'
# One accent plus semantic colours; nothing else may appear in a frame.
ACCENT, TEAL, GREEN, AMBER, RED, VIOLET = '#5eb6ff', '#3fd8c0', '#54d9a1', '#f3b65b', '#ff7285', '#b097ff'
PALETTE = ('BASE', 'SURFACE', 'RAISED', 'TRACK', 'INK', 'INK2', 'INK3', 'INK4',
           'ACCENT', 'TEAL', 'GREEN', 'AMBER', 'RED', 'VIOLET')

SPIN = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'


class Seg(NamedTuple):
    text: str
    fg: str = INK
    bg: str = BASE
    bold: bool = False


def mix(fg, bg, share):
    """Opaque stand-in for `fg` at `share` opacity over `bg` (terminals have no alpha)."""
    return '#' + ''.join(f'{round(int(bg[i:i+2], 16)*(1-share) + int(fg[i:i+2], 16)*share):02x}' for i in (1, 3, 5))


def size(segs):
    return sum(cell_len(s.text) for s in segs)


def crop(segs, width):
    if width <= 0:
        return []
    if size(segs) <= width:
        return list(segs)
    out, room = [], width-1
    for seg in segs:
        cells = cell_len(seg.text)
        if cells > room:
            out.append(seg._replace(text=set_cell_size(seg.text, room)+'…'))
            break
        out.append(seg)
        room -= cells
    return out


def fit(segs, width, bg, right=False):
    segs = crop(segs, width)
    blank = [Seg(' '*(width-size(segs)), bg=bg)]
    return blank+segs if right else segs+blank


def first_fit(width, *options):
    """Richest variant that fits; a narrower window drops whole facts instead of clipping them."""
    return next((o for o in options if size(o) <= width), options[-1])


def wrap(text, width):
    lines, line = [], ''
    for char in ' '.join(text.split()):
        if cell_len(line+char) > width:
            lines.append(line)
            line = ''
        line += char
    return lines + [line]


class Row:
    """Left and right aligned runs on one background, laid out to an exact cell width."""
    def __init__(self, bg=BASE):
        self.bg, self.left, self.right = bg, [], []

    def add(self, text, fg=INK, bold=False):
        self.left.append(Seg(str(text), fg, self.bg, bold))
        return self

    def end(self, text, fg=INK, bold=False):
        self.right.append(Seg(str(text), fg, self.bg, bold))
        return self

    def extend(self, segs):
        self.left.extend(segs)
        return self

    def cells(self, width):
        # The right side is an extra: it goes first when the two sides collide.
        right = self.right if size(self.left)+size(self.right) < width else []
        left = crop(self.left, width)
        return left + [Seg(' '*(width-size(left)-size(right)), bg=self.bg)] + right


def card(rows, width, edges, pad, height=0):
    """Rows on a raised surface; `height` stretches the body so cards standing side by side end together."""
    rows = list(rows) + [Row(SURFACE) for _ in range(height-len(rows))]
    side = [Seg(' '*pad, bg=SURFACE)]
    body = [side + row.cells(width-2*pad) + side for row in rows]
    if not edges:
        return body
    # Half-block edges give the card half a row of breathing room above and below.
    return [[Seg('▄'*width, SURFACE, BASE)]] + body + [[Seg('▀'*width, SURFACE, BASE)]]


def beside(blocks, widths, gutter):
    rows = []
    for i in range(max(map(len, blocks))):
        row = []
        for j, (block, width) in enumerate(zip(blocks, widths)):
            row += [Seg(' '*gutter)]*bool(j) + (block[i] if i < len(block) else [Seg(' '*width)])
        rows.append(row)
    return rows


def bar(fraction, width, color, bg, to=None):
    """Heavy fill on a light track (still readable without colour), half-cell precision;
    any non-zero value keeps a visible sliver."""
    fraction = min(1, max(0, fraction))
    halves = max(round(fraction*width*2), bool(fraction))
    full, half = divmod(halves, 2)
    shade = (lambda i: mix(to, color, i/max(1, width-1))) if to else (lambda i: color)
    return [Seg('━', shade(i), bg) for i in range(full)] + [Seg('╸', shade(full), bg)]*half \
        + [Seg('─'*(width-full-half), TRACK, bg)]


def badge(text, color, bg):
    return Seg(f' {text} ', color, mix(color, bg, .16), True)


def num(value, spec=',', suffix=''):
    return '—' if value is None else format(value, spec)+suffix


def seconds(value):
    return '—' if value is None else f'{value:.2f}' if value < 10 else f'{value:.1f}' if value < 100 else f'{value:.0f}'


def clock(value):
    if value is None:
        return '—'
    value = int(value)
    return f'{value//3600}:{value//60%60:02d}:{value%60:02d}' if value >= 3600 else f'{value//60:02d}:{value%60:02d}'


def ago(value):
    for unit, span in (('d', 86400), ('h', 3600), ('m', 60)):
        if value >= span:
            return f'{int(value//span)}{unit}'
    return f'{int(value)}s'


def tone(value):
    return INK3 if not value else GREEN if value > 0 else RED
