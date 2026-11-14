"""Hypothesis families: judge the neighbourhood of every formula an author proposes, not the formula.

An author writes a few formulas per hypothesis, each one point in a space of windows, thresholds and fields picked by
hand; on this data a single point proves little (its score is about half estimation noise, and the author already
kept the best of what it tried). So every formula (the seed) is expanded and its neighbourhood is judged:
  grid     every window and every scalar moved along a ladder of round values: +-2 rungs when the formula has at most
           4 such slots (5^k points), +-1 rung beyond (3^k), a seeded sample of GRID_CAP points past that
  fields   every field replaced, one at a time, by each other field of its category: does the formula need THIS
           field, or would any series of that kind do?
  twins    a seeded sample of the grid again with every signal delayed 1-3 years (Engine null shift): same structure,
           costs and exposure, no information about the days it trades - how often admission passes on noise here
Every variant goes through production admission against the current portfolio and is read as the change it makes
to the portfolio: full-period Sharpe (the gain, primary), CAGR and max drawdown. Verdict per seed:
  成立   the grid passes at least MIN_PASS_RATE, at least MIN_TWIN_RATIO times its twins' rate, and the plateau centre
         (the grid point whose one-rung neighbours do best) passes and holds a position on at least MIN_ATTEND of days
  稀疏   all of that except the attendance: the gain comes from a handful of days
  尖峰   the seed itself passes but its neighbourhood does not: a lucky point, not a hypothesis
  无效   neither
The number of variants is the multiple-testing count of the round.
"""
from __future__ import annotations

from collections import defaultdict
import itertools
import random
import statistics

from foundry.expr.parse import TOKEN, validate
from dataclasses import dataclass
from foundry.candidates import Formula
from foundry.research_types import ResearchContext
from foundry.ops.registry import OPS, WINDOW_LADDER

GRID_CAP = 2187                                      # 3^7 grid points per seed; sampled beyond
TWIN_SAMPLE = 256                                    # twins only estimate the pass rate on noise
# 2026-09-22 local smoke, 3 machine-mined members against the production library: twins pass 0-8%, the two with a
# supported neighbourhood 37% / 64% of their grid, the lone spike (seed ranks 100% in its own grid) 20%.
MIN_PASS_RATE = .25
MIN_TWIN_RATIO = 3.
MIN_ATTEND = .05                                     # a book in cash on 95%+ of days buys its Sharpe with a few lucky days
SEED = 7
SCALARS = (-2, -1, -.5, -.3, -.2, -.1, -.05, 0, .05, .1, .2, .3, .5, .8, 1, 1.5, 2, 3, 5)


def slots(expr):
    """Tokens of the formula and, per token, (role, enclosing operator): role 'window' / 'scalar' / 'field' / None."""
    tokens, i = [], 0
    while i < len(expr):
        m = TOKEN.match(expr, i)
        tokens.append((m.lastgroup, m.group(m.lastgroup)))
        i = m.end()
    roles, stack = [], []
    for j, (kind, text) in enumerate(tokens):
        role, op = None, None
        if kind == 'id' and (j + 1 == len(tokens) or tokens[j + 1][1] != '('):
            role = 'field'
        elif kind == 'id':
            stack.append([text, 0])
        elif text == ',':
            stack[-1][1] += 1
        elif text == ')':
            stack.pop()
        elif kind == 'num':
            op, arg = stack[-1]
            k = OPS[op].kind
            if k in ('ts', 'ts_param') and arg == 1 or k in ('pair', 'pair_param') and arg == 2:
                role = 'window'
            elif not (k == 'pair_param' and arg == 3):   # the regression's return type is a category, not a level
                role = 'scalar'
        roles.append((role, op))
    return tokens, roles


def ladder(value, role, op, steps):
    """(the value with `steps` round values on either side, index of the value)."""
    if role == 'window':
        kind = OPS[op].kind
        cap = WINDOW_LADDER[-1]
        rungs, value = [w for w in WINDOW_LADDER if w <= cap], int(value)
    else:
        rungs = list(SCALARS)
    below = [r for r in rungs if r < value][-steps:]
    return below + [value] + [r for r in rungs if r > value][:steps], len(below)


def text(tokens, values):
    return ''.join(f'{v:g}' if isinstance(v, (int, float)) else v for v in (values.get(j, t) for j, (_, t) in enumerate(tokens)))


def neighbourhood(expr, categories):
    """(grid [(expr, index vector)], seed index vector, field variants [(expr, field, replacement)])."""
    tokens, roles = slots(expr)
    numeric = [j for j, (role, _) in enumerate(roles) if role in ('window', 'scalar')]
    steps = 2 if len(numeric) <= 4 else 1
    axes = [ladder(float(tokens[j][1]), *roles[j], steps) for j in numeric]
    origin = tuple(at for _, at in axes)
    points = list(itertools.product(*[range(len(values)) for values, _ in axes]))
    if len(points) > GRID_CAP:
        points = [origin] + random.Random(SEED).sample([p for p in points if p != origin], GRID_CAP - 1)
    grid = [(text(tokens, {j: axes[a][0][p[a]] for a, j in enumerate(numeric)}), p) for p in points]
    fields = [(text(tokens, {j: other}), tokens[j][1], other)
              for j, (role, _) in enumerate(roles) if role == 'field'
              for other in categories[tokens[j][1]] if other != tokens[j][1]]
    return grid, origin, fields


@dataclass(frozen=True)
class FamilyPoint:
    expr: str
    passed: bool
    sharpe: float | None
    maxdd: float | None
    attend: float | None
    gain: float | None
    d_cagr: float | None
    d_maxdd: float | None


@dataclass(frozen=True)
class Spread:
    n: int
    pass_rate: float
    median: float | None
    median_cagr: float | None
    median_maxdd: float | None
    p25: float | None
    p75: float | None


@dataclass(frozen=True)
class PlateauPoint:
    point: FamilyPoint
    neighbourhood_median: float
    neighbours: int


@dataclass(frozen=True)
class FieldSummary:
    field: str
    substitutes: int
    passed: int
    median_gain: float | None


@dataclass(frozen=True)
class SkippedFamily:
    id: str
    expression: str
    hypothesis: str
    error: str


@dataclass(frozen=True)
class FamilySummary:
    id: str
    hypothesis: str
    seed: FamilyPoint
    grid: Spread
    twins: Spread
    verdict: str
    seed_rank: float | None
    centre: PlateauPoint | None
    fields: tuple[FieldSummary, ...]


@dataclass(frozen=True)
class FamilyResult:
    context: ResearchContext
    families: tuple[FamilySummary, ...]
    variants: int
    skipped: tuple[SkippedFamily, ...]


def evaluate(research, exprs, delayed, context_id):
    out = {}
    for lo in range(0, len(exprs), research.engine.candidate_batch):
        part = exprs[lo:lo + research.engine.candidate_batch]
        sources = tuple(Formula(expression) for expression in part)
        batch = research.evaluate_noise_device(sources, context_id) if delayed else research.evaluate_device(sources, context_id, report=True)
        for row, verdict in zip(batch.rows, batch.verdicts):
            if verdict.stopped_at == 'G1':
                out[row.source.expression] = FamilyPoint(row.source.expression, False, None, None, None, None, None, None)
                continue
            p, m, ref = verdict.proposal, row.metrics, research.portfolio.reference
            days = sum(y.days for y in m.yearly)
            out[row.source.expression] = FamilyPoint(row.source.expression, verdict.passed, m.sharpe, m.maxdd,
                sum(y.attend * y.days for y in m.yearly) / days,
                p.sharpe - ref.sharpe if p.valid else None,
                p.cagr - ref.cagr if p.valid else None,
                p.maxdd - ref.maxdd if p.valid else None)
    return out


def spread(rows):
    valid = [r for r in rows if r.gain is not None]
    gains = [r.gain for r in valid]
    return Spread(len(rows), sum(r.passed for r in rows) / len(rows),
        statistics.median(gains) if valid else None,
        statistics.median(r.d_cagr for r in valid) if valid else None,
        statistics.median(r.d_maxdd for r in valid) if valid else None,
        statistics.quantiles(gains, n=4)[0] if len(gains) > 1 else None,
        statistics.quantiles(gains, n=4)[2] if len(gains) > 1 else None)


def plateau(grid, rows):
    index = {p: rows[e].gain for e, p in grid if rows[e].gain is not None}
    points = []
    for expr, point in grid:
        moves = [point] + [point[:a] + (point[a] + d,) + point[a + 1:] for a in range(len(point)) for d in (-1, 1)]
        near = [index[q] for q in moves if q in index]
        if near:
            points.append((statistics.median(near), expr, len(near)))
    if points:
        median, expr, neighbours = max(points, key=lambda point: point[0])
        return PlateauPoint(rows[expr], median, neighbours)
    return None


def expand(research, seeds, context_id):
    engine = research.engine
    categories = {f: names for names in engine.panel.categories().values() for f in names}
    plans, skipped = [], []
    for seed in seeds:
        checked = validate(seed.expression, engine.vocabulary, engine.expr_cfg.max_cost)
        if not checked['ok']:
            skipped.append(SkippedFamily(seed.id, seed.expression, seed.hypothesis, checked['error']))
            continue
        grid, origin, fields = neighbourhood(seed.expression.strip(), categories)
        plans.append((seed, grid, fields, next(e for e, p in grid if p == origin)))
    real_exprs = sorted({e for _, grid, fields, _ in plans for e in [*(x[0] for x in grid), *(x[0] for x in fields)]})
    rng = random.Random(SEED)
    twins = {seed.id: [e for e, _ in grid] if len(grid) <= TWIN_SAMPLE else rng.sample([e for e, _ in grid], TWIN_SAMPLE)
             for seed, grid, _, _ in plans}
    twin_exprs = sorted({e for expressions in twins.values() for e in expressions})
    real = evaluate(research, real_exprs, False, context_id)
    twin = evaluate(research, twin_exprs, True, context_id)
    families = []
    for seed, grid, fields, seed_expr in plans:
        g = spread([real[e] for e, _ in grid])
        t = spread([twin[e] for e in twins[seed.id]])
        centre = plateau(grid, real)
        holds = g.pass_rate >= MIN_PASS_RATE and g.pass_rate >= MIN_TWIN_RATIO * t.pass_rate and centre is not None and centre.point.passed
        verdict = ('成立' if centre.point.attend >= MIN_ATTEND else '稀疏') if holds else '尖峰' if real[seed_expr].passed else '无效'
        gains = [real[e].gain for e, _ in grid if real[e].gain is not None]
        seed_gain = real[seed_expr].gain
        seed_rank = sum(gain <= seed_gain for gain in gains) / len(gains) if seed_gain is not None and gains else None
        by_field = defaultdict(list)
        for expr, field, _ in fields:
            by_field[field].append(real[expr])
        field_summaries = []
        for field, rows in by_field.items():
            valid = [row.gain for row in rows if row.gain is not None]
            field_summaries.append(FieldSummary(field, len(rows), sum(row.passed for row in rows),
                                                statistics.median(valid) if valid else None))
        families.append(FamilySummary(seed.id, seed.hypothesis, real[seed_expr], g, t, verdict, seed_rank,
                                      centre, tuple(field_summaries)))
    return FamilyResult(research.context, tuple(families), len(real) + len(twin), tuple(skipped))
