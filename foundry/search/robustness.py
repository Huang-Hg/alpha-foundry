"""Candidate families: shared formula and model plateau and delayed-twin judgments.

An author writes a few formulas per hypothesis, each one point in a space of windows, thresholds and fields picked by
hand; on this data a single point proves little (its score is about half estimation noise, and the author already
kept the best of what it tried). So every formula (the seed) is expanded and its neighbourhood is judged:
  grid     every window and every scalar moved along a ladder of round values: +-2 rungs when the formula has at most
           4 such slots (5^k points), +-1 rung beyond (3^k), a seeded sample of GRID_CAP points past that
  fields   every field replaced, one at a time, by each other field of its category: does the formula need THIS
           field, or would any series of that kind do?
  twins    a seeded sample of the grid again with every signal delayed 1-3 years (stateless forward delay): same structure,
           costs and exposure, no information about the days it trades - how often admission passes on noise here
Every variant goes through production admission against the current portfolio and is read as the change it makes
to the portfolio: full-period Sharpe (the gain, primary), CAGR and max drawdown. Verdict per seed:
  holds    the grid passes at least MIN_PASS_RATE, at least MIN_TWIN_RATIO times its twins' rate, and the plateau centre
           (the grid point whose one-rung neighbours do best) passes and holds a position on at least MIN_ATTEND of days
  sparse   all of that except the attendance: the gain comes from a handful of days
  spike    the seed itself passes but its neighbourhood does not: a lucky point, not a hypothesis
  invalid  neither
The number of variants is the multiple-testing count of the round.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
import hashlib
import itertools
import random
import statistics

from foundry.types import Candidate, CandidateGrid, DeviceBatch, FamilySeed, Formula, PortfolioView, ResearchContext
from foundry.study.session import ResidentPort
from foundry.types import FamilyPoint, Spread, PlateauPoint, FieldSummary, SkippedFamily, FamilySummary, FamilyResult
from foundry.ops.registry import FORMULA_TOKEN, OPS, WINDOW_LADDER, SCALAR_LADDER

GRID_CAP = 2187                                      # 3^7 grid points per seed; sampled beyond
TWIN_SAMPLE = 256                                    # twins only estimate the pass rate on noise
MIN_PASS_RATE = .25
MIN_TWIN_RATIO = 3.
MIN_ATTEND = .05                                     # a book in cash on 95%+ of days buys its Sharpe with a few lucky days
SEED = 7


def slots(expr: str) -> tuple[list[tuple[str, str]], list[tuple[str | None, str | None]]]:
    """Tokens of the formula and, per token, (role, enclosing operator): role 'window' / 'scalar' / 'field' / None."""
    tokens, i = [], 0
    while i < len(expr):
        m = FORMULA_TOKEN.match(expr, i)
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


def ladder(value: float, role: str, op: str | None, steps: int) -> tuple[list[int | float], int]:
    """(the value with `steps` round values on either side, index of the value)."""
    if role == 'window':
        cap = WINDOW_LADDER[-1]
        rungs, value = [w for w in WINDOW_LADDER if w <= cap], int(value)
    else:
        rungs = list(SCALAR_LADDER)
    below = [r for r in rungs if r < value][-steps:]
    return below + [value] + [r for r in rungs if r > value][:steps], len(below)


def text(tokens: Sequence[tuple[str, str]], values: Mapping[int, str | int | float]) -> str:
    return ''.join(f'{v:g}' if isinstance(v, (int, float)) else v for v in (values.get(j, t) for j, (_, t) in enumerate(tokens)))


def neighbourhood(expr: str, categories: Mapping[str, Sequence[str]]) -> tuple[
        list[tuple[str, tuple[int, ...]]], tuple[int, ...], list[tuple[str, str, str]]]:
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


def formula_candidate(expression: str) -> Candidate:
    return Candidate(hashlib.sha256(expression.encode()).hexdigest(), Formula(expression))


def points(candidates: Sequence[Candidate], batch: DeviceBatch, ref: PortfolioView) -> dict[Candidate, FamilyPoint]:
    out = {}
    for candidate, row, verdict in zip(candidates, batch.rows, batch.verdicts):
        if verdict.stopped_at == 'G1':
            out[candidate] = FamilyPoint(candidate, False, None, None, None, None, None, None)
            continue
        p, m = verdict.proposal, row.metrics
        days = sum(y.days for y in m.yearly)
        out[candidate] = FamilyPoint(candidate, verdict.passed, m.sharpe, m.maxdd,
            sum(y.attend * y.days for y in m.yearly) / days,
            p.sharpe - ref.sharpe if p.valid else None,
            p.cagr - ref.cagr if p.valid else None,
            p.maxdd - ref.maxdd if p.valid else None)
    return out


def spread(rows: Sequence[FamilyPoint]) -> Spread:
    valid = [r for r in rows if r.gain is not None]
    gains = [r.gain for r in valid]
    return Spread(len(rows), sum(r.passed for r in rows) / len(rows),
        statistics.median(gains) if valid else None,
        statistics.median(r.d_cagr for r in valid) if valid else None,
        statistics.median(r.d_maxdd for r in valid) if valid else None,
        statistics.quantiles(gains, n=4)[0] if len(gains) > 1 else None,
        statistics.quantiles(gains, n=4)[2] if len(gains) > 1 else None)


def plateau(grid: Sequence[tuple[Candidate, tuple[int, ...]]],
            rows: Mapping[Candidate, FamilyPoint]) -> PlateauPoint | None:
    index = {p: rows[c].gain for c, p in grid if rows[c].gain is not None}
    centres = []
    for candidate, point in grid:
        moves = [point] + [point[:a] + (point[a] + d,) + point[a + 1:] for a in range(len(point)) for d in (-1, 1)]
        near = [index[q] for q in moves if q in index]
        if near:
            centres.append((statistics.median(near), candidate, len(near)))
    if centres:
        median, candidate, neighbours = max(centres, key=lambda point: point[0])
        return PlateauPoint(rows[candidate], median, neighbours)
    return None


@dataclass(frozen=True)
class PlannedFamily:
    id: str
    hypothesis: str
    seed: Candidate
    grid: tuple[tuple[Candidate, tuple[int, ...]], ...]
    fields: tuple[tuple[Candidate, str, str], ...]


@dataclass(frozen=True)
class FamilyPlan:
    seeds: tuple[PlannedFamily, ...]
    skipped: tuple[SkippedFamily, ...]
    candidates: tuple[Candidate, ...]
    twin_candidates: tuple[Candidate, ...]
    twins: Mapping[str, tuple[Candidate, ...]]


def _candidate_key(candidate: Candidate) -> tuple[str, str, str]:
    source = candidate.source
    return (source.kind, source.expression if isinstance(source, Formula) else candidate.id, repr(source))


def _explicit_grid(grid: CandidateGrid) -> PlannedFamily:
    if not grid.variants or len(grid.variants) != len(grid.parameter_coords):
        raise ValueError('candidate grid requires one coordinate per variant')
    if any(not isinstance(candidate, Candidate) for candidate in grid.variants):
        raise TypeError('candidate grid variants must contain typed candidates')
    if any(not isinstance(candidate.id, str) or not candidate.id for candidate in grid.variants):
        raise ValueError('candidate grid IDs must be nonempty strings')
    if grid.variants.count(grid.seed) != 1 or len({c.id for c in grid.variants}) != len(grid.variants):
        raise ValueError('candidate grid must include its seed once and use unique candidate IDs')
    if not grid.parameter_coords[0] or len({len(point) for point in grid.parameter_coords}) != 1:
        raise ValueError('candidate grid coordinates must have the same positive dimension')
    if any(type(rung) is not int for point in grid.parameter_coords for rung in point):
        raise TypeError('candidate grid coordinates must contain integer rung indices')
    if len(set(grid.parameter_coords)) != len(grid.parameter_coords):
        raise ValueError('candidate grid coordinates must be unique')
    return PlannedFamily(grid.seed.id, grid.hypothesis, grid.seed,
                         tuple(zip(grid.variants, grid.parameter_coords)), ())


def plan(seeds: Sequence[FamilySeed | CandidateGrid], categories: Mapping[str, Sequence[str]],
         formula_error: Callable[[str], str | None]) -> FamilyPlan:
    plans, skipped = [], []
    identities = [seed.id if isinstance(seed, FamilySeed) else seed.seed.id for seed in seeds]
    if any(not isinstance(identity, str) or not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError('family IDs must be nonempty and unique')
    for seed in seeds:
        if isinstance(seed, CandidateGrid):
            plans.append(_explicit_grid(seed))
            continue
        error = formula_error(seed.expression)
        if error is not None:
            skipped.append(SkippedFamily(seed.id, Candidate(seed.id, Formula(seed.expression)), seed.hypothesis, error))
            continue
        grid, origin, fields = neighbourhood(seed.expression.strip(), categories)
        plans.append(PlannedFamily(seed.id, seed.hypothesis,
            formula_candidate(next(e for e, p in grid if p == origin)),
            tuple((formula_candidate(e), p) for e, p in grid),
            tuple((formula_candidate(e), field, other) for e, field, other in fields)))
    real = sorted({c for p in plans for c in [*(v[0] for v in p.grid), *(v[0] for v in p.fields)]}, key=_candidate_key)
    rng = random.Random(SEED)
    twins = {p.id: tuple(c for c, _ in p.grid) if len(p.grid) <= TWIN_SAMPLE
             else tuple(rng.sample([c for c, _ in p.grid], TWIN_SAMPLE)) for p in plans}
    twin = sorted({c for variants in twins.values() for c in variants}, key=_candidate_key)
    return FamilyPlan(tuple(plans), tuple(skipped), tuple(real), tuple(twin), twins)


def judge(planned: FamilyPlan, real: Mapping[Candidate, FamilyPoint], twin: Mapping[Candidate, FamilyPoint],
          context: ResearchContext) -> FamilyResult:
    families = []
    for family in planned.seeds:
        g = spread([real[c] for c, _ in family.grid])
        t = spread([twin[c] for c in planned.twins[family.id]])
        centre = plateau(family.grid, real)
        holds = g.pass_rate >= MIN_PASS_RATE and g.pass_rate >= MIN_TWIN_RATIO * t.pass_rate and centre is not None and centre.point.passed
        verdict = ('成立' if centre.point.attend >= MIN_ATTEND else '稀疏') if holds else '尖峰' if real[family.seed].passed else '无效'
        gains = [real[c].gain for c, _ in family.grid if real[c].gain is not None]
        seed_gain = real[family.seed].gain
        seed_rank = sum(gain <= seed_gain for gain in gains) / len(gains) if seed_gain is not None and gains else None
        by_field = defaultdict(list)
        for candidate, field, _ in family.fields:
            by_field[field].append(real[candidate])
        field_summaries = []
        for field, rows in by_field.items():
            valid = [row.gain for row in rows if row.gain is not None]
            field_summaries.append(FieldSummary(field, len(rows), sum(row.passed for row in rows),
                                                statistics.median(valid) if valid else None))
        families.append(FamilySummary(family.id, family.hypothesis, real[family.seed], g, t, verdict, seed_rank,
                                      centre, tuple(field_summaries)))
    return FamilyResult(context, tuple(families), len(real) + len(twin), planned.skipped)


def judge_family(resident: ResidentPort, seeds: Sequence[FamilySeed | CandidateGrid]) -> FamilyResult:
    """Judge explicit candidate lattices and formula families against one context."""
    context = resident.context
    categories = {field: names for names in resident.categories.values() for field in names}
    planned = plan(seeds, categories, resident.formula_error)
    real: dict[Candidate, FamilyPoint] = {}
    twins: dict[Candidate, FamilyPoint] = {}
    for candidates, delayed, output in ((planned.candidates, False, real), (planned.twin_candidates, True, twins)):
        for lo in range(0, len(candidates), resident.candidate_batch):
            part = candidates[lo:lo + resident.candidate_batch]
            batch = resident.evaluate(part, report=True, delayed=delayed)
            output.update(points(part, batch, resident.reference))
    return judge(planned, real, twins, context)
