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
from foundry.ops.cuda import unsupported
from foundry.ops.registry import OPS, WINDOW_LADDER

GRID_CAP = 2187                                      # 3^7 grid points per seed; sampled beyond
TWIN_SAMPLE = 256                                    # twins only estimate the pass rate on noise
# 2026-09-22 local smoke, 3 machine-mined members against the production library: twins pass 0-8%, the two with a
# supported neighbourhood 37% / 64% of their grid, the lone spike (seed ranks 100% in its own grid) 20%.
MIN_PASS_RATE = .25
MIN_TWIN_RATIO = 3.
MIN_ATTEND = .05                                     # a book in cash on 95%+ of days buys its Sharpe with a few lucky days
SEED = 7
PAIR_WINDOW_MAX = 365                                # neighbourhoods stay portable to cards with a smaller pair-kernel tile
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
        cap = PAIR_WINDOW_MAX if kind in ('pair', 'pair_param') else OPS[op].causal_window or WINDOW_LADDER[-1]
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


def evaluate(portfolio, exprs, delayed, tick):
    """expr -> compact row of production admission; `delayed` evaluates the noise twins."""
    engine, out = portfolio.engine, {}
    for lo in range(0, len(exprs), engine.candidate_batch):
        tick('GPU 邻域孪生' if delayed else 'GPU 邻域展开', lo, len(exprs))
        batch = engine.formulas(exprs[lo:lo + engine.candidate_batch], null_shift=delayed)
        for r, g in zip(batch, portfolio.admit(batch)):
            c = g.get('combo') or {}
            yearly = r.get('yearly') or []
            days = sum(y['days'] for y in yearly)
            after = 'full_sharpe_after' in c
            out[r['expr']] = dict(passed=bool(g['passed']), sharpe=r.get('sharpe'), maxdd=r.get('maxdd'),
                                  attend=sum(y['attend'] * y['days'] for y in yearly) / days if days else None,
                                  rho=c.get('rho'), weight=c.get('w'),
                                  gain=c['full_sharpe_after'] - c['full_sharpe'] if after else None,
                                  d_cagr=c['cagr_after'] - c['cagr'] if after else None,
                                  d_maxdd=c['maxdd_after'] - c['maxdd'] if after else None)   # > 0: shallower
    return out


def spread(rows):
    gains = [r['gain'] for r in rows if r['gain'] is not None]
    return dict(n=len(rows), pass_rate=sum(r['passed'] for r in rows) / len(rows),
                median=statistics.median(gains) if gains else None,
                median_cagr=statistics.median(r['d_cagr'] for r in rows if r['gain'] is not None) if gains else None,
                median_maxdd=statistics.median(r['d_maxdd'] for r in rows if r['gain'] is not None) if gains else None,
                p25=statistics.quantiles(gains, n=4)[0] if len(gains) > 1 else None,
                p75=statistics.quantiles(gains, n=4)[2] if len(gains) > 1 else None)


def plateau(grid, rows):
    """Grid point whose immediate neighbours (itself and every point one rung away along one axis) have the highest
    median gain: the middle of a flat good region rather than its tallest spike."""
    index = {p: rows[e]['gain'] for e, p in grid if rows[e]['gain'] is not None}
    best = None
    for e, p in grid:
        moves = [p] + [p[:a] + (p[a] + d,) + p[a + 1:] for a in range(len(p)) for d in (-1, 1)]
        near = [index[q] for q in moves if q in index]
        if near and (best is None or statistics.median(near) > best[1]):
            best = (e, statistics.median(near), len(near))
    return best


def expand(portfolio, seeds, tick=lambda stage, done, total: None):
    """seeds [{expr, hypothesis}] -> dict(families=[summary per seed], variants=evaluated count, skipped=[...])."""
    vocabulary = portfolio.engine.vocabulary
    categories = {f: names for names in portfolio.engine.panel.categories().values() for f in names}

    def runs(e):
        v = validate(e, vocabulary)
        return v['ok'] and not unsupported(v['tree'])

    plans, skipped = [], []
    for s in seeds:
        if not runs(s['expr']):
            v = validate(s['expr'], vocabulary)
            skipped.append(dict(s, error=v.get('error') or '窗口超出本卡上限'))
            continue
        grid, origin, fields = neighbourhood(s['expr'].strip(), categories)
        grid = [(e, p) for e, p in grid if runs(e)]
        plans.append(dict(s, seed=next(e for e, p in grid if p == origin), grid=grid,
                          fields=[(e, a, b) for e, a, b in fields if runs(e)]))
    real_exprs = sorted({e for p in plans for e, _ in p['grid']} | {e for p in plans for e, _, _ in p['fields']})
    rng = random.Random(SEED)
    twin_of = {id(p): [e for e, _ in p['grid']] if len(p['grid']) <= TWIN_SAMPLE else rng.sample([e for e, _ in p['grid']], TWIN_SAMPLE)
               for p in plans}
    twin_exprs = sorted({e for es in twin_of.values() for e in es})
    real = evaluate(portfolio, real_exprs, False, tick)
    twin = evaluate(portfolio, twin_exprs, True, tick)
    families = []
    for p in plans:
        grid_rows, twin_rows, seed = [real[e] for e, _ in p['grid']], [twin[e] for e in twin_of[id(p)]], real[p['seed']]
        gains = [r['gain'] for r in grid_rows if r['gain'] is not None]
        centre = plateau(p['grid'], real)
        by_field = defaultdict(list)
        for e, field, _ in p['fields']:
            by_field[field].append(real[e])
        g, t = spread(grid_rows), spread(twin_rows)
        holds = (g['pass_rate'] >= MIN_PASS_RATE and g['pass_rate'] >= MIN_TWIN_RATIO * t['pass_rate']
                 and centre is not None and real[centre[0]]['passed'])
        families.append(dict(
            hypothesis=p.get('hypothesis', ''), seed=dict(seed, expr=p['seed']), grid=g, twins=t,
            verdict=('成立' if real[centre[0]]['attend'] >= MIN_ATTEND else '稀疏') if holds
            else '尖峰' if seed['passed'] else '无效',
            seed_rank=sum(x <= seed['gain'] for x in gains) / len(gains) if seed['gain'] is not None and gains else None,
            centre=dict(real[centre[0]], expr=centre[0], neighbourhood_median=centre[1], neighbours=centre[2]) if centre else None,
            fields={f: dict(substitutes=len(rows), passed=sum(r['passed'] for r in rows),
                            median_gain=statistics.median(x) if (x := [r['gain'] for r in rows if r['gain'] is not None]) else None)
                    for f, rows in by_field.items()}))
    tick('收尾', 1, 1)
    return dict(families=families, variants=len(real) + len(twin), skipped=skipped)
