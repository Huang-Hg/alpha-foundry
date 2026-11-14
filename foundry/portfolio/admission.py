"""Portfolio admission: a candidate is judged only by what it does to the portfolio.

Each candidate book is tried on a grid of weights 1/N..1 plus the analytic suggestion (tangency of P + aA);
every trial is the real blended book U + a*A capped to the market's gross exposure budget, backtested with costs
(trades inside the portfolio net out). A trial is feasible when
  - the recency-weighted Sharpe rises by at least [portfolio] min_sharpe_gain and the full-period Sharpe does not fall
    (Sharpe is the primary dimension and is never traded), and
  - full-period CAGR and recency-weighted log growth each change relatively at least as much as the max drawdown
    depth: dx >= |x_ref| * (|DD| / |DD_ref| - 1) (same drawdown: return must not fall; 10% deeper: 10% more return;
    10% shallower: 10% may go).
The feasible trial with the best balanced utility (tanh of Sharpe gain and growth gain) wins, ties by CAGR. A full
library admits no new member at any weight (adding never replaces).

Every query, weight trial and commit uses the study's declared numerical policy.
Admission never schedules a second simulation in a different precision.
"""
from __future__ import annotations

from foundry.research_types import ConstraintCheck, Proposal, Verdict, WeightMetrics


from foundry.core import metrics as M


KEYS = ('sharpe', 'full_sharpe', 'cagr', 'recent_growth')
LABELS = ('近期加权 Sharpe 增量不足', '全期 Sharpe 下降', '全期 CAGR 不足以抵偿回撤变化', '近期加权对数增长不足以抵偿回撤变化')


def return_requirements(dd, ref):
    """Minimum change of (recency-weighted Sharpe, full Sharpe, CAGR, recent log growth) for a portfolio whose max
    drawdown becomes dd (drawdowns negative). Return and drawdown trade against each other; Sharpe never does."""
    xp, cfg = ref.xp, ref.cfg
    zero = xp.zeros_like(dd)
    if ref.maxdd == 0:
        # The relative-depth limit: finite growth cannot buy a new drawdown from a zero-drawdown reference.
        growth = xp.where(dd == 0, zero, xp.inf)
        return xp.stack((zero + cfg.min_sharpe_gain, zero, growth, growth), axis=-1)
    deeper = dd / ref.maxdd - 1
    return xp.stack((zero + cfg.min_sharpe_gain, zero, abs(ref.cagr) * deeper, abs(ref.recent_growth) * deeper), axis=-1)


def weight_metrics(rr, dd, ref, ppy, shape):
    """One set of returns, constraints and utility for admission and joint reweighting."""
    xp, cfg = ref.xp, ref.cfg
    valid = M.sound(rr, xp)
    moments = ref.measure(rr, ref.p, ppy)
    ws = moments.weighted_sharpe.reshape(shape)
    sh = moments.sharpe.reshape(shape)
    cagr = moments.cagr.reshape(shape)
    growth = moments.recent_growth.reshape(shape)
    dd = dd.reshape(shape)
    delta = xp.stack((ws - ref.wsharpe, sh - ref.sharpe, cagr - ref.cagr,
                      growth - ref.recent_growth), axis=-1)
    required = return_requirements(dd, ref)
    checks = delta >= required - cfg.metric_atol
    utility = .5 * xp.tanh(delta[..., 0] / cfg.sharpe_scale) + .5 * xp.tanh(delta[..., 3] / cfg.growth_scale)
    return WeightMetrics(valid.reshape(shape), ws, sh, cagr, growth, dd, delta, required, checks, utility)


def proposals(engine, rows, ref):
    """Weight trials, constraints and the chosen weight on the explicit backend."""
    with engine.resources.scope():
        xp, cfg = engine.xp, ref.cfg
        W = xp.ascontiguousarray(xp.stack([r.book for r in rows]))
        R = xp.stack([r.returns for r in rows])
        q, p = ref.rr, ref.p
        ma, mq = xp.sum(R * p, axis=1), xp.sum(q * p)
        va, vq = xp.sum((R - ma[:, None]) ** 2 * p, axis=1), xp.sum((q - mq) ** 2 * p)
        cov = xp.sum((R - ma[:, None]) * (q - mq) * p, axis=1)
        v_a, v_p = ma * vq - mq * cov, mq * va - ma * cov
        # Tangency of the additive pair P + aA is a* = v_a/v_p. A dominating candidate gets the largest weight; one with
        # nothing to add repeats grid point 1 and is dropped as a duplicate trial.
        both = (v_a > 0) & (v_p > 0)
        suggestion = xp.where(both, v_a / xp.where(both, v_p, 1), 1)
        suggestion = xp.where(suggestion <= 1, suggestion, 1)
        B, K = len(rows), cfg.weight_steps + 1
        grid = xp.broadcast_to(xp.arange(1, K, dtype=xp.float64) / (K - 1), (B, K - 1))
        weights = xp.sort(xp.concatenate((grid, suggestion[:, None]), axis=1), axis=1)
        unique = xp.concatenate((xp.ones((B, 1), dtype=xp.bool_), weights[:, 1:] != weights[:, :-1]), axis=1)
        owners = xp.repeat(xp.arange(B, dtype=xp.int32), K)
        qW = xp.asarray(ref.U, dtype=xp.float64)
        result = engine.blend(W, qW, owners, weights.ravel())
        rr, stats = result.returns, result.statistics
        ppy = engine.ppy
        measured = weight_metrics(rr, stats[:, 3], ref, ppy, (B, K))
        ws, sh, cagr, growth, dd = measured.wsharpe, measured.sharpe, measured.cagr, measured.recent_growth, measured.maxdd
        delta, required, checks, utility = measured.delta, measured.required, measured.checks, measured.utility
        valid = measured.valid & unique
        new = xp.asarray([r.source not in ref.keys for r in rows])
        capacity = xp.broadcast_to(~((ref.n >= cfg.max_members) & new)[:, None], weights.shape)
        feasible = checks.all(2) & valid & capacity
        passed = feasible.any(1)
        score = xp.where(feasible, utility, -xp.inf)
        # Project metric tolerance through tanh's Lipschitz bound. Within the
        # same utility/CAGR tolerance, sorted trials choose the smallest weight.
        utility_atol = .5 * cfg.metric_atol * (1 / cfg.sharpe_scale + 1 / cfg.growth_scale)
        tie = feasible & (utility >= score.max(1)[:, None] - utility_atol)
        best = xp.argmax(tie & (cagr >= xp.where(tie, cagr, -xp.inf).max(1)[:, None] - cfg.metric_atol), axis=1)
        margins = delta - required
        scales = xp.asarray([cfg.violation_scale_wsharpe, cfg.violation_scale_sharpe, cfg.violation_scale_cagr,
                             cfg.violation_scale_growth])
        violation = xp.maximum(-margins / scales, 0).sum(2) + xp.where(capacity, 0., 1.)
        minimum = xp.where(valid, violation, xp.inf).min(1)
        # Project the same metric tolerance into the normalized violation units.
        # Sorted weights make numerically tied rejected trials choose the smallest.
        tie_atol = cfg.metric_atol * (1 / scales).sum()
        near = xp.argmax(valid & (violation <= minimum[:, None] + tie_atol), axis=1)
        chosen = xp.where(passed, best, near)
        ix = xp.arange(B)
        mine = xp.where(va > 0, ma / xp.sqrt(xp.where(va > 0, va, 1)) * ppy ** 0.5, 0)
        rho = xp.where((va > 0) & (vq > 0), cov / xp.sqrt(xp.where((va > 0) & (vq > 0), va * vq, 1)), 0)
        need = rho * ref.wsharpe + 0.6 * xp.sqrt(xp.maximum(1 - rho * rho, 0))
        report = xp.stack((passed, valid.any(1), weights[ix, chosen], ws[ix, chosen], sh[ix, chosen], cagr[ix, chosen],
                           growth[ix, chosen], dd[ix, chosen], unique.sum(1), capacity[ix, chosen], mine, rho, need), axis=1)
        report = xp.concatenate((report, delta[ix, chosen], checks[ix, chosen], utility[ix, chosen, None],
                                 violation[ix, chosen, None], required[ix, chosen]), axis=1)
        out = []
        for row, v in zip(rows, engine.to_host(report).tolist()):
            row.view = (v[10], v[12], v[11])
            if not v[1]:
                out.append(Proposal(False, (), ('所有试探组合均产生非有限收益或破产',), 0., int(v[8]),
                                    f'{engine.spec.backend}-{engine.precision}', ref.version, False))
                continue
            checks = [ConstraintCheck(key, bool(v[17 + j]), v[13 + j]) for j, key in enumerate(KEYS)]
            vetoes = [f'{label} ({v[13 + j]:+.6f},要求 ≥ {v[23 + j]:+.6f})' for j, label in enumerate(LABELS) if not v[17 + j]]
            if not v[9]:
                checks.append(ConstraintCheck('capacity', False, float(cfg.max_members - ref.n - 1)))
                vetoes.append('组合成员数超过上限')
            if not v[0]:
                vetoes.insert(0, f'{int(v[8])} 个权重点未找到可行组合;以下为约束违反最小的试探点')
            out.append(Proposal(bool(v[0]), tuple(checks), tuple(vetoes), v[2], int(v[8]),
                                f'{engine.spec.backend}-{engine.precision}', ref.version, True,
                                v[3], v[4], v[5], v[6], v[7], v[21], v[22]))
        return out


def admit(engine, rows, ref):
    """Apply the same admission constraints to every caller's batch."""
    results = []
    for lo in range(0, len(rows), engine.candidate_batch):
        part = rows[lo:lo + engine.candidate_batch]
        indices = [i for i, row in enumerate(part) if row.ok and row.valid_returns]
        proposals_by_index = dict(zip(indices, proposals(engine, [part[i] for i in indices], ref))) if indices else {}
        for i, row in enumerate(part):
            if not row.ok:
                results.append(Verdict(row.source, False, 'G1', 'invalid_candidate', row.error, None))
            elif not row.valid_returns:
                results.append(Verdict(row.source, False, 'G1', 'invalid_returns', '无效收益序列:非有限值或破产', None))
            else:
                p = proposals_by_index[i]
                message = '结构有效;加入后组合 Sharpe 提升且全期/近期收益抵得过回撤变化' if p.passed else '; '.join(p.vetoes)
                results.append(Verdict(row.source, p.passed, None if p.passed else 'G2', 'admitted' if p.passed else 'constraints_failed', message, p))
    return tuple(results)
