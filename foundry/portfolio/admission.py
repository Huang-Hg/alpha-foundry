"""Portfolio admission (protocol combo-cuda-v6): a candidate is judged only by what it does to the portfolio.

Each candidate book is tried on a grid of weights 1/N..1 plus the analytic suggestion (tangency of P + aA);
every trial is the real blended book (U + a*A) / max(1, exposure), backtested with costs (trades inside the
portfolio net out). A trial is feasible when
  - the recency-weighted Sharpe rises by at least [portfolio] min_sharpe_gain and the full-period Sharpe does not fall
    (Sharpe is the primary dimension and is never traded), and
  - full-period CAGR and recency-weighted log growth each change relatively at least as much as the max drawdown
    depth: dx >= |x_ref| * (|DD| / |DD_ref| - 1) (same drawdown: return must not fall; 10% deeper: 10% more return;
    10% shallower: 10% may go).
The feasible trial with the best balanced utility (tanh of Sharpe gain and growth gain) wins, ties by CAGR. A full
library admits no new member at any weight (adding never replaces).

Precision: trials run with FP32 daily state and FP64 statistics; every candidate is calibrated against FP64 at two
weights, and any candidate whose widened error envelope could still pass - or whose calibration is unstable, out of
budget or out of domain - is recomputed in FP64. Only FP64 results can pass. The envelope is an empirical guard with
a safety factor, not an interval proof.
"""
from __future__ import annotations

import math

import cupy as cp

from foundry.config import CudaConfig
from foundry.core import metrics as M
from foundry.portfolio.library import PROTOCOL
from foundry.portfolio.reference import CFG

CUDA = CudaConfig()
KEYS = ('sharpe', 'full_sharpe', 'cagr', 'recent_growth')
LABELS = ('近期加权 Sharpe 增量不足', '全期 Sharpe 下降', '全期 CAGR 不足以抵偿回撤变化', '近期加权对数增长不足以抵偿回撤变化')


def return_requirements(dd, ref):
    """Minimum change of (recency-weighted Sharpe, full Sharpe, CAGR, recent log growth) for a portfolio whose max
    drawdown becomes dd (drawdowns negative). Return and drawdown trade against each other; Sharpe never does."""
    deeper = dd / ref.maxdd - 1
    zero = cp.zeros_like(dd)
    return cp.stack((zero + CFG.min_sharpe_gain, zero, abs(ref.cagr) * deeper, abs(ref.recent_growth) * deeper), axis=-1)


def sharpe_upper(mean, sd, error, ppy):
    """Largest Sharpe reachable under an absolute per-day return perturbation of `error`."""
    m, lo, hi = mean + error, sd - error, sd + error
    return cp.where(lo <= 0, cp.inf, cp.where(m >= 0, m / cp.maximum(lo, 1e-30), m / hi) * ppy ** 0.5)


def _refine(engine, W, qW, weights, owners, rr, stats, ref):
    """Calibrate every candidate, then recompute every possibly feasible trial in FP64 (in place)."""
    B, K = weights.shape
    ix = cp.arange(B)
    cube = rr.reshape(B, K, -1)
    mid = cp.argmin(cp.abs(weights - .5), axis=1)
    probe, _ = engine.blend(W, qW, cp.tile(ix.astype(cp.int32), 2), cp.concatenate((weights[ix, mid], weights[:, -1])), 'f64')
    if ref.mixed_error is None:
        qr, _ = engine.simulate(qW[None], 'mixed')
        ref.mixed_error = cp.max(cp.abs(qr[0, engine.start:] - ref.rr))
    error = cp.maximum(cp.max(cp.abs(probe[:B] - cube[ix, mid]), axis=1), cp.max(cp.abs(probe[B:] - cube[:, -1]), axis=1))
    error = cp.maximum(error, ref.mixed_error)
    budget = cp.maximum(CUDA.mixed_return_atol, CUDA.mixed_error_safety * error)
    e = cp.repeat(budget, K)
    p = ref.p
    mu = cp.sum(rr * p, axis=1)
    wsd = cp.sqrt(cp.sum((rr - mu[:, None]) ** 2 * p, axis=1))
    fullsd = rr.std(1)
    possible = ((sharpe_upper(mu, wsd, e, engine.ppy) >= ref.wsharpe + CFG.min_sharpe_gain - 1e-10) &
                (sharpe_upper(rr.mean(1), fullsd, e, engine.ppy) >= ref.sharpe - 1e-10)).reshape(B, K).any(1)
    # The return requirements move with the approximate drawdown, so only the Sharpe bounds screen.
    invalid = ((~cp.isfinite(rr)).any(1) | (rr <= -1).any(1) | (wsd <= e) | (fullsd <= e)).reshape(B, K).any(1)
    outside = (~cp.isfinite(error)) | (error > CUDA.mixed_return_atol)
    # The calibrated domain is a nonnegative, unlevered candidate; the portfolio's raw sum may exceed full exposure.
    domain = (W < 0).any(axis=(1, 2)) | (W.sum(2) > 1 + 1e-10).any(1)
    refine = possible | invalid | outside | domain | (qW < 0).any()
    selected = cp.flatnonzero(refine)
    if len(selected):
        chosen = (selected[:, None] * K + cp.arange(K)).ravel()
        exact, exact_stats = engine.blend(W, qW, owners[chosen], weights.ravel()[chosen], 'f64')
        # A probe outside the widened envelope voids this batch's approximate rejections: recompute all in FP64.
        measured = cp.max(cp.abs(exact - rr[chosen]), axis=1)
        breach = (~cp.isfinite(measured)) | (measured > cp.repeat(budget[selected], K))
        rr[chosen], stats[chosen] = exact, exact_stats
        if bool(breach.any()):
            rr[:], stats[:] = engine.blend(W, qW, owners, weights.ravel(), 'f64')
            refine[:] = True
            ref.force_f64 = True
    return cp.asnumpy(cp.stack((refine, error, budget, outside, invalid), axis=1)).tolist()


def proposals(engine, rows, ref, precision=None):
    """Weight trials, constraints and the chosen weight for every row (all on CUDA)."""
    precision = CUDA.precision if precision is None else precision
    if precision == 'auto' and ref.force_f64:
        out = proposals(engine, rows, ref, 'f64')
        for r in out:
            r['precision'] = dict(mode='auto', refined=True, calibration_breach=True,
                                  fp64_trials=CFG.weight_steps + 1, fp32_trials=0)
        return out
    W = cp.ascontiguousarray(cp.stack([r['_W'] for r in rows]))
    R = cp.stack([r['_rr'] for r in rows])
    q, p = ref.rr, ref.p
    ma, mq = cp.sum(R * p, axis=1), cp.sum(q * p)
    va, vq = cp.sum((R - ma[:, None]) ** 2 * p, axis=1), cp.sum((q - mq) ** 2 * p)
    cov = cp.sum((R - ma[:, None]) * (q - mq) * p, axis=1)
    v_a, v_p = ma * vq - mq * cov, mq * va - ma * cov
    # Tangency of the additive pair P + aA is a* = v_a/v_p. A dominating candidate gets the largest weight; one with
    # nothing to add repeats grid point 1 and is dropped as a duplicate trial.
    both = (v_a > 0) & (v_p > 0)
    suggestion = cp.where(both, v_a / cp.where(both, v_p, 1), 1)
    suggestion = cp.where(suggestion <= 1, suggestion, 1)
    B, K = len(rows), CFG.weight_steps + 1
    grid = cp.broadcast_to(cp.arange(1, K, dtype=cp.float64) / (K - 1), (B, K - 1))
    weights = cp.sort(cp.concatenate((grid, suggestion[:, None]), axis=1), axis=1)
    unique = cp.concatenate((cp.ones((B, 1), dtype=cp.bool_), weights[:, 1:] != weights[:, :-1]), axis=1)
    owners = cp.repeat(cp.arange(B, dtype=cp.int32), K)
    qW = cp.asarray(ref.U, dtype=cp.float64)
    rr, stats = engine.blend(W, qW, owners, weights.ravel(), 'mixed' if precision == 'auto' else precision)
    diagnostics = _refine(engine, W, qW, weights, owners, rr, stats, ref) if precision == 'auto' else None
    valid = M.sound(rr)
    ppy = engine.ppy
    ws = M.weighted_sharpe(rr, p, ppy).reshape(B, K)
    sh = M.sharpe(rr, ppy).reshape(B, K)
    cagr = M.cagr(rr, ppy).reshape(B, K)
    growth = M.recent_growth(rr, p, ppy).reshape(B, K)
    dd = stats[:, 3].reshape(B, K)
    delta = cp.stack((ws - ref.wsharpe, sh - ref.sharpe, cagr - ref.cagr, growth - ref.recent_growth), axis=2)
    required = return_requirements(dd, ref)
    checks = delta >= required - 1e-10
    valid = valid.reshape(B, K) & unique
    new = cp.asarray([r['expr'] not in ref.keys for r in rows])
    capacity = cp.broadcast_to(~((ref.n >= CFG.max_members) & new)[:, None], weights.shape)
    feasible = checks.all(2) & valid & capacity
    passed = feasible.any(1)
    utility = .5 * cp.tanh(delta[:, :, 0] / CFG.sharpe_scale) + .5 * cp.tanh(delta[:, :, 3] / CFG.growth_scale)
    score = cp.where(feasible, utility, -cp.inf)
    tie = feasible & (utility == score.max(1)[:, None])
    best = cp.argmax(tie & (cagr == cp.where(tie, cagr, -cp.inf).max(1)[:, None]), axis=1)
    margins = delta - required
    scales = cp.asarray([CFG.violation_scale_wsharpe, CFG.violation_scale_sharpe, CFG.violation_scale_cagr,
                         CFG.violation_scale_growth])
    violation = cp.maximum(-margins / scales, 0).sum(2) + cp.where(capacity, 0., 1.)
    near = cp.argmin(cp.where(valid, violation, cp.inf), axis=1)
    chosen = cp.where(passed, best, near)
    ix = cp.arange(B)
    mine = cp.where(va > 0, ma / cp.sqrt(cp.where(va > 0, va, 1)) * ppy ** 0.5, 0)
    rho = cp.where((va > 0) & (vq > 0), cov / cp.sqrt(cp.where((va > 0) & (vq > 0), va * vq, 1)), 0)
    need = rho * ref.wsharpe + 0.6 * cp.sqrt(cp.maximum(1 - rho * rho, 0))
    report = cp.stack((passed, valid.any(1), weights[ix, chosen], ws[ix, chosen], sh[ix, chosen], cagr[ix, chosen],
                       growth[ix, chosen], dd[ix, chosen], unique.sum(1), capacity[ix, chosen], mine, rho, need), axis=1)
    report = cp.concatenate((report, delta[ix, chosen], checks[ix, chosen], utility[ix, chosen, None],
                             violation[ix, chosen, None], required[ix, chosen]), axis=1)
    out = []
    for row, v in zip(rows, cp.asnumpy(report).tolist()):
        row['_view'] = (v[10], v[12], v[11])                   # own Sharpe, required-Sharpe diagnostic, correlation
        if not v[1]:
            out.append(dict(passed=False, checks={}, vetoes=['所有试探组合均产生非有限收益或破产'], weight=0.,
                            trials=int(v[8]), backend=f'cuda-{precision}', reference=ref))
            continue
        ck = {key: (bool(v[17 + j]), v[13 + j]) for j, key in enumerate(KEYS)}
        vetoes = [f'{label} ({v[13 + j]:+.6f},要求 ≥ {v[23 + j]:+.6f})' for j, label in enumerate(LABELS) if not v[17 + j]]
        if not v[9]:
            ck['capacity'] = (False, float(CFG.max_members - ref.n - 1))
            vetoes.append('组合成员数超过上限')
        if not v[0]:
            vetoes.insert(0, f'{int(v[8])} 个权重点未找到可行组合;以下为约束违反最小的试探点')
        out.append(dict(passed=bool(v[0]), checks=ck, vetoes=vetoes, weight=v[2], wsharpe=v[3], sharpe=v[4], cagr=v[5],
                        recent_growth=v[6], maxdd=v[7], trials=int(v[8]), utility=v[21], violation=v[22],
                        backend=f'cuda-{precision}', reference=ref))
    if diagnostics is not None:
        for r, d in zip(out, diagnostics):
            r.update(backend='cuda-f64' if d[0] else 'cuda-mixed',
                     precision=dict(mode='auto', refined=bool(d[0]), calibration_error=d[1], return_envelope=d[2],
                                    over_budget=bool(d[3]), unstable=bool(d[4]), fp64_trials=K * int(d[0]) + 2, fp32_trials=K))
            assert not r['passed'] or d[0], 'an approximate pass escaped precision refinement'
    return out


def _summary(proposal, ref):
    """The portfolio before and, at the chosen weight, after adding the candidate."""
    mine, _, rho = proposal['row']['_view']
    out = dict(n=ref.n, seed=ref.seed, sharpe=ref.wsharpe, full_sharpe=ref.sharpe, cagr=ref.cagr, maxdd=ref.maxdd,
               mine=mine, rho=rho, w=proposal['weight'], trials=proposal['trials'], backend=proposal['backend'],
               feasible=proposal['passed'])
    if 'precision' in proposal:
        out['precision'] = proposal['precision']
    if 'wsharpe' in proposal:
        out.update(sharpe_after=proposal['wsharpe'], full_sharpe_after=proposal['sharpe'], cagr_after=proposal['cagr'],
                   maxdd_after=proposal['maxdd'], delta_sharpe=proposal['checks']['sharpe'][1],
                   delta_cagr=proposal['checks']['cagr'][1],
                   checks={k: dict(passed=bool(v[0]), delta=v[1]) for k, v in proposal['checks'].items()},
                   submission_ready=proposal['passed'] and proposal['sharpe'] >= CFG.submission_sharpe)
    return out


def admit(engine, rows, ref):
    """Verdicts of formula rows (Engine.formulas) against the reference, in candidate batches."""
    assert all(r.get('precision', 'f64') == 'f64' for r in rows), 'experimental mixed rows need FP64 validation before admission'
    results = []
    for lo in range(0, len(rows), engine.candidate_batch):
        part = rows[lo:lo + engine.candidate_batch]
        why = [r['error'] if not r['ok'] else None if r['_valid_rr'] else '无效收益序列:非有限值或破产' for r in part]
        idx = [i for i, w in enumerate(why) if w is None]
        done = dict(zip(idx, proposals(engine, [part[i] for i in idx], ref))) if idx else {}
        for i, r in enumerate(part):
            if i not in done:
                results.append(dict(passed=False, stopped_at='G1', expr=r['expr'], protocol=PROTOCOL,
                                    layers={'G1_structure': dict(passed=False, vetoes=[why[i]])}))
                continue
            p = dict(done[i], row=r)
            results.append(dict(passed=p['passed'], stopped_at=None if p['passed'] else 'G2', expr=r['expr'], protocol=PROTOCOL,
                                layers=dict(G1_structure=dict(passed=True, vetoes=[]), G2_performance=p),
                                combo=_summary(p, ref)))
    return results


def explain(verdict):
    if verdict['passed']:
        return '结构有效;加入后组合 Sharpe 提升且全期/近期收益抵得过回撤变化'
    return '; '.join(str(v) for layer in verdict['layers'].values() for v in layer.get('vetoes', []))


def _json(x):
    if isinstance(x, dict):
        return {k: _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


REPORTED = ('sharpe', 'cagr', 'ann_ret', 'maxdd', 'total_ret', 'turnover', 'exposure', 'margin_bps', 'max_single_weight',
            'concentration', 'span', 'n_days', 'yearly')


def rowify(row, verdict):
    """The JSON row of a formula and its verdict (reports, service responses)."""
    out = dict(protocol=PROTOCOL, expr=row['expr'], ok=bool(row['ok']), error=row.get('error'), passed=bool(verdict['passed']),
               stopped_at=verdict['stopped_at'], explain=explain(verdict), accepted=False,
               backend=verdict.get('combo', {}).get('backend', 'cuda-f64'))
    out.update({k: row[k] for k in REPORTED if k in row})
    if 'combo' in verdict:
        out['combo'] = verdict['combo']
    return _json(out)
