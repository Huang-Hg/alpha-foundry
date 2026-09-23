"""Single-alpha objective: a candidate is scored on its own book and admitted by a (correlation, Sharpe) gate.

Gate G2: the Sharpe a candidate needs as a function of its correlation, required = max(unknown, known).

unknown - the portfolio the pool will end up in is not known when an alpha is mined, so the pool is taken as
exchangeable: N = [portfolio] max_members alphas, each with Sharpe S and pairwise correlation rho. Their maximum-Sharpe
portfolio is the equal-weight one, with
    Sharpe_N(S, rho) = S * sqrt(N / (1 + (N-1)*rho)),
so alphas of one kind can build a portfolio of Sharpe T exactly when
    S >= T * sqrt((1 + (N-1)*rho) / N):
rising in rho (correlated alphas add no breadth, each must carry the target alone: S -> T as rho -> 1) and low for
uncorrelated ones (S -> T/sqrt(N)). T is fixed by the anchor ([single] anchor_correlation, anchor_sharpe): an alpha
at correlation 0.7 needs Sharpe 1.5, i.e. T = 1.78 for N = 40. rho of a candidate = its mean correlation with the
members of the pool (what enters the variance of an equal-weight portfolio); the 0.9 cap on any single correlation
stays a separate check.

known - against the CURRENT portfolio P (Sharpe S_p, correlation rho_p with the candidate A) the best mix of the two
has Sharpe^2 = S_p^2 + AR^2, AR = (S_a - rho_p*S_p)/sqrt(1-rho_p^2) (Treynor-Black: AR is the Sharpe of the part of A
that P does not hold), so candidates on S_a = rho_p*S_p + kappa*sqrt(1-rho_p^2) add the same. That mix gives P the
weight ~ S_p - rho_p*S_a, negative beyond rho* = S_p/sqrt(S_p^2+kappa^2); books add with weights in (0,1] here, the
portfolio cannot be shorted, so from rho* on the best mix is A alone and the requirement stays at its peak
sqrt(S_p^2+kappa^2) instead of falling back to S_p. kappa prices estimation noise: with the empirical-Bayes posterior
mean E[AR | observed] = m + s*(observed - m) the expected portfolio Sharpe rises by D = [portfolio] min_sharpe_gain when
    observed AR >= kappa = m + (sqrt((S_p + D)^2 - S_p^2) - m)/s.
m and s are measured against the current reference on a fixed population of ordinary candidates (uniform random
policy of the search grammar, seeded) and their delayed twins (Engine.formulas(null_shift=True): same books, signals
1-3 years late, so no information about the days they trade):
    m = mean observed AR of the population,   s = 1 - Var(AR of the twins)/Var(AR of the population).
The twins' spread is what estimation noise (plus the spread of plain cost and exposure structure) produces on these
sparse, fat-tailed books - 2026-09-21 it was 1.8x the textbook 1/sqrt(years). As the pool improves, m falls and s
shrinks, kappa rises, and this part closes the gate by itself once a candidate's apparent edge is mostly noise.
Both parts never fall as their correlation rises. Against the equal-weight start (both correlations are then the one
with the market) the requirement is 0.69 / 0.89 / 1.15 / 1.39 / 1.50 / 1.60 / 1.69 at rho 0 / .2 / .4 / .6 / .7 / .8 / .9.

Gate G3: every one of the last [single] recent_years calendar years must have Sharpe > recent_year_sharpe.
Gates by number (report flags and counters g1, g2, g3): G1 a sound causal book, G2 a non-empty book with
Sharpe >= required, G3 = G2 and the recent-years rule. After G3 only the single-correlation cap and duplicates remain.

Reward. Outside the gate: -6 - min(4, V), V = scaled shortfalls of the gate conditions. Inside: 1 + 3*U,
U = u_S*(w_S + w_D*u_D + w_R*u_R): Sharpe multiplies the score (main dimension), drawdown and return modulate it.
Admitted alphas join the pool at weight 1; the periodic CEM reweighting is the same one the portfolio objective uses.
"""
from __future__ import annotations

from collections import OrderedDict
import math

import cupy as cp

from foundry.config import ExprConfig, SearchConfig, SingleConfig
from foundry.core import metrics as M
from foundry.ops import cuda
from foundry.portfolio.reference import CFG as PORTFOLIO
from foundry.search.evaluator import Evaluator
from foundry.search.policy import Policy

CFG = SingleConfig()
_POPULATION = []


def population(engine):
    """(daily returns of ordinary candidates, of their delayed twins), device-resident, built once per process."""
    if not _POPULATION:
        search = SearchConfig()
        policy = Policy(engine.vocabulary, search.seed, search.max_tokens, ExprConfig().max_cost, search.max_ts_depth,
                        (cuda.MAX_WINDOW, cuda.MAX_PAIR_WINDOW))
        expressions = list(dict.fromkeys(policy.sample(CFG.calibration_sample, 0, [0.] * 16, random=True)['expressions']))
        for delayed in (False, True):
            parts = []
            for lo in range(0, len(expressions), engine.candidate_batch):
                rows = engine.formulas(expressions[lo:lo + engine.candidate_batch], report=False, null_shift=delayed)
                parts.append(cp.stack([r['_rr'] for r in rows if r['ok'] and r['_valid_rr']]))
            R = cp.concatenate(parts)
            _POPULATION.append(R[R.std(1) > 1e-12])               # all-cash books have no Sharpe to speak of
    return _POPULATION


def prior(engine, reference):
    """(m, s) of the module docstring against this reference."""
    real, twins = population(engine)
    observed, noise = measures(engine, real, reference)[4], measures(engine, twins, reference)[4]
    return float(observed.mean()), 1 - float(noise.var() / observed.var())


def required_appraisal(reference, mean, shrinkage):
    gain = ((reference.sharpe + PORTFOLIO.min_sharpe_gain) ** 2 - reference.sharpe ** 2) ** .5
    # No spread beyond noise: the posterior is the (negative) prior mean whatever is observed, nothing can pass.
    return mean + (gain - mean) / shrinkage if shrinkage > 0 else float('inf')


def pool_target():
    """T of the module docstring: the portfolio Sharpe that N exchangeable alphas at the anchor build."""
    n = PORTFOLIO.max_members
    return CFG.anchor_sharpe * (n / (1 + (n - 1) * CFG.anchor_correlation)) ** .5


def mean_correlation(R, members):
    """Mean correlation of every candidate's daily returns (B, T) with the members' (n, T); 0 for an all-cash book."""
    a, b = R - R.mean(1, keepdims=True), members - members.mean(1, keepdims=True)
    norm = cp.sqrt((a * a).sum(1))[:, None] * cp.sqrt((b * b).sum(1))[None]
    return cp.where(norm > 0, a @ b.T / cp.where(norm > 0, norm, 1), 0).mean(1)


def required_sharpe(rho_portfolio, rho_members, reference, kappa):
    """Device array: max(unknown, known) of the module docstring."""
    n, sp = PORTFOLIO.max_members, reference.sharpe
    unknown = pool_target() * cp.sqrt((1 + (n - 1) * cp.maximum(rho_members, 0)) / n)   # exchangeable alphas cannot average below 0
    if not math.isfinite(kappa):
        return cp.full_like(unknown, cp.inf)
    peak = (sp * sp + kappa * kappa) ** .5
    known = cp.where(rho_portfolio * peak <= sp, rho_portfolio * sp + kappa * cp.sqrt(cp.maximum(1 - rho_portfolio ** 2, 0)), peak)
    return cp.maximum(unknown, known)


def measures(engine, R, reference):
    """Device arrays per candidate: Sharpe, CAGR, Sharpe of each recent calendar year (B, recent_years),
    correlation with the reference returns, appraisal ratio against them."""
    sharpe = M.sharpe(R, engine.ppy)
    cagr = M.cagr(R, engine.ppy)
    years = engine.years.sharpe(R, engine.ppy, last=CFG.recent_years)
    centered, base = R - R.mean(1, keepdims=True), reference.rr - reference.rr.mean()
    norm = cp.sqrt((centered * centered).sum(1) * (base * base).sum())
    rho = cp.where(norm > 0, centered @ base / cp.where(norm > 0, norm, 1), 0)
    appraisal = (sharpe - rho * reference.sharpe) / cp.sqrt(cp.maximum(1 - rho * rho, 1e-12))
    return sharpe, cagr, years, rho, appraisal


def utility(sharpe, maxdd, cagr):
    u_s = cp.tanh(cp.maximum(sharpe, 0) / CFG.sharpe_scale)
    u_d = cp.maximum(0, 1 - cp.abs(maxdd) / CFG.drawdown_full)
    u_r = cp.tanh(cp.maximum(cagr, 0) / CFG.return_scale)
    return u_s * (CFG.weight_sharpe + CFG.weight_drawdown * u_d + CFG.weight_return * u_r)


class SingleEvaluator(Evaluator):
    COUNTERS = Evaluator.COUNTERS + ('g3',)                       # G1 sound book > G2 Sharpe >= required(correlation) > G3 every recent year too

    def __init__(self, portfolio, archive, rho_max=.9):
        super().__init__(portfolio, archive, rho_max)
        mean, shrinkage = prior(self.engine, self.ref)            # the reference is frozen for the epoch, so is the gate
        self.kappa = required_appraisal(self.ref, mean, shrinkage)
        self.gate = dict(prior_mean=mean, shrinkage=shrinkage, reference_sharpe=self.ref.sharpe,
                         required_appraisal=self.kappa if math.isfinite(self.kappa) else None,
                         anchor_correlation=CFG.anchor_correlation, anchor_sharpe=CFG.anchor_sharpe,
                         pool_size=PORTFOLIO.max_members, pool_target=pool_target(),
                         recent_years=CFG.recent_years, recent_year_sharpe=CFG.recent_year_sharpe)

    def evaluate(self, expressions):
        self.check_context()
        self.counters['requested'] += len(expressions)
        fresh = [x for x in dict.fromkeys(expressions) if x not in self.cache]
        for lo in range(0, len(fresh), self.engine.candidate_batch):
            rows = self.engine.formulas(fresh[lo:lo + self.engine.candidate_batch], report=False,
                                        null_shift=self.portfolio.null_shift)
            self.counters['unique_evaluations'] += len(rows)
            for r in rows:
                self.cache[r['expr']] = dict(expr=r['expr'], log_reward=-12., passed=False, g1=False,
                                             g2=False, g3=False, rho=None, utility=None, error=r['error'])
            valid = [r for r in rows if r['ok'] and r['_valid_rr']]
            if valid:
                self._score(valid)
        return [self.cache[x] for x in expressions]

    def _score(self, rows):
        W = cp.stack([r['_W'] for r in rows])
        R = cp.stack([r['_rr'] for r in rows])
        maxdd = cp.stack([r['_maxdd'] for r in rows])
        _, nonzero, rho, low_corr, novelty = self._behavior(R)
        sharpe, cagr, years, rho_p, appraisal = measures(self.engine, R, self.ref)
        rho_m = mean_correlation(R, self.member_returns)
        required = required_sharpe(rho_p, rho_m, self.ref, self.kappa)
        room = self.ref.n < PORTFOLIO.max_members
        worth = nonzero & (sharpe >= required)
        gate = worth & (years > CFG.recent_year_sharpe).all(1)
        feasible = gate & low_corr & room
        violation = (cp.maximum(required - sharpe, 0) / CFG.violation_scale_sharpe
                     + cp.maximum(CFG.recent_year_sharpe - years, 0).sum(1) / CFG.violation_scale_year
                     + cp.maximum(0, rho - self.rho_max) + (~nonzero) + (0 if room else 1))
        score = utility(sharpe, maxdd, cagr)
        reward = cp.where(feasible, 1 + 3 * score, -6 - cp.minimum(4, violation))
        keys = self._fingerprints(W)
        host = cp.asnumpy(cp.concatenate((cp.stack((reward, feasible, gate, rho, novelty, nonzero, score, sharpe, cagr,
                                                    maxdd, rho_p, appraisal, required, worth, rho_m), axis=1), years), axis=1)).tolist()
        staged = OrderedDict()
        for i, (r, v, key) in enumerate(zip(rows, host, keys)):
            duplicate = self._duplicate(key, W[i])
            accepted = bool(v[1]) and not duplicate
            open_gate = math.isfinite(v[12])                        # a closed gate has no finite required Sharpe to report
            report = dict(expr=r['expr'], log_reward=v[0] if not duplicate else -10., passed=accepted, g1=True,
                          g2=bool(v[13]), g3=bool(v[2]), rho=v[3], novelty=v[4], duplicate=duplicate, utility=v[6], weight=1.,
                          delta=dict(sharpe=v[7] - v[12] if open_gate else None, cagr=v[8]), sharpe=v[7], cagr=v[8],
                          maxdd=v[9], rho_portfolio=v[10], rho_members=v[14], appraisal=v[11],
                          required_sharpe=v[12] if open_gate else None, recent_years=v[15:], backend='cuda-f64')
            self.cache[r['expr']] = report
            self.counters['g1'] += 1
            self.counters['g2'] += int(v[13])
            self.counters['g3'] += int(v[2])
            self.counters['low_corr'] += int(accepted)
            self.counters['behavior_duplicates'] += int(duplicate)
            if accepted and (self.best is None or v[6] > self.best[1]['utility']):
                self.best = (dict(r, _W=r['_W'].copy(), _rr=r['_rr'].copy()), report)
            if not duplicate and v[5]:
                staged[key] = i
                staged.move_to_end(key)
        self._stage(staged, W, R, rows)

    def accept(self):
        self.check_context()
        if self.best is None:
            return None
        row, report = self.best
        self.engine.remember(row['expr'], row['_W'], row['_rr'], self.portfolio.null_shift)
        self.portfolio.library.add(row['expr'], 1.)
        self.portfolio.changed()
        return dict(expr=row['expr'], weight=1., utility=report['utility'])
