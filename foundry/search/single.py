"""Single-alpha proposal rewards, with final admission owned by Research.

Correlation and recent-year diagnostics shape search rewards only. They cannot
admit a candidate rejected by the shared portfolio rules or choose commit weight.
The calibration population is local to an archive and immutable study snapshot.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
from foundry.candidates import Formula

from foundry.core import metrics as M
from foundry.search.evaluator import Evaluator, MiningResult
from foundry.search.policy import Policy


@dataclass(frozen=True)
class SingleGate:
    prior_mean: float
    shrinkage: float
    reference_sharpe: float
    required_appraisal: float
    anchor_correlation: float
    anchor_sharpe: float
    pool_size: int
    pool_target: float
    recent_years: int
    recent_year_sharpe: float


def population(research, archive):
    """Ordinary candidates and forward-delayed twins for this study snapshot."""
    engine, context = research.engine, research.context
    key = context.snapshot.id
    if key not in archive.populations:
        settings, xp = engine.settings, engine.xp
        search, cfg = settings.search, settings.single
        policy = Policy(engine.vocabulary, search.seed, search.max_tokens, settings.expr.max_cost,
                        search.max_ts_depth, (engine.backend.max_window, engine.backend.max_pair_window))
        expressions = list(dict.fromkeys(policy.sample(cfg.calibration_sample, 0, [0.] * 16, random=True)['expressions']))
        populations = []
        for delayed in (False, True):
            parts = []
            for lo in range(0, len(expressions), engine.candidate_batch):
                sources = tuple(Formula(expression) for expression in expressions[lo:lo + engine.candidate_batch])
                batch = (research.evaluate_noise_device(sources, context.id) if delayed
                         else research.evaluate_device(sources, context.id))
                rows = [row.returns for row in batch.rows if row.ok and row.valid_returns]
                parts.append(xp.stack(rows) if rows else xp.empty((0, len(engine.dates) - engine.start)))
            returns = xp.concatenate(parts)
            populations.append(returns[returns.std(1) > 1e-12])
        archive.populations[key] = tuple(populations)
    return archive.populations[key]


def prior(research, archive, reference):
    real, twins = population(research, archive)
    engine = research.engine
    observed, noise = measures(engine, real, reference)[4], measures(engine, twins, reference)[4]
    return float(observed.mean()), 1 - float(noise.var() / observed.var())


def required_appraisal(reference, mean, shrinkage, portfolio_cfg):
    gain = ((reference.sharpe + portfolio_cfg.min_sharpe_gain) ** 2 - reference.sharpe ** 2) ** .5
    return mean + (gain - mean) / shrinkage if shrinkage > 0 else float('inf')


def pool_target(single_cfg, portfolio_cfg):
    n = portfolio_cfg.max_members
    return single_cfg.anchor_sharpe * (n / (1 + (n - 1) * single_cfg.anchor_correlation)) ** .5


def mean_correlation(returns, members, xp):
    a, b = returns - returns.mean(1, keepdims=True), members - members.mean(1, keepdims=True)
    norm = xp.sqrt((a * a).sum(1))[:, None] * xp.sqrt((b * b).sum(1))[None]
    return xp.where(norm > 0, a @ b.T / xp.where(norm > 0, norm, 1), 0).mean(1)


def required_sharpe(rho_portfolio, rho_members, reference, kappa, single_cfg, portfolio_cfg, xp):
    n, sp = portfolio_cfg.max_members, reference.sharpe
    unknown = pool_target(single_cfg, portfolio_cfg) * xp.sqrt((1 + (n - 1) * xp.maximum(rho_members, 0)) / n)
    if not math.isfinite(kappa):
        return xp.full_like(unknown, xp.inf)
    peak = (sp * sp + kappa * kappa) ** .5
    known = xp.where(rho_portfolio * peak <= sp,
                     rho_portfolio * sp + kappa * xp.sqrt(xp.maximum(1 - rho_portfolio ** 2, 0)), peak)
    return xp.maximum(unknown, known)


def measures(engine, returns, reference):
    xp, cfg = engine.xp, engine.settings.single
    sharpe = M.sharpe(returns, engine.ppy, xp)
    cagr = M.cagr(returns, engine.ppy, xp)
    years = engine.years.sharpe(returns, engine.ppy, last=cfg.recent_years)
    centered, base = returns - returns.mean(1, keepdims=True), reference.rr - reference.rr.mean()
    norm = xp.sqrt((centered * centered).sum(1) * (base * base).sum())
    rho = xp.where(norm > 0, centered @ base / xp.where(norm > 0, norm, 1), 0)
    appraisal = (sharpe - rho * reference.sharpe) / xp.sqrt(xp.maximum(1 - rho * rho, 1e-12))
    return sharpe, cagr, years, rho, appraisal


def utility(sharpe, maxdd, cagr, cfg, xp):
    u_s = xp.tanh(xp.maximum(sharpe, 0) / cfg.sharpe_scale)
    u_d = xp.maximum(0, 1 - xp.abs(maxdd) / cfg.drawdown_full)
    u_r = xp.tanh(xp.maximum(cagr, 0) / cfg.return_scale)
    return u_s * (cfg.weight_sharpe + cfg.weight_drawdown * u_d + cfg.weight_return * u_r)


class SingleEvaluator(Evaluator):
    COUNTERS = Evaluator.COUNTERS + ('g3',)

    def __init__(self, research, archive, rho_max):
        super().__init__(research, archive, rho_max)
        self.cfg, self.portfolio_cfg = self.engine.settings.single, self.engine.portfolio_cfg
        mean, shrinkage = prior(research, archive, self.ref)
        self.kappa = required_appraisal(self.ref, mean, shrinkage, self.portfolio_cfg)
        self.gate = SingleGate(mean, shrinkage, self.ref.sharpe, self.kappa, self.cfg.anchor_correlation,
                               self.cfg.anchor_sharpe, self.portfolio_cfg.max_members,
                               pool_target(self.cfg, self.portfolio_cfg), self.cfg.recent_years,
                               self.cfg.recent_year_sharpe)

    def _valid_batch(self, pairs):
        xp, cfg = self.engine.xp, self.cfg
        rows, verdicts = zip(*pairs)
        books = xp.stack([row.book for row in rows])
        returns = xp.stack([row.returns for row in rows])
        maxdd = xp.stack([xp.asarray(row.maxdd) for row in rows])
        _, nonzero, rho, low_corr, novelty = self._behavior(returns)
        sharpe, cagr, years, rho_p, appraisal = measures(self.engine, returns, self.ref)
        rho_m = mean_correlation(returns, self.member_returns, xp)
        required = required_sharpe(rho_p, rho_m, self.ref, self.kappa, cfg, self.portfolio_cfg, xp)
        room = self.ref.n < self.portfolio_cfg.max_members
        worth = nonzero & (sharpe >= required)
        gate = worth & (years > cfg.recent_year_sharpe).all(1)
        admitted = xp.asarray([verdict.passed for verdict in verdicts])
        feasible = gate & low_corr & room & admitted
        violation = (xp.maximum(required - sharpe, 0) / cfg.violation_scale_sharpe
                     + xp.maximum(cfg.recent_year_sharpe - years, 0).sum(1) / cfg.violation_scale_year
                     + xp.maximum(0, rho - self.rho_max) + (~nonzero) + (~admitted) + (0 if room else 1))
        score = utility(sharpe, maxdd, cagr, cfg, xp)
        reward = xp.where(feasible, 1 + 3 * score, -6 - xp.minimum(4, violation))
        host = self.engine.to_host(xp.concatenate((xp.stack((reward, feasible, gate, rho, novelty, nonzero,
                         score, sharpe, cagr, maxdd, rho_p, appraisal, required, worth, rho_m), axis=1), years), axis=1)).tolist()
        staged = OrderedDict()
        for i, (row, verdict, values, key) in enumerate(zip(rows, verdicts, host, self.engine.fingerprints(books))):
            duplicate = self._duplicate(key, books[i])
            accepted = bool(values[1]) and not duplicate
            proposal = verdict.proposal
            self.cache[row.source.expression] = MiningResult(row.source.expression, values[0] if not duplicate else -10., accepted, True,
                bool(values[13]), rho=values[3], utility=values[6], novelty=values[4], duplicate=duplicate,
                weight=proposal.weight, delta=proposal.checks, precision=proposal.precision, backend=proposal.backend,
                g3=bool(values[2]), sharpe=values[7], cagr=values[8], maxdd=values[9], rho_portfolio=values[10],
                rho_members=values[14], appraisal=values[11], required_sharpe=values[12], recent_years=tuple(values[15:]))
            self.counters['g1'] += 1
            self.counters['g2'] += int(values[13])
            self.counters['g3'] += int(values[2])
            self.counters['low_corr'] += int(accepted)
            self.counters['behavior_duplicates'] += int(duplicate)
            self.counters['weight_trials'] += proposal.trials
            self.counters['fp64_refined'] += int(proposal.backend.endswith('-f64'))
            if accepted and values[6] > self.best_score:
                self.best, self.best_score = row.source.expression, values[6]
            if not duplicate and values[5]:
                staged[key] = i
                staged.move_to_end(key)
        self._stage(staged, books, returns, rows)
