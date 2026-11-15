"""Named portfolio gaps, context features and explicit noise calibration."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType

from foundry.config import PortfolioConfig, SingleConfig
from foundry.search.archive import Archive, standardized
from foundry.search.proposer.random import RandomProposer
from foundry.study.session import ResidentPort
from foundry.types import (
    Candidate,
    ContextConflict,
    ContextFeature,
    Diagnostics,
    DrawdownWindow,
    FloatArray,
    PortfolioView,
    SingleGate,
)

CONTEXT_NAMES = (
    'weighted_sharpe_scaled', 'sharpe_scaled', 'recent_growth', 'geometric_growth',
    'weight_concentration', 'weighted_member_volatility', 'mean_exposure', 'member_fraction',
    'return_block_1', 'return_block_2', 'return_block_3', 'return_block_4',
    'return_block_5', 'return_block_6', 'archive_fraction', 'archive_correlation',
)


@dataclass(frozen=True)
class CalibrationPopulation:
    """Immutable resident return populations; an owner reuses them by snapshot."""
    snapshot: str
    observed: FloatArray
    delayed: FloatArray


@dataclass(frozen=True)
class SingleMeasures:
    sharpe: FloatArray
    cagr: FloatArray
    years: FloatArray
    rho_portfolio: FloatArray
    appraisal: FloatArray


def prepare_calibration(port: ResidentPort, candidates: tuple[Candidate, ...]) -> CalibrationPopulation:
    """Evaluate explicitly supplied calibration candidates and their causal twins."""
    if not candidates:
        raise ValueError('calibration requires a nonempty candidate population')
    populations = []
    with port.scope():
        xp = port.xp
        for delayed in (False, True):
            parts = []
            for lo in range(0, len(candidates), port.candidate_batch):
                batch = port.evaluate(candidates[lo:lo + port.candidate_batch], report=delayed, delayed=delayed)
                rows = [row.returns for row in batch.rows if row.ok and row.valid_returns]
                parts.append(xp.stack(rows) if rows else xp.empty((0, len(port.reference.rr))))
            returns = xp.concatenate(parts)
            populations.append(returns[returns.std(1) > 1e-12])
    return CalibrationPopulation(port.context.snapshot.id, *populations)


def measures(port: ResidentPort, returns: FloatArray, reference: PortfolioView) -> SingleMeasures:
    xp, cfg = port.xp, port.settings.single
    moments = port.measure_returns(returns)
    sharpe, cagr = moments.sharpe, moments.cagr
    years = port.yearly_sharpe(returns, last=cfg.recent_years)
    centered, base = returns - returns.mean(1, keepdims=True), reference.rr - reference.rr.mean()
    norm = xp.sqrt((centered * centered).sum(1) * (base * base).sum())
    rho = xp.where(norm > 0, centered @ base / xp.where(norm > 0, norm, 1), 0)
    appraisal = (sharpe - rho * reference.sharpe) / xp.sqrt(xp.maximum(1 - rho * rho, 1e-12))
    return SingleMeasures(sharpe, cagr, years, rho, appraisal)


def prior(port: ResidentPort, population: CalibrationPopulation, reference: PortfolioView) -> tuple[float, float]:
    if population.snapshot != port.context.snapshot.id:
        raise ContextConflict('calibration population belongs to a different study snapshot')
    observed = measures(port, population.observed, reference).appraisal
    noise = measures(port, population.delayed, reference).appraisal
    return float(observed.mean()), 1 - float(noise.var() / observed.var())


def required_appraisal(reference: PortfolioView, mean: float, shrinkage: float,
                       portfolio_cfg: PortfolioConfig) -> float:
    gain = ((reference.sharpe + portfolio_cfg.min_sharpe_gain) ** 2 - reference.sharpe ** 2) ** .5
    return mean + (gain - mean) / shrinkage if shrinkage > 0 else float('inf')


def pool_target(single_cfg: SingleConfig, portfolio_cfg: PortfolioConfig) -> float:
    n = portfolio_cfg.max_members
    return single_cfg.anchor_sharpe * (n / (1 + (n - 1) * single_cfg.anchor_correlation)) ** .5


def mean_correlation(returns: FloatArray, members: FloatArray, xp: ModuleType) -> FloatArray:
    a, b = returns - returns.mean(1, keepdims=True), members - members.mean(1, keepdims=True)
    norm = xp.sqrt((a * a).sum(1))[:, None] * xp.sqrt((b * b).sum(1))[None]
    return xp.where(norm > 0, a @ b.T / xp.where(norm > 0, norm, 1), 0).mean(1)


def required_sharpe(rho_portfolio: FloatArray, rho_members: FloatArray, reference: PortfolioView,
                    kappa: float, single_cfg: SingleConfig, portfolio_cfg: PortfolioConfig,
                    xp: ModuleType) -> FloatArray:
    n, sp = portfolio_cfg.max_members, reference.sharpe
    unknown = pool_target(single_cfg, portfolio_cfg) * xp.sqrt((1 + (n - 1) * xp.maximum(rho_members, 0)) / n)
    if not math.isfinite(kappa):
        return xp.full_like(unknown, xp.inf)
    peak = (sp * sp + kappa * kappa) ** .5
    known = xp.where(rho_portfolio * peak <= sp,
                     rho_portfolio * sp + kappa * xp.sqrt(xp.maximum(1 - rho_portfolio ** 2, 0)), peak)
    return xp.maximum(unknown, known)


def calibrate(port: ResidentPort, population: CalibrationPopulation) -> SingleGate:
    with port.scope():
        cfg, portfolio_cfg, reference = port.settings.single, port.settings.portfolio, port.reference
        mean, shrinkage = prior(port, population, reference)
        kappa = required_appraisal(reference, mean, shrinkage, portfolio_cfg)
        return SingleGate(mean, shrinkage, reference.sharpe, kappa, cfg.anchor_correlation,
                          cfg.anchor_sharpe, portfolio_cfg.max_members, pool_target(cfg, portfolio_cfg),
                          cfg.recent_years, cfg.recent_year_sharpe)


def drawdown_window(port: ResidentPort, returns: FloatArray) -> DrawdownWindow:
    xp, dates = port.xp, port.stat_dates
    wealth = xp.concatenate((xp.ones(1), xp.cumprod(1 + returns)))
    peaks = xp.maximum.accumulate(wealth)
    ratios = wealth / peaks - 1
    trough = int(xp.argmin(ratios))
    peak = int(xp.argmax(wealth[:trough + 1]))
    recovered = xp.flatnonzero(wealth[trough:] >= wealth[peak])
    recovery = trough + int(recovered[0]) if len(recovered) else None
    return DrawdownWindow(None if peak == 0 else dates[peak - 1], dates[max(0, trough - 1)],
                          None if recovery is None else dates[max(0, recovery - 1)], float(ratios[trough]))


def diagnose(port: ResidentPort, archive: Archive | None = None, *, gate: SingleGate | None = None) -> Diagnostics:
    """Describe portfolio gaps without changing the portfolio or archive."""
    with port.scope():
        xp, ref = port.xp, port.reference
        weights, _, member_returns = port.member_states()
        r, weights = ref.rr, xp.asarray(weights)
        reference_vectors, _ = standardized(xp.concatenate((ref.rr[None], member_returns)), xp)
        archive_vectors = xp.empty((0, len(ref.rr))) if archive is None else archive.vectors(port)
        block = xp.minimum(xp.arange(len(r)) * 6 // len(r), 5)
        means = xp.bincount(block, weights=r, minlength=6) / xp.maximum(xp.bincount(block, minlength=6), 1)
        sig = xp.sqrt(xp.mean(r * r))
        member_vol = member_returns.std(1)
        values = xp.concatenate((xp.asarray([ref.wsharpe / 3, ref.sharpe / 3, ref.recent_growth, ref.cagr]),
            xp.stack((xp.sum(weights * weights), xp.sum(weights * member_vol) * port.ppy ** .5,
                      xp.mean(xp.sum(ref.W, axis=1)), xp.asarray(len(weights) / 40))),
            means / xp.maximum(sig, 1e-8),
            xp.stack((xp.asarray(0. if archive is None else len(archive.items) / max(1, archive.capacity)),
                      xp.mean(archive_vectors @ reference_vectors[0]) if len(archive_vectors) else xp.asarray(0.)))))
        encoded = port.to_host(xp.tanh(xp.nan_to_num(values))).tolist()
        vectors, _ = standardized(member_returns, xp)
        correlations = port.to_host(xp.clip(vectors @ vectors.T, -1, 1)).tolist()
        return Diagnostics(port.context, tuple(ContextFeature(name, value) for name, value in zip(CONTEXT_NAMES, encoded)),
                           drawdown_window(port, ref.rr), tuple(tuple(row) for row in correlations),
                           max(0, port.settings.portfolio.max_members - ref.n), gate)


def sample_calibration(port: ResidentPort) -> CalibrationPopulation:
    """Sample the configured deterministic grammar population for single-alpha diagnostics."""
    settings, cfg = port.settings, port.settings.search
    proposer = RandomProposer(port.vocabulary, cfg.seed, cfg.max_tokens, settings.expr.max_cost,
                              cfg.max_ts_depth, port.window_caps, len(CONTEXT_NAMES))
    batch = proposer.sample(settings.single.calibration_sample, 0, [0.] * len(CONTEXT_NAMES))
    return prepare_calibration(port, tuple(dict.fromkeys(batch.candidates)))


def diagnose_current(port: ResidentPort) -> Diagnostics:
    """Resolve the selected objective's diagnostics using the same calibration math as mining."""
    gate = calibrate(port, sample_calibration(port)) if port.settings.search.objective == 'single' else None
    return diagnose(port, gate=gate)
