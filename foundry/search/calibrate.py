"""python -m foundry calibrate RUN_DIR: constraint-violation sizes of a finished search run -> reward scales.

The infeasible-branch reward is  log R = -6 - min(4, V + V_corr),  V = sum_j max(0, -m_j)/s_j
over the four admission margins m_j (recency-weighted Sharpe gain, full Sharpe, CAGR, recent
log growth). If the s_j are small against real damage sizes, V sits on the cap, the reward is
the same -10 for most candidates and neither TB nor the paired teacher can rank them.
Rule: s_j = one common percentile of constraint j's violation over distinct, G1-valid, non-cash
candidates. Among the percentiles in RULES, take the one that spreads candidates widest over
the 4-nat band (V at p90 minus V at p1) while leaving at most MAX_CAPPED of them on the cap.
Cash books get +1 on top of V by construction and are left out so that a fifth of identical
samples does not pin the percentiles. Pure JSON (RUN_DIR/epochs.jsonl), no GPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

from foundry.config import SearchConfig

Scales = tuple[float, float, float, float]

RULES = (.5, .75, .9, .99)
MAX_CAPPED = .10
KEYS = ('sharpe', 'full_sharpe', 'cagr', 'recent_growth')
NAMES = ('violation_scale_wsharpe', 'violation_scale_sharpe', 'violation_scale_cagr', 'violation_scale_growth')


@dataclass(frozen=True)
class CalibrationRule:
    percentile: float
    scales: Scales
    status: Literal['feasible', 'saturated', 'nonpositive_scale', 'nonfinite_scale']
    capped_share: float | None
    spread: float | None


@dataclass(frozen=True)
class CalibrationReport:
    status: Literal['calibrated', 'empty', 'no_feasible_rule']
    candidates: int
    cash_books: int
    unscalable_candidates: int
    rules: tuple[CalibrationRule, ...]
    selected: CalibrationRule | None


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def saturation(rows, scales):
    totals = [sum(v / s for v, s in zip(row, scales)) for row in rows]
    return totals, sum(t >= 4 for t in totals) / len(totals)


def report(run_dir: Path, cfg: SearchConfig) -> CalibrationReport:
    """Report empirical rule feasibility without inventing scales for absent evidence."""
    seen, rows, cash, unscalable = set(), [], 0, 0
    with (run_dir / 'epochs.jsonl').open(encoding='utf-8') as f:
        for line in f:
            for r in json.loads(line)['report']['rows']:
                identity = json.dumps(r['candidate'], sort_keys=True)
                if not r['g1'] or not r['delta'] or identity in seen:
                    continue
                seen.add(identity)
                if r['rho'] == 0:               # undefined correlation = all-cash / constant book
                    cash += 1
                    continue
                checks = {check['name']: check for check in r['delta']}
                if not all(math.isfinite(checks[key]['delta']) and
                           (checks[key]['required'] is None or math.isfinite(checks[key]['required']))
                           for key in KEYS):
                    raise ValueError('Calibration checks must contain finite deltas and finite or unattainable requirements')
                if any(checks[key]['required'] is None for key in KEYS):
                    unscalable += 1
                    continue
                rows.append([max(0., checks[key]['required'] - checks[key]['delta']) for key in KEYS])
    print(f'{len(rows):,} distinct scalable non-cash candidates '
          f'({cash:,} cash books and {unscalable:,} unattainable candidates left out)')
    if not rows:
        status = 'no_feasible_rule' if unscalable else 'empty'
        print(f'Calibration status: {status}; no scale recommendation is available.')
        return CalibrationReport(status, 0, cash, unscalable, (), None)
    print(f'{"violation of":16s} {"share>0":>8s} {"p25":>9s} {"p50":>9s} {"p75":>9s} {"p90":>9s} {"p99":>9s}')
    for j, key in enumerate(KEYS):
        column = [row[j] for row in rows]
        print(f'{key:16s} {sum(v > 0 for v in column) / len(column):8.1%} '
              + ' '.join(f'{quantile(column, q):9.4f}' for q in (.25, .5, .75, .9, .99)))
    old = (cfg.violation_scale_wsharpe, cfg.violation_scale_sharpe, cfg.violation_scale_cagr, cfg.violation_scale_growth)
    rules = []
    for q in RULES:
        scales = tuple(quantile([row[j] for row in rows], q) for j in range(4))
        if not all(math.isfinite(scale) for scale in scales):
            rules.append(CalibrationRule(q, scales, 'nonfinite_scale', None, None))
        elif min(scales) <= 0:
            rules.append(CalibrationRule(q, scales, 'nonpositive_scale', None, None))
        else:
            totals, capped = saturation(rows, scales)
            spread = quantile(totals, .9) - quantile(totals, .01)
            rules.append(CalibrationRule(q, scales, 'feasible' if capped <= MAX_CAPPED else 'saturated',
                                         capped, spread))
    feasible = [rule for rule in rules if rule.status == 'feasible']
    selected = max(feasible, key=lambda rule: rule.spread) if feasible else None
    print(f'\n{"scales":16s} {"wsharpe":>8s} {"sharpe":>8s} {"cagr":>8s} {"growth":>8s} {"on cap":>8s}   V at p1 / p10 / p50 / p90 of candidates')
    display = [('current parameters', old)]
    for rule in rules:
        label = f'p{rule.percentile * 100:.0f} rule' + ' <-' * (rule is selected)
        if rule.status in ('nonpositive_scale', 'nonfinite_scale'):
            print(f'{label:16s} not evaluable: {rule.status}')
        else:
            display.append((label, rule.scales))
    for label, scales in display:
        totals, capped = saturation(rows, scales)
        print(f'{label:16s} ' + ' '.join(f'{s:8.4f}' for s in scales) + f' {capped:8.1%}   '
              + ' / '.join(f'{quantile(totals, q):.2f}' for q in (.01, .1, .5, .9)))
    if selected is None:
        print('\nCalibration status: no_feasible_rule; no scale recommendation is available.')
        return CalibrationReport('no_feasible_rule', len(rows), cash, unscalable, tuple(rules), None)
    print('\nfoundry.Search(')
    for name, value in zip(NAMES, selected.scales):
        print(f'    {name}={value:.2g},')
    print(')')
    return CalibrationReport('calibrated', len(rows), cash, unscalable, tuple(rules), selected)
