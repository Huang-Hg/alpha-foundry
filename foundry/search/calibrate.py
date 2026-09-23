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
from pathlib import Path

from foundry.config import PortfolioConfig

RULES = (.5, .75, .9, .99)
MAX_CAPPED = .10
KEYS = ('sharpe', 'full_sharpe', 'cagr', 'recent_growth')
NAMES = ('violation_scale_wsharpe', 'violation_scale_sharpe', 'violation_scale_cagr', 'violation_scale_growth')


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def saturation(rows, scales):
    totals = [sum(v / s for v, s in zip(row, scales)) for row in rows]
    return totals, sum(t >= 4 for t in totals) / len(totals)


def report(run_dir: Path):
    cfg = PortfolioConfig()
    gain = (cfg.min_sharpe_gain, 0., 0., 0.)
    seen, rows, cash = set(), [], 0
    with (run_dir / 'epochs.jsonl').open(encoding='utf-8') as f:
        for line in f:
            for r in json.loads(line)['reports']:
                if not r.get('g1') or not r.get('delta') or r['expr'] in seen:
                    continue
                seen.add(r['expr'])
                if r['rho'] == 0:               # undefined correlation = all-cash / constant book
                    cash += 1
                    continue
                rows.append([max(0., g - r['delta'][k]) for k, g in zip(KEYS, gain)])
    print(f'{len(rows):,} distinct valid non-cash candidates ({cash:,} cash books left out)')
    print(f'{"violation of":16s} {"share>0":>8s} {"p25":>9s} {"p50":>9s} {"p75":>9s} {"p90":>9s} {"p99":>9s}')
    for j, key in enumerate(KEYS):
        column = [row[j] for row in rows]
        print(f'{key:16s} {sum(v > 0 for v in column) / len(column):8.1%} '
              + ' '.join(f'{quantile(column, q):9.4f}' for q in (.25, .5, .75, .9, .99)))
    old = (cfg.violation_scale_wsharpe, cfg.violation_scale_sharpe, cfg.violation_scale_cagr, cfg.violation_scale_growth)
    rules = {q: tuple(quantile([row[j] for row in rows], q) for j in range(4)) for q in RULES}
    spread = {}
    for q, scales in rules.items():
        totals, capped = saturation(rows, scales)
        if capped <= MAX_CAPPED:
            spread[q] = quantile(totals, .9) - quantile(totals, .01)
    chosen = max(spread, key=spread.get)
    print(f'\n{"scales":16s} {"wsharpe":>8s} {"sharpe":>8s} {"cagr":>8s} {"growth":>8s} {"on cap":>8s}   V at p1 / p10 / p50 / p90 of candidates')
    for label, scales in [('config.ini now', old)] + [(f'p{q * 100:.0f} rule' + ' <-' * (q == chosen), v) for q, v in rules.items()]:
        totals, capped = saturation(rows, scales)
        print(f'{label:16s} ' + ' '.join(f'{s:8.4f}' for s in scales) + f' {capped:8.1%}   '
              + ' / '.join(f'{quantile(totals, q):.2f}' for q in (.01, .1, .5, .9)))
    print('\n[portfolio]')
    for name, value in zip(NAMES, rules[chosen]):
        print(f'{name} = {value:.2g}')
