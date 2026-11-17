# L4: remove repeated replay between program evaluation and commit

[简体中文](README.zh-CN.md) · [Summary JSON](summary.json) · [Stage timings](stages.csv)

For 256 stateful programs, query plus ordered commit fell from **42.301 s to 20.851 s (2.03×)**. The complete open-through-close lifecycle fell from **44.496 s to 23.116 s (1.92×)**. Returns, metrics, fill/risk summaries, weights, and admission decisions match bit-for-bit.

## Matched before/after measurements

| Workload | Query before → after (s) | Commit before → after (s) | Query + commit before → after (s) | Speedup | Open through close before → after (s) | Peak device MiB before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `orders-16-shared` | 7.4374 → 6.8756 | 6.8356 → 0.0012 | 14.2729 → 6.8768 | 2.076× | 16.0970 → 8.6815 | 218 → 216 |
| `orders-16-independent` | 8.5230 → 8.5982 | 8.5235 → 0.0013 | 17.0465 → 8.5995 | 1.982× | 18.6756 → 10.2327 | 222 → 222 |
| `orders-256-shared` | 21.1883 → 20.8361 | 21.1126 → 0.0152 | 42.3009 → 20.8513 | 2.029× | 44.4965 → 23.1158 | 250 → 250 |

**Scope: every stress candidate in this batch was rejected.** The ~2× speedup comes from reusing the session's recent host evaluation and weight-trial report when `commit` runs under an unchanged `ResearchContext`. Once an early candidate is admitted, subsequent candidates still replay against the updated portfolio—see [L4 consecutive admissions and reallocation](../program-transitions-l4-20261117/README.md) for state-changing commits. Regression tests cover accepted-prefix invalidation, RPC parity, caller-mutated reports, released leases, and failed atomic writes.

## Method and correctness

Measured on a single NVIDIA L4 (23,034 MiB, driver 580.82.07), one run per condition, with identical harness, native kernels, data, and parameters: 8 synthetic crypto spot assets, 2 days of 5-minute bars, 192 fifteen-minute decisions, 576 events, and 48 hourly valuation periods. Paired FP32 arithmetic, `Admission(weight_steps=4)`, unchanged production defaults.

- Return arrays are bit-identical; maximum absolute error and decision differences across all reported fields are zero.
- 521 tests and 8,778 subtests passed (1 legacy private production baseline skipped without re-recording).
- Four daily formulas in each of US equities, A-shares, crypto spot, and perpetuals match identically in metrics, risk, admission, and ordered weights.
- At most one candidate batch of host report metadata is retained—never returns, positions, live accounts, or historical batches.
- Complete lifecycle covers load/plan/JIT, candidate generation, registration, evaluation, commit, retention, export, release, and close. Peak memory is a 20 ms NVML sampling lower bound.

## Evidence

Published summaries live in [summary.json](summary.json) and [stages.csv](stages.csv); raw logs and archives remain in local `var/runs/` (recorded at baseline `cc0f2b4` vs. `521ac5a` using the local `pytest` / `pytest-benchmark` suite).
