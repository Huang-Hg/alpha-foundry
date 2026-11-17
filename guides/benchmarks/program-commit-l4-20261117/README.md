# L4: remove repeated replay between program evaluation and commit

[简体中文](README.zh-CN.md) · [Summary](summary.json) · [Stage timings](stages.csv)

For 256 stateful programs, query plus ordered commit fell from **42.301 s to 20.851 s (2.03×)**. The complete open-through-close lifecycle fell from **44.496 s to 23.116 s (1.92×)**. Returns, metrics, fill/risk summaries, weights, and admission decisions match.

## Matched before/after measurements

| Workload | Query before → after (s) | Commit before → after (s) | Query + commit before → after (s) | Speedup | Open through close before → after (s) | Peak device MiB before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `orders-16-shared` | 7.4374 → 6.8756 | 6.8356 → 0.0012 | 14.2729 → 6.8768 | 2.076× | 16.0970 → 8.6815 | 218 → 216 |
| `orders-16-independent` | 8.5230 → 8.5982 | 8.5235 → 0.0013 | 17.0465 → 8.5995 | 1.982× | 18.6756 → 10.2327 | 222 → 222 |
| `orders-256-shared` | 21.1883 → 20.8361 | 21.1126 → 0.0152 | 42.3009 → 20.8513 | 2.029× | 44.4965 → 23.1158 | 250 → 250 |

**Scope: every stress candidate was rejected.** The ~2× gain removes the second standalone evaluation and weight trials when `ResearchContext` is unchanged. Acceptance changes the portfolio and still requires suffix replay (measured separately in [consecutive admissions](../program-transitions-l4-20261117/README.md)); this table does not imply that every mining workload doubles in speed. Regression checks cover accepted-prefix invalidation, RPC, caller-mutated reports, released leases, and failed atomic writes.

## Method and correctness

One NVIDIA L4 (23,034 MiB, driver 580.82.07), one run per condition, with identical harness, native kernels, data, and parameters. Eight synthetic crypto spot assets, two days of 5-minute bars, 192 fifteen-minute decisions, 576 events, and 48 hourly statistics. Paired FP32 arithmetic, `Admission(weight_steps=4)`, unchanged production defaults. No new CPU timing baseline was collected.

- Return arrays are bit-identical; every reported numerical absolute error and decision difference is zero.
- 521 tests and 8,778 subtests passed; one legacy private production baseline was skipped without re-recording.
- Four daily formulas in each of US equities, A-shares, crypto spot, and perpetuals match in metrics, risk, admission, and ordered weights.
- Only the latest host report batch is retained—no returns, positions, live accounts, or historical batches.
- Lifecycle timing includes load/plan/JIT, generation, registration, evaluation, commit, retention, export, evidence writes, release, and close. Peak memory is a 20 ms sampling lower bound.
- All adjacent checks reused one L4 session, released after evidence download. Single observations are not statistical confidence intervals.

## Reproduction notes

Sizes 16 and 256 ran as consecutive commands so the small batch gated the larger one without repeated measurements. The baseline restored the two `study` files from `cc0f2b4` while sharing the current harness and all other files. Published metrics are in [summary.json](summary.json) and [stages.csv](stages.csv); raw logs and archives stay in local `var/runs/`.
