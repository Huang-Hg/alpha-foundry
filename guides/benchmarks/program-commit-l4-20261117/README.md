# L4: remove repeated replay between program evaluation and commit

[简体中文](README.zh-CN.md)

For 256 stateful programs, query plus ordered commit fell from **42.301 s to 20.851 s (2.03×)**. The complete open-through-close lifecycle fell from **44.496 s to 23.116 s (1.92×)**. Returns, metrics, fill/risk summaries, weights and admission decisions match exactly.

## Matched before/after measurements

| Workload | Query before → after (s) | Commit before → after (s) | Query + commit before → after (s) | Speedup | Open through close before → after (s) | Peak device MiB before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| orders-16-shared | 7.4374 → 6.8756 | 6.8356 → 0.0012 | 14.2729 → 6.8768 | 2.076× | 16.0970 → 8.6815 | 218 → 216 |
| orders-16-independent | 8.5230 → 8.5982 | 8.5235 → 0.0013 | 17.0465 → 8.5995 | 1.982× | 18.6756 → 10.2327 | 222 → 222 |
| orders-256-shared | 21.1883 → 20.8361 | 21.1126 → 0.0152 | 42.3009 → 20.8513 | 2.029× | 44.4965 → 23.1158 | 250 → 250 |

**Scope: every stress candidate was rejected.** The roughly 2× gain removes the second standalone evaluation and weight trials under an unchanged context. Acceptance changes the portfolio and still requires suffix replay; this is not a claim that every mining workload doubles in speed. Regression covers that branch, RPC, mutated public reports, released leases and failed atomic writes.

## Method and correctness

One NVIDIA L4 (23034 MiB, driver 580.82.07), one run per condition, with identical harness, native kernels, data and parameters. Eight synthetic crypto spot assets, two days of five-minute bars, 192 fifteen-minute decisions, 576 events and 48 hourly statistics. Paired arithmetic, explicitly four weight steps, unchanged production defaults. No new CPU timing baseline.

- Return arrays are bit-identical; every reported numerical absolute error and decision difference is zero.
- 521 tests and 8778 subtests passed; one legacy private production baseline skipped, without rerecording it.
- Four daily formulas in each of US equities, A-shares, crypto spot and perpetuals match exactly in metrics, risk, admission and ordered weights.
- Only the latest host report batch is retained: no returns, positions, accounts or historical batches.
- Lifecycle includes load/plan/JIT, production, registration, evaluation, commit, retention, export, evidence writes, release and close. Peak memory is a 20ms sampling lower bound.
- All adjacent checks used one L4 session, released after verified evidence download. Single observations are not statistical confidence intervals.

## Reproduce

```bash
python -m foundry benchmark BEFORE --suite programs --sizes 16 256 --backends cuda
# Run the first command with the baseline study implementation, then use the updated checkout.
python -m foundry benchmark AFTER --suite programs --sizes 16 256 --backends cuda --reference BEFORE
```

This run split 16 and 256 into consecutive commands so the small comparison gated the large one, without repeated measurements. The baseline restores the two study files from `d96fe84`; the harness and all other files are shared.

[Full results](summary.json) · [Every stage](stages.csv) · [Original cloud evidence](raw.tar.gz) · [Source manifest](deployment.json) · [Measured source](source.tar.gz) · [Baseline delta](baseline-delta.tar.gz) · [Static report binding review](source-review.json)
