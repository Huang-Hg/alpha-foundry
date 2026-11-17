# L4: reuse program evaluation reviews during commit

[简体中文](README.zh-CN.md) · [Summary JSON](summary.json) · [Stage timings](stages.csv)

When `ResearchContext` has not changed between `evaluate` and `commit`, reusing the evaluation review avoids replaying rejected stateful programs a second time during `commit`. On 256 stateful programs, query plus ordered commit dropped from `42.301 s` to `20.851 s` (`2.03×`), and open-through-close dropped from `44.496 s` to `23.116 s` (`1.92×`), with identical returns, metrics, fills, weights, and decisions.

| Workload | Query before → after (s) | Commit before → after (s) | Query + commit before → after (s) | Speedup | Open through close before → after (s) | Peak device MiB before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `orders-16-shared` | 7.4374 → 6.8756 | 6.8356 → 0.0012 | 14.2729 → 6.8768 | 2.076× | 16.0970 → 8.6815 | 218 → 216 |
| `orders-16-independent` | 8.5230 → 8.5982 | 8.5235 → 0.0013 | 17.0465 → 8.5995 | 1.982× | 18.6756 → 10.2327 | 222 → 222 |
| `orders-256-shared` | 21.1883 → 20.8361 | 21.1126 → 0.0152 | 42.3009 → 20.8513 | 2.029× | 44.4965 → 23.1158 | 250 → 250 |

All candidates in these three stress batches were rejected, so `ResearchContext` stayed constant throughout `commit`. Once an earlier candidate is admitted and changes the portfolio, remaining candidates still replay against the updated portfolio (measured separately in [consecutive admissions](../program-transitions-l4-20261117/README.md)), so this `~2×` ratio applies specifically to unchanged-context commits.

Measured on one NVIDIA L4 (23,034 MiB, driver 580.82.07, one run per condition) over 8 synthetic crypto spot assets, 2 days of 5-minute bars, 192 fifteen-minute decisions, 576 events, and 48 hourly valuation periods with paired-FP32 arithmetic and `Admission(weight_steps=4)`. Return arrays are bit-identical (`0` metric error and `0` decision differences across 521 regression checks). Only the latest host report summary is cached—no return series, holdings, live accounts, or historical batches are retained. Peak memory in [summary.json](summary.json) is a 20 ms sampling lower bound.
