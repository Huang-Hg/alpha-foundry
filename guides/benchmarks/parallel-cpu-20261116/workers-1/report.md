# Standalone backend measurements

**PASS** · cpu

This records an independent baseline, not cross-device errors, speedups or full regressions.

OpenMP threads: 1; OpenBLAS threads: 1.

[Metadata](benchmark.json) · [Summary](summary.json) · [Operators](operators.csv) · [Flows](flows.csv) · [Checks](checks.html)

| Candidates | Evaluate / s | Commit / s | Sum / s | Lifecycle / s | Valid / accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.686726 | 2.426299 | 4.113025 | 4.250920 | 64 / 11 |
| 4096 | 107.636748 | 108.023318 | 215.660066 | 215.810862 | 4058 / 28 |

One warm-up and one measurement per condition; compare only matching data, rules and compute sources.
