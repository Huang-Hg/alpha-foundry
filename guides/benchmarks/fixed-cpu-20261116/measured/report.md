# Standalone backend measurements

**PASS** · cpu

This records an independent baseline, not cross-device errors, speedups or full regressions.

OpenMP threads: 8; OpenBLAS threads: 8.

Metadata (local artifact) · [Summary](summary.json) · [Operators](operators.csv) · [Flows](flows.csv) · Checks (local artifact)

| Candidates | Evaluate / s | Commit / s | Sum / s | Lifecycle / s | Valid / accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 14.181038 | 12.903873 | 27.084911 | 27.314381 | 254 / 17 |
| 1024 | 56.662372 | 43.760687 | 100.423059 | 100.644347 | 1013 / 21 |
| 4096 | 218.121768 | 181.853170 | 399.974939 | 400.219412 | 4058 / 28 |

One warm-up and one measurement per condition; compare only matching data, rules and compute sources.
