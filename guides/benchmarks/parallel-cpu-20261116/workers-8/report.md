# Standalone backend measurements

**PASS** · cpu

This records an independent baseline, not cross-device errors, speedups or full regressions.

OpenMP threads: 1; OpenBLAS threads: 1.

Metadata (local artifact) · [Summary](summary.json) · [Operators](operators.csv) · [Flows](flows.csv) · Checks (local artifact)

| Candidates | Evaluate / s | Commit / s | Sum / s | Lifecycle / s | Valid / accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.869723 | 1.227880 | 2.097603 | 2.244804 | 64 / 11 |
| 4096 | 54.997264 | 51.647825 | 106.645089 | 106.782930 | 4058 / 28 |

One warm-up and one measurement per condition; compare only matching data, rules and compute sources.
