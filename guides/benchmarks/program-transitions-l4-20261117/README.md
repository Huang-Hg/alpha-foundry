# L4: consecutive admissions and portfolio reallocation

[简体中文](README.zh-CN.md)

One L4, one run per before/after condition. All three workflows must change portfolio state; the full regression suite was not repeated.

| Workload (registered / evaluated) | Research before → after (s) | Speedup | Lifecycle before → after (s) | New admissions | Weight change |
| --- | ---: | ---: | ---: | ---: | --- |
| Admissions / shared (4/4) | 82.735 → 55.700 | 1.49× | 110.121 → 57.554 | 2 | [1.0] → [2.0] |
| Reallocation / shared (4/2) | 55.931 → 37.582 | 1.49× | 62.149 → 43.767 | 0 | [0.25, 0.25] → [0.5, 0.5] |
| Reallocation / independent (4/2) | 82.768 → 55.302 | 1.50× | 91.586 → 63.898 | 0 | [0.25, 0.25] → [0.5, 0.5] |
| Admissions / shared (16/16) | 70.818 → 55.568 | 1.27× | 72.659 → 57.371 | 2 | [1.0] → [2.0] |
| Reallocation / shared (16/14) | 56.422 → 37.464 | 1.51× | 62.629 → 43.638 | 0 | [0.25, 0.25] → [0.5, 0.5] |
| Reallocation / independent (16/14) | 83.046 → 55.380 | 1.50× | 91.605 → 63.958 | 0 | [0.25, 0.25] → [0.5, 0.5] |

Candidate-return maximum absolute error: **0**. Metric maximum scaled error: **0**. Decision differences: **0**. Weight feasibility, selected vectors, commit decisions and pre/post-allocation portfolio returns pass comparison.

The first two programs consecutively replace/increase rising-instrument exposure; suffix candidates use the successor portfolio. Reallocation starts from an explicitly constructed historical two-member library at 0.25/0.25, not counted as new admissions. Independent capital stays within one. Every case checks stale-report conflict and submission after reevaluation.

Research timing includes initial evaluation/ordered commit, pre-weight evaluation, weight query/commit, stale rejection and post-weight evaluation/commit. Lifecycle also includes historical setup, load/JIT, registration, exports, release and close. Peak memory is a 20ms sampling lower bound. Eight synthetic crypto spot assets, 576 events, 192 decisions and 48 statistics; default fees and admission thresholds, explicitly four weight steps. Single observations are not confidence intervals or profitability claims.

Focused smoke: 32 passing checks. pytest and pytest-benchmark execute each condition once without warmup repetition. Raw artifacts stay local; GitHub retains this summary, machine-readable metrics and source checksums.

The initial small job failed canonical JSON validation before measurement. The local driver was corrected and resumed on the same L4; the failed setup log is preserved.

The first four-candidate baseline also pays cold-start/compilation costs; later conditions naturally reuse device and compilation caches. The sixteen-candidate measurements better describe steady workflow cost. Shared-account weights scale program allocation; 2.0 does not mean 200% capital exposure.

[Metrics and every measured stage](summary.json)
