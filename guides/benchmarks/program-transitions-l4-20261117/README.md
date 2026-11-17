# L4: consecutive admissions and portfolio reallocation

[简体中文](README.zh-CN.md) · [Metrics and measured stages](summary.json)

One L4, one run per before/after condition. All three workflows mutate portfolio state; the full regression suite was not repeated.

| Workload (registered / evaluated) | Research before → after (s) | Speedup | Lifecycle before → after (s) | New admissions | Weight change |
| --- | ---: | ---: | ---: | ---: | --- |
| Admissions / shared (4/4) | 82.735 → 55.700 | 1.49× | 110.121 → 57.554 | 2 | [1.0] → [2.0] |
| Reallocation / shared (4/2) | 55.931 → 37.582 | 1.49× | 62.149 → 43.767 | 0 | [0.25, 0.25] → [0.5, 0.5] |
| Reallocation / independent (4/2) | 82.768 → 55.302 | 1.50× | 91.586 → 63.898 | 0 | [0.25, 0.25] → [0.5, 0.5] |
| Admissions / shared (16/16) | 70.818 → 55.568 | 1.27× | 72.659 → 57.371 | 2 | [1.0] → [2.0] |
| Reallocation / shared (16/14) | 56.422 → 37.464 | 1.51× | 62.629 → 43.638 | 0 | [0.25, 0.25] → [0.5, 0.5] |
| Reallocation / independent (16/14) | 83.046 → 55.380 | 1.50× | 91.605 → 63.958 | 0 | [0.25, 0.25] → [0.5, 0.5] |

Candidate-return maximum absolute error: **0**. Metric maximum scaled error: **0**. Decision differences: **0**. Weight feasibility, selected vectors, commit decisions, and pre/post-allocation portfolio returns match across before and after runs.

The first two programs consecutively replace or increase rising-instrument exposure; trailing candidates evaluate against the updated portfolio. Reallocation starts from an explicitly constructed historical two-member library at `0.25 / 0.25`, not counted as new admissions. Independent-account capital stays within `1.0`. Every condition checks stale-report `ContextConflict` rejection and submission after re-evaluation.

Research timing covers initial evaluation, ordered commit, pre-reweight evaluation, weight query/commit, stale-report rejection, and post-reweight evaluation/commit. Lifecycle timing adds historical setup, load/JIT, registration, exports, release, and close. Peak memory in [summary.json](summary.json) is a 20 ms sampling lower bound. Inputs: 8 synthetic crypto spot assets, 576 events, 192 decisions, and 48 valuation periods, with default fees/thresholds and `Admission(weight_steps=4)`. Single observations are not confidence intervals or profitability claims.

Focused smoke passed 32 checks under `pytest` and `pytest-benchmark` (one run per condition, without warm-up repetition). Raw artifacts stay in local `var/runs/` (including the failed initial setup log before a canonical-JSON fix in the local driver on the same L4); GitHub retains this summary and [summary.json](summary.json).

The first 4-candidate baseline includes cold-start and compilation costs, whereas later conditions reuse device and compilation caches; the 16-candidate rows better reflect steady-state workflow cost. Under shared accounts, weights scale emitted order quantities (`2.0` does not imply 200% capital leverage).
