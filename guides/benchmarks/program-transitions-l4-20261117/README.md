# L4: consecutive admissions and portfolio reallocation

[简体中文](README.zh-CN.md) · [Summary JSON](summary.json)

Measured on one NVIDIA L4 (one run per before/after condition) across workflows that mutate portfolio state through consecutive admissions and two-member historical reweighting.

| Workload (registered / evaluated) | Research before → after (s) | Speedup | Lifecycle before → after (s) | New admissions | Weight change |
| --- | ---: | ---: | ---: | ---: | --- |
| Admissions / shared (4/4) | 82.735 → 55.700 | 1.49× | 110.121 → 57.554 | 2 | [1.0] → [2.0] |
| Reallocation / shared (4/2) | 55.931 → 37.582 | 1.49× | 62.149 → 43.767 | 0 | [0.25, 0.25] → [0.5, 0.5] |
| Reallocation / independent (4/2) | 82.768 → 55.302 | 1.50× | 91.586 → 63.898 | 0 | [0.25, 0.25] → [0.5, 0.5] |
| Admissions / shared (16/16) | 70.818 → 55.568 | 1.27× | 72.659 → 57.371 | 2 | [1.0] → [2.0] |
| Reallocation / shared (16/14) | 56.422 → 37.464 | 1.51× | 62.629 → 43.638 | 0 | [0.25, 0.25] → [0.5, 0.5] |
| Reallocation / independent (16/14) | 83.046 → 55.380 | 1.50× | 91.605 → 63.958 | 0 | [0.25, 0.25] → [0.5, 0.5] |

Before and after runs match with `0` candidate-return error, `0` scaled metric error, and `0` decision differences across weight feasibility, selected vectors, commit results, and pre/post-allocation portfolio returns.

In the consecutive-admission runs, the first two programs sequentially replace or expand rising-asset exposure while remaining candidates re-evaluate against the updated portfolio. Reallocation runs start from a two-member portfolio initialized at `0.25 / 0.25` (not counted as new admissions), keeping independent-account capital within `1.0` and verifying `ContextConflict` rejection on stale reports prior to re-evaluation.

Research timing covers initial evaluation, ordered commit, pre-reweight evaluation, weight query/commit, stale-report rejection, and post-reweight evaluation/commit; lifecycle timing adds setup, load/JIT, registration, exports, release, and close. Inputs cover 8 synthetic crypto spot assets, 576 events, 192 decisions, and 48 valuation periods with `Admission(weight_steps=4)`; peak memory in [summary.json](summary.json) is sampled every 20 ms. The first 4-candidate row includes initial cold-start and compilation overhead, so the 16-candidate rows better reflect steady-state cost. Under shared accounts, weights scale emitted order quantities (`2.0` is an order-size multiplier rather than 200% capital leverage).
