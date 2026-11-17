# L4: timed strategies and public integration

[简体中文](README.zh-CN.md) · [Summary JSON](summary.json) · [Stage timings](stages.csv)

**The paired-FP32 workflow passed all accuracy and decision gates; the single-FP32 (`mixed`) control failed and remains excluded from full-scale runs.** Measured on one NVIDIA L4 (23,034 MiB) over 8 synthetic crypto assets, 2 days of 5-minute base bars, 192 fifteen-minute decisions, 576 execution events, and 48 hourly valuation periods, with one measured run per condition.

## Full-scale results

| Candidates | Query + ordered commit | Complete lifecycle | Return max abs error | Metric max scaled error | Decision differences | Sampled peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 arrays | 3.718 s | 7.962 s | 3.71e-14 | 3.31e-08 | 0 | 226 MiB |
| 256 stateful programs | 41.886 s | 43.769 s | 0 | 1.15e-10 | 0 | 250 MiB |

Errors are measured against CPU correctness replay on identical data, candidates, and submission order (`abs(error) / max(1, abs(reference))` with unchanged tolerances). Standalone program evaluations generated 346 fills (`None` on target-only executors). Neither stress batch passed hard admission; real admissions were verified in the independent client run below.

| Stage | 4096 arrays | 256 programs |
| --- | ---: | ---: |
| Open / load / plan / JIT | 0.659 s | 1.519 s |
| Candidate generation | 0.098 s | 0.001 s |
| Registration | 0.613 s | 0.004 s |
| Evaluation + weight trials | 2.062 s | 20.928 s |
| Ordered commit | 1.657 s | 20.958 s |
| Explicit retention | 0.040 s | 0.000 s |
| Report / artifact writes | 2.782 s | 0.355 s |

Active resident leases are bounded at 128 array candidates and 256 program candidates without materializing a monolithic `4096 × D × S` tensor or caching historical batches. Program runs used `Admission(weight_steps=4)` (production default remains 20).

## Precision control: why single-FP32 (`mixed`) failed

On the 64-array `mixed` control, maximum return error was `4.27e-7`, pushing annualized return and Sharpe beyond tolerance and flipping two admission sub-checks (one at query, one at commit). On the 16-program shared/independent `mixed` controls, trial weights and metrics diverged and final benchmark-portfolio fill counts differed by 18 fills—even though the admitted candidate IDs happened to match.

All failed control records are preserved in [summary.json](summary.json). Because paired-FP32 16-program CPU/CUDA returns were bit-identical (`1.15e-10` max scaled metric error, zero decision differences), full-scale benchmarks ran exclusively on the preselected paired-FP32 path without runtime precision fallback.

## Independent client & regression coverage

- **Public-API client check:** an independent client importing only `foundry` ran 4 mixed candidates through GPU reward feedback, ordered commit, `retain`/`release`, and session reopen in `115.708 s` (two array candidates admitted sequentially; stateful program executed buy and close across 2 fills).
- **Regression suite:** 508 checks passed (1 legacy private production baseline skipped). Four daily formulas across all four markets matched prior milestone `6c94ead` in metrics, risk, decisions, and weights; CPU/CUDA parity passed for program pause/resume checkpoints, partial valuation periods, resting orders, funding, and liquidation.
- **Provenance:** full-scale measurement source SHA-256 `37dfd8be024b8699de70f8f2910fd506ad324d07d1444971737ffca4cf1e9db7` (post-report harness/doc revision `33dcd2166f6b2640625e35acb2d56d519a216d44cb003459905ff460d4f932df`); raw archive retained locally in `var/runs/` (`2fd88c355afce5f5ffe37ae2977bea271887b1240e1bcfd5ad9b888b80353129`).
