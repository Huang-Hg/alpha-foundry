# L4: timed strategies and order programs

[简体中文](README.zh-CN.md) · [Summary JSON](summary.json) · [Stage timings](stages.csv)

Measured on one NVIDIA L4 (23,034 MiB) over 8 synthetic crypto assets, 2 days of 5-minute base bars, 192 decisions at 15-minute intervals, 576 execution events, and 48 hourly measurements (one run per condition). Paired-FP32 runs passed CPU/CUDA numerical and admission parity; single-FP32 (`mixed`) controls exceeded metric tolerances and were excluded from full-scale runs.

## Full-scale results

| Candidates | Query + ordered commit | Complete lifecycle | Return max abs error | Metric max scaled error | Decision differences | Sampled peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 arrays | 3.718 s | 7.962 s | 3.71e-14 | 3.31e-08 | 0 | 226 MiB |
| 256 stateful programs | 41.886 s | 43.769 s | 0 | 1.15e-10 | 0 | 250 MiB |

Errors compare paired CPU replay on identical data, candidates, and submission order (`abs(error) / max(1, abs(reference))`). Standalone program evaluations produced 346 fills (`None` on target-only executors). Query covers candidate execution, statistics, and portfolio weight trials; commit re-judges candidates in order and publishes the portfolio; complete lifecycle adds open/load/JIT, candidate generation, registration, retention, report writes, and close (excluding separate CPU replay). Peak memory is sampled every 20 ms.

| Stage | 4096 arrays | 256 programs |
| --- | ---: | ---: |
| Open / load / plan / JIT | 0.659 s | 1.519 s |
| Candidate generation | 0.098 s | 0.001 s |
| Registration | 0.613 s | 0.004 s |
| Evaluation with trials | 2.062 s | 20.928 s |
| Ordered commit | 1.657 s | 20.958 s |
| Explicit retention | 0.040 s | 0.000 s |
| Output writes | 2.782 s | 0.355 s |

Active resident batches are capped at 128 array candidates and 256 program states without materializing a `4096 × D × S` tensor. Program runs set `Admission(weight_steps=4)` rather than the default 20-step grid.

## Single-FP32 (`mixed`) controls

On the 64-array `mixed` control, maximum return error reached `4.27e-7`, pushing annualized return and Sharpe past tolerance and flipping two admission sub-checks (one at query, one at commit). On the 16-program shared and independent `mixed` controls, metrics and trial weights diverged and final portfolio fill counts differed by 18 fills despite matching admitted IDs. All failed control records remain in [summary.json](summary.json). Paired-FP32 16-program CPU/CUDA returns were bit-identical (`1.15e-10` max scaled metric error, `0` decision differences).

An end-to-end integration client using only public `foundry` APIs also exercised GPU reward feedback, sequential array admissions, stateful program buy/close fills, `retain`/`release`, and session reopen (`115.708 s`), alongside 508 passing regression checks across all four markets.
