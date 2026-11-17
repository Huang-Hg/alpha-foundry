# L4: timed strategies and public integration

[简体中文](README.zh-CN.md) · [Summary JSON](summary.json) · [Stage timings](stages.csv)

**The paired-FP32 workflow passed; the mixed single-FP32 control did not qualify.** Measured on one NVIDIA L4 (23,034 MiB) over 8 synthetic crypto assets, 2 days of 5-minute base bars, 192 decisions at 15-minute intervals, 576 execution events, and 48 hourly measurements. Each condition ran once, without extra warm-ups or repeated seeds.

## Full-scale results

| Candidates | Query + ordered commit | Complete lifecycle | Return max abs error | Metric max scaled error | Decision differences | Sampled peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 arrays | 3.718 s | 7.962 s | 3.71e-14 | 3.31e-08 | 0 | 226 MiB |
| 256 stateful programs | 41.886 s | 43.769 s | 0 | 1.15e-10 | 0 | 250 MiB |

Errors compare paired CPU correctness replay using the same data, candidates, and submission order; the GPU host CPU is not a speed baseline. Scaled metric error is `abs(error) / max(1, abs(reference))` with unchanged tolerances. Standalone program evaluations produced 346 actual fills; target-executor fill counts are unavailable (`None`). Neither stress batch admitted a candidate; the independent integration case below exercised actual admissions.

Query covers candidate execution, statistics, and portfolio weight trials; commit re-judges candidates in order and atomically publishes the portfolio. Complete lifecycle adds session open/load/JIT, candidate generation, registration, required storage, explicit retention, evidence writes, and close, while excluding separate CPU correctness audits. Peak memory is a 20 ms sampling lower bound.

| Stage | 4096 arrays | 256 programs |
| --- | ---: | ---: |
| Open / load / plan / JIT | 0.659 s | 1.519 s |
| Candidate production | 0.098 s | 0.001 s |
| Registration | 0.613 s | 0.004 s |
| Evaluation with trials | 2.062 s | 20.928 s |
| Ordered commit | 1.657 s | 20.958 s |
| Explicit retention | 0.040 s | 0.000 s |
| Evidence writes | 2.782 s | 0.355 s |

Program replay remains the main cost; evidence writes are also material for the array batch. Active arrays are capped at 128 candidates and program state at 256, without constructing a full `4096 × D × S` tensor or caching historical batches. Program conditions explicitly use `Admission(weight_steps=4)`; production defaults remain unchanged, so these timings do not predict the larger default trial grid.

## Precision control: failures remain visible

The 64-array mixed single-FP32 control had a maximum return error of `4.27e-7`, pushing annualized and Sharpe metrics past tolerance and flipping two admission sub-checks (the same candidate at query and commit). Shared and independent 16-program mixed controls also differed in metrics and trial weights, and final benchmark-portfolio fill counts differed by 18; identical accepted sets therefore did not establish metric or event equivalence.

All failed controls remain in [summary.json](summary.json). Because paired-FP32 16-program CPU/CUDA returns were bit-identical (`1.15e-10` maximum scaled metric error, zero decision differences), full scale ran only the preselected paired-FP32 path. No candidate triggered runtime precision switching or recomputation, and unqualified mixed full conditions were not run. This is not a 1.5× qualification of a wide-precision path.

## Independent client and regression

- **Public integration:** an independent client written from the public guides (importing only `foundry`) ran 4 mixed candidates through GPU feedback updates, direct commit, retention/release, and reopen in `115.708 s`. Two non-formula array factors were admitted in order at weight 1 (the second replacing the first), and the stateful program bought and closed actual inventory in 2 fills. One in-sample chooser update demonstrates the feedback API, not model training performance or out-of-sample profitability.
- **Regression:** 508 checks passed and 1 legacy private-production baseline was skipped. Four daily formulas per market matched `6c94ead` in metrics, risk, decisions, and ordered weights, excluding protocol identities and the added null fill-count field. CPU/CUDA checks covered program pause/resume, incomplete statistics periods, pending orders, funding, and liquidation. Two initial smoke fixture issues were corrected with targeted reruns; failed logs remain retained locally.
- **Provenance:** full-scale measurement source SHA-256 `37dfd8be024b8699de70f8f2910fd506ad324d07d1444971737ffca4cf1e9db7` (post-measurement qualification/report revision `33dcd2166f6b2640625e35acb2d56d519a216d44cb003459905ff460d4f932df`, engine arithmetic unchanged). Complete raw artifacts are retained locally (`2fd88c355afce5f5ffe37ae2977bea271887b1240e1bcfd5ad9b888b80353129`).
