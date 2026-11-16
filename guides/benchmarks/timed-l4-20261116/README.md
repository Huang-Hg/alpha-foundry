# L4: timed strategies and public integration

[简体中文](README.zh-CN.md) · [Detailed evidence](summary.json) · [Stage timings](stages.csv)

**The paired FP32 workflow passed; the mixed FP32 control did not qualify.** One NVIDIA L4 (23034 MiB), eight synthetic crypto assets, two days of 5-minute bars, 192 decisions at 15-minute intervals, 576 execution events and 48 hourly measurements. Each condition ran once, without extra warmups or repeated seeds.

## Full-scale results

| Candidates | Query + ordered commit | Complete lifecycle | Return max abs error | Metric max scaled error | Decision differences | Sampled peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 arrays | 3.718 s | 7.962 s | 3.71e-14 | 3.31e-08 | 0 | 226 MiB |
| 256 stateful programs | 41.886 s | 43.769 s | 0 | 1.15e-10 | 0 | 250 MiB |

Errors compare paired CPU correctness replay using the same data, candidates and order. The GPU host CPU is not a speed baseline. Scaled metric error is abs(error)/max(1,abs(reference)); existing tolerances were unchanged. Standalone program evaluations produced 346 actual fills. Target-executor fill counts are unavailable. Neither stress batch admitted a candidate; the independent integration case below exercised actual admissions.

Query includes candidate execution, statistics and portfolio weight trials. Commit judges candidates in order and atomically publishes the portfolio. Complete lifecycle also includes open/load/JIT, production, registration, required storage, explicit retention, evidence writes and close. Separate CPU correctness audits are excluded. Peak memory is a 20 ms sampling lower bound.

| Stage | 4096 arrays | 256 programs |
| --- | ---: | ---: |
| Open / load / plan / JIT | 0.659 s | 1.519 s |
| Candidate production | 0.098 s | 0.001 s |
| Registration | 0.613 s | 0.004 s |
| Evaluation with trials | 2.062 s | 20.928 s |
| Ordered commit | 1.657 s | 20.958 s |
| Explicit retention | 0.040 s | 0.000 s |
| Evidence writes | 2.782 s | 0.355 s |

Program replay remains the main cost; evidence writes are also material for the array batch. Active arrays are capped at 128 candidates and program state at 256. No complete 4096-by-decision-by-asset input or historical candidate cache is constructed. Program conditions explicitly use Admission(weight_steps=4); production defaults remain unchanged, so these timings do not predict the larger default trial grid.

## Precision control: failures remain visible

The 64-array mixed control had a maximum return error of 4.27e-7, but annualized and Sharpe metrics exceeded tolerance. Two admission subchecks differed: the same candidate at query and commit. Shared/independent 16-program controls also differed in metrics and trial weights; final benchmark-portfolio fill counts differed by 18. Identical accepted sets therefore did not establish metric or event equivalence.

All original failures remain in the evidence. The paired 16-program CPU/CUDA returns were bit-identical, maximum scaled metric error was 1.15e-10 and decisions matched. Full scale therefore used the preselected paired production path only. No candidate triggered precision switching or recomputation, and the unqualified mixed full conditions were not run. This is not a 1.5x qualification of a wide-precision path.

## Independent Agent and regression

GPT-6.1 Sol wrote the integration client from public guides, importing only foundry from the project. Four mixed candidates completed GPU feedback learning, direct commit, retention/release and reopen in 115.708 s. Two non-formula array factors were admitted in order at weight one; the second replaced the first. The stateful program bought and closed actual inventory in two fills. One in-sample chooser update demonstrates the feedback API, not model training performance or out-of-sample profitability.

508 checks passed; one legacy private-production baseline was skipped. Four daily formulas per market matched R4 6c94ead exactly in metrics, risk, decisions and ordered weights, excluding protocol identities and the new null fill-count field. CPU/CUDA checks cover program pause/resume, incomplete statistics periods, pending orders, funding and liquidation. Two initial smoke fixture issues were corrected with targeted reruns; failed logs remain retained.

## Reproduce

```bash
python -m foundry benchmark var/runs/timed-smoke --suite timed --sizes 64 --backends cuda
python -m foundry benchmark var/runs/timed-full --suite timed --sizes 4096 --backends cuda --qualification var/runs/timed-smoke/qualification.json
```

Run only on authorized cloud compute. Qualification binds source data, parameters and existing numerical evidence. Full scale uses the same data and rules. GPU measurements, CPU correctness replay and reporting remain separate, and qualification never overwrites failed raw controls. The benchmark generates data inside its output directory.

Final source SHA256: `33dcd2166f6b2640625e35acb2d56d519a216d44cb003459905ff460d4f932df`. Full measurement source: `37dfd8be024b8699de70f8f2910fd506ad324d07d1444971737ffca4cf1e9db7`.

After measurement, changes were limited to qualification, CPU-reference/report code and client explanatory text; engine arithmetic and GPU candidate generation were unchanged. The complete raw archive is retained locally with SHA256: `2fd88c355afce5f5ffe37ae2977bea271887b1240e1bcfd5ad9b888b80353129`.
