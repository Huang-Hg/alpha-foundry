# Fixed Colab CPU: parallel candidates and weight trials

[简体中文](README.zh-CN.md) · [Flow timings](flows.csv) · [Summary JSON](summary.json) · [1-worker report](workers-1/report.md) · [8-worker report](workers-8/report.md)

On one fixed CPU-only Colab instance, 4,096 formulas take **106.645 s instead of 215.660 s (2.02×)** with 8 workers. The 1-worker and 8-worker reports have zero numerical differences and identical candidate order, admission decisions, and final weights (4,058 valid, 28 admitted).

## End-to-end timing

Evaluation covers operators, allocation, accounts, statistics, and admission weight trials. Ordered commit covers required re-evaluation and trials against the evolving portfolio before publication. Combined excludes session open, loading, and warm-up; the final column spans open through close.

| 4096 formulas | Evaluate | Ordered commit | Combined | Open through close |
| --- | ---: | ---: | ---: | ---: |
| CPU (`workers=1`) | 107.637 s | 108.023 s | 215.660 s | 215.811 s |
| CPU (`workers=8`) | 54.997 s | 51.648 s | 106.645 s | 106.783 s |
| NVIDIA A100 | 2.931 s | 3.760 s | 6.691 s | 6.934 s |
| NVIDIA L4 | 3.710 s | 4.100 s | 7.809 s | 8.098 s |

Relative to this fixed 8-worker CPU run, the archived A100 and L4 runs are **15.94×** and **13.66×** faster on 4,096 formulas. At 64 formulas, CPU moves from `4.113 s` to `2.098 s` (`0.421 s` on A100, `0.403 s` on L4).

## Concurrency and memory

- **Host:** AMD EPYC 7B12, 8 vCPUs (guest topology: 4 cores × 2 threads, affinity `0–7`, no cgroup CPU-time quota), same instance, binaries, and candidate order for both worker counts. Process CPU time / wall time rose from `0.99` to **`3.50` cores** on the 4,096-formula run.
- **Execution:** one process shares immutable panels across worker threads; native operators and account kernels release the GIL while inner `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` stay at `1`. Portfolio evolution remains single-writer.
- **Memory bounds:** CPU candidate batches are 32 (GPU: 256), both with 32 MiB compact return/report caches and at most 8 outstanding CPU tasks; no per-process panel copies, retained expression intermediates, or full trial-book caches are used. Temporary trial books scale as `O(workers × T × S)`. Periodic `VmHWM` observations reached `612.5 MiB` (`1` worker) and `636.4 MiB` (`8` workers); these are observed lower bounds, not complete peak tracing.

## Validation and limits

- **Smoke and worker checks:** smoke passed 69 checks (6,693 subtests) in `9.58 s` with bitwise serial/parallel parity for cash accounts and perpetual events. Each worker condition passed 158 checks (6,660 subtests) across all 66 operators (124 window cases) and 64/4,096-formula workflows, using 1 warm-up and 1 measured round without repeated seeds.
- **Numerical comparison:** worker-report maximum absolute error is `0`. Maximum GPU/CPU scaled error is `7.40983567e-07` (`abs(actual - reference) / max(1, abs(reference))`), with zero candidate-order, admission, reason-code, or selected-weight mismatches.
- **Scope limits:** this run is not a full GPU/RPC/private-production regression; private baselines were neither changed nor rerecorded. A100 and L4 rows reuse the archived 2026-11-16 runs on identical inputs (`269 × 500` US equities, `0.0005` costs, close decisions with next-open execution, native CPU arithmetic vs. fixed paired-FP32 CUDA, no result-dependent recalculation); they are cross-device reference points rather than a remeasurement of the updated GPU scheduling wrapper. Earlier 400 s Xeon measurements used different CPU hardware and are not used as a speedup denominator. Raw logs and source archives remain in local `var/runs/`.
