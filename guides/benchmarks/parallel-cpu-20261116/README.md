# Fixed Colab CPU: parallel candidates and weight trials

[English](README.md) · [简体中文](README.zh-CN.md)

On one fixed CPU-only Colab instance, 4096 formulas take **106.645 s instead of 215.660 s: 2.02× faster** with 8 workers. The 1/8-worker reports have zero numerical differences and identical candidate order, admission decisions and final weights: 4058 valid, 28 admitted.

## End-to-end timing

Evaluation includes operators, allocation, accounts, statistics and admission weight trials. Ordered commit includes necessary reevaluation and trials against the evolving portfolio before publication. Evaluation + commit excludes open/loading/warm-up; the final column separately includes open through close.

| 4096 formulas | Evaluate | Ordered commit | Combined | Open through close |
| --- | ---: | ---: | ---: | ---: |
| CPU 1 worker | 107.637 s | 108.023 s | 215.660 s | 215.811 s |
| CPU 8 worker | 54.997 s | 51.648 s | 106.645 s | 106.783 s |
| A100 | 2.931 s | 3.760 s | 6.691 s | 6.934 s |
| L4 | 3.710 s | 4.100 s | 7.809 s | 8.098 s |

A100/L4 are **15.94× / 13.66×** faster than this fixed parallel CPU. At 64 formulas, CPU improves from 4.113 s to 2.098 s; A100/L4 take 0.421/0.403 s.

## Concurrency and memory

**AMD EPYC 7B12, 8 vCPUs (guest topology: 4 cores × 2 threads)**, affinity 0–7, no cgroup CPU-time quota. Both worker counts use the same instance, source, binaries and candidate order. Process CPU time / wall time rises from 0.99 to **3.50 cores** for the large workload.

One process shares immutable panels across worker threads. Native operators release the GIL; account kernels already did. Inner OpenMP and BLAS pools use one thread. Portfolio evolution remains an ordered single writer.

CPU candidate batches are 32; GPU batches are 256. Both use 32 MiB compact return/report caches. At most eight CPU tasks are outstanding. No panel-per-process copies, retained expression intermediates or full trial-book cache are used. Temporary trial books scale as O(workers*T*S), excluding required return/event outputs. Periodic VmHWM observations reached 612.5/636.4 MiB for 1/8 workers; these are observed lower bounds, not complete peak tracing.

## Validation and limits

- Smoke: 69 checks and 6693 subtests passed in 9.58 s, covering concurrency, failure draining, four-market accounts, funding and liquidation. Serial/parallel cash accounts and perpetual events pass bitwise comparisons.
- Each worker condition: 158 checks and 6660 subtests passed, including 124 cases across all 66 operators and 64/4096-formula workflows. One warm-up and one measured round per condition; no repeated seeds.
- Worker-report maximum absolute error: 0. Previous CPU and A100/L4 semantic reports also pass, with no candidate-order, admission, reason-code or selected-weight mismatches.
- Maximum GPU/CPU scaled error: 7.40983567e-07, using abs(actual-reference)/max(1,abs(reference)). Per-field errors and limits are in comparisons.json.
- This is not a full GPU/RPC/private-production regression. Private baselines were not changed or rerecorded.

A100/L4 reuse the 2026-11-16 measured runs on identical data, formulas and trading rules; no new GPU was allocated. CUDA numerical kernels and admission rules are unchanged. Shared formula scheduling only moves the original stream/cache partition into the resource owner. Reviewed source differences and GPU report hashes are recorded. These are archived cross-device measurements, not a claim that the new GPU scheduling wrapper was remeasured. The earlier 400 s Xeon result uses different CPU hardware and is not an optimization denominator.

Inputs: 269 daily rows × 500 US equities, fixed 4096-formula corpus, 0.0005 costs, close decisions with next-open execution. CPU uses its native arithmetic and CUDA fixed paired FP32; there is no result-dependent recalculation.

## Reproduction and evidence

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m foundry \
  --parameters '{"market":"us_equity","cpu":{"workers":8}}' \
  benchmark var/runs/cpu-parallel-new --backends cpu --sizes 64 4096
```

[Flow timings](flows.csv) · Semantic comparisons (local artifact) · [Summary](summary.json) · Source (local archive) · Raw CPU reports and logs (local artifact) · Checksums (local artifact)

[1 worker](workers-1/report.md) · [8 workers](workers-8/report.md) · [GPU evidence](../compact-multigpu-20261116/README.md)
