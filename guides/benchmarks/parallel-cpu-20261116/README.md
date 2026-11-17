# Fixed Colab CPU: parallel candidates and weight trials

[简体中文](README.zh-CN.md) · [Flow timings](flows.csv) · [Summary JSON](summary.json) · [1-worker report](workers-1/report.md) · [8-worker report](workers-8/report.md)

On one fixed CPU-only Colab runtime, 4,096 formulas complete evaluation plus ordered commit in **106.645 s instead of 215.660 s (2.02× faster)** with 8 workers. Single-worker and 8-worker reports have zero numerical differences and identical candidate order, admission decisions, and final weights (4,058 valid, 28 admitted).

## End-to-end timing

Evaluation covers operators, allocation, account simulation, statistics, and admission weight trials. Ordered commit covers required re-evaluation and trials against the evolving portfolio prior to atomic publication. The final column adds session open, data loading, and teardown.

| 4096 formulas | Evaluate | Ordered commit | Combined | Open through close |
| --- | ---: | ---: | ---: | ---: |
| CPU (`workers=1`) | 107.637 s | 108.023 s | 215.660 s | 215.811 s |
| CPU (`workers=8`) | 54.997 s | 51.648 s | 106.645 s | 106.783 s |
| NVIDIA A100 | 2.931 s | 3.760 s | 6.691 s | 6.934 s |
| NVIDIA L4 | 3.710 s | 4.100 s | 7.809 s | 8.098 s |

Against this fixed parallel CPU baseline, A100 and L4 achieve **15.94×** and **13.66×** speedups on 4,096 formulas. At 64 formulas, CPU drops from `4.113 s` to `2.098 s` (`0.421 s` on A100, `0.403 s` on L4).

## Concurrency and memory

- **Host:** AMD EPYC 7B12, 8 vCPUs (4 cores × 2 threads, affinity `0–7`, no cgroup CPU quota). Process CPU-time / wall-time ratio rose from `0.99` to **`3.50` cores** on the 4,096-formula run.
- **Execution model:** a single process shares immutable market panels across worker threads while native C++ operator and account kernels release the GIL (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` inside each worker). Portfolio commits remain single-writer to preserve deterministic ordering.
- **Memory bounds:** `CPU(candidate_batch=32, measurement_cache_mib=32)` vs. `CUDA(candidate_batch=256, measurement_cache_mib=32)`. Temporary trial books scale as `O(workers × T × S)`; sampled `VmHWM` reached `612.5 MiB` (1 worker) and `636.4 MiB` (8 workers).

## Validation scope

- **CPU smoke:** 69 checks (6,693 subtests) passed in `9.58 s`, verifying serial/parallel bitwise parity across cash accounts and isolated perpetual liquidations.
- **Per-worker benchmark:** 158 checks (6,660 subtests) passed per condition across all 66 operators (124 window cases) and 64/4,096-formula workflows, with 1 warm-up and 1 measured run.
- **GPU comparison:** A100 and L4 rows reference the matched 2026-11-16 measurements (`269 × 500` US equities, `cost_rate=0.0005`, max GPU/CPU scaled error `7.41e-7`). Raw CPU/GPU logs and archives are retained locally in `var/runs/`.
