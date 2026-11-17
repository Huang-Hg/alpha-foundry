# Fixed Colab CPU: parallel candidate evaluation and weight trials

[简体中文](README.zh-CN.md) · [Flow timings](flows.csv) · [Summary JSON](summary.json) · [1-worker report](workers-1/report.md) · [8-worker report](workers-8/report.md)

On one CPU-only Colab instance (AMD EPYC 7B12, 8 vCPUs, 4 cores × 2 threads, affinity `0–7`), 4,096 US equity formulas completed evaluation plus ordered commit in `106.645 s` with 8 workers compared with `215.660 s` on 1 worker (`2.02×`). Single-worker and 8-worker reports match with `0` numerical error and identical candidate ordering, admission decisions, and final weights (`4,058` valid, `28` admitted).

| 4096 formulas | Evaluate | Ordered commit | Combined | Open through close |
| --- | ---: | ---: | ---: | ---: |
| CPU (`workers=1`) | 107.637 s | 108.023 s | 215.660 s | 215.811 s |
| CPU (`workers=8`) | 54.997 s | 51.648 s | 106.645 s | 106.783 s |
| NVIDIA A100 | 2.931 s | 3.760 s | 6.691 s | 6.934 s |
| NVIDIA L4 | 3.710 s | 4.100 s | 7.809 s | 8.098 s |

Relative to this 8-worker CPU run, the archived A100 and L4 runs on identical inputs (`269 × 500` US equities, `cost_rate=0.0005`, close decisions with next-open execution) are `15.94×` and `13.66×` faster. At 64 formulas, CPU drops from `4.113 s` to `2.098 s` (`0.421 s` on A100, `0.403 s` on L4).

One process shares read-only market panels across worker threads while C++ operator and account kernels release the GIL (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` inside each worker), raising process CPU-time / wall-time from `0.99` to `3.50` cores on the 4,096-formula run. Portfolio commits remain single-writer. CPU batches are capped at 32 candidates (GPU: 256) with 32 MiB return/report caches; periodic `VmHWM` samples reached `612.5 MiB` (`1` worker) and `636.4 MiB` (`8` workers).

Each worker condition passed 158 checks (6,660 subtests) across all 66 operators (124 window cases) and 64/4,096-formula workflows after 1 warm-up round, alongside 69 concurrency/account smoke checks (6,693 subtests). Maximum GPU/CPU scaled metric error against the archived A100/L4 runs is `7.40983567e-07` (`abs(actual - reference) / max(1, abs(reference))`).
