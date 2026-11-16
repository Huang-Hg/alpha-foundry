# Compact caches and measured GPU comparisons

[English](README.md) · [中文](README.zh-CN.md) · [Operators](operators.csv) · [Flows](flows.csv) · [Comparisons](comparisons.json) · [Source](source.tar.gz) · [Raw reports](details.tar.gz)

On identical US-equity data (269 days × 500 assets) and 4096 formulas, **PRO 6000 recorded the fastest evaluate + ordered commit: 2.7855 s**. Every device used a 32 MiB measurement cache and regenerated candidate books in bounded batches.

| GPU | evaluate / s | commit / s | evaluate + commit / s | lifecycle / s | sampled device peak / MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRO 6000 | 1.252132 | 1.533382 | 2.785514 | 2.887840 | 8156.2 |
| A100 | 2.931202 | 3.759916 | 6.691118 | 6.934042 | 3615.9 |
| L4 | 3.709663 | 4.099566 | 7.809228 | 8.097634 | 3382.0 |

Target operators (window 20, microseconds, including host dispatch and CUDA synchronization):

| GPU | ts_slope CPU | ts_slope CUDA | ts_av_diff CPU | ts_av_diff CUDA |
| --- | ---: | ---: | ---: | ---: |
| PRO 6000 | 130.85 | 152.25 | 87.63 | 83.86 |
| A100 | 314.27 | 153.57 | 208.10 | 125.01 |
| L4 | 312.36 | 179.61 | 209.18 | 114.46 |

Every device produced 4058 valid candidates and 28 admissions. Sources, formula ordering, data snapshots, parameters and initial equal-weight portfolio matched. CSV records every lifecycle stage; each device benchmark.json preserves warm-up and setup/JIT stages.

## Memory and latency tradeoff

The previous 8192 MiB full-book cache recorded a device peak of 11886.2 MiB on PRO 6000; this version recorded 8156.2 MiB. Evaluate + commit changed from 2.3515 s to 2.7855 s. All 4096-formula report values, snapshots, decisions and final weights match the previous version exactly. Historical measurements came from the previous Colab PRO 6000 instance, so this is a same-model comparison.

The cache retains (T,) returns and scalar reports, not candidate (T,S) books or syntax trees. Accepted members own individual copies, preventing views from retaining entire candidate batches. Persistent caches occupy O(C + (b+M)TS), where C is the cache budget, b the execution batch, M the member bound, and T/S days/assets. Temporary expression and reweight workspaces additionally depend on bounded operator cost and trial_batch. Public reports require O(N) host output; explicit resident-device APIs return caller-owned O(NTS) output.

| GPU | candidates | evaluate + commit / s | sampled device peak / MiB |
| --- | ---: | ---: | ---: |
| PRO 6000 | 64 | 0.159292 | 5828.2 |
| PRO 6000 | 256 | 0.634396 | 7668.2 |
| PRO 6000 | 1024 | 0.775167 | 8062.2 |
| PRO 6000 | 4096 | 2.785514 | 8156.2 |
| A100 | 64 | 0.420813 | 1249.9 |
| A100 | 256 | 1.830468 | 3089.9 |
| A100 | 1024 | 2.013612 | 3521.9 |
| A100 | 4096 | 6.691118 | 3615.9 |
| L4 | 64 | 0.403065 | 1016.0 |
| L4 | 256 | 2.093350 | 2856.0 |
| L4 | 1024 | 2.217054 | 3288.0 |
| L4 | 4096 | 7.809228 | 3382.0 |

Device-wide memory is sampled every 50 ms and is a lower bound on the actual peak. It includes the driver, allocator reservations and pools left by earlier checks; it is neither cache size nor an isolated hardware comparison. PRO 6000 ran full regressions before measurements; other measurement processes ran only benchmarks.

## Correctness and measurement scope

All 66 operators and 124 window conditions passed CPU comparisons on each GPU, including signs and NaN/infinity masks. Maximum absolute report change across GPUs and the compact-cache change was 7.28e-12; maximum scaled report error against the PRO host CPU was 7.41e-07. No numerical tolerance or decision check failed. comparisons.json records field-level maxima.

PRO 6000 full acceptance: 563 passed, 1 skipped, 7186 passed subtests, including funding and liquidation boundary regressions. Each device first ran four-market candidate, cache/CAS and operator smoke checks, then one warm-up and one measured round per condition. Other GPUs did not repeat the complete perpetual boundary suite. No seed repeats. The private production baseline was unchanged and explicitly skipped; arithmetic was unchanged and the separate 1.5× single-precision budget was not requalified.

Candidate batch 256, trial batch 16384, eight streams, paired arithmetic; OMP/OpenBLAS each used two threads. Colab hosts have different CPUs and end-to-end timings include Python/host work, so these single observations are not pure GPU-core rankings. benchmark.json/device.txt preserve hardware/software details. The H100 allocation returned HTTP 503; no H100 measurements exist.

Reproduce the complete PRO run:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m foundry --parameters '{"market":"us_equity"}' benchmark var/runs/compact --sizes 64 256 1024 4096
```

Other GPUs used the same pytest workloads, selecting operators and CUDA public flows to avoid repeating the costly CPU batch flow. cloud-job.py preserves exact commands. Source hashes identify the measured working tree; Git metadata in benchmark.json describes the Colab base checkout.
