# Fixed Colab CPU reference: A100 and L4

[English](README.md) · [中文](README.zh-CN.md) · [All operators](operators.csv) · [Stages](flows.csv) · Errors (local artifact) · CPU metadata (local artifact) · Thread pools (local artifact)

The fixed reference is a dedicated **Colab CPU high-RAM runtime: Intel Xeon 2.20GHz, eight vCPUs (four cores, two threads/core)**. CPU affinity is 0–7 with no cgroup quota; both OpenMP and OpenBLAS loaded eight-thread pools. No GPU-attached host CPU supplies the denominator.

4096 formulas, 269 days × 500 US equities, 32 MiB measurement cache:

| Backend | Evaluate / s | Commit / s | Sum / s | Speedup vs fixed CPU |
| --- | ---: | ---: | ---: | ---: |
| CPU · 8 threads | 218.121768 | 181.853170 | 399.974939 | 1× |
| A100 | 2.931202 | 3.759916 | 6.691118 | 59.78× |
| L4 | 3.709663 | 4.099566 | 7.809228 | 51.22× |

flows.csv also records complete lifecycle timing, including open/close. The sum above covers evaluation and ordered commit, including operators, account statistics, admission and portfolio updates.

| GPU | Candidates | Fixed CPU / s | GPU / s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| A100 | 64 | 6.870593 | 0.420813 | 16.33× |
| A100 | 256 | 27.084911 | 1.830468 | 14.80× |
| A100 | 1024 | 100.423059 | 2.013612 | 49.87× |
| A100 | 4096 | 399.974939 | 6.691118 | 59.78× |
| L4 | 64 | 6.870593 | 0.403065 | 17.05× |
| L4 | 256 | 27.084911 | 2.093350 | 12.94× |
| L4 | 1024 | 100.423059 | 2.217054 | 45.30× |
| L4 | 4096 | 399.974939 | 7.809228 | 51.22× |

## Operator comparison

Fixed CPU reference, window 20, microseconds; CSV contains all 124 operator/window conditions.

| Operator | CPU · 8 threads | A100 | L4 |
| --- | ---: | ---: | ---: |
| `ts_slope` | 201.30 | 153.57 | 179.61 |
| `ts_av_diff` | 205.49 | 125.01 | 114.46 |

## Results and scope

Every 4096-formula flow produced 4058 valid candidates and 28 admissions. Comparisons against the fixed CPU reference found 0 decision differences, 0 numerical violations and maximum scaled report error 7.41e-07.

CPU completed a 64-formula smoke, then 256/1024/4096 flows and all 66 operators. Every condition has one warm-up and one measured round, with no seed repeats. GPU timings and raw reports are reused from the preceding A100/L4 run; compute kernels, data/execution snapshots, parameters and formula ordering match. Changes only enable standalone benchmark execution and metadata; production numerical kernels are unchanged.

Flow metrics, weights and admission are compared directly with the new eight-thread CPU reports. Operator latency ratios use the fixed CPU timings; historical GPU operator errors retain their original contemporaneous CPU reference because elementwise outputs were not saved. Standalone CPU measurements do not represent full multi-market regressions, a private production audit or a new precision-budget qualification.

Eight threads are enabled for native parallel work, but serial formula dispatch and ordered commit prevent continuous eight-core occupancy. Thread counts were not tuned. GPU wall time also includes its Colab host dispatch. This measures the implementation and selected Colab configurations, not hardware performance ceilings.

## Reproduction and evidence

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE python -m foundry --parameters '{"market":"us_equity"}' benchmark var/runs/cpu-reference --backends cpu --sizes 64 256 1024 4096
```

Measured CPU metadata (local artifact) · Smoke (local artifact) · Checks (local artifact) · CPU raw reports (local artifact) · Measured source (local archive) · Exact cloud job (local artifact) · [GPU evidence](../compact-multigpu-20261116/README.md)
