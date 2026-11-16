# TMA history reuse and faster evaluate-to-commit

[English](README.md) · [中文](README.zh-CN.md) · [66 operators](optimized/operators.csv) · [Full report](optimized/report.md) · [Decision comparison](before-after.json) · [Measured source](source.tar.gz)

On the same Colab PRO 6000 and US-equity input (269 days × 500 assets), evaluate plus ordered commit for 4096 formulas fell from **2.6609 s to 2.3515 s, a 1.13× speedup**.

| Operator (window 20) | CPU / μs | Old CUDA / μs | New CUDA / μs | CUDA gain |
| --- | ---: | ---: | ---: | ---: |
| `ts_slope` | 132.87 | 233.17 | 152.52 | 1.53× |
| `ts_av_diff` | 89.07 | 118.29 | 82.74 | 1.43× |

See [target-operators.csv](target-operators.csv) for windows 2/20/120. Timings include transpose packing, TMA transfer and host synchronization, not just kernel execution.

`ts_av_diff` beats CPU in all three measured windows. `ts_slope` improves by 1.46–1.66× over the previous CUDA implementation, but still trails CPU by approximately 8–33%. These single observations do not establish statistical rankings.

| 4096-formula CUDA stage | Before / s | After / s |
| --- | ---: | ---: |
| open_s | 0.074404 | 0.070756 |
| evaluate_s | 1.172996 | 1.064080 |
| commit_s | 1.487911 | 1.287382 |
| evaluate_commit_s | 2.660907 | 2.351462 |
| close_s | 0.035720 | 0.044667 |
| lifecycle_s | 2.771104 | 2.466954 |

Sampled device-wide peak memory changed from 5824 MiB to 11886 MiB. These are sampled lower bounds, not allocator peaks. Retention trades memory for less repeated computation.

TMA bulk asynchronous global-to-shared transfers and transaction barriers stage four interleaved assets per warp, each with eight output tiles. Histories are reused in shared memory; a separate pointwise step normalizes outputs without changing addition or division order. The actual CuPy cache cubin contains `UBLKCP.S.G`; see [gpu-feature.json](gpu-feature.json). Arithmetic precision is unchanged.

Implementation reference: [NVIDIA asynchronous copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html). TMA is available from compute capability 9.0; earlier devices use cooperative loading of the same layout. Histories exceeding shared-memory capacity use global reads of the same recurrence, planned before launch. Other devices were not benchmarked.

Measurement retention now uses an allocation-group memory budget instead of a single execution batch. Set `CUDA(measurement_cache_mib=8192)` to tune it, or 0 to disable it; GPU startup caps the budget at 20% of free memory. Keys still bind snapshot, source, precision and delay. Commit preserves CAS and rejudges against the new portfolio after each acceptance. Four-market checks verify that commit reuses queried measurements and matches cold-cache ordered commit.

Before/after 4096-formula reports: maximum numeric change 0, identical decisions, weights and final portfolio; 4058 valid and 28 admitted. Full acceptance: 561 passed, 1 skipped, 5 warnings, 7184 subtests passed in 245.72s. All 66 operators and 124 window cases passed error and sign/NaN/infinity-mask checks; CPU/CUDA flows passed for 1/64/4096 candidates.

This session began with smoke and the baseline. The first layout failed the performance gate ([record](target/benchmark.json)); moving normalization out of the recurrence passed the targeted check ([record](target-normalize/benchmark.json)), followed by one final complete regression/benchmark. Each run used one warm-up and one measured call per condition, without repeated seeds. Baseline: `e0a281f`, with only the build script normalized to LF. The unchanged private production baseline was explicitly skipped; the single-FP32 1.5× precision budget was not requalified. Short-history results do not establish long-history speed; elementary launch overhead and version-dependent admission work remain.
