# Optimizing CUDA operators that lag CPU

[English](README.md) · [中文](README.zh-CN.md) · [Before](baseline/report.md) · [After](optimized/report.md) · [Before/after decisions](before-after.json) · [Measured source](source.tar.gz)

On the same Colab PRO 6000 and US-equity inputs (269 days x 500 assets), this run compares all 66 operators and 124 window cases before and after the change. Evaluate plus ordered commit for 4096 formulas changed from **2.8047 s to 2.7699 s**.

| Operator (window 20) | CPU / us | CUDA before / us | CUDA after / us | CUDA gain |
| --- | ---: | ---: | ---: | ---: |
| `ts_std` | 240.05 | 437.27 | 233.44 | 1.87x |
| `ts_slope` | 132.51 | 325.81 | 236.41 | 1.38x |
| `ts_av_diff` | 87.60 | 195.70 | 121.42 | 1.61x |
| `ts_backfill` | 33.13 | 76.33 | 22.51 | 3.39x |
| `ts_ref` | 27.72 | 28.21 | 19.05 | 1.48x |
| `ts_delta` | 37.16 | 34.71 | 20.53 | 1.69x |
| `ts_corr` | 1101.91 | 872.47 | 421.53 | 2.07x |
| `ts_cov` | 701.61 | 448.81 | 318.29 | 1.41x |

The table uses microseconds and the CPU control from the optimized round. All windows 2/20/120 are in [comparison.csv](comparison.csv). CUDA still trails CPU in 24 cases: abs, add_const, div, gt, inv, is_nan, left_tail, mul_const, neg, not, not_equal, right_tail, sign, square, sub, ts_av_diff, ts_backfill, ts_skew, ts_slope, ts_std. Single observations do not establish statistical rankings; the approximately 1.2% end-to-end change is not evidence of a stable improvement.

Before/after CUDA reports for 1/64/4096 candidates match in decisions, weights and final portfolios; maximum absolute change across reported numeric fields is 0. CPU/CUDA operator errors, sign/NaN/infinity masks and admission checks passed in both rounds. Valid candidates: 4058; admitted: 28.

Recurrence kernels partition output rows while replaying history in the original order, preserving cancellation after extreme observations leave a window. Backfill uses parallel backward lookups; shift and difference each use one launch. Trading rules, admission parameters and arithmetic policy are unchanged. Launch latency remains a limitation for simple elementwise calls; this change does not implement formula fusion or claim every operator beats CPU.

Acceptance: 560 passed, 7159 subtests passed, one private-baseline skip; targeted smoke: 15 passed and 74 subtests passed. Baseline source comes from local commit `5d4d0e1`; the measured optimized source is preserved in source.tar.gz with per-file hashes in benchmark.json. Colab was released after evidence retrieval; no active sessions remain.

Each condition has one warm-up and one measured call, with one complete run before and after the edit and no repeated seeds. Loading/JIT is separated from evaluate plus commit. The private production baseline is unchanged and explicitly skipped; the separate same-device single-FP32 1.5x budget remains unevaluated. Results do not establish performance on other devices or longer histories.
