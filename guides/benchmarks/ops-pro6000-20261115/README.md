# CUDA rolling operator optimization

[简体中文](README.zh-CN.md) · [Comparison CSV](comparison.csv) · [Before](baseline/report.md) · [After](optimized/report.md)

Measured on one RTX PRO 6000 over the `269 days × 500 assets` US-equity fixture, comparing all 66 operators across 124 window cases before and after tiling the rolling CUDA kernels. End-to-end evaluation plus ordered commit for 4,096 formulas moved from `2.8047 s` to `2.7699 s` (`~1.2%`, within single-run noise).

| Operator (`window=20`) | CPU (μs) | CUDA before (μs) | CUDA after (μs) | CUDA ratio |
| --- | ---: | ---: | ---: | ---: |
| `ts_std` | 240.05 | 437.27 | 233.44 | 1.87× |
| `ts_slope` | 132.51 | 325.81 | 236.41 | 1.38× |
| `ts_av_diff` | 87.60 | 195.70 | 121.42 | 1.61× |
| `ts_backfill` | 33.13 | 76.33 | 22.51 | 3.39× |
| `ts_ref` | 27.72 | 28.21 | 19.05 | 1.48× |
| `ts_delta` | 37.16 | 34.71 | 20.53 | 1.69× |
| `ts_corr` | 1101.91 | 872.47 | 421.53 | 2.07× |
| `ts_cov` | 701.61 | 448.81 | 318.29 | 1.41× |

Full `2 / 20 / 120` window measurements are in [comparison.csv](comparison.csv) (1 warm-up and 1 measured call per case). On a standalone `(269, 500)` panel, kernel launch overhead still leaves single-call CUDA slower than CPU in 24 cases across 20 lightweight elementwise and short-window operators (`abs`, `add_const`, `div`, `gt`, `inv`, `is_nan`, `left_tail`, `mul_const`, `neg`, `not`, `not_equal`, `right_tail`, `sign`, `square`, `sub`, `ts_av_diff`, `ts_backfill`, `ts_skew`, `ts_slope`, `ts_std`).

Time-partitioned rolling kernels replay each tile in chronological order so extreme values leaving a window do not leave cancellation residuals; `ts_backfill` uses parallel backward lookups, and `ts_ref` / `ts_delta` each fuse into one launch. Before and after CUDA runs on `1 / 64 / 4,096` formulas match with `0` numerical delta across admission decisions, weights, and final portfolios (`4,058` valid, `28` admitted), with 560 tests (7,159 subtests) passing.
