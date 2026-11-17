# Optimizing CUDA rolling operators

[简体中文](README.zh-CN.md) · [Comparison CSV](comparison.csv) · [Baseline report](baseline/report.md) · [Optimized report](optimized/report.md)

Measured on the same Colab RTX PRO 6000 and US-equity fixture (`269 days × 500 assets`), comparing all 66 operators across 124 window conditions before and after tiled rolling-kernel optimization. End-to-end evaluation plus ordered commit for 4,096 formulas moved from **2.8047 s to 2.7699 s**.

| Operator (`window=20`) | CPU (μs) | CUDA before (μs) | CUDA after (μs) | CUDA speedup |
| --- | ---: | ---: | ---: | ---: |
| `ts_std` | 240.05 | 437.27 | 233.44 | 1.87× |
| `ts_slope` | 132.51 | 325.81 | 236.41 | 1.38× |
| `ts_av_diff` | 87.60 | 195.70 | 121.42 | 1.61× |
| `ts_backfill` | 33.13 | 76.33 | 22.51 | 3.39× |
| `ts_ref` | 27.72 | 28.21 | 19.05 | 1.48× |
| `ts_delta` | 37.16 | 34.71 | 20.53 | 1.69× |
| `ts_corr` | 1101.91 | 872.47 | 421.53 | 2.07× |
| `ts_cov` | 701.61 | 448.81 | 318.29 | 1.41× |

Full `2 / 20 / 120` window measurements are in [comparison.csv](comparison.csv). On a single `(269, 500)` panel, kernel launch latency still leaves standalone single-call CUDA slower than CPU on lightweight elementwise operators (`abs`, `add_const`, `div`, `gt`, `inv`, `is_nan`, `left_tail`, `mul_const`, `neg`, `not`, `not_equal`, `right_tail`, `sign`, `square`, `sub`) and a few short-panel rolling cases (`ts_av_diff`, `ts_backfill`, `ts_skew`, `ts_slope`, `ts_std`).

## Numerical parity and method

- **Bitwise report parity:** before/after CUDA runs on `1 / 64 / 4096` formulas produced identical admission decisions, selected weights, and final portfolios (`4,058` valid, `28` admitted, `0` max absolute report delta).
- **Kernel design:** time-partitioned rolling kernels preserve chronological accumulation within each tile so extreme values exiting a window cannot leave catastrophic cancellation residuals; `ts_backfill` uses parallel backward lookups, and `ts_ref` / `ts_delta` fuse into single launches.
- **Validation:** 560 tests (7,159 subtests) passed on PRO 6000 (1 legacy private production baseline skipped), with raw artifacts retained locally in `var/runs/ops-opt-evidence.tar.gz`.
