# Optimizing CUDA operators that lag CPU

[简体中文](README.zh-CN.md) · [Comparison CSV](comparison.csv) · [Before](baseline/report.md) · [After](optimized/report.md)

On the same Colab RTX PRO 6000 and US-equity fixture (`269 days × 500 assets`), this run compares all 66 operators across 124 window cases before and after the kernel changes. Evaluation plus ordered commit for 4,096 formulas moved from **2.8047 s to 2.7699 s**.

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

Timings are in microseconds using the CPU control from the optimized round; all windows (`2 / 20 / 120`) appear in [comparison.csv](comparison.csv). CUDA still trails CPU in 24 standalone cases across 20 operators (`abs`, `add_const`, `div`, `gt`, `inv`, `is_nan`, `left_tail`, `mul_const`, `neg`, `not`, `not_equal`, `right_tail`, `sign`, `square`, `sub`, `ts_av_diff`, `ts_backfill`, `ts_skew`, `ts_slope`, `ts_std`). Single observations do not establish statistical rankings; the ~1.2% end-to-end change is not evidence of a stable improvement.

## Numerical parity and scope limits

- **Report equivalence:** before/after CUDA reports for `1 / 64 / 4,096` candidates match in admission decisions, weights, and final portfolios (`4,058` valid, `28` admitted, `0` maximum absolute change across reported numeric fields). CPU/CUDA operator error bounds, sign/NaN/infinity masks, and admission checks passed in both rounds.
- **Kernel scope:** recurrence kernels partition output rows while replaying history in chronological order so extreme observations leaving a window do not leave cancellation residuals; `ts_backfill` uses parallel backward lookups, and `ts_ref` / `ts_delta` each use one launch. Trading rules, admission parameters, and arithmetic policy are unchanged. Kernel launch latency remains a bottleneck for simple elementwise calls; this change does not implement formula fusion or claim that every operator beats CPU.
- **Methodology and limits:** each condition ran 1 warm-up and 1 measured call (one complete pass before and after the edit, no repeated seeds), separating loading/JIT from evaluate + commit. Regression checks passed 560 tests and 7,159 subtests (15 targeted smoke tests / 74 subtests) with 1 unchanged private-baseline skip; baseline source was commit `5d4d0e1` and raw artifacts remain in local `var/runs/ops-opt-evidence.tar.gz`. The separate same-device single-FP32 1.5× budget was not evaluated, and these numbers do not establish performance on other devices or longer histories.
