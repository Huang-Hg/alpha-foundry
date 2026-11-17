# CPU vs CUDA: public API benchmarks

This report compares CPU and CUDA latency, numerical error and admission decisions for identical strategies.

**Result: PASS** · NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.82.07

Market: `us_equity`; panel: **269 × 500 (dates × assets, including warm-up history)**; study: 2025-07-01 through 2026-06-30; CPU OpenMP threads: 2.

[JSON](summary.json) · pytest JSON (local artifact) · HTML (local artifact) · JUnit (local artifact) · [CSV](operators.csv)

## Complete call: open → evaluate → commit → portfolio → close

Evaluate computes formulas, positions, backtest metrics and proposed weights. Commit updates the portfolio in input order. Every round starts from the same equal-weight baseline in memory.

| Candidates | CPU full call / s | CUDA full call / s | Speedup | CPU evaluate+commit / s | CUDA evaluate+commit / s | Valid / admitted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0916 | 0.0883 | 1.04× | 0.0230 | 0.0164 | 1 / 1 |
| 64 | 1.4883 | 0.2321 | 6.41× | 1.4189 | 0.1548 | 64 / 11 |
| 4096 | 95.0245 | 2.8767 | 33.03× | 94.9451 | 2.7699 | 4058 / 28 |

Full call includes opening, final portfolio inspection and closing. Inspection is timed separately from evaluate+ordered commit. See [flows.csv](flows.csv) for stages.

## Execution precision plans

| Candidates | Backend | Requested | Effective | Reason |
| ---: | --- | --- | --- | --- |
| 1 | cpu | paired | native | validated_native_path |
| 1 | cuda | paired | paired | fixed_study_policy |
| 64 | cpu | paired | native | validated_native_path |
| 64 | cuda | paired | paired | fixed_study_policy |
| 4096 | cpu | paired | native | validated_native_path |
| 4096 | cuda | paired | paired | fixed_study_policy |

These are fixed runtime account-execution plans. CPU/CUDA speedup is not a same-device FP32 precision comparison; the 1.5× precision budget is not requalified here.

## Are the results consistent?

| Candidates | Decision mismatches | Finite-value mismatches | Numerical violations | Result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 0 | 0 | PASS |
| 64 | 0 | 0 | 0 | PASS |
| 4096 | 0 | 0 | 0 | PASS |

Worst errors across evaluate and commit for the largest batch (4096 candidates):

| Quantity | Max absolute error | Max scaled error |
| --- | ---: | ---: |
| Sharpe | 1.836e-07 | 1.698e-07 |
| CAGR | 4.811e-13 | 4.811e-13 |
| Max drawdown | 2.184e-13 | 2.184e-13 |
| Selected weight | 1.099e-10 | 1.099e-10 |
| Return concentration | 2.321e-05 | 7.410e-07 |

Validity, admitted IDs, reason codes, constraint booleans and weights are compared, together with final portfolio members, weights and metrics. Where perpetual risk reports exist, funding totals, liquidation counts and first-liquidation times are checked; this comparison does not include the complete hourly event bitmap. Device labels, version hashes and rounded prose are not semantic decisions.

Metric tolerance is `1e-6 + 1e-6 × |CPU value|`; weight absolute tolerance is `1e-7`. Decisions must match exactly. summary.json contains errors and worst candidate IDs.

## Per-operator latency

Both backends use the same real panel. The table shows window 20 for rolling operators and the single case otherwise. Both columns measure host wall time; CUDA synchronizes before returning. operators.csv includes windows 2/20/120.

| Operator | CPU / μs | CUDA / μs | Speedup | Max absolute error | Max scaled error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `abs` | 10.4 | 15.8 | 0.66× | 0.00e+00 | 0.00e+00 |
| `neg` | 9.8 | 14.7 | 0.67× | 0.00e+00 | 0.00e+00 |
| `sign` | 12.0 | 15.6 | 0.77× | 0.00e+00 | 0.00e+00 |
| `log` | 351.9 | 20.4 | 17.27× | 4.44e-16 | 2.20e-16 |
| `square` | 10.3 | 15.3 | 0.67× | 0.00e+00 | 0.00e+00 |
| `sqrt` | 263.9 | 16.7 | 15.82× | 0.00e+00 | 0.00e+00 |
| `tanh` | 866.7 | 21.4 | 40.42× | 2.22e-16 | 2.22e-16 |
| `inv` | 11.4 | 15.6 | 0.73× | 0.00e+00 | 0.00e+00 |
| `s_log_1p` | 744.6 | 36.1 | 20.63× | 1.11e-16 | 1.11e-16 |
| `not` | 11.1 | 14.6 | 0.76× | 0.00e+00 | 0.00e+00 |
| `is_nan` | 9.5 | 14.5 | 0.66× | 0.00e+00 | 0.00e+00 |
| `add` | 16.1 | 15.6 | 1.03× | 0.00e+00 | 0.00e+00 |
| `sub` | 14.6 | 15.0 | 0.97× | 0.00e+00 | 0.00e+00 |
| `mul` | 14.6 | 14.5 | 1.01× | 0.00e+00 | 0.00e+00 |
| `div` | 14.9 | 15.7 | 0.95× | 0.00e+00 | 0.00e+00 |
| `max` | 15.3 | 14.9 | 1.03× | 0.00e+00 | 0.00e+00 |
| `min` | 15.2 | 14.5 | 1.05× | 0.00e+00 | 0.00e+00 |
| `gt` | 15.6 | 15.8 | 0.99× | 0.00e+00 | 0.00e+00 |
| `lt` | 15.3 | 14.7 | 1.04× | 0.00e+00 | 0.00e+00 |
| `and` | 16.4 | 15.0 | 1.09× | 0.00e+00 | 0.00e+00 |
| `or` | 16.2 | 15.0 | 1.08× | 0.00e+00 | 0.00e+00 |
| `equal` | 15.5 | 15.1 | 1.03× | 0.00e+00 | 0.00e+00 |
| `not_equal` | 15.5 | 17.2 | 0.90× | 0.00e+00 | 0.00e+00 |
| `greater_equal` | 15.4 | 15.1 | 1.02× | 0.00e+00 | 0.00e+00 |
| `less_equal` | 15.4 | 15.1 | 1.02× | 0.00e+00 | 0.00e+00 |
| `add_const` | 9.9 | 18.5 | 0.53× | 0.00e+00 | 0.00e+00 |
| `mul_const` | 9.9 | 17.1 | 0.58× | 0.00e+00 | 0.00e+00 |
| `pow_const` | 123.2 | 34.6 | 3.57× | 1.11e-16 | 1.11e-16 |
| `left_tail` | 10.1 | 13.2 | 0.77× | 0.00e+00 | 0.00e+00 |
| `right_tail` | 9.4 | 15.0 | 0.63× | 0.00e+00 | 0.00e+00 |
| `hump` | 147.6 | 107.5 | 1.37× | 0.00e+00 | 0.00e+00 |
| `if_else` | 21.5 | 16.5 | 1.31× | 0.00e+00 | 0.00e+00 |
| `trade_when` | 211.0 | 173.2 | 1.22× | 0.00e+00 | 0.00e+00 |
| `cs_rank` | 2298.0 | 49.5 | 46.38× | 0.00e+00 | 0.00e+00 |
| `cs_zscore` | 119.6 | 23.8 | 5.02× | 0.00e+00 | 0.00e+00 |
| `cs_demean` | 60.3 | 20.9 | 2.88× | 0.00e+00 | 0.00e+00 |
| `cs_scale` | 73.1 | 22.7 | 3.22× | 0.00e+00 | 0.00e+00 |
| `ts_mean` | 505.5 | 40.1 | 12.62× | 0.00e+00 | 0.00e+00 |
| `ts_std` | 240.1 | 233.4 | 1.03× | 0.00e+00 | 0.00e+00 |
| `ts_max` | 948.5 | 37.2 | 25.50× | 0.00e+00 | 0.00e+00 |
| `ts_min` | 956.8 | 36.9 | 25.92× | 0.00e+00 | 0.00e+00 |
| `ts_sum` | 590.9 | 37.8 | 15.62× | 0.00e+00 | 0.00e+00 |
| `ts_rank` | 3106.5 | 46.4 | 66.95× | 0.00e+00 | 0.00e+00 |
| `ts_arg_max` | 971.6 | 37.4 | 25.97× | 0.00e+00 | 0.00e+00 |
| `ts_arg_min` | 978.0 | 38.0 | 25.73× | 0.00e+00 | 0.00e+00 |
| `ts_ema` | 132.1 | 66.9 | 1.98× | 0.00e+00 | 0.00e+00 |
| `ts_wma` | 410.8 | 43.6 | 9.43× | 0.00e+00 | 0.00e+00 |
| `ts_ref` | 27.7 | 19.0 | 1.46× | 0.00e+00 | 0.00e+00 |
| `ts_delta` | 37.2 | 20.5 | 1.81× | 0.00e+00 | 0.00e+00 |
| `ts_skew` | 2132.1 | 110.6 | 19.28× | 2.22e-16 | 2.22e-16 |
| `ts_kurt` | 1412.4 | 88.4 | 15.98× | 0.00e+00 | 0.00e+00 |
| `ts_mad` | 1019.6 | 53.1 | 19.22× | 0.00e+00 | 0.00e+00 |
| `ts_slope` | 132.5 | 236.4 | 0.56× | 0.00e+00 | 0.00e+00 |
| `ts_chanpos` | 459.0 | 44.6 | 10.28× | 0.00e+00 | 0.00e+00 |
| `ts_ddown` | 400.5 | 35.0 | 11.43× | 4.44e-16 | 2.81e-16 |
| `ts_zscore` | 1018.1 | 44.9 | 22.66× | 0.00e+00 | 0.00e+00 |
| `ts_snr` | 1024.3 | 43.8 | 23.40× | 0.00e+00 | 0.00e+00 |
| `ts_backfill` | 33.1 | 22.5 | 1.47× | 0.00e+00 | 0.00e+00 |
| `ts_median` | 2902.2 | 88.2 | 32.90× | 0.00e+00 | 0.00e+00 |
| `ts_av_diff` | 87.6 | 121.4 | 0.72× | 0.00e+00 | 0.00e+00 |
| `ts_decay_exp_window` | 669.0 | 37.9 | 17.63× | 0.00e+00 | 0.00e+00 |
| `ts_scale` | 506.0 | 34.9 | 14.51× | 0.00e+00 | 0.00e+00 |
| `ts_quantile` | 4350.0 | 64.2 | 67.79× | 0.00e+00 | 0.00e+00 |
| `ts_corr` | 1101.9 | 421.5 | 2.61× | 0.00e+00 | 0.00e+00 |
| `ts_cov` | 701.6 | 318.3 | 2.20× | 0.00e+00 | 0.00e+00 |
| `ts_regression` | 1014.0 | 66.3 | 15.28× | 0.00e+00 | 0.00e+00 |

## Interpretation and reproduction

- One warm-up and one measured round per condition. This is one observation, not a confidence interval; avoid ranking small microsecond differences.
- Warm-up and timing use fresh sessions without private-cache changes. Warm-up stages are recorded separately. Other cases may already have compiled kernels, so this is not a cold-start claim.
- Operator upload/preparation is recorded outside operator timing. Full calls include loading/preparation but exclude installation, native builds, RPC and portfolio disk persistence.
- Device-wide memory is sampled every 50 ms, including context and both warm-up and measured calls; short peaks may be missed.
- This compares production CPU/CUDA backends, not a single-arithmetic experiment; it does not requalify the 1.5× precision budget.
- Validation cases: 7720; failures: 0; errors: 0; skipped: 1.
- Private production baseline (actual JUnit outcome): skipped; passed 0, skipped 1, failures/errors 0.
- Operator/window coverage: 124/124; missing []; unexpected []

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=2 python -m foundry benchmark var/runs/benchmark-new
```

Run on the authorized GPU with a matching CuPy package. JSON records parameters, source hashes, data/execution snapshots, corpus identity and hardware.
