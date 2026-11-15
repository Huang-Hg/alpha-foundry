# CPU vs CUDA: public API benchmarks

This report compares CPU and CUDA latency, numerical error and admission decisions for identical strategies.

**Result: PASS** · NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.82.07

Market: `us_equity`; panel: **269 × 500 (dates × assets, including warm-up history)**; study: 2025-07-01 through 2026-06-30; CPU OpenMP threads: 2.

[JSON](summary.json) · [pytest JSON](benchmark.json) · [HTML](checks.html) · [JUnit](junit.xml) · [CSV](operators.csv)

## Complete call: open → evaluate → commit → portfolio → close

Evaluate computes formulas, positions, backtest metrics and proposed weights. Commit updates the portfolio in input order. Every round starts from the same equal-weight baseline in memory.

| Candidates | CPU full call / s | CUDA full call / s | Speedup | CPU evaluate+commit / s | CUDA evaluate+commit / s | Valid / admitted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0874 | 0.0868 | 1.01× | 0.0205 | 0.0161 | 1 / 1 |
| 64 | 1.4425 | 0.2309 | 6.25× | 1.3745 | 0.1539 | 64 / 11 |
| 4096 | 93.7237 | 2.8791 | 32.55× | 93.6434 | 2.7724 | 4058 / 28 |

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
| Selected weight | 8.135e-10 | 8.135e-10 |
| Return concentration | 2.321e-05 | 7.410e-07 |

Validity, admitted IDs, reason codes, constraint booleans and weights are compared, together with final portfolio members, weights and metrics. Where perpetual risk reports exist, funding totals, liquidation counts and first-liquidation times are checked; this comparison does not include the complete hourly event bitmap. Device labels, version hashes and rounded prose are not semantic decisions.

Metric tolerance is `1e-6 + 1e-6 × |CPU value|`; weight absolute tolerance is `1e-7`. Decisions must match exactly. summary.json contains errors and worst candidate IDs.

## Per-operator latency

Both backends use the same real panel. The table shows window 20 for rolling operators and the single case otherwise. Both columns measure host wall time; CUDA synchronizes before returning. operators.csv includes windows 2/20/120.

| Operator | CPU / μs | CUDA / μs | Speedup | Max absolute error | Max scaled error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `abs` | 20.6 | 17.2 | 1.19× | 0.00e+00 | 0.00e+00 |
| `neg` | 20.4 | 16.0 | 1.27× | 0.00e+00 | 0.00e+00 |
| `sign` | 21.9 | 16.8 | 1.31× | 0.00e+00 | 0.00e+00 |
| `log` | 487.3 | 21.6 | 22.54× | 4.44e-16 | 2.20e-16 |
| `square` | 20.4 | 16.6 | 1.23× | 0.00e+00 | 0.00e+00 |
| `sqrt` | 317.3 | 17.6 | 18.03× | 0.00e+00 | 0.00e+00 |
| `tanh` | 862.4 | 21.2 | 40.60× | 2.22e-16 | 2.22e-16 |
| `inv` | 11.4 | 14.9 | 0.76× | 0.00e+00 | 0.00e+00 |
| `s_log_1p` | 734.9 | 34.9 | 21.07× | 1.11e-16 | 1.11e-16 |
| `not` | 11.0 | 14.0 | 0.78× | 0.00e+00 | 0.00e+00 |
| `is_nan` | 9.8 | 14.2 | 0.69× | 0.00e+00 | 0.00e+00 |
| `add` | 15.9 | 15.2 | 1.05× | 0.00e+00 | 0.00e+00 |
| `sub` | 15.2 | 14.8 | 1.03× | 0.00e+00 | 0.00e+00 |
| `mul` | 15.4 | 14.7 | 1.05× | 0.00e+00 | 0.00e+00 |
| `div` | 15.2 | 17.7 | 0.86× | 0.00e+00 | 0.00e+00 |
| `max` | 15.6 | 14.9 | 1.04× | 0.00e+00 | 0.00e+00 |
| `min` | 15.6 | 14.7 | 1.06× | 0.00e+00 | 0.00e+00 |
| `gt` | 15.6 | 15.8 | 0.99× | 0.00e+00 | 0.00e+00 |
| `lt` | 15.7 | 14.8 | 1.06× | 0.00e+00 | 0.00e+00 |
| `and` | 16.7 | 13.4 | 1.25× | 0.00e+00 | 0.00e+00 |
| `or` | 16.2 | 14.9 | 1.08× | 0.00e+00 | 0.00e+00 |
| `equal` | 15.7 | 14.7 | 1.07× | 0.00e+00 | 0.00e+00 |
| `not_equal` | 15.5 | 14.9 | 1.04× | 0.00e+00 | 0.00e+00 |
| `greater_equal` | 15.6 | 15.2 | 1.03× | 0.00e+00 | 0.00e+00 |
| `less_equal` | 15.8 | 15.7 | 1.01× | 0.00e+00 | 0.00e+00 |
| `add_const` | 10.1 | 18.3 | 0.55× | 0.00e+00 | 0.00e+00 |
| `mul_const` | 10.0 | 17.1 | 0.59× | 0.00e+00 | 0.00e+00 |
| `pow_const` | 124.0 | 32.9 | 3.77× | 1.11e-16 | 1.11e-16 |
| `left_tail` | 11.5 | 15.4 | 0.75× | 0.00e+00 | 0.00e+00 |
| `right_tail` | 11.2 | 15.0 | 0.74× | 0.00e+00 | 0.00e+00 |
| `hump` | 154.4 | 105.4 | 1.46× | 0.00e+00 | 0.00e+00 |
| `if_else` | 21.8 | 16.0 | 1.36× | 0.00e+00 | 0.00e+00 |
| `trade_when` | 209.6 | 169.6 | 1.24× | 0.00e+00 | 0.00e+00 |
| `cs_rank` | 2292.9 | 49.4 | 46.42× | 0.00e+00 | 0.00e+00 |
| `cs_zscore` | 122.4 | 23.8 | 5.14× | 0.00e+00 | 0.00e+00 |
| `cs_demean` | 68.3 | 18.2 | 3.76× | 0.00e+00 | 0.00e+00 |
| `cs_scale` | 74.7 | 22.8 | 3.28× | 0.00e+00 | 0.00e+00 |
| `ts_mean` | 524.9 | 41.0 | 12.79× | 0.00e+00 | 0.00e+00 |
| `ts_std` | 238.9 | 425.7 | 0.56× | 0.00e+00 | 0.00e+00 |
| `ts_max` | 956.4 | 36.9 | 25.90× | 0.00e+00 | 0.00e+00 |
| `ts_min` | 955.1 | 37.9 | 25.18× | 0.00e+00 | 0.00e+00 |
| `ts_sum` | 467.8 | 37.6 | 12.45× | 0.00e+00 | 0.00e+00 |
| `ts_rank` | 3164.4 | 43.6 | 72.63× | 0.00e+00 | 0.00e+00 |
| `ts_arg_max` | 967.2 | 37.0 | 26.16× | 0.00e+00 | 0.00e+00 |
| `ts_arg_min` | 960.1 | 37.6 | 25.52× | 0.00e+00 | 0.00e+00 |
| `ts_ema` | 144.4 | 65.1 | 2.22× | 0.00e+00 | 0.00e+00 |
| `ts_wma` | 411.7 | 44.8 | 9.18× | 0.00e+00 | 0.00e+00 |
| `ts_ref` | 31.9 | 29.4 | 1.08× | 0.00e+00 | 0.00e+00 |
| `ts_delta` | 39.6 | 34.7 | 1.14× | 0.00e+00 | 0.00e+00 |
| `ts_skew` | 2139.3 | 108.7 | 19.68× | 2.22e-16 | 2.22e-16 |
| `ts_kurt` | 1407.6 | 88.4 | 15.92× | 0.00e+00 | 0.00e+00 |
| `ts_mad` | 1030.3 | 51.9 | 19.86× | 0.00e+00 | 0.00e+00 |
| `ts_slope` | 133.9 | 319.9 | 0.42× | 0.00e+00 | 0.00e+00 |
| `ts_chanpos` | 449.1 | 35.6 | 12.60× | 0.00e+00 | 0.00e+00 |
| `ts_ddown` | 406.0 | 35.0 | 11.59× | 4.44e-16 | 2.81e-16 |
| `ts_zscore` | 992.4 | 44.2 | 22.43× | 0.00e+00 | 0.00e+00 |
| `ts_snr` | 1018.0 | 42.3 | 24.09× | 0.00e+00 | 0.00e+00 |
| `ts_backfill` | 40.7 | 75.4 | 0.54× | 0.00e+00 | 0.00e+00 |
| `ts_median` | 2924.0 | 85.3 | 34.28× | 0.00e+00 | 0.00e+00 |
| `ts_av_diff` | 89.6 | 193.5 | 0.46× | 0.00e+00 | 0.00e+00 |
| `ts_decay_exp_window` | 657.4 | 37.4 | 17.58× | 0.00e+00 | 0.00e+00 |
| `ts_scale` | 503.4 | 33.2 | 15.14× | 0.00e+00 | 0.00e+00 |
| `ts_quantile` | 4306.1 | 63.5 | 67.82× | 0.00e+00 | 0.00e+00 |
| `ts_corr` | 1069.6 | 858.1 | 1.25× | 0.00e+00 | 0.00e+00 |
| `ts_cov` | 653.4 | 438.9 | 1.49× | 0.00e+00 | 0.00e+00 |
| `ts_regression` | 1016.4 | 65.9 | 15.41× | 0.00e+00 | 0.00e+00 |

## Interpretation and reproduction

- One warm-up and one measured round per condition. This is one observation, not a confidence interval; avoid ranking small microsecond differences.
- Warm-up and timing use fresh sessions without private-cache changes. Warm-up stages are recorded separately. Other cases may already have compiled kernels, so this is not a cold-start claim.
- Operator upload/preparation is recorded outside operator timing. Full calls include loading/preparation but exclude installation, native builds, RPC and portfolio disk persistence.
- Device-wide memory is sampled every 50 ms, including context and both warm-up and measured calls; short peaks may be missed.
- This compares production CPU/CUDA backends, not a single-arithmetic experiment; it does not requalify the 1.5× precision budget.
- Validation cases: 7574; failures: 0; errors: 0; skipped: 1.
- Private production baseline (actual JUnit outcome): skipped; passed 0, skipped 1, failures/errors 0.
- Operator/window coverage: 124/124; missing []; unexpected []

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=2 python -m foundry benchmark var/runs/benchmark-new
```

Run on the authorized GPU with a matching CuPy package. JSON records parameters, source hashes, data/execution snapshots, corpus identity and hardware.
