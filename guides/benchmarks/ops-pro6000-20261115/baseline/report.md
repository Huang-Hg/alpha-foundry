# CPU vs CUDA: public API benchmarks

This report compares CPU and CUDA latency, numerical error and admission decisions for identical strategies.

**Result: PASS** · NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.82.07

Market: `us_equity`; panel: **269 × 500 (dates × assets, including warm-up history)**; study: 2025-07-01 through 2026-06-30; CPU OpenMP threads: 2.

[JSON](summary.json) · pytest JSON (local artifact) · HTML (local artifact) · JUnit (local artifact) · [CSV](operators.csv)

## Complete call: open → evaluate → commit → portfolio → close

Evaluate computes formulas, positions, backtest metrics and proposed weights. Commit updates the portfolio in input order. Every round starts from the same equal-weight baseline in memory.

| Candidates | CPU full call / s | CUDA full call / s | Speedup | CPU evaluate+commit / s | CUDA evaluate+commit / s | Valid / admitted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0914 | 0.0865 | 1.06× | 0.0228 | 0.0163 | 1 / 1 |
| 64 | 1.4602 | 0.2317 | 6.30× | 1.3912 | 0.1559 | 64 / 11 |
| 4096 | 93.3117 | 2.9108 | 32.06× | 93.2330 | 2.8047 | 4058 / 28 |

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
| `abs` | 10.3 | 15.6 | 0.66× | 0.00e+00 | 0.00e+00 |
| `neg` | 9.5 | 15.0 | 0.64× | 0.00e+00 | 0.00e+00 |
| `sign` | 11.8 | 15.2 | 0.77× | 0.00e+00 | 0.00e+00 |
| `log` | 361.3 | 20.0 | 18.04× | 4.44e-16 | 2.20e-16 |
| `square` | 10.1 | 13.6 | 0.75× | 0.00e+00 | 0.00e+00 |
| `sqrt` | 259.2 | 16.7 | 15.54× | 0.00e+00 | 0.00e+00 |
| `tanh` | 859.1 | 21.6 | 39.73× | 2.22e-16 | 2.22e-16 |
| `inv` | 11.0 | 15.4 | 0.71× | 0.00e+00 | 0.00e+00 |
| `s_log_1p` | 744.7 | 35.1 | 21.22× | 1.11e-16 | 1.11e-16 |
| `not` | 10.5 | 14.7 | 0.72× | 0.00e+00 | 0.00e+00 |
| `is_nan` | 9.2 | 14.7 | 0.63× | 0.00e+00 | 0.00e+00 |
| `add` | 15.0 | 15.5 | 0.97× | 0.00e+00 | 0.00e+00 |
| `sub` | 15.2 | 15.6 | 0.97× | 0.00e+00 | 0.00e+00 |
| `mul` | 14.9 | 15.0 | 0.99× | 0.00e+00 | 0.00e+00 |
| `div` | 14.9 | 15.9 | 0.93× | 0.00e+00 | 0.00e+00 |
| `max` | 15.0 | 13.1 | 1.15× | 0.00e+00 | 0.00e+00 |
| `min` | 15.5 | 15.2 | 1.02× | 0.00e+00 | 0.00e+00 |
| `gt` | 15.2 | 15.6 | 0.98× | 0.00e+00 | 0.00e+00 |
| `lt` | 15.3 | 15.1 | 1.01× | 0.00e+00 | 0.00e+00 |
| `and` | 15.7 | 15.3 | 1.02× | 0.00e+00 | 0.00e+00 |
| `or` | 16.0 | 15.5 | 1.03× | 0.00e+00 | 0.00e+00 |
| `equal` | 15.0 | 13.2 | 1.14× | 0.00e+00 | 0.00e+00 |
| `not_equal` | 15.2 | 13.2 | 1.15× | 0.00e+00 | 0.00e+00 |
| `greater_equal` | 15.4 | 14.9 | 1.03× | 0.00e+00 | 0.00e+00 |
| `less_equal` | 15.4 | 15.1 | 1.02× | 0.00e+00 | 0.00e+00 |
| `add_const` | 9.7 | 18.4 | 0.52× | 0.00e+00 | 0.00e+00 |
| `mul_const` | 9.6 | 18.1 | 0.53× | 0.00e+00 | 0.00e+00 |
| `pow_const` | 122.9 | 32.8 | 3.74× | 1.11e-16 | 1.11e-16 |
| `left_tail` | 10.5 | 13.4 | 0.78× | 0.00e+00 | 0.00e+00 |
| `right_tail` | 9.4 | 14.9 | 0.63× | 0.00e+00 | 0.00e+00 |
| `hump` | 147.4 | 108.0 | 1.36× | 0.00e+00 | 0.00e+00 |
| `if_else` | 21.6 | 16.2 | 1.33× | 0.00e+00 | 0.00e+00 |
| `trade_when` | 210.1 | 173.6 | 1.21× | 0.00e+00 | 0.00e+00 |
| `cs_rank` | 2286.2 | 50.7 | 45.07× | 0.00e+00 | 0.00e+00 |
| `cs_zscore` | 119.6 | 24.0 | 4.97× | 0.00e+00 | 0.00e+00 |
| `cs_demean` | 60.2 | 21.0 | 2.86× | 0.00e+00 | 0.00e+00 |
| `cs_scale` | 72.3 | 23.8 | 3.04× | 0.00e+00 | 0.00e+00 |
| `ts_mean` | 513.3 | 40.5 | 12.67× | 0.00e+00 | 0.00e+00 |
| `ts_std` | 238.8 | 437.3 | 0.55× | 0.00e+00 | 0.00e+00 |
| `ts_max` | 963.8 | 38.1 | 25.28× | 0.00e+00 | 0.00e+00 |
| `ts_min` | 956.8 | 37.8 | 25.30× | 0.00e+00 | 0.00e+00 |
| `ts_sum` | 657.0 | 37.7 | 17.43× | 0.00e+00 | 0.00e+00 |
| `ts_rank` | 3123.0 | 44.5 | 70.12× | 0.00e+00 | 0.00e+00 |
| `ts_arg_max` | 956.4 | 37.6 | 25.47× | 0.00e+00 | 0.00e+00 |
| `ts_arg_min` | 961.9 | 37.9 | 25.37× | 0.00e+00 | 0.00e+00 |
| `ts_ema` | 133.4 | 66.6 | 2.00× | 0.00e+00 | 0.00e+00 |
| `ts_wma` | 410.1 | 44.1 | 9.30× | 0.00e+00 | 0.00e+00 |
| `ts_ref` | 31.9 | 28.2 | 1.13× | 0.00e+00 | 0.00e+00 |
| `ts_delta` | 37.1 | 34.7 | 1.07× | 0.00e+00 | 0.00e+00 |
| `ts_skew` | 2110.0 | 108.9 | 19.37× | 2.22e-16 | 2.22e-16 |
| `ts_kurt` | 1413.5 | 86.9 | 16.27× | 0.00e+00 | 0.00e+00 |
| `ts_mad` | 1017.5 | 53.9 | 18.88× | 0.00e+00 | 0.00e+00 |
| `ts_slope` | 132.4 | 325.8 | 0.41× | 0.00e+00 | 0.00e+00 |
| `ts_chanpos` | 443.6 | 37.0 | 11.99× | 0.00e+00 | 0.00e+00 |
| `ts_ddown` | 406.0 | 35.4 | 11.48× | 4.44e-16 | 2.81e-16 |
| `ts_zscore` | 977.5 | 44.5 | 21.94× | 0.00e+00 | 0.00e+00 |
| `ts_snr` | 1027.6 | 43.3 | 23.71× | 0.00e+00 | 0.00e+00 |
| `ts_backfill` | 34.8 | 76.3 | 0.46× | 0.00e+00 | 0.00e+00 |
| `ts_median` | 2931.1 | 89.5 | 32.77× | 0.00e+00 | 0.00e+00 |
| `ts_av_diff` | 86.9 | 195.7 | 0.44× | 0.00e+00 | 0.00e+00 |
| `ts_decay_exp_window` | 674.6 | 38.9 | 17.34× | 0.00e+00 | 0.00e+00 |
| `ts_scale` | 525.2 | 35.9 | 14.65× | 0.00e+00 | 0.00e+00 |
| `ts_quantile` | 4280.5 | 63.2 | 67.71× | 0.00e+00 | 0.00e+00 |
| `ts_corr` | 1021.2 | 872.5 | 1.17× | 0.00e+00 | 0.00e+00 |
| `ts_cov` | 686.1 | 448.8 | 1.53× | 0.00e+00 | 0.00e+00 |
| `ts_regression` | 1009.9 | 65.9 | 15.31× | 0.00e+00 | 0.00e+00 |

## Interpretation and reproduction

- One warm-up and one measured round per condition. This is one observation, not a confidence interval; avoid ranking small microsecond differences.
- Warm-up and timing use fresh sessions without private-cache changes. Warm-up stages are recorded separately. Other cases may already have compiled kernels, so this is not a cold-start claim.
- Operator upload/preparation is recorded outside operator timing. Full calls include loading/preparation but exclude installation, native builds, RPC and portfolio disk persistence.
- Device-wide memory is sampled every 50 ms, including context and both warm-up and measured calls; short peaks may be missed.
- This compares production CPU/CUDA backends, not a single-arithmetic experiment; it does not requalify the 1.5× precision budget.
- Validation cases: 7676; failures: 0; errors: 0; skipped: 1.
- Private production baseline (actual JUnit outcome): skipped; passed 0, skipped 1, failures/errors 0.
- Operator/window coverage: 124/124; missing []; unexpected []

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=2 python -m foundry benchmark var/runs/benchmark-new
```

Run on the authorized GPU with a matching CuPy package. JSON records parameters, source hashes, data/execution snapshots, corpus identity and hardware.
