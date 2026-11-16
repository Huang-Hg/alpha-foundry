# CPU vs CUDA: public API benchmarks

This report compares CPU and CUDA latency, numerical error and admission decisions for identical strategies.

**Result: PASS** · NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.82.07

Market: `us_equity`; panel: **269 × 500 (dates × assets, including warm-up history)**; study: 2025-07-01 through 2026-06-30; CPU OpenMP threads: 2.

[JSON](summary.json) · [pytest JSON](benchmark.json) · [HTML](checks.html) · [JUnit](junit.xml) · [CSV](operators.csv)

## Complete call: open → evaluate → commit → portfolio → close

Evaluate computes formulas, positions, backtest metrics and proposed weights. Commit updates the portfolio in input order. Every round starts from the same equal-weight baseline in memory.

| Candidates | CPU full call / s | CUDA full call / s | Speedup | CPU evaluate+commit / s | CUDA evaluate+commit / s | Valid / admitted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0932 | 0.0891 | 1.05× | 0.0237 | 0.0163 | 1 / 1 |
| 64 | 1.5394 | 0.2336 | 6.59× | 1.4678 | 0.1550 | 64 / 11 |
| 4096 | 88.7475 | 2.4670 | 35.97× | 88.6675 | 2.3515 | 4058 / 28 |

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
| `abs` | 10.4 | 16.0 | 0.65× | 0.00e+00 | 0.00e+00 |
| `neg` | 9.3 | 15.3 | 0.61× | 0.00e+00 | 0.00e+00 |
| `sign` | 12.2 | 15.5 | 0.79× | 0.00e+00 | 0.00e+00 |
| `log` | 360.5 | 20.7 | 17.46× | 4.44e-16 | 2.20e-16 |
| `square` | 9.9 | 14.4 | 0.69× | 0.00e+00 | 0.00e+00 |
| `sqrt` | 256.9 | 17.3 | 14.88× | 0.00e+00 | 0.00e+00 |
| `tanh` | 887.1 | 21.6 | 40.99× | 2.22e-16 | 2.22e-16 |
| `inv` | 10.8 | 15.8 | 0.68× | 0.00e+00 | 0.00e+00 |
| `s_log_1p` | 742.8 | 35.8 | 20.77× | 1.11e-16 | 1.11e-16 |
| `not` | 10.3 | 14.9 | 0.69× | 0.00e+00 | 0.00e+00 |
| `is_nan` | 9.0 | 14.8 | 0.61× | 0.00e+00 | 0.00e+00 |
| `add` | 13.7 | 15.5 | 0.88× | 0.00e+00 | 0.00e+00 |
| `sub` | 13.8 | 15.2 | 0.91× | 0.00e+00 | 0.00e+00 |
| `mul` | 13.9 | 15.3 | 0.91× | 0.00e+00 | 0.00e+00 |
| `div` | 14.5 | 18.1 | 0.80× | 0.00e+00 | 0.00e+00 |
| `max` | 14.4 | 23.1 | 0.62× | 0.00e+00 | 0.00e+00 |
| `min` | 14.5 | 15.4 | 0.94× | 0.00e+00 | 0.00e+00 |
| `gt` | 14.6 | 15.6 | 0.93× | 0.00e+00 | 0.00e+00 |
| `lt` | 14.5 | 14.7 | 0.98× | 0.00e+00 | 0.00e+00 |
| `and` | 15.3 | 15.5 | 0.99× | 0.00e+00 | 0.00e+00 |
| `or` | 15.5 | 13.4 | 1.16× | 0.00e+00 | 0.00e+00 |
| `equal` | 14.6 | 13.6 | 1.08× | 0.00e+00 | 0.00e+00 |
| `not_equal` | 14.4 | 13.3 | 1.09× | 0.00e+00 | 0.00e+00 |
| `greater_equal` | 14.6 | 15.1 | 0.97× | 0.00e+00 | 0.00e+00 |
| `less_equal` | 14.4 | 15.3 | 0.94× | 0.00e+00 | 0.00e+00 |
| `add_const` | 9.6 | 16.6 | 0.58× | 0.00e+00 | 0.00e+00 |
| `mul_const` | 9.5 | 17.9 | 0.53× | 0.00e+00 | 0.00e+00 |
| `pow_const` | 146.6 | 32.9 | 4.46× | 1.11e-16 | 1.11e-16 |
| `left_tail` | 11.7 | 14.4 | 0.81× | 0.00e+00 | 0.00e+00 |
| `right_tail` | 9.4 | 15.5 | 0.60× | 0.00e+00 | 0.00e+00 |
| `hump` | 146.6 | 107.0 | 1.37× | 0.00e+00 | 0.00e+00 |
| `if_else` | 22.0 | 16.4 | 1.34× | 0.00e+00 | 0.00e+00 |
| `trade_when` | 214.6 | 171.0 | 1.26× | 0.00e+00 | 0.00e+00 |
| `cs_rank` | 2312.0 | 50.6 | 45.71× | 0.00e+00 | 0.00e+00 |
| `cs_zscore` | 119.4 | 24.2 | 4.94× | 0.00e+00 | 0.00e+00 |
| `cs_demean` | 60.6 | 21.0 | 2.88× | 0.00e+00 | 0.00e+00 |
| `cs_scale` | 72.5 | 23.3 | 3.11× | 0.00e+00 | 0.00e+00 |
| `ts_mean` | 526.4 | 40.7 | 12.93× | 0.00e+00 | 0.00e+00 |
| `ts_std` | 238.9 | 230.1 | 1.04× | 0.00e+00 | 0.00e+00 |
| `ts_max` | 938.0 | 43.9 | 21.36× | 0.00e+00 | 0.00e+00 |
| `ts_min` | 957.1 | 37.7 | 25.41× | 0.00e+00 | 0.00e+00 |
| `ts_sum` | 540.9 | 37.7 | 14.35× | 0.00e+00 | 0.00e+00 |
| `ts_rank` | 3157.8 | 45.0 | 70.16× | 0.00e+00 | 0.00e+00 |
| `ts_arg_max` | 954.9 | 37.5 | 25.48× | 0.00e+00 | 0.00e+00 |
| `ts_arg_min` | 974.6 | 36.9 | 26.39× | 0.00e+00 | 0.00e+00 |
| `ts_ema` | 144.6 | 67.6 | 2.14× | 0.00e+00 | 0.00e+00 |
| `ts_wma` | 474.1 | 46.4 | 10.23× | 0.00e+00 | 0.00e+00 |
| `ts_ref` | 34.8 | 19.2 | 1.81× | 0.00e+00 | 0.00e+00 |
| `ts_delta` | 39.7 | 20.7 | 1.92× | 0.00e+00 | 0.00e+00 |
| `ts_skew` | 2156.7 | 109.4 | 19.72× | 2.22e-16 | 2.22e-16 |
| `ts_kurt` | 1410.7 | 90.8 | 15.53× | 0.00e+00 | 0.00e+00 |
| `ts_mad` | 1022.1 | 52.8 | 19.34× | 0.00e+00 | 0.00e+00 |
| `ts_slope` | 132.9 | 152.5 | 0.87× | 0.00e+00 | 0.00e+00 |
| `ts_chanpos` | 439.2 | 36.5 | 12.02× | 0.00e+00 | 0.00e+00 |
| `ts_ddown` | 407.7 | 35.4 | 11.52× | 4.44e-16 | 2.81e-16 |
| `ts_zscore` | 1023.4 | 45.8 | 22.36× | 0.00e+00 | 0.00e+00 |
| `ts_snr` | 996.0 | 44.9 | 22.18× | 0.00e+00 | 0.00e+00 |
| `ts_backfill` | 37.3 | 24.0 | 1.55× | 0.00e+00 | 0.00e+00 |
| `ts_median` | 2987.4 | 88.3 | 33.85× | 0.00e+00 | 0.00e+00 |
| `ts_av_diff` | 89.1 | 82.7 | 1.08× | 0.00e+00 | 0.00e+00 |
| `ts_decay_exp_window` | 667.7 | 39.3 | 17.01× | 0.00e+00 | 0.00e+00 |
| `ts_scale` | 509.9 | 35.9 | 14.19× | 0.00e+00 | 0.00e+00 |
| `ts_quantile` | 4286.4 | 65.6 | 65.35× | 0.00e+00 | 0.00e+00 |
| `ts_corr` | 1074.5 | 416.1 | 2.58× | 0.00e+00 | 0.00e+00 |
| `ts_cov` | 614.4 | 314.0 | 1.96× | 0.00e+00 | 0.00e+00 |
| `ts_regression` | 1020.6 | 67.0 | 15.23× | 0.00e+00 | 0.00e+00 |

## Interpretation and reproduction

- One warm-up and one measured round per condition. This is one observation, not a confidence interval; avoid ranking small microsecond differences.
- Warm-up and timing use fresh sessions without private-cache changes. Warm-up stages are recorded separately. Other cases may already have compiled kernels, so this is not a cold-start claim.
- Operator upload/preparation is recorded outside operator timing. Full calls include loading/preparation but exclude installation, native builds, RPC and portfolio disk persistence.
- Device-wide memory is sampled every 50 ms, including context and both warm-up and measured calls; short peaks may be missed.
- This compares production CPU/CUDA backends, not a single-arithmetic experiment; it does not requalify the 1.5× precision budget.
- Validation cases: 7746; failures: 0; errors: 0; skipped: 1.
- Private production baseline (actual JUnit outcome): skipped; passed 0, skipped 1, failures/errors 0.
- Operator/window coverage: 124/124; missing []; unexpected []

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=2 python -m foundry benchmark var/runs/benchmark-new
```

Run on the authorized GPU with a matching CuPy package. JSON records parameters, source hashes, data/execution snapshots, corpus identity and hardware.
