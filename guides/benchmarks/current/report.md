# CPU vs CUDA: public API benchmarks

This report compares CPU and CUDA latency, numerical error and admission decisions for identical strategies.

**Result: PASS** · NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.82.07

Market: `us_equity`; panel: **269 × 500 (dates × assets, including warm-up history)**; study: 2025-07-01 through 2026-06-30; CPU OpenMP threads: 2.

[JSON](summary.json) · pytest JSON (local artifact) · HTML (local artifact) · JUnit (local artifact) · [CSV](operators.csv)

## Complete call: open → evaluate → commit → close

Evaluate computes formulas, positions, backtest metrics and proposed weights. Commit updates the portfolio in input order. Every round starts from the same equal-weight baseline in memory.

| Candidates | CPU full call / s | CUDA full call / s | Speedup | CPU evaluate+commit / s | CUDA evaluate+commit / s | Valid / admitted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0939 | 0.0933 | 1.01× | 0.0242 | 0.0162 | 1 / 1 |
| 64 | 1.5949 | 0.2339 | 6.82× | 1.5249 | 0.1536 | 64 / 11 |
| 4096 | 97.6857 | 2.9508 | 33.10× | 97.6000 | 2.8427 | 4058 / 28 |

Full call includes opening and closing the session. Evaluate+commit is a subset, not total latency. See [flows.csv](flows.csv) for stages.

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

Validity, admitted IDs, reason codes, constraint booleans, liquidation events and weights are compared. Device labels, version hashes and rounded prose are not semantic decisions.

Metric tolerance is `1e-6 + 1e-6 × |CPU value|`; weight absolute tolerance is `1e-7`. Decisions must match exactly. summary.json contains errors and worst candidate IDs.

## Per-operator latency

Both backends use the same real panel. The table shows window 20 for rolling operators and the single case otherwise. Both columns measure host wall time; CUDA synchronizes before returning. operators.csv includes windows 2/20/120.

| Operator | CPU / μs | CUDA / μs | Speedup | Max absolute error | Max scaled error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `abs` | 11.6 | 30.8 | 0.38× | 0.00e+00 | 0.00e+00 |
| `neg` | 11.5 | 19.9 | 0.58× | 0.00e+00 | 0.00e+00 |
| `sign` | 13.0 | 19.6 | 0.66× | 0.00e+00 | 0.00e+00 |
| `log` | 355.2 | 23.1 | 15.40× | 4.44e-16 | 2.20e-16 |
| `square` | 11.4 | 19.1 | 0.60× | 0.00e+00 | 0.00e+00 |
| `sqrt` | 258.9 | 20.8 | 12.47× | 0.00e+00 | 0.00e+00 |
| `tanh` | 875.5 | 25.9 | 33.83× | 2.22e-16 | 2.22e-16 |
| `inv` | 12.2 | 18.5 | 0.66× | 0.00e+00 | 0.00e+00 |
| `s_log_1p` | 744.2 | 39.1 | 19.03× | 1.11e-16 | 1.11e-16 |
| `not` | 12.2 | 19.1 | 0.64× | 0.00e+00 | 0.00e+00 |
| `is_nan` | 10.3 | 18.1 | 0.57× | 0.00e+00 | 0.00e+00 |
| `add` | 16.0 | 21.4 | 0.75× | 0.00e+00 | 0.00e+00 |
| `sub` | 16.1 | 21.5 | 0.75× | 0.00e+00 | 0.00e+00 |
| `mul` | 15.8 | 14.6 | 1.08× | 0.00e+00 | 0.00e+00 |
| `div` | 16.2 | 21.7 | 0.75× | 0.00e+00 | 0.00e+00 |
| `max` | 16.5 | 21.4 | 0.77× | 0.00e+00 | 0.00e+00 |
| `min` | 16.5 | 20.8 | 0.79× | 0.00e+00 | 0.00e+00 |
| `gt` | 16.1 | 18.9 | 0.85× | 0.00e+00 | 0.00e+00 |
| `lt` | 16.8 | 17.0 | 0.99× | 0.00e+00 | 0.00e+00 |
| `and` | 16.9 | 19.6 | 0.86× | 0.00e+00 | 0.00e+00 |
| `or` | 17.0 | 18.8 | 0.91× | 0.00e+00 | 0.00e+00 |
| `equal` | 16.4 | 18.6 | 0.88× | 0.00e+00 | 0.00e+00 |
| `not_equal` | 16.8 | 19.0 | 0.88× | 0.00e+00 | 0.00e+00 |
| `greater_equal` | 16.7 | 19.9 | 0.84× | 0.00e+00 | 0.00e+00 |
| `less_equal` | 16.7 | 19.6 | 0.85× | 0.00e+00 | 0.00e+00 |
| `add_const` | 11.1 | 32.6 | 0.34× | 0.00e+00 | 0.00e+00 |
| `mul_const` | 11.5 | 43.6 | 0.26× | 0.00e+00 | 0.00e+00 |
| `pow_const` | 135.1 | 38.5 | 3.51× | 1.11e-16 | 1.11e-16 |
| `left_tail` | 11.8 | 20.5 | 0.57× | 0.00e+00 | 0.00e+00 |
| `right_tail` | 10.2 | 20.5 | 0.50× | 0.00e+00 | 0.00e+00 |
| `hump` | 148.5 | 111.2 | 1.34× | 0.00e+00 | 0.00e+00 |
| `if_else` | 23.0 | 21.0 | 1.09× | 0.00e+00 | 0.00e+00 |
| `trade_when` | 221.3 | 173.8 | 1.27× | 0.00e+00 | 0.00e+00 |
| `cs_rank` | 2300.6 | 51.0 | 45.13× | 0.00e+00 | 0.00e+00 |
| `cs_zscore` | 122.6 | 25.6 | 4.78× | 0.00e+00 | 0.00e+00 |
| `cs_demean` | 63.5 | 20.9 | 3.03× | 0.00e+00 | 0.00e+00 |
| `cs_scale` | 75.7 | 24.2 | 3.13× | 0.00e+00 | 0.00e+00 |
| `ts_mean` | 517.3 | 39.9 | 12.96× | 0.00e+00 | 0.00e+00 |
| `ts_std` | 238.9 | 429.0 | 0.56× | 0.00e+00 | 0.00e+00 |
| `ts_max` | 958.9 | 37.3 | 25.73× | 0.00e+00 | 0.00e+00 |
| `ts_min` | 967.5 | 37.2 | 26.03× | 0.00e+00 | 0.00e+00 |
| `ts_sum` | 475.4 | 38.3 | 12.42× | 0.00e+00 | 0.00e+00 |
| `ts_rank` | 3143.0 | 44.0 | 71.35× | 0.00e+00 | 0.00e+00 |
| `ts_arg_max` | 971.0 | 36.7 | 26.44× | 0.00e+00 | 0.00e+00 |
| `ts_arg_min` | 974.0 | 38.4 | 25.39× | 0.00e+00 | 0.00e+00 |
| `ts_ema` | 133.4 | 66.4 | 2.01× | 0.00e+00 | 0.00e+00 |
| `ts_wma` | 417.8 | 44.1 | 9.47× | 0.00e+00 | 0.00e+00 |
| `ts_ref` | 35.7 | 30.1 | 1.19× | 0.00e+00 | 0.00e+00 |
| `ts_delta` | 39.8 | 35.3 | 1.13× | 0.00e+00 | 0.00e+00 |
| `ts_skew` | 2070.8 | 107.6 | 19.24× | 2.22e-16 | 2.22e-16 |
| `ts_kurt` | 1409.6 | 86.4 | 16.32× | 0.00e+00 | 0.00e+00 |
| `ts_mad` | 1067.3 | 51.9 | 20.55× | 0.00e+00 | 0.00e+00 |
| `ts_slope` | 132.4 | 322.4 | 0.41× | 0.00e+00 | 0.00e+00 |
| `ts_chanpos` | 450.1 | 35.1 | 12.82× | 0.00e+00 | 0.00e+00 |
| `ts_ddown` | 404.0 | 34.6 | 11.66× | 4.44e-16 | 2.81e-16 |
| `ts_zscore` | 1023.6 | 44.3 | 23.10× | 0.00e+00 | 0.00e+00 |
| `ts_snr` | 977.0 | 43.0 | 22.71× | 0.00e+00 | 0.00e+00 |
| `ts_backfill` | 38.4 | 76.1 | 0.50× | 0.00e+00 | 0.00e+00 |
| `ts_median` | 2940.6 | 84.4 | 34.86× | 0.00e+00 | 0.00e+00 |
| `ts_av_diff` | 101.7 | 189.7 | 0.54× | 0.00e+00 | 0.00e+00 |
| `ts_decay_exp_window` | 676.0 | 37.8 | 17.87× | 0.00e+00 | 0.00e+00 |
| `ts_scale` | 506.4 | 34.5 | 14.66× | 0.00e+00 | 0.00e+00 |
| `ts_quantile` | 4305.6 | 61.3 | 70.24× | 0.00e+00 | 0.00e+00 |
| `ts_corr` | 1136.2 | 865.7 | 1.31× | 0.00e+00 | 0.00e+00 |
| `ts_cov` | 687.9 | 439.5 | 1.56× | 0.00e+00 | 0.00e+00 |
| `ts_regression` | 1027.5 | 64.6 | 15.90× | 0.00e+00 | 0.00e+00 |

## Interpretation and reproduction

- One warm-up and one measured round per condition. This is one observation, not a confidence interval; avoid ranking small microsecond differences.
- Warm-up and timing use fresh sessions without private-cache changes. Warm-up stages are recorded separately. Other cases may already have compiled kernels, so this is not a cold-start claim.
- Operator upload/preparation is recorded outside operator timing. Full calls include loading/preparation but exclude installation, native builds, RPC and portfolio disk persistence.
- Device-wide memory is sampled every 50 ms, including context and both warm-up and measured calls; short peaks may be missed.
- This compares production CPU/CUDA backends, not a single-arithmetic experiment; it does not requalify the 1.5× precision budget.
- Validation cases: 1335; failures: 0; errors: 0; skipped: 1.
- The obsolete private production baseline is explicitly skipped and preserved. PIT, A-share, spot, funding and liquidation checks still run.

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=2 python -m foundry benchmark output/benchmark-new
```

Run on the authorized GPU with a matching CuPy package. JSON records parameters, source hashes, data/execution snapshots, corpus identity and hardware.
