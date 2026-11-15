# Historical PRO 6000 validation under the former 2× budget



The fixed-policy implementation passed 182 tests on Colab RTX PRO 6000 Blackwell Server Edition (97,887 MiB, driver 580.82.07; Python 3.13.16, NumPy 2.4.6, CuPy 14.0.1). Builds, smoke, CPU references, GPU tests and endpoint timing all ran there. The legacy private `production/regression.npz` lacks the current protocol/study identity: `test_production` was excluded and that artifact was not rewritten. This is not a claim of bitwise reproduction of the legacy portfolio.

The comparison runs one warmed query followed by an ordered commit from the same initial portfolio, with identical candidates, order and market rules. Formula caches are cleared after warm-up. Loading, registration and JIT are outside the ratio. Each condition is measured once; the results qualify these workloads, not every device or input size.

The **single-FP32 probe** changes GPU formula arithmetic and account arithmetic to single components. Both sides retain the same paired statistics, wide interchange arrays and preplanned small-perpetual CPU dispatch. It is an arithmetic-cost comparison, not an all-FP32 production implementation. The production formula operators and perpetual account recurrence retain native double arithmetic; cash accounts and CUDA statistics use paired FP32. Queries and commits never trigger precision reruns.

| Workload | Single probe, seconds | Production, seconds | Ratio | Accepted |
| --- | ---: | ---: | ---: | ---: |
| US: 500 assets × 269 days, 64 formulas | 0.05081 | 0.08930 | 1.758× | 6 |
| A-share: 365 assets × 601 days, 64 formulas | 0.10562 | 0.20479 | 1.939× | 8 |
| Crypto spot: 73 assets × 4,018 days, 64 formulas | 0.22727 | 0.42082 | 1.852× | 5 |
| Perpetual: 64 contracts, 64 formulas | 0.45509 | 0.83460 | 1.834× | 7 |
| Perpetual: 64 contracts, 64 target books | 0.91915 | 1.56882 | 1.707× | 17 |
| Perpetual: 64 contracts, 4,096 target books | 15.25573 | 26.91782 | 1.764× | 39 |

All times in the table are **query + commit**. Perpetual inputs retain 365 days and all 8,760 hourly events, with a 184-day evaluation window. The large target-book case spends 10.52891 seconds on query and 16.38891 on commit; prepare and registration add 0.84255 and 1.24893 seconds separately. Device memory at completion is 11,666 MiB; pool reservation is 10,958 MiB. These are end-state readings, not peak measurements.

Against the independent CPU reference, all six conditions have identical admission/rejection decisions, reason codes, constraint booleans, liquidation counts/first timestamps and bankruptcy flags. Maximum absolute differences across query and commit reports are: Sharpe `7.32e-11`, CAGR `2.18e-12`, drawdown `3.48e-13`, and selected weight `5.03e-12`. The 4,096-book funding difference is below `1.00e-15` and trading-fee difference below `2.03e-15`, in initial-equity units. Other report quantities have different scales: the largest normalized scalar error is `1.71e-9`, using `abs(error)/max(1, abs(reference))`; this is not a universal absolute-error bound. Direct CUDA tests additionally check event-stage bitmaps and adjacent liquidation thresholds. Array correctness reuses the unchanged CPU reference collected on the preceding authorized PRO 6000 worker; no cross-worker CPU speedup is claimed.

The single probe changes some selected weights and constraint checks; in spot it even exchanges accepted candidate identities while leaving the accepted count at five. It therefore serves only as a speed denominator. Accuracy is judged against the independent CPU implementation, never from equal accepted counts alone.

Paired arithmetic uses Quick-Two-Sum only for ordered normalization and FMA product residuals. Division forms its residual directly with FMA and preserves round-to-nearest division; an exact unit denominator returns the original pair. This removed redundant pair construction and reduced the previously over-budget cash workloads without lowering precision or changing admission tolerances. Tensor-core funding-prefix experiments are not enabled in production: they did not establish an end-to-end improvement for the chronological account model.
