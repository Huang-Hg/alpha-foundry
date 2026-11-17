# Order accounts: fees, cross margin, and intraday equities

[简体中文](README.zh-CN.md) · [Summary JSON](summary.json) · [Program API](../../programs.md)

Measured on one NVIDIA L4 (23,034 MiB, driver 580.82.07, CuPy CUDA 13 14.0.1) across per-fill maker/taker fees, pooled USDT cross margin, and regular-session US/A-share cash orders. 125 functional and numerical parity checks passed across CPU and CUDA on synthetic fixtures.

| Workload | Result |
| --- | ---: |
| Accounts × events × assets | 17 × 16 × 2 |
| Arithmetic | Paired FP32 |
| Account replay | 214.25 ms |
| Ending equity max abs error | 6.93e-12 quote units |
| Ending quantity max abs error | 0 |
| Measured rounds / iterations | 1 / 1 |

Replay timing includes market-buffer setup, account creation, order commands, detached feedback copies, assertions, and teardown; kernel compilation/JIT runs outside the timed block. Peak VRAM was not sampled in this run, and this micro-benchmark does not measure full portfolio query-and-commit throughput or GPU-vs-CPU speedup.

Coverage includes marketable vs. resting limit orders, partial-fill fee roles, and maker rebates; profitable-leg cross-margin collateral, funding-triggered full-account liquidation, and resting-order cancellation; A-share T+1 sellable inventory, odd lots, and split-adjusted position closes; and US T+1 sale-proceeds settlement, session order expiry, holidays, lunch breaks, early closes, and DST transitions. Equity formula and quantity-array evaluations match across CPU and CUDA within `rtol=2e-5`, `atol=2e-6`.

Cross-margin liquidation uses a full-account bar model rather than exchange-specific tiered maintenance brackets. Equity order accounts model regular-session cash trading; dividend entitlement, short borrowing, and auction limits are documented in [guides/data.md](../../data.md#equity-intraday-cash-accounts).
