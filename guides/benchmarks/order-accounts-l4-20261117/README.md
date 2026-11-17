# Order accounts: fees, cross margin and intraday equities

[简体中文](README.zh-CN.md) · [Results](summary.json) · [Program API](../../programs.md)

The same Colab NVIDIA L4 session validated per-fill maker/taker fees, pooled USDT cross margin, and regular-session US/A-share cash orders. **125 focused checks passed**, including CPU/CUDA functional parity, public formula/array evaluation and commit decisions, PIT, calendars, settlement, and checkpoint restoration. No real-market performance or profitability claim is made from these synthetic fixtures.

| Single measured CUDA workload | Result |
| --- | ---: |
| Accounts × events × assets | 17 × 16 × 2 |
| Arithmetic | Paired FP32 |
| Complete account replay | **214.25 ms** |
| Ending equity maximum absolute error | **6.93e-12** quote units |
| Ending quantity maximum absolute error | **0** |
| Measured rounds / iterations | 1 / 1 |

Replay includes market-buffer preparation, account creation, commands, detached feedback, assertions and account close. Kernel preparation/JIT is outside the timed function. This is not a full research query-and-commit benchmark, a comparison against the old implementation, or a GPU-versus-CPU speed comparison. Peak VRAM was not sampled in this run.

The checks cover marketable versus resting limits, partial-fill fee roles and maker rebates; profitable-leg collateral support, funding-triggered account liquidation and order cancellation; A-share T+1 inventory, odd lots and split-adjusted targets; US T+1 cash, session expiry, holidays, lunch breaks, early closes and DST. Public equity formula/quantity-array results and admission decisions agree across CPU/CUDA within metric tolerances (`rtol=2e-5`, `atol=2e-6`).

The first collection attempt stopped before tests because another installed `tests` package shadowed the local suite. The next run passed 118 checks; six others hit an incompatible CUDA package installation and one retained an obsolete equity-rejection assertion. After repairing those issues, only the seven affected checks ran again and passed. The 214.25 ms timing was measured once after acceptance. Native compilation and Ruff checks passed; the private production baseline was neither read nor rewritten.

The device was an L4 with 23034 MiB and driver 580.82.07; the final environment used CuPy CUDA 13 14.0.1. All 119 uploaded core source files matched local SHA-256 digests. Raw evidence remains in local `var/runs/order-accounts-l4/evidence.tar.gz`; the session was released after verification.

Cross liquidation is an explicitly declared full-account Bar research model, not an exchange-specific risk-tier replica. Equity orders cover regular-session cash accounts; dividend entitlement, short borrowing, auctions and other limitations are detailed in the [data contract](../../data.md#equity-intraday-cash-accounts).
