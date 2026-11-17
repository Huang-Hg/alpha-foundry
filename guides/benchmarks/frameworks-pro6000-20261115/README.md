# Foundry against Spectre, vectorbt, and Freqtrade

[简体中文](README.zh-CN.md) · [Full report](report.md) · [CSV](summary.csv) · [Method and environment](../../framework-benchmark.md)

Foundry was fastest on this restricted daily spot workload using its **CPU** backend; Foundry CUDA did not beat Foundry CPU at the tested sizes. At 64 candidates, Foundry CUDA completed in `0.145 s` versus vectorbt Numba `0.268 s`, Freqtrade `14.023 s`, and Spectre `66.445 s`; at 1 candidate, vectorbt was faster than Foundry CUDA (`0.015–0.016 s` vs. `0.027 s`). Each cell comes from one cold warm-up and one valid measured round on the same Colab RTX PRO 6000 Blackwell Server (`2` CPU threads), not a general ranking of the projects.

Common input: **8 crypto spot assets, 549 UTC daily bars, and 365 trading days** (`BTC`, `ETH`, `SOL`, `BNB`, `XRP`, `ADA`, `DOGE`, `LTC`), one independent account per `(candidate, asset)` pair. A causal SMA pulse buys at `open[t+1]` using 25% prior-close cash and sells at `open[t+2]`, up to **512 accounts and 48,032 fills**, tested at both zero fees and 10 bps per side.

## End-to-end results (seconds, 10 bps per side)

Includes case loading, native indicators/accounts, common metrics, and `.npz` result writes; excludes imports, warm-up, environment inventory, and correctness comparison.

| Implementation | 1 candidate × 8 assets | 16 × 8 | 64 × 8 | 64 × 8 native pipeline | Sampled GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foundry CPU | 0.008776 | 0.048034 | 0.136946 | 0.063351 | 0 MiB |
| Foundry CUDA (paired-FP32 account) | 0.027278 | 0.064833 | 0.145133 | 0.071795 | 558 MiB |
| vectorbt 1.2.1 (Numba) | 0.015752 | 0.230781 | 0.268384 | 0.199743 | 0 MiB |
| vectorbt 1.2.1 (Rust) | 0.015157 | 0.085072 | 0.397096 | 0.193573 | 0 MiB |
| Freqtrade 2026.10 | 0.348844 | 4.285429 | 14.023404 | 13.947977 | 0 MiB |
| Spectre (CUDA factors + CPU accounts) | 1.080832 | 16.276312 | 66.444601 | 66.375656 | 666 MiB |

The native-pipeline column covers each adapter's native indicator/account execution and materialization before common metric/array output. At 64 candidates, vectorbt's Rust native pipeline was slightly faster than Numba (`0.194 s` vs. `0.200 s`), while its observed output phase was longer (`0.203 s` vs. `0.068 s`); this run does not establish why or a stable ranking between those two backends. GPU memory is sampled every 20 ms (including context and allocator caches) and can miss shorter peaks.

## Numerical results and scope limits

- **Trade parity:** all 36 measured conditions had zero entry-decision and filled-event identity mismatches; all accounts finished flat and passed cash/ledger reconciliation.
- **Zero-fee and 10 bps errors:** against Foundry CPU, zero-fee maximum normalized NAV error was `2.14e-13` for Foundry CUDA and at most `4.89e-15` for external frameworks (gate `1e-8`). At 10 bps, Foundry CUDA reached `1.57e-13` (NAV) and `2.31e-12` (Sharpe); Spectre reached `9.13e-8` / `1.87e-6` (`< 0.10` currency units per `1,000,000` initial cash) because its native cent-rounded commission was preserved without claiming bit-identity.
- **Non-strategy control:** the 512-account fee-bearing batch had median total return `-7.26%` (`-19.50%` to `+10.67%`); this synthetic pulse workload tests execution equivalence, not strategy profitability.
- **Architectural tradeoffs and non-claims:** Foundry's target-mode cash kernel exposes returns rather than an independent native order ledger (Foundry fills here are reconstructed from native NAV and declared one-bar targets, `native_ledger=False`), and CUDA launch/setup overhead still penalizes small batches. Spectre and Freqtrade execute full sequential event/wallet engines whose capabilities carry extra cost on narrow target pulses. This run does not compare shared-wallet portfolio scheduling, stop orders, live execution, A-share restrictions, perpetual liquidation, Foundry discovery/admission, or the separate 1.5× precision budget, nor does it establish a general GPU advantage.
- **Audit notes:** an earlier timing attempt was discarded because package-version enumeration ran inside the timer (none of its timings appear here). Subsequent architecture-review fixes passed 90 smoke checks and 303 non-performance regression checks (7,117 subtests, 1 private-baseline skip; 251 separate performance items were not rerun). Per-condition tables live in [report.md](report.md) and [summary.csv](summary.csv); raw arrays and archives remain in local `var/runs/framework-evidence.tar.gz`.
