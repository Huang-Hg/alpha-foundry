# Foundry vs. Spectre, vectorbt, and Freqtrade

[简体中文](README.zh-CN.md) · [Full report](report.md) · [CSV](summary.csv) · [Method](../../framework-benchmark.md)

On this controlled daily crypto-spot SMA pulse workload, Foundry's **CPU** backend recorded the lowest end-to-end latency across tested batch sizes; Foundry CUDA did not beat Foundry CPU at these sizes. At 64 candidates, Foundry CUDA finished in `0.145 s` compared with vectorbt Numba `0.268 s`, Freqtrade `14.023 s`, and Spectre `66.445 s`; at 1 candidate, vectorbt was faster than Foundry CUDA (`0.015–0.016 s` vs. `0.027 s`). Each number comes from one cold warm-up and one measured round on the same RTX PRO 6000 Blackwell Server (`2` CPU threads), reflecting this specific workload rather than a general ranking across frameworks.

Input fixture: 8 crypto spot pairs (`BTC`, `ETH`, `SOL`, `BNB`, `XRP`, `ADA`, `DOGE`, `LTC`), 549 UTC daily bars, and 365 trading days, with one independent account per `(candidate, asset)` pair. A causal SMA breakout buys at `open[t+1]` with 25% prior-close cash and exits at `open[t+2]`, up to 512 accounts and 48,032 fills, tested at zero fees and 10 bps per side.

## End-to-end latency (seconds, 10 bps per side)

Includes fixture loading, native indicators and accounts, common metric reduction, and `.npz` output writes (excludes imports, warm-up, and post-run verification).

| Implementation | 1 × 8 accounts | 16 × 8 | 64 × 8 | 64 × 8 native pipeline | Sampled GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foundry CPU | 0.008776 | 0.048034 | 0.136946 | 0.063351 | 0 MiB |
| Foundry CUDA (paired-FP32 account) | 0.027278 | 0.064833 | 0.145133 | 0.071795 | 558 MiB |
| vectorbt 1.2.1 (Numba) | 0.015752 | 0.230781 | 0.268384 | 0.199743 | 0 MiB |
| vectorbt 1.2.1 (Rust) | 0.015157 | 0.085072 | 0.397096 | 0.193573 | 0 MiB |
| Freqtrade 2026.10 | 0.348844 | 4.285429 | 14.023404 | 13.947977 | 0 MiB |
| Spectre (CUDA factors + CPU accounts) | 1.080832 | 16.276312 | 66.444601 | 66.375656 | 666 MiB |

The native-pipeline column measures each framework's indicator and account execution plus common-output materialization prior to metric reduction and disk writes. At 64 candidates, vectorbt's Rust pipeline ran slightly faster than Numba (`0.194 s` vs. `0.200 s`) while spending longer in the output phase (`0.203 s` vs. `0.068 s`). GPU memory is sampled every 20 ms (including context and allocator caches) and may miss briefer peaks.

## Numerical parity and scope

Across all 36 measured conditions, entry decisions and filled-event identities (`candidate`, `asset`, `date`, `side`) had zero mismatches, and every account ended flat with reconciled cash and ledger balances. Against Foundry CPU, zero-fee maximum normalized NAV error was `2.14e-13` for Foundry CUDA and at most `4.89e-15` for the external frameworks (threshold `1e-8`). At 10 bps, Foundry CUDA reached `1.57e-13` (NAV) and `2.31e-12` (Sharpe); Spectre reached `9.13e-8` / `1.87e-6` (`< 0.10` quote units per `1,000,000` initial cash) due to its native cent-rounded commission. Median total return across the 512 fee-bearing accounts was `-7.26%` (`-19.50%` to `+10.67%`).

In this target-mode test, Foundry's cash kernel outputs return series directly (fills are reconstructed from native NAV and one-bar targets, `native_ledger=False`), whereas Spectre and Freqtrade step full sequential event/wallet engines that carry extra overhead on simple target pulses. This benchmark does not compare shared-wallet portfolio scheduling, stop orders, live trading, equity rules, perpetual liquidation, or Foundry's order-account engine; per-condition tables are in [report.md](report.md) and [summary.csv](summary.csv).
