# Foundry vs. Spectre, vectorbt, and Freqtrade

[简体中文](README.zh-CN.md) · [Detailed report](report.md) · [Summary CSV](summary.csv) · [Methodology](../../framework-benchmark.md)

On this controlled daily crypto-spot SMA workload, **Foundry CPU achieved the lowest latency across tested sizes**, while Foundry CUDA beat all three external frameworks at 16 and 64 candidates (`0.145 s` at 64 candidates vs. `0.268 s` for vectorbt Numba, `14.023 s` for Freqtrade, and `66.445 s` for Spectre). At 1 candidate, CUDA launch overhead left Foundry CUDA slower than vectorbt. Each cell reflects one measured run on an RTX PRO 6000 Blackwell Server (`OMP/BLAS/Numba/Rayon = 2` threads), not a universal ranking across workloads.

Input fixture: **8 crypto spot assets, 549 UTC daily bars, 365 trading days** (`BTC`, `ETH`, `SOL`, `BNB`, `XRP`, `ADA`, `DOGE`, `LTC`), independent accounts per `(candidate, asset)` pair, 25% prior-close cash entry at `open[t+1]` and exit at `open[t+2]`, up to **512 accounts and 48,032 fills**.

## End-to-end latency (seconds, 10 bps per side)

Includes fixture loading, native indicators and account execution, common metric computation, and `.npz` output writes (excludes imports, warm-up, environment inspection, and post-run verification).

| Implementation | 1 × 8 accounts | 16 × 8 | 64 × 8 | 64 × 8 native pipeline | Sampled GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foundry CPU | 0.008776 | 0.048034 | 0.136946 | 0.063351 | 0 MiB |
| Foundry CUDA (paired-FP32 account) | 0.027278 | 0.064833 | 0.145133 | 0.071795 | 558 MiB |
| vectorbt 1.2.1 (Numba) | 0.015752 | 0.230781 | 0.268384 | 0.199743 | 0 MiB |
| vectorbt 1.2.1 (Rust) | 0.015157 | 0.085072 | 0.397096 | 0.193573 | 0 MiB |
| Freqtrade 2026.10 | 0.348844 | 4.285429 | 14.023404 | 13.947977 | 0 MiB |
| Spectre (CUDA factors + CPU blotter) | 1.080832 | 16.276312 | 66.444601 | 66.375656 | 666 MiB |

## Numerical and execution parity

- **Exact trade parity:** across all 36 measured conditions, entry decisions and filled-order identities (`candidate`, `asset`, `date`, `side`) had **zero mismatches**; every account finished flat with reconciled cash and ledger balances.
- **Zero-fee NAV accuracy:** max normalized NAV error vs. Foundry CPU was **`2.14e-13`** for Foundry CUDA and at most **`4.89e-15`** for external frameworks (well within the `1e-8` gate).
- **10 bps fee condition:** Foundry CUDA max normalized NAV / Sharpe error was **`1.57e-13` / `2.31e-12`**. Spectre's native cent-rounded commission produced **`9.13e-8` / `1.87e-6`** (`< 0.10` quote units per `1,000,000` initial cash) and is reported separately rather than patched upstream.
- **Tradeoffs:** Foundry's target-mode cash kernel is compact and fast, whereas its fills in this target-mode test are reconstructed from NAV (`native_ledger=False`); Spectre and Freqtrade execute full sequential event/wallet machinery that carries extra overhead on simple target pulses.

Published per-condition numbers live in [report.md](report.md) and [summary.csv](summary.csv); complete per-account arrays and environment logs are retained locally in `var/runs/framework-evidence.tar.gz`.
