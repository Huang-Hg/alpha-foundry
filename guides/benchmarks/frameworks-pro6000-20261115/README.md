# Foundry against Spectre, vectorbt and Freqtrade

[简体中文](README.zh-CN.md) · [Full report](report.md) · [CSV](summary.csv) · [Method and reproduction](../../framework-benchmark.md)

Foundry was fastest on this restricted daily spot workload using its **CPU** backend. CUDA did not beat Foundry CPU at the tested sizes. At 64 candidates, Foundry CUDA completed in 0.145 s versus vectorbt Numba 0.268 s, Freqtrade 14.023 s and Spectre 66.445 s. At one candidate, vectorbt was faster than Foundry CUDA. These are observations from one measured round per condition, not a general ranking of the projects.

The common input has **8 assets, 549 daily bars and 365 trading days**: BTC, ETH, SOL, BNB, XRP, ADA, DOGE and LTC. Each candidate/asset has an independent account. A causal SMA pulse buys at the next open and sells at the following open, with 25% prior-close cash allocation. The largest batch contains **512 accounts and 48,032 fills**. Both zero fees and 10 bps per side were tested on the same Colab RTX PRO 6000 Blackwell Server with two requested CPU threads.

## End-to-end results

Seconds, 10 bps per side; includes case loading, native indicators/accounts, common metrics and result-array writing. Imports, warm-up, environment inventory and correctness comparison are excluded.

| Implementation | 1 candidate × 8 assets | 16 × 8 | 64 × 8 | 64 × 8 native pipeline | Sampled GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foundry CPU | 0.008776 | 0.048034 | 0.136946 | 0.063351 | 0 MiB |
| Foundry CUDA, paired FP32 account | 0.027278 | 0.064833 | 0.145133 | 0.071795 | 558 MiB |
| vectorbt 1.2.1, Numba | 0.015752 | 0.230781 | 0.268384 | 0.199743 | 0 MiB |
| vectorbt 1.2.1, Rust | 0.015157 | 0.085072 | 0.397096 | 0.193573 | 0 MiB |
| Freqtrade 2026.10 | 0.348844 | 4.285429 | 14.023404 | 13.947977 | 0 MiB |
| Spectre, CUDA factors + CPU accounts | 1.080832 | 16.276312 | 66.444601 | 66.375656 | 666 MiB |

The native-pipeline column includes each adapter's native indicator and account work plus materialization of the common result, before common metric/array output. Rust's native pipeline was slightly faster than Numba at 64 candidates, but its observed output phase was longer (0.203 s versus 0.068 s). This run does not establish why or a stable ranking between those backends. GPU memory is sampled every 20 ms and can miss shorter peaks; it includes context and allocator caches.

## Numerical and trading results

- All 36 measured conditions had **zero entry-decision and filled-event identity mismatches**. All completed flat; native cash/ledger checks passed.
- Zero-fee maximum normalized NAV error was **2.14e-13** for Foundry CUDA and at most **4.89e-15** for the external frameworks, against Foundry CPU. The declared limit was 1e-8.
- At 10 bps, Foundry CUDA's largest normalized NAV/Sharpe errors were **1.57e-13 / 2.31e-12**. Spectre's were **9.13e-8 / 1.87e-6**; its native commission rounding to cents was preserved. The NAV discrepancy is under 0.10 currency units per 1,000,000 initial cash. The fee-bearing results are not claimed bit-identical.
- The 512-account fee-bearing control had median total return **−7.26%**, ranging from **−19.50% to +10.67%**. This artificial workload tests execution equivalence, not a recommended trading strategy.

Foundry's strengths here are the compact native account loop and batched numerical path. Its gaps are clear: CUDA launch/setup overhead still hurts small jobs, and its cash executor exposes returns rather than an independent native order ledger. Foundry fills in this report are explicitly reconstructed from native NAV and the declared one-bar targets. Spectre and vectorbt supply native account histories; Freqtrade supplies native orders/wallets, from which close NAV is marked and reconciled.

The competitors execute their real native engines. Spectre batches its factors, then uses native saved-factor caches and sequential event accounts; Freqtrade uses sequential native independent wallets and offline exchange metadata. Those event-driven capabilities carry overhead that this narrow target-allocation task does not need. The experiment does **not** compare shared-wallet portfolio scheduling, stop orders, live execution, A-share restrictions, perpetual liquidation, Foundry discovery/admission or the separate 1.5× precision budget. It does not establish a universal GPU advantage.

## Evidence

[comparison.json](comparison.json) contains per-account metrics, exact mismatch counts, fee residuals, all versions and timing stages. [plan.json](plan.json) fixes the inputs and source hashes. [source.tar.gz](source.tar.gz) preserves the exact measured source bytes, before the subsequent architecture-review corrections. [contracts.xml](contracts.xml) records the preceding 48-test contract gate and its subtests. [SHA256SUMS](SHA256SUMS) covers the published artifacts. Full native arrays and setup logs remain in the local `var/runs/framework-evidence.tar.gz` archive.

An earlier timing attempt was invalidated because dependency-version enumeration was inside the timer; none of its speed numbers appear here. Each published condition has one cold warm-up and one valid measured round. Tests and benchmarks ran only on Colab.

The subsequent architecture-review corrections passed [90 smoke checks](review-smoke.xml), followed by [303 non-performance regression checks](review-regression.xml), 7117 subtests and one explicit private-baseline skip. The 251 deselected items were Foundry's separate performance measurements and were not rerun for these review corrections. [review-source.json](review-source.json) identifies that later validated source; it does not replace the benchmark's measured source archive. Local/cloud raw-byte parity passed for all 150 deployment files. The Colab instance was released after evidence retrieval.
