# Native framework end-to-end comparison

Each candidate and asset has an independent cash account. Native SMA signals decide at close, enter at the next open and exit at the following open. This deliberately constrained one-bar strategy does not establish general trading-feature equivalence or benchmark Foundry discovery and admission.

End-to-end timing includes case loading, native indicators and simulation, metrics and result writing. Process startup and cold warm-up are excluded. Adapter import time is listed separately; cold time is the first adapter call, including its lazy imports and JIT. Each framework uses its native execution path; devices and versions are listed explicitly.

Error cells show maximum absolute / maximum scaled error; scaled error is |actual-reference| / max(1, |reference|). NAV is divided by initial cash. The zero-fee NAV limit is 1e-8; fee cases report differences without claiming numerical equivalence. SMA comparisons cover trade_start through the third-last row, excluding terminal delayed-output gaps.

Environment: Linux-6.6.159+-x86_64-with-glibc2.39; GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.82.07; threads: 2; 1 warm-up and 1 measured round per condition. Full CPU information is retained in comparison.json.

## case-1-0.npz

1 candidates × 8 assets × 549 days; one-way fee 0; reference: foundry-cpu.

### Speed (seconds)

| Implementation | Device | Import | Cold | Load | Compute | Write | End-to-end | Speed vs reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | cpu | 0.0459302 | 0.00392578 | 0.00053372 | 0.00246923 | 0.00583032 | 0.00883349 | 1.000× |
| foundry-cuda | cuda | 0.0454632 | 0.332723 | 0.000756279 | 0.0201583 | 0.00643102 | 0.0273459 | 0.323× |
| freqtrade | cpu | 0.942812 | 0.362775 | 0.000633309 | 0.347883 | 0.00250656 | 0.351023 | 0.025× |
| spectre-cuda | cuda | 0.766295 | 1.35308 | 0.000708749 | 1.044 | 0.0025983 | 1.04731 | 0.008× |
| vectorbt-numba | cpu | 0.0001105 | 1.23494 | 0.00066671 | 0.0124966 | 0.00249945 | 0.0156629 | 0.564× |
| vectorbt-rust | rust | 0.00010991 | 1.12952 | 0.00063296 | 0.012061 | 0.0024434 | 0.0151375 | 0.584× |

### Decisions and numerical differences

| Implementation | Decision mismatches | Fill-key mismatches | Unmatched fills | SMA abs / scaled | Normalized NAV error | Zero-fee NAV check |
| --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | 0 | 0 | 0 | 0 / 0 | 0 | PASS |
| foundry-cuda | 0 | 0 | 0 | 0 / 0 | 2.13718e-13 | PASS |
| freqtrade | 0 | 0 | 0 | 2.91038e-11 / 2.7692e-16 | 4.88498e-15 | PASS |
| spectre-cuda | 0 | 0 | 0 | 1.45519e-11 / 2.21115e-16 | 1.9984e-15 | PASS |
| vectorbt-numba | 0 | 0 | 0 | 2.38651e-09 / 3.20541e-14 | 1.9984e-15 | PASS |
| vectorbt-rust | 0 | 0 | 0 | 1.74623e-10 / 4.05285e-15 | 1.9984e-15 | PASS |

| Implementation | Matched fills | Quantity abs / scaled | Price abs / scaled | Fee abs / scaled | Positions abs / scaled |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | 1410 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| foundry-cuda | 1410 | 2.02097e-07 / 2.14198e-13 | 0 / 0 | 0 / 0 | 2.02097e-07 / 2.14198e-13 |
| freqtrade | 1410 | 7.24103e-08 / 6.62187e-14 | 0 / 0 | 0 / 0 | 7.24103e-08 / 6.62187e-14 |
| spectre-cuda | 1410 | 2.56114e-09 / 2.05564e-15 | 0 / 0 | 0 / 0 | 2.56114e-09 / 2.05564e-15 |
| vectorbt-numba | 1410 | 9.31323e-10 / 2.05564e-15 | 0 / 0 | 0 / 0 | 9.31323e-10 / 2.05564e-15 |
| vectorbt-rust | 1410 | 9.31323e-10 / 2.05564e-15 | 0 / 0 | 0 / 0 | 9.31323e-10 / 2.05564e-15 |

### Native ledger reconciliation

Native filled orders reconstruct holdings and NAV only for reconciliation, never to generate trades. A false independent-ledger flag means this is not independent evidence. Cent-rounding residuals are shown separately: fee differences propagate into cash and later quantities, so the whole NAV difference cannot be classified as floating-point error.

| Implementation | Independent ledger | Normalized NAV reconciliation | Final cash reconciliation | Unrounded fee residual | Cent-rounded fee residual |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | False | 1.66533e-15 | 2.09548e-09 | 0 | 0 |
| foundry-cuda | False | 2.16716e-13 | 2.16882e-07 | 0 | 0 |
| freqtrade | True | 0 | 5.23869e-08 | 0 | 0 |
| spectre-cuda | True | 1.33227e-15 | 0 | 0 | 0 |
| vectorbt-numba | True | 1.11022e-15 | 0 | 0 | 0 |
| vectorbt-rust | True | 1.11022e-15 | 0 | 0 | 0 |

### Common metrics

Cells show account median [minimum, maximum]; returns and drawdown are fractions. All use 365-day annualization and population volatility; zero-volatility Sharpe is zero. Per-account values and metric errors against the reference are in comparison.json.

| Implementation | Total return | CAGR | Sharpe | Max drawdown |
| --- | --- | --- | --- | --- |
| foundry-cpu | 0.00191688 [-0.0900101, 0.0832049] | 0.00191688 [-0.0900101, 0.0832049] | 0.0737953 [-0.875207, 0.487393] | 0.102031 [0.0615711, 0.125007] |
| foundry-cuda | 0.00191688 [-0.0900101, 0.0832049] | 0.00191688 [-0.0900101, 0.0832049] | 0.0737953 [-0.875207, 0.487393] | 0.102031 [0.0615711, 0.125007] |
| freqtrade | 0.00191688 [-0.0900101, 0.0832049] | 0.00191688 [-0.0900101, 0.0832049] | 0.0737953 [-0.875207, 0.487393] | 0.102031 [0.0615711, 0.125007] |
| spectre-cuda | 0.00191688 [-0.0900101, 0.0832049] | 0.00191688 [-0.0900101, 0.0832049] | 0.0737953 [-0.875207, 0.487393] | 0.102031 [0.0615711, 0.125007] |
| vectorbt-numba | 0.00191688 [-0.0900101, 0.0832049] | 0.00191688 [-0.0900101, 0.0832049] | 0.0737953 [-0.875207, 0.487393] | 0.102031 [0.0615711, 0.125007] |
| vectorbt-rust | 0.00191688 [-0.0900101, 0.0832049] | 0.00191688 [-0.0900101, 0.0832049] | 0.0737953 [-0.875207, 0.487393] | 0.102031 [0.0615711, 0.125007] |

### Implementation records

- **foundry-cpu**: Foundry native ts_mean and cash executor (cpu, native); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **foundry-cuda**: Foundry native ts_mean and cash executor (cuda, paired); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **freqtrade**: Freqtrade 2026.10 Backtesting + pandas rolling + native LocalTrade/Order/Wallets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **spectre-cuda**: spectre@c059665ba33b3d98f523ee55fdecf5105af146f6: cuda SMA float64; native CPU SimulationBlotter; native cent-rounded fees; `{"numpy": "2.5.4", "pandas": "3.1.1", "spectre": "0.5b1.dev541+gc059665ba", "torch": "2.15.0"}`
- **vectorbt-numba**: vectorbt 1.2.1: native MA and Portfolio.from_orders (numba); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **vectorbt-rust**: vectorbt 1.2.1: native MA and Portfolio.from_orders (rust); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`

## case-1-10.npz

1 candidates × 8 assets × 549 days; one-way fee 0.001; reference: foundry-cpu.

### Speed (seconds)

| Implementation | Device | Import | Cold | Load | Compute | Write | End-to-end | Speed vs reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | cpu | 0.0456206 | 0.00386998 | 0.00049509 | 0.00248842 | 0.00579237 | 0.00877612 | 1.000× |
| foundry-cuda | cuda | 0.0458679 | 0.33134 | 0.000733179 | 0.02014 | 0.00640429 | 0.0272777 | 0.322× |
| freqtrade | cpu | 0.941048 | 0.360766 | 0.000659089 | 0.345701 | 0.00248433 | 0.348844 | 0.025× |
| spectre-cuda | cuda | 0.776084 | 1.39076 | 0.000728849 | 1.0775 | 0.0026056 | 1.08083 | 0.008× |
| vectorbt-numba | cpu | 0.00011308 | 1.2484 | 0.000688009 | 0.0125902 | 0.00247368 | 0.0157522 | 0.557× |
| vectorbt-rust | rust | 0.000113849 | 1.14681 | 0.000680779 | 0.0119854 | 0.0024904 | 0.0151568 | 0.579× |

### Decisions and numerical differences

| Implementation | Decision mismatches | Fill-key mismatches | Unmatched fills | SMA abs / scaled | Normalized NAV error | Zero-fee NAV check |
| --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | 0 | 0 | 0 | 0 / 0 | 0 | n/a |
| foundry-cuda | 0 | 0 | 0 | 0 / 0 | 1.4988e-13 | n/a |
| freqtrade | 0 | 0 | 0 | 2.91038e-11 / 2.7692e-16 | 5.32907e-15 | n/a |
| spectre-cuda | 0 | 0 | 0 | 1.45519e-11 / 2.21115e-16 | 7.45306e-08 | n/a |
| vectorbt-numba | 0 | 0 | 0 | 2.38651e-09 / 3.20541e-14 | 2.66454e-15 | n/a |
| vectorbt-rust | 0 | 0 | 0 | 1.74623e-10 / 4.05285e-15 | 2.66454e-15 | n/a |

| Implementation | Matched fills | Quantity abs / scaled | Price abs / scaled | Fee abs / scaled | Positions abs / scaled |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | 1410 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| foundry-cuda | 1410 | 1.46916e-07 / 1.36215e-13 | 0 / 0 | 3.60956e-11 / 1.3624e-13 | 1.46916e-07 / 1.36215e-13 |
| freqtrade | 1410 | 2.8871e-08 / 5.70294e-14 | 0 / 0 | 1.39266e-11 / 5.70943e-14 | 2.8871e-08 / 5.70294e-14 |
| spectre-cuda | 1410 | 0.0455009 / 7.49001e-08 | 0 / 0 | 0.00499607 / 2.32248e-05 | 0.0455009 / 7.49001e-08 |
| vectorbt-numba | 1410 | 1.16415e-09 / 2.59412e-15 | 0 / 0 | 6.82121e-13 / 2.67011e-15 | 1.16415e-09 / 2.59412e-15 |
| vectorbt-rust | 1410 | 1.16415e-09 / 2.59412e-15 | 0 / 0 | 6.82121e-13 / 2.67011e-15 | 1.16415e-09 / 2.59412e-15 |

### Native ledger reconciliation

Native filled orders reconstruct holdings and NAV only for reconciliation, never to generate trades. A false independent-ledger flag means this is not independent evidence. Cent-rounding residuals are shown separately: fee differences propagate into cash and later quantities, so the whole NAV difference cannot be classified as floating-point error.

| Implementation | Independent ledger | Normalized NAV reconciliation | Final cash reconciliation | Unrounded fee residual | Cent-rounded fee residual |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | False | 1.77636e-15 | 1.62981e-09 | 0 | 0.00499607 |
| foundry-cuda | False | 1.49436e-13 | 1.43307e-07 | 0 | 0.00499607 |
| freqtrade | True | 0 | 4.49363e-08 | 0 | 0.00499607 |
| spectre-cuda | True | 1.11022e-15 | 0 | 0.0049968 | 0 |
| vectorbt-numba | True | 1.11022e-15 | 0 | 0 | 0.00499607 |
| vectorbt-rust | True | 1.11022e-15 | 0 | 0 | 0.00499607 |

### Common metrics

Cells show account median [minimum, maximum]; returns and drawdown are fractions. All use 365-day annualization and population volatility; zero-volatility Sharpe is zero. Per-account values and metric errors against the reference are in comparison.json.

| Implementation | Total return | CAGR | Sharpe | Max drawdown |
| --- | --- | --- | --- | --- |
| foundry-cpu | -0.0417824 [-0.12784, 0.0438124] | -0.0417824 [-0.12784, 0.0438124] | -0.39052 [-1.29039, 0.3045] | 0.112568 [0.0745481, 0.152311] |
| foundry-cuda | -0.0417824 [-0.12784, 0.0438124] | -0.0417824 [-0.12784, 0.0438124] | -0.39052 [-1.29039, 0.3045] | 0.112568 [0.0745481, 0.152311] |
| freqtrade | -0.0417824 [-0.12784, 0.0438124] | -0.0417824 [-0.12784, 0.0438124] | -0.39052 [-1.29039, 0.3045] | 0.112568 [0.0745481, 0.152311] |
| spectre-cuda | -0.0417824 [-0.12784, 0.0438124] | -0.0417824 [-0.12784, 0.0438124] | -0.39052 [-1.29039, 0.3045] | 0.112568 [0.0745481, 0.152311] |
| vectorbt-numba | -0.0417824 [-0.12784, 0.0438124] | -0.0417824 [-0.12784, 0.0438124] | -0.39052 [-1.29039, 0.3045] | 0.112568 [0.0745481, 0.152311] |
| vectorbt-rust | -0.0417824 [-0.12784, 0.0438124] | -0.0417824 [-0.12784, 0.0438124] | -0.39052 [-1.29039, 0.3045] | 0.112568 [0.0745481, 0.152311] |

### Implementation records

- **foundry-cpu**: Foundry native ts_mean and cash executor (cpu, native); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **foundry-cuda**: Foundry native ts_mean and cash executor (cuda, paired); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **freqtrade**: Freqtrade 2026.10 Backtesting + pandas rolling + native LocalTrade/Order/Wallets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **spectre-cuda**: spectre@c059665ba33b3d98f523ee55fdecf5105af146f6: cuda SMA float64; native CPU SimulationBlotter; native cent-rounded fees; `{"numpy": "2.5.4", "pandas": "3.1.1", "spectre": "0.5b1.dev541+gc059665ba", "torch": "2.15.0"}`
- **vectorbt-numba**: vectorbt 1.2.1: native MA and Portfolio.from_orders (numba); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **vectorbt-rust**: vectorbt 1.2.1: native MA and Portfolio.from_orders (rust); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`

## case-16-0.npz

16 candidates × 8 assets × 549 days; one-way fee 0; reference: foundry-cpu.

### Speed (seconds)

| Implementation | Device | Import | Cold | Load | Compute | Write | End-to-end | Speed vs reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | cpu | 0.0459098 | 0.0240045 | 0.000631829 | 0.0205909 | 0.0268026 | 0.0480257 | 1.000× |
| foundry-cuda | cuda | 0.0454476 | 0.348161 | 0.00075883 | 0.0362142 | 0.0275211 | 0.0644943 | 0.745× |
| freqtrade | cpu | 0.932556 | 4.3064 | 0.000703729 | 4.21292 | 0.0233809 | 4.23701 | 0.011× |
| spectre-cuda | cuda | 0.770012 | 15.9952 | 0.000760649 | 15.8862 | 0.0223131 | 15.9093 | 0.003× |
| vectorbt-numba | cpu | 0.000115 | 1.299 | 0.00071899 | 0.0654234 | 0.154807 | 0.22095 | 0.217× |
| vectorbt-rust | rust | 0.00012812 | 1.19532 | 0.000706149 | 0.0622654 | 0.0225238 | 0.0854957 | 0.562× |

### Decisions and numerical differences

| Implementation | Decision mismatches | Fill-key mismatches | Unmatched fills | SMA abs / scaled | Normalized NAV error | Zero-fee NAV check |
| --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | 0 | 0 | 0 | 0 / 0 | 0 | PASS |
| foundry-cuda | 0 | 0 | 0 | 0 / 0 | 2.13718e-13 | PASS |
| freqtrade | 0 | 0 | 0 | 2.91038e-11 / 5.98061e-16 | 4.88498e-15 | PASS |
| spectre-cuda | 0 | 0 | 0 | 1.45519e-11 / 7.97415e-16 | 1.9984e-15 | PASS |
| vectorbt-numba | 0 | 0 | 0 | 2.38651e-09 / 3.20541e-14 | 2.22045e-15 | PASS |
| vectorbt-rust | 0 | 0 | 0 | 2.61934e-10 / 4.05285e-15 | 2.22045e-15 | PASS |

| Implementation | Matched fills | Quantity abs / scaled | Price abs / scaled | Fee abs / scaled | Positions abs / scaled |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | 16246 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| foundry-cuda | 16246 | 2.02097e-07 / 2.14198e-13 | 0 / 0 | 0 / 0 | 2.02097e-07 / 2.14198e-13 |
| freqtrade | 16246 | 7.24103e-08 / 6.62187e-14 | 0 / 0 | 0 / 0 | 7.24103e-08 / 6.62187e-14 |
| spectre-cuda | 16246 | 2.56114e-09 / 2.05564e-15 | 0 / 0 | 0 / 0 | 2.56114e-09 / 2.05564e-15 |
| vectorbt-numba | 16246 | 2.09548e-09 / 2.30208e-15 | 0 / 0 | 0 / 0 | 2.09548e-09 / 2.30208e-15 |
| vectorbt-rust | 16246 | 2.09548e-09 / 2.30208e-15 | 0 / 0 | 0 / 0 | 2.09548e-09 / 2.30208e-15 |

### Native ledger reconciliation

Native filled orders reconstruct holdings and NAV only for reconciliation, never to generate trades. A false independent-ledger flag means this is not independent evidence. Cent-rounding residuals are shown separately: fee differences propagate into cash and later quantities, so the whole NAV difference cannot be classified as floating-point error.

| Implementation | Independent ledger | Normalized NAV reconciliation | Final cash reconciliation | Unrounded fee residual | Cent-rounded fee residual |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | False | 1.66533e-15 | 2.21189e-09 | 0 | 0 |
| foundry-cuda | False | 2.16716e-13 | 2.16882e-07 | 0 | 0 |
| freqtrade | True | 0 | 5.75092e-08 | 0 | 0 |
| spectre-cuda | True | 1.44329e-15 | 0 | 0 | 0 |
| vectorbt-numba | True | 1.44329e-15 | 0 | 0 | 0 |
| vectorbt-rust | True | 1.44329e-15 | 0 | 0 | 0 |

### Common metrics

Cells show account median [minimum, maximum]; returns and drawdown are fractions. All use 365-day annualization and population volatility; zero-volatility Sharpe is zero. Per-account values and metric errors against the reference are in comparison.json.

| Implementation | Total return | CAGR | Sharpe | Max drawdown |
| --- | --- | --- | --- | --- |
| foundry-cpu | -0.0609619 [-0.161463, 0.0832049] | -0.0609619 [-0.161463, 0.0832049] | -0.832901 [-2.24511, 0.487393] | 0.0928262 [0.0456172, 0.169736] |
| foundry-cuda | -0.0609619 [-0.161463, 0.0832049] | -0.0609619 [-0.161463, 0.0832049] | -0.832901 [-2.24511, 0.487393] | 0.0928262 [0.0456172, 0.169736] |
| freqtrade | -0.0609619 [-0.161463, 0.0832049] | -0.0609619 [-0.161463, 0.0832049] | -0.832901 [-2.24511, 0.487393] | 0.0928262 [0.0456172, 0.169736] |
| spectre-cuda | -0.0609619 [-0.161463, 0.0832049] | -0.0609619 [-0.161463, 0.0832049] | -0.832901 [-2.24511, 0.487393] | 0.0928262 [0.0456172, 0.169736] |
| vectorbt-numba | -0.0609619 [-0.161463, 0.0832049] | -0.0609619 [-0.161463, 0.0832049] | -0.832901 [-2.24511, 0.487393] | 0.0928262 [0.0456172, 0.169736] |
| vectorbt-rust | -0.0609619 [-0.161463, 0.0832049] | -0.0609619 [-0.161463, 0.0832049] | -0.832901 [-2.24511, 0.487393] | 0.0928262 [0.0456172, 0.169736] |

### Implementation records

- **foundry-cpu**: Foundry native ts_mean and cash executor (cpu, native); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **foundry-cuda**: Foundry native ts_mean and cash executor (cuda, paired); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **freqtrade**: Freqtrade 2026.10 Backtesting + pandas rolling + native LocalTrade/Order/Wallets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **spectre-cuda**: spectre@c059665ba33b3d98f523ee55fdecf5105af146f6: cuda SMA float64; native CPU SimulationBlotter; native cent-rounded fees; `{"numpy": "2.5.4", "pandas": "3.1.1", "spectre": "0.5b1.dev541+gc059665ba", "torch": "2.15.0"}`
- **vectorbt-numba**: vectorbt 1.2.1: native MA and Portfolio.from_orders (numba); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **vectorbt-rust**: vectorbt 1.2.1: native MA and Portfolio.from_orders (rust); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`

## case-16-10.npz

16 candidates × 8 assets × 549 days; one-way fee 0.001; reference: foundry-cpu.

### Speed (seconds)

| Implementation | Device | Import | Cold | Load | Compute | Write | End-to-end | Speed vs reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | cpu | 0.0460364 | 0.0238416 | 0.000594449 | 0.0204521 | 0.0269874 | 0.0480344 | 1.000× |
| foundry-cuda | cuda | 0.0459275 | 0.344594 | 0.000789159 | 0.0366084 | 0.0274356 | 0.0648334 | 0.741× |
| freqtrade | cpu | 0.941705 | 4.34368 | 0.00073573 | 4.26119 | 0.0234995 | 4.28543 | 0.011× |
| spectre-cuda | cuda | 0.770184 | 16.3203 | 0.000736099 | 16.2527 | 0.0228791 | 16.2763 | 0.003× |
| vectorbt-numba | cpu | 0.00011079 | 1.28935 | 0.000699859 | 0.065945 | 0.164135 | 0.230781 | 0.208× |
| vectorbt-rust | rust | 0.00011191 | 1.18346 | 0.00068292 | 0.0621264 | 0.0222618 | 0.0850715 | 0.565× |

### Decisions and numerical differences

| Implementation | Decision mismatches | Fill-key mismatches | Unmatched fills | SMA abs / scaled | Normalized NAV error | Zero-fee NAV check |
| --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | 0 | 0 | 0 | 0 / 0 | 0 | n/a |
| foundry-cuda | 0 | 0 | 0 | 0 / 0 | 1.4988e-13 | n/a |
| freqtrade | 0 | 0 | 0 | 2.91038e-11 / 5.98061e-16 | 6.32827e-15 | n/a |
| spectre-cuda | 0 | 0 | 0 | 1.45519e-11 / 7.97415e-16 | 7.45306e-08 | n/a |
| vectorbt-numba | 0 | 0 | 0 | 2.38651e-09 / 3.20541e-14 | 2.66454e-15 | n/a |
| vectorbt-rust | 0 | 0 | 0 | 2.61934e-10 / 4.05285e-15 | 2.66454e-15 | n/a |

| Implementation | Matched fills | Quantity abs / scaled | Price abs / scaled | Fee abs / scaled | Positions abs / scaled |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | 16246 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| foundry-cuda | 16246 | 2.37254e-07 / 1.37027e-13 | 0 / 0 | 3.60956e-11 / 1.37069e-13 | 2.37254e-07 / 1.37027e-13 |
| freqtrade | 16246 | 1.06404e-07 / 7.152e-14 | 0 / 0 | 1.7792e-11 / 7.15172e-14 | 1.06404e-07 / 7.152e-14 |
| spectre-cuda | 16246 | 0.0824727 / 7.49001e-08 | 0 / 0 | 0.00501105 / 2.64612e-05 | 0.0824727 / 7.49001e-08 |
| vectorbt-numba | 16246 | 3.0268e-09 / 2.78997e-15 | 0 / 0 | 6.82121e-13 / 2.92313e-15 | 3.0268e-09 / 2.78997e-15 |
| vectorbt-rust | 16246 | 3.0268e-09 / 2.78997e-15 | 0 / 0 | 6.82121e-13 / 2.92313e-15 | 3.0268e-09 / 2.78997e-15 |

### Native ledger reconciliation

Native filled orders reconstruct holdings and NAV only for reconciliation, never to generate trades. A false independent-ledger flag means this is not independent evidence. Cent-rounding residuals are shown separately: fee differences propagate into cash and later quantities, so the whole NAV difference cannot be classified as floating-point error.

| Implementation | Independent ledger | Normalized NAV reconciliation | Final cash reconciliation | Unrounded fee residual | Cent-rounded fee residual |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | False | 2.55351e-15 | 2.21189e-09 | 0 | 0.00499998 |
| foundry-cuda | False | 1.49436e-13 | 1.43307e-07 | 0 | 0.00499998 |
| freqtrade | True | 0 | 6.34464e-08 | 0 | 0.00499998 |
| spectre-cuda | True | 1.55431e-15 | 0 | 0.00499996 | 0 |
| vectorbt-numba | True | 1.33227e-15 | 0 | 0 | 0.00499998 |
| vectorbt-rust | True | 1.33227e-15 | 0 | 0 | 0.00499998 |

### Common metrics

Cells show account median [minimum, maximum]; returns and drawdown are fractions. All use 365-day annualization and population volatility; zero-volatility Sharpe is zero. Per-account values and metric errors against the reference are in comparison.json.

| Implementation | Total return | CAGR | Sharpe | Max drawdown |
| --- | --- | --- | --- | --- |
| foundry-cpu | -0.0885936 [-0.181279, 0.0438124] | -0.0885936 [-0.181279, 0.0438124] | -1.25219 [-2.66844, 0.3045] | 0.107476 [0.0603455, 0.189087] |
| foundry-cuda | -0.0885936 [-0.181279, 0.0438124] | -0.0885936 [-0.181279, 0.0438124] | -1.25219 [-2.66844, 0.3045] | 0.107476 [0.0603455, 0.189087] |
| freqtrade | -0.0885936 [-0.181279, 0.0438124] | -0.0885936 [-0.181279, 0.0438124] | -1.25219 [-2.66844, 0.3045] | 0.107476 [0.0603455, 0.189087] |
| spectre-cuda | -0.0885936 [-0.181279, 0.0438124] | -0.0885936 [-0.181279, 0.0438124] | -1.25219 [-2.66844, 0.3045] | 0.107476 [0.0603455, 0.189087] |
| vectorbt-numba | -0.0885936 [-0.181279, 0.0438124] | -0.0885936 [-0.181279, 0.0438124] | -1.25219 [-2.66844, 0.3045] | 0.107476 [0.0603455, 0.189087] |
| vectorbt-rust | -0.0885936 [-0.181279, 0.0438124] | -0.0885936 [-0.181279, 0.0438124] | -1.25219 [-2.66844, 0.3045] | 0.107476 [0.0603455, 0.189087] |

### Implementation records

- **foundry-cpu**: Foundry native ts_mean and cash executor (cpu, native); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **foundry-cuda**: Foundry native ts_mean and cash executor (cuda, paired); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **freqtrade**: Freqtrade 2026.10 Backtesting + pandas rolling + native LocalTrade/Order/Wallets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **spectre-cuda**: spectre@c059665ba33b3d98f523ee55fdecf5105af146f6: cuda SMA float64; native CPU SimulationBlotter; native cent-rounded fees; `{"numpy": "2.5.4", "pandas": "3.1.1", "spectre": "0.5b1.dev541+gc059665ba", "torch": "2.15.0"}`
- **vectorbt-numba**: vectorbt 1.2.1: native MA and Portfolio.from_orders (numba); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **vectorbt-rust**: vectorbt 1.2.1: native MA and Portfolio.from_orders (rust); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`

## case-64-0.npz

64 candidates × 8 assets × 549 days; one-way fee 0; reference: foundry-cpu.

### Speed (seconds)

| Implementation | Device | Import | Cold | Load | Compute | Write | End-to-end | Speed vs reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | cpu | 0.0453684 | 0.0679497 | 0.000736079 | 0.0632273 | 0.0731507 | 0.137115 | 1.000× |
| foundry-cuda | cuda | 0.0457661 | 0.391824 | 0.000822569 | 0.0730185 | 0.0721139 | 0.145955 | 0.939× |
| freqtrade | cpu | 0.937277 | 13.7501 | 0.000808819 | 13.8084 | 0.0731952 | 13.8824 | 0.010× |
| spectre-cuda | cuda | 0.770368 | 67.0536 | 0.000791519 | 67.3607 | 0.0696941 | 67.4312 | 0.002× |
| vectorbt-numba | cpu | 0.0001114 | 1.56016 | 0.000819339 | 0.199131 | 0.0680938 | 0.268044 | 0.512× |
| vectorbt-rust | rust | 0.00011121 | 1.31347 | 0.000775019 | 0.190677 | 0.198358 | 0.38981 | 0.352× |

### Decisions and numerical differences

| Implementation | Decision mismatches | Fill-key mismatches | Unmatched fills | SMA abs / scaled | Normalized NAV error | Zero-fee NAV check |
| --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | 0 | 0 | 0 | 0 / 0 | 0 | PASS |
| foundry-cuda | 0 | 0 | 0 | 0 / 0 | 2.13718e-13 | PASS |
| freqtrade | 0 | 0 | 0 | 2.91038e-11 / 5.98061e-16 | 4.88498e-15 | PASS |
| spectre-cuda | 0 | 0 | 0 | 1.45519e-11 / 7.97415e-16 | 1.9984e-15 | PASS |
| vectorbt-numba | 0 | 0 | 0 | 2.38651e-09 / 3.20541e-14 | 2.22045e-15 | PASS |
| vectorbt-rust | 0 | 0 | 0 | 2.61934e-10 / 4.05285e-15 | 2.22045e-15 | PASS |

| Implementation | Matched fills | Quantity abs / scaled | Price abs / scaled | Fee abs / scaled | Positions abs / scaled |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | 48032 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| foundry-cuda | 48032 | 3.04775e-07 / 2.14198e-13 | 0 / 0 | 0 / 0 | 3.04775e-07 / 2.14198e-13 |
| freqtrade | 48032 | 7.98609e-08 / 6.62187e-14 | 0 / 0 | 0 / 0 | 7.98609e-08 / 6.62187e-14 |
| spectre-cuda | 48032 | 2.56114e-09 / 2.05564e-15 | 0 / 0 | 0 / 0 | 2.56114e-09 / 2.05564e-15 |
| vectorbt-numba | 48032 | 2.09548e-09 / 2.30208e-15 | 0 / 0 | 0 / 0 | 2.09548e-09 / 2.30208e-15 |
| vectorbt-rust | 48032 | 2.09548e-09 / 2.30208e-15 | 0 / 0 | 0 / 0 | 2.09548e-09 / 2.30208e-15 |

### Native ledger reconciliation

Native filled orders reconstruct holdings and NAV only for reconciliation, never to generate trades. A false independent-ledger flag means this is not independent evidence. Cent-rounding residuals are shown separately: fee differences propagate into cash and later quantities, so the whole NAV difference cannot be classified as floating-point error.

| Implementation | Independent ledger | Normalized NAV reconciliation | Final cash reconciliation | Unrounded fee residual | Cent-rounded fee residual |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | False | 1.77636e-15 | 2.21189e-09 | 0 | 0 |
| foundry-cuda | False | 2.16716e-13 | 2.16882e-07 | 0 | 0 |
| freqtrade | True | 0 | 5.75092e-08 | 0 | 0 |
| spectre-cuda | True | 1.44329e-15 | 0 | 0 | 0 |
| vectorbt-numba | True | 1.55431e-15 | 0 | 0 | 0 |
| vectorbt-rust | True | 1.55431e-15 | 0 | 0 | 0 |

### Common metrics

Cells show account median [minimum, maximum]; returns and drawdown are fractions. All use 365-day annualization and population volatility; zero-volatility Sharpe is zero. Per-account values and metric errors against the reference are in comparison.json.

| Implementation | Total return | CAGR | Sharpe | Max drawdown |
| --- | --- | --- | --- | --- |
| foundry-cpu | -0.049662 [-0.178827, 0.137623] | -0.049662 [-0.178827, 0.137623] | -0.83562 [-2.48938, 1.45361] | 0.0735815 [0, 0.186929] |
| foundry-cuda | -0.049662 [-0.178827, 0.137623] | -0.049662 [-0.178827, 0.137623] | -0.83562 [-2.48938, 1.45361] | 0.0735815 [0, 0.186929] |
| freqtrade | -0.049662 [-0.178827, 0.137623] | -0.049662 [-0.178827, 0.137623] | -0.83562 [-2.48938, 1.45361] | 0.0735815 [0, 0.186929] |
| spectre-cuda | -0.049662 [-0.178827, 0.137623] | -0.049662 [-0.178827, 0.137623] | -0.83562 [-2.48938, 1.45361] | 0.0735815 [0, 0.186929] |
| vectorbt-numba | -0.049662 [-0.178827, 0.137623] | -0.049662 [-0.178827, 0.137623] | -0.83562 [-2.48938, 1.45361] | 0.0735815 [0, 0.186929] |
| vectorbt-rust | -0.049662 [-0.178827, 0.137623] | -0.049662 [-0.178827, 0.137623] | -0.83562 [-2.48938, 1.45361] | 0.0735815 [0, 0.186929] |

### Implementation records

- **foundry-cpu**: Foundry native ts_mean and cash executor (cpu, native); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **foundry-cuda**: Foundry native ts_mean and cash executor (cuda, paired); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **freqtrade**: Freqtrade 2026.10 Backtesting + pandas rolling + native LocalTrade/Order/Wallets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **spectre-cuda**: spectre@c059665ba33b3d98f523ee55fdecf5105af146f6: cuda SMA float64; native CPU SimulationBlotter; native cent-rounded fees; `{"numpy": "2.5.4", "pandas": "3.1.1", "spectre": "0.5b1.dev541+gc059665ba", "torch": "2.15.0"}`
- **vectorbt-numba**: vectorbt 1.2.1: native MA and Portfolio.from_orders (numba); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **vectorbt-rust**: vectorbt 1.2.1: native MA and Portfolio.from_orders (rust); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`

## case-64-10.npz

64 candidates × 8 assets × 549 days; one-way fee 0.001; reference: foundry-cpu.

### Speed (seconds)

| Implementation | Device | Import | Cold | Load | Compute | Write | End-to-end | Speed vs reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | cpu | 0.0456236 | 0.068644 | 0.000736799 | 0.0633513 | 0.0728574 | 0.136946 | 1.000× |
| foundry-cuda | cuda | 0.046104 | 0.391295 | 0.000830769 | 0.0717947 | 0.0725076 | 0.145133 | 0.944× |
| freqtrade | cpu | 0.945953 | 13.9072 | 0.000840199 | 13.948 | 0.0745872 | 14.0234 | 0.010× |
| spectre-cuda | cuda | 0.766918 | 65.5944 | 0.00083758 | 66.3757 | 0.0681072 | 66.4446 | 0.002× |
| vectorbt-numba | cpu | 0.00011402 | 1.59665 | 0.000803539 | 0.199743 | 0.067837 | 0.268384 | 0.510× |
| vectorbt-rust | rust | 0.00011451 | 1.33778 | 0.000802109 | 0.193573 | 0.20272 | 0.397096 | 0.345× |

### Decisions and numerical differences

| Implementation | Decision mismatches | Fill-key mismatches | Unmatched fills | SMA abs / scaled | Normalized NAV error | Zero-fee NAV check |
| --- | --- | --- | --- | --- | --- | --- |
| foundry-cpu | 0 | 0 | 0 | 0 / 0 | 0 | n/a |
| foundry-cuda | 0 | 0 | 0 | 0 / 0 | 1.57208e-13 | n/a |
| freqtrade | 0 | 0 | 0 | 2.91038e-11 / 5.98061e-16 | 6.32827e-15 | n/a |
| spectre-cuda | 0 | 0 | 0 | 1.45519e-11 / 7.97415e-16 | 9.13435e-08 | n/a |
| vectorbt-numba | 0 | 0 | 0 | 2.38651e-09 / 3.20541e-14 | 2.66454e-15 | n/a |
| vectorbt-rust | 0 | 0 | 0 | 2.61934e-10 / 4.05285e-15 | 2.66454e-15 | n/a |

| Implementation | Matched fills | Quantity abs / scaled | Price abs / scaled | Fee abs / scaled | Positions abs / scaled |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | 48032 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| foundry-cuda | 48032 | 2.37254e-07 / 1.53836e-13 | 0 / 0 | 3.91083e-11 / 1.53904e-13 | 2.37254e-07 / 1.53836e-13 |
| freqtrade | 48032 | 1.06404e-07 / 7.152e-14 | 0 / 0 | 1.7792e-11 / 7.15172e-14 | 1.06404e-07 / 7.152e-14 |
| spectre-cuda | 48032 | 0.0824727 / 9.56391e-08 | 0 / 0 | 0.0050236 / 2.93397e-05 | 0.0824727 / 9.56391e-08 |
| vectorbt-numba | 48032 | 3.0268e-09 / 2.78997e-15 | 0 / 0 | 6.82121e-13 / 2.92313e-15 | 3.0268e-09 / 2.78997e-15 |
| vectorbt-rust | 48032 | 3.0268e-09 / 2.78997e-15 | 0 / 0 | 6.82121e-13 / 2.92313e-15 | 3.0268e-09 / 2.78997e-15 |

### Native ledger reconciliation

Native filled orders reconstruct holdings and NAV only for reconciliation, never to generate trades. A false independent-ledger flag means this is not independent evidence. Cent-rounding residuals are shown separately: fee differences propagate into cash and later quantities, so the whole NAV difference cannot be classified as floating-point error.

| Implementation | Independent ledger | Normalized NAV reconciliation | Final cash reconciliation | Unrounded fee residual | Cent-rounded fee residual |
| --- | --- | --- | --- | --- | --- |
| foundry-cpu | False | 2.55351e-15 | 2.32831e-09 | 0 | 0.00499998 |
| foundry-cuda | False | 1.62093e-13 | 1.61584e-07 | 0 | 0.00499998 |
| freqtrade | True | 0 | 6.34464e-08 | 0 | 0.00499998 |
| spectre-cuda | True | 1.55431e-15 | 0 | 0.005 | 0 |
| vectorbt-numba | True | 1.33227e-15 | 0 | 0 | 0.00499998 |
| vectorbt-rust | True | 1.33227e-15 | 0 | 0 | 0.00499998 |

### Common metrics

Cells show account median [minimum, maximum]; returns and drawdown are fractions. All use 365-day annualization and population volatility; zero-volatility Sharpe is zero. Per-account values and metric errors against the reference are in comparison.json.

| Implementation | Total return | CAGR | Sharpe | Max drawdown |
| --- | --- | --- | --- | --- |
| foundry-cpu | -0.0725957 [-0.19501, 0.106691] | -0.0725957 [-0.19501, 0.106691] | -1.22584 [-2.85155, 1.35645] | 0.0875913 [0, 0.202687] |
| foundry-cuda | -0.0725957 [-0.19501, 0.106691] | -0.0725957 [-0.19501, 0.106691] | -1.22584 [-2.85155, 1.35645] | 0.0875913 [0, 0.202687] |
| freqtrade | -0.0725957 [-0.19501, 0.106691] | -0.0725957 [-0.19501, 0.106691] | -1.22584 [-2.85155, 1.35645] | 0.0875913 [0, 0.202687] |
| spectre-cuda | -0.0725957 [-0.19501, 0.106691] | -0.0725957 [-0.19501, 0.106691] | -1.22584 [-2.85155, 1.35645] | 0.0875913 [0, 0.202687] |
| vectorbt-numba | -0.0725957 [-0.19501, 0.106691] | -0.0725957 [-0.19501, 0.106691] | -1.22584 [-2.85155, 1.35645] | 0.0875913 [0, 0.202687] |
| vectorbt-rust | -0.0725957 [-0.19501, 0.106691] | -0.0725957 [-0.19501, 0.106691] | -1.22584 [-2.85155, 1.35645] | 0.0875913 [0, 0.202687] |

### Implementation records

- **foundry-cpu**: Foundry native ts_mean and cash executor (cpu, native); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **foundry-cuda**: Foundry native ts_mean and cash executor (cuda, paired); fills reconstructed from native NAV and declared one-bar targets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **freqtrade**: Freqtrade 2026.10 Backtesting + pandas rolling + native LocalTrade/Order/Wallets; `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **spectre-cuda**: spectre@c059665ba33b3d98f523ee55fdecf5105af146f6: cuda SMA float64; native CPU SimulationBlotter; native cent-rounded fees; `{"numpy": "2.5.4", "pandas": "3.1.1", "spectre": "0.5b1.dev541+gc059665ba", "torch": "2.15.0"}`
- **vectorbt-numba**: vectorbt 1.2.1: native MA and Portfolio.from_orders (numba); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
- **vectorbt-rust**: vectorbt 1.2.1: native MA and Portfolio.from_orders (rust); `{"TA-Lib": "0.8.3", "cupy-cuda13x": "14.2.0", "freqtrade": "2026.10", "numba": "0.68.1", "numpy": "2.5.4", "pandas": "3.1.1", "vectorbt": "1.2.1", "vectorbt-rust": "1.2.1"}`
