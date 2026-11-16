# Comparing native backtest frameworks

[简体中文](framework-benchmark.zh-CN.md) · [Benchmark guide](benchmarks/README.md)

This benchmark compares Foundry, Spectre, vectorbt and Freqtrade on the same deliberately small trading contract. It measures each implementation's indicators, native account execution and output materialization. It does not compare strategy-discovery algorithms, Foundry portfolio admission or live trading, and it does not establish that the frameworks support the same markets or trading features.

This page defines the method and reproduction environment. It contains no performance results; completed run artifacts provide those measurements.

## Common data and strategy

The source is the preserved `data/crypto_spot_daily/panel.parquet` snapshot. The fixture includes daily UTC OHLCV from 2024-07-01 through 2025-12-31, with trading starting on 2025-01-01. Earlier rows supply indicator warm-up. The fixed asset order is BTC, ETH, SOL, BNB, XRP, ADA, DOGE and LTC; a smaller run uses its prefix. Selection does not depend on strategy returns. Missing observations, nonpositive prices or inconsistent OHLC bounds fail preparation instead of being filled.

Candidates use SMA windows `(5,10,20,30,40,60,90,120)` and thresholds `(-.02,-.01,0,.01,.02,.03,.04,.05)`, ordered by threshold and then window. Runs of 1, 16 and 64 candidates take prefixes of this grid. Every framework receives identical float64 quote arrays; factor output precision is reported rather than assumed.

Each candidate/asset pair has a separate account with initial cash of 1,000,000 quote-currency units. There is no shared cash, cross-asset ranking or portfolio optimization.

1. At close on row `t`, enter if `close[t] > SMA(window)[t] * (1 + threshold)`, `t >= trade_start`, and `t % 3 == 0`. Row numbering starts at the beginning of the fixture, including warm-up.
2. The order quantity is `0.25 * flat_cash[t] / close[t]`. It buys at `open[t+1]`; this is a prior-close allocation, not 25% of cash divided by the future opening price.
3. Sell the whole position at `open[t+2]`. Entries are disabled on the last two rows, so completed runs need no terminal forced liquidation.
4. Quantities are continuous. There is no leverage, borrowing charge, slippage, minimum order or exchange lot-size restriction. A cash-reserve check rejects an opening gap that cannot fund the declared entry and fees.
5. Run zero fees and 10 basis points per side as separate conditions. Preserve each framework's native fee behavior.

A fixed liquid universe and a vendor snapshot are useful for controlled execution comparisons. They do not supply survivorship-free membership or complete publication/revision history, and this experiment is not evidence of investment performance.

## What each implementation executes

| Implementation | Indicators | Account execution and evidence |
| --- | --- | --- |
| Foundry CPU | Native `ts_mean` | Native cash executor; fills reconstructed from its NAV and the declared one-bar targets |
| Foundry CUDA | Native CUDA `ts_mean` | Native paired-FP32 cash executor; reconstructed fills are explicitly marked non-independent |
| Spectre CUDA | One native `FactorEngine` batch of unique SMA windows and prior-close prices | Native `trading.run_backtest` / `SimulationBlotter` on CPU; actual transactions, positions, cash and daily account history |
| vectorbt Numba / Rust | Native `MA.run` | Native `Portfolio.from_orders`, independent accounts, native order records and account arrays |
| Freqtrade | Strategy indicators through pandas rolling inside native strategy analysis | Native `Backtesting`, orders and wallets; daily holdings/NAV marked from actual fills, checked against native final balance |

Spectre computes factors once for the condition, then reuses its native saved-factor format across independent accounts. Cache creation, reading and cleanup remain inside the timed call. Its automatic one-bar factor delay is applied exactly once; indicator values are mapped back to decision dates for error comparisons. GPU factor computation does not make its Python account loop a GPU backtester.

Spectre's native `order_target` preserves fractional quantities, but its blotter rounds each commission to two decimal places. We retain this behavior without patching upstream code. Those fee differences change later cash and order quantities, so they are reported separately from floating-point error. Zero-fee cases provide the stricter account-equivalence control.

Freqtrade uses an explicit offline spot-venue metadata adapter: continuous quantities/prices, no minimum size and no exchange network discovery. Its actual order, trade and wallet implementation still executes the strategy. This fixture does not model an exchange's real lot sizes, liquidity or matching engine.

## Timing boundaries

Each framework/device/size/fee condition runs in its own process. One cold call warms imports, JIT and runtime caches; one measured call then starts with fresh account state and newly created condition caches. The experiment does not take the fastest of repeated trials.

The measured interval is contiguous: reload the common case, execute native indicators and accounts, materialize outputs, calculate common metrics and write `result.npz`. Adapter imports, process startup, fixture preparation and the cold warm-up are reported separately or outside that interval. NVML sampler startup/shutdown, installed-distribution enumeration, correctness validation and final metadata-JSON publication are outside the measured interval. Invalid output still fails the worker before success metadata is published.

The runner requests two OpenMP, BLAS, Numba and Rayon threads. GPU work is synchronized before its results are used. Package versions, Python version, device, CPU information, case hashes and timing stages accompany the results. One observation can describe that run; small differences are not a statistical ranking or a guarantee on another device.

NVML samples the worker's GPU residency every 20 ms during the measured round. The reported peak includes the process context and native allocator caches; it is not an allocation delta. Sampling can miss a brief peak, so this is a lower bound on instantaneous peak GPU use. Host peak RSS covers the process lifetime, including imports and warm-up, rather than only the measured call. Neither number measures memory allocated by unrelated processes.

## Correctness and report interpretation

- Compare SMA values on decision rows from `trade_start` through the third-last fixture row. Initial warm-up and unavailable terminal delayed outputs do not affect that comparison.
- Compare entry flags and filled-order identities exactly: candidate, asset, date and side. Report unmatched fills separately from quantity, price and fee errors.
- Compare close NAV normalized by initial cash, positions and final cash. Independently reconcile native fills where available. Foundry's reconstructed fills are marked `native_ledger=False`, so their reconciliation is not an independent order-ledger proof.
- Use the same 365-day annualization, population volatility (`ddof=0`), CAGR and positive maximum-drawdown magnitude. Zero-volatility Sharpe is defined as zero. These are common reporting formulas over each implementation's outputs.
- Zero-fee normalized NAV has a declared maximum absolute error threshold of `1e-8`. This check alone does not excuse decision or fill mismatches. Fee cases report differences without claiming numerical equivalence; show both unrounded and cent-rounded fee residuals.

Absolute and scaled errors are both retained. Scaled error means `abs(actual-reference) / max(1, abs(reference))`. The reporting reference is Foundry CPU, not an external ground-truth simulator. Passing this limited workload does not validate all operators, perpetual funding/liquidation, A-share restrictions, PIT provenance or research admission rules. Cross-framework speedups do not requalify Foundry's separate same-device 1.5× precision budget.

## Reproduce on the authorized cloud runtime

Use Linux or WSL2 with compatible NVIDIA drivers and native build tools. This comparison was prepared with separate Python environments because the pinned Spectre revision requires Python 3.14. The `frameworks` and `spectre` dependency groups in [pyproject.toml](../pyproject.toml) preserve the direct packages verified during setup; run metadata records the complete observed installed distribution versions. These groups are not a full transitive lockfile. Run the following commands from the repository root and keep the two environments separate.

```bash
python -m pip install --upgrade "pip>=25.1" uv
python -m uv python install 3.13.16 3.14.10
python -m uv venv --python 3.13.16 /content/framework-main
python -m pip --python /content/framework-main/bin/python install --group frameworks
python -m uv venv --python 3.14.10 /content/framework-spectre
python -m pip --python /content/framework-spectre/bin/python install --group spectre
source /content/framework-main/bin/activate
bash build.sh
```

The [main environment](../pyproject.toml) pins vectorbt/vectorbt-rust 1.2.1 and Freqtrade 2026.10. The [Spectre environment](../pyproject.toml) pins upstream commit `c059665ba33b3d98f523ee55fdecf5105af146f6`. Its verified Linux Torch installation reports `2.15.0+cu132`; package metadata reports `2.15.0`. The PyPI package named `spectre` is unrelated: use the pinned Git URL. Dependencies are installed without bypassing the upstream requirements.

Start with a small smoke on a fresh output directory:

```bash
python -m foundry benchmark var/runs/framework-smoke --suite frameworks \
  --sizes 1 --assets 1 --spectre-python /content/framework-spectre/bin/python
```

After smoke acceptance, run the declared grid:

```bash
python -m foundry benchmark var/runs/framework-comparison --suite frameworks \
  --sizes 1 16 64 --assets 8 --spectre-python /content/framework-spectre/bin/python
```

Do not pass research parameters such as `market` or `backend`: this suite owns its published common contract and compares the configured implementations. Output directories must be new. Logs and incomplete artifacts remain available if a worker fails; a partial run is not a finished comparison.

The Python entry is `foundry.benchmark(output, suite="frameworks", sizes=(1,16,64), assets=8, spectre_python=...)`. Results include a plan, common cases, per-implementation logs and arrays, version/timing metadata, `comparison.json` and bilingual reports. Keep each result attached to its measured source revision and dependency record; do not blend historical results into a new run.
