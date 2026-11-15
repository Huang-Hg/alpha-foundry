# Foundry research and interfaces

[Project overview](../README.md) · [简体中文](research.zh-CN.md)

Point-in-time research for formulas, model signals and target exposures. Markets include US equities, A-shares, daily crypto spot and USDT linear isolated perpetuals. A-shares retain price-limit, suspension, sell-tax and next-session execution coverage. Linux CPU and CUDA execute explicit market specifications; cash accounts and CUDA statistics use paired FP32. Numerical policies are fixed before evaluation; no automatic precision replay is performed.

## Bring your own data

The bundled datasets are replaceable research fixtures. Pass `market`, `data_dir`, `start` and `end` to `foundry.open()` to use your own daily tables. CSV, database and API exports can be converted to the formats below without a vendor account or configuration file.

### Daily files and columns

All `panel.parquet` tables use long-form rows with `date`, `symbol`, `open`, `high`, `low`, `close` and `volume`. Store `date` as a timezone-naive, midnight datetime representing the exchange session (UTC calendar date for spot), `symbol` as a string and numerical columns as numbers. Preserve leading zeros in A-share identifiers.

| Market | Additional panel columns | Other required files |
| --- | --- | --- |
| `us_equity` | `adj_factor` | `membership.parquet` |
| `a_share` | `published`, `exec_open`, `exec_close` | `membership.parquet`, `manifest.json` |
| `crypto_spot` | Boolean `eligible` | `manifest.json` |

- **US equities:** the engine multiplies OHLC by `adj_factor`. Supply raw OHLC with the corresponding factors, or already-adjusted OHLC with factors of 1; do not adjust twice. Record the adjustment method and any missing historical vintages.
- **A-shares:** OHLCV supplies raw signal prices; `exec_open` and `exec_close` supply cumulative historically adjusted execution prices. `published` is a session date: the reader freezes each bar using the latest version published by that bar's close. A later revision does not replace the old decision's bar. This is a daily availability convention, not an intraday publication clock.
- **Eligibility:** equity `membership.parquet` has `symbol`, `start`, `end`, with inclusive start, exclusive end and null end for an open interval. The reader assumes these intervals were known at the relevant closes; it does not reconstruct missing announcement history. Keep quotes outside membership intervals so existing positions can still be valued. Spot eligibility is stored per daily row; missing eligibility becomes false.
- **History and gaps:** retain history before the configured research start for rolling windows. Use one observation per date/symbol, apart from the explicit A-share publication versions. Leave missing quotes missing; never fill earlier rows from a future quote or infer historical constituents from today's list.

### Example: your US equity CSV files

Prepare `ohlcv.csv` with the US columns above and `membership.csv` with the eligibility columns. This conversion creates a new dataset directory:

```python
from pathlib import Path
import pandas as pd

directory = Path("data/my_us_equity")
directory.mkdir(parents=True)
panel = pd.read_csv("ohlcv.csv", parse_dates=["date"], dtype={"symbol": str})
members = pd.read_csv("membership.csv", parse_dates=["start", "end"],
                      dtype={"symbol": str})
panel.to_parquet(directory / "panel.parquet", index=False)
members.to_parquet(directory / "membership.parquet", index=False)
```

Open foundry.open(market="us_equity", data_dir="./data/my_us_equity", start=..., end=...) within your data coverage. session.describe() lists the available formula fields. Your dataset uses the same public API as the bundled snapshots.

### Source metadata and custom features

A-share and spot manifests must contain the seven `DataContract` fields: `schema`, `source`, `retrieved_at`, `features`, `execution_prices`, `publication`, `membership`. Use the bundled manifests as format examples, but record your own source, price conventions and actual availability limits. Do not copy a PIT claim or fabricate a publication timestamp.

The current US file reader reports a fixed legacy data contract; adding a custom `manifest.json` alone does not change that reported provenance. Keep your source record with the data. Exposing a custom US provenance contract requires updating the data adapter; numeric snapshot identity still derives from the loaded panel and execution specification.

Additional Parquet columns are not automatically exposed as formula fields. Feed externally computed model outputs through `ArrayBatch`, or explicitly extend the data adapter and market field definitions. The integration boundaries are [`foundry/market/data.py`](../foundry/market/data.py), [`MarketData`](../foundry/types.py) and the market's `compile` method; changing a data source does not require rewriting the backtest kernels.

### Perpetual data

The current reader uses a Binance-style hourly dataset contract: `trade.parquet`, `mark.parquet`, `funding.parquet` and `manifest.json`, with artifact hashes, ordered symbols and funding-interval transition evidence where required. The bundled three-contract and 64-contract snapshots show this layout. Daily OHLC alone is insufficient for funding and liquidation simulation.

Use `foundry.import_perpetual(output, symbols="BTCUSDT ETHUSDT SOLUSDT", history_start="2025-01-01", start="2025-07-01", end="2025-12-31")` for explicit acquisition parameters. The following CLI uses the default symbols and dates:

```bash
python -m foundry import-perpetual data/crypto_perpetual_new
```

The importer preserves source records and verifies archive checksums. Research the result with `foundry.open(market="crypto_perpetual", data_dir="./data/crypto_perpetual_new", start=..., end=...)`. Existing snapshots take their asset order from the manifest; `symbols` controls new imports. Custom sources require the verified event contract or an explicit adapter and execution specification. See [market/perpetual.py](../foundry/market/perpetual.py) and [crypto execution](#crypto-execution).

## Public Python API

Start with `import foundry`. `foundry.open(...)` owns a local compute session; `foundry.connect((host, port))` connects to an existing service. Both offer the same reports, model registration, neighbourhood exploration and portfolio operations. Native Windows Python can act as a remote client; local numerical computation uses WSL2. Linux/Colab runs directly. Importing the package loads no data and initializes no GPU.

```python
import foundry

with foundry.open(
    "production/my_portfolio.json",
    market="us_equity", backend="cuda",
    start="2025-07-01", end="2026-06-30",
    cuda=foundry.CUDA(candidate_batch=64),
    admission=foundry.Admission(min_sharpe_gain=0.02),
) as session:
    report = session.evaluate({"trend": "gt(ma_dist_20,0)"})
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

`evaluate` accepts a formula string, sequence of formulas or `{candidate_id: formula}` mapping. Each `report.rows` entry contains `metrics`, `risk`, `decision` and an ID. `commit(report)` rechecks admission against the reviewed portfolio version; evaluation does not change the portfolio. Callers do not construct stores, snapshots or version tokens.

Pass parameters directly; there is no INI or implicit configuration file. Common arguments are `market`, `backend`, `data_dir`, `start` and `end`. Market-specific fees and leverage use `cost_rate`, `commission`, `stamp_tax_sell`, `leverage`, `maintenance_rate` and related fields. Advanced groups are `foundry.CUDA`, `Admission`, `Operators`, `Search`, `Single` and `Monitor`, with one set of defaults in `foundry/config.py`. `foundry.schema()` returns the parameter schema and market defaults; `session.describe()["parameters"]` reports resolved values.

Parameter objects are immutable; open another session to change settings. `foundry.mine(output, **parameters)` and `foundry.serve(portfolio, **parameters)` use the same parameters. The CLI accepts JSON with the same schema, for example `python -m foundry --parameters '{"market":"a_share","backend":"cuda"}' fields`, without creating a configuration file.

### External models and target positions

`session.grid` supplies ordered dates, assets and their data identity. This zero-signal example uses no observations or fitted model; replace the values and cutoffs with your actual producer's outputs:

```python
import numpy as np
import foundry

with foundry.open() as session:
    grid = session.grid
    days, assets = len(grid.dates), len(grid.assets)
    values = np.zeros((1, days, assets), dtype=np.float32)
    unused = np.full((1, days), -1, dtype=np.int64)
    candidates = session.register(
        values, kind="signals", ids=("model-v1",), grid=grid,
        producer="external-program-v1",
        observed_through=unused, fit_through=unused,
    )
    report = session.evaluate(candidates)
```

Choose `signals`, `target_weights` or `target_exposure`; units and market constraints are in the [candidate contract](#candidate-interface). Both cutoffs are integer row indices shaped `(candidates, dates)`. They must honestly describe consumed information and cannot exceed the decision row; `-1` means no observations or fit were used. Remote sessions upload NumPy arrays automatically without server paths. In-process sessions also accept CuPy arrays.

### Portfolio operations and lifecycle

- `session.judge_family(formulas, hypothesis="trend")` explores parameter and field neighbourhoods.
- `session.portfolio()` returns members, weights and metrics; `baseline` identifies a new study's equal-weight comparison baseline.
- `session.trial_weights(matrix)` evaluates finite nonnegative `(trials, members)` component weights in `portfolio()` order. These are not per-asset target positions.
- `session.commit_weights(trials, index)` submits a selected trial through the shared admission rules. All-zero weights mean cash, not a restored equal-weight position.
- `foundry.export(report, path)` writes JSON and creates parent directories.

Omit a portfolio path for in-memory research. With a path, the portfolio is published atomically and external arrays use `Admission.artifact_dir`. Stale reports fail without automatic retries; call `session.refresh()` after another writer changes the portfolio, then evaluate again. Leaving `with` releases local resources; closing a remote client does not stop its shared service. In-process miners can use `session.evaluate_device(inputs)` to retain device arrays; remote calls use structured reports.

## Architecture

The modular monolith separates the responsibilities below. [Architecture checks](../validation/test_architecture.py) reject dependency cycles, forbidden imports and import-time side effects.

| Owner | Responsibility |
| --- | --- |
| types, config | Studies, snapshots, backend capabilities, typed sources/results and storage ports |
| numeric, ops | Shared statistics, operator contracts, formula parsing and CPU/CUDA implementations |
| market | Data reading, PIT panels and immutable trading-rule compilation |
| backtest | Cash and hourly isolated-perpetual accounts over explicit execution arrays |
| portfolio | Tensor-only allocation, admission and portfolio weight solvers |
| study | Resource ownership, bounded caches, atomic storage and session evaluation/commit use cases |
| search, CLI/RPC | Proposals and transport using the same research decisions |
| monitor | Published status only; no data, engine or GPU ownership |

Public entry points resolve explicit parameters into immutable settings once. Imports do not read data, probe GPUs or allocate resources. Sessions own and release computation resources; domain dependencies are passed explicitly.

Study fixes the research interval. Its snapshot also binds data, execution specifications, numerical parameters and explicit computation semantics versions. Query results carry the portfolio version. Evaluation never writes a portfolio; commit is a separate compare-and-swap command that atomically publishes one replacement. Device-resident batch evaluation and CLI/RPC serialization share typed use cases. The exported JSON Schema is the machine contract; `python -m foundry --help` is the command list.

Old wire, portfolio and checkpoint protocols are rejected explicitly. There are no compatibility shims or automatic baseline rewrites. Obsolete Quantiacs source, submission notebooks, configuration and platform-specific tests are removed. Historical Quantiacs data, including data/qnt_crypto, source manifests and private production artifacts, remain preserved; preservation does not imply complete PIT provenance.

## Candidate interface

`research-v13` uses `Candidate(id, source)`. The tagged source is `Formula(expression)`, `ArraySource(kind, digest, snapshot, producer, complexity)`, or the explicit `EqualWeight()` portfolio seed. Formula miners and family exploration construct `Formula` objects. External model outputs enter through `session.register(batch)` and then use the same evaluation, weight trials, admission and commit rules.

Public sessions assemble runtimes and stores. `evaluate` returns structured reports, while in-process `evaluate_device` retains device positions and returns. Queries do not change the portfolio; `commit` publishes an atomic successor. Registration takes an owned copy, and persisted array members are verified when a portfolio reopens.

`ArrayBatch` in `foundry.types` carries `kind`, `ids`, `values`, `snapshot`, `producer`, `dates`, `assets`, `observed_through` and `fit_through`. Values are NumPy/CuPy `float32` or `float64` with shape `(B,T,S)` on the complete ordered study grid. For cash markets, `signals` use the same `clip(0,1)/eligible_count` conversion as formulas; NaN represents missing input. `target_weights` are finite nonnegative portfolio fractions, with zero weight on ineligible assets and total daily exposure at most 1. An excess within `Admission.weight_sum_atol` is normalized to 1; a larger excess is rejected. These weights still pass through market execution rules and next-open timing.

Both cutoff arrays are `int64 (B,T)` inclusive row indices, from `-1` through the decision row. `-1` means no observed input or no fitted model. The engine rejects forward cutoffs, wrong axes and a mismatched snapshot. These are producer declarations: they do not prove that external training or feature generation avoided leakage. `producer` must identify the model/program version. `session.describe()` publishes the axes, snapshot and these limits; use `context.snapshot.id` in the batch and `context.id` for query/command concurrency.

Registration transfers the payload to host storage once for hashing and publication. The recent input cache holds at most `candidate_batch` candidates; repeated evaluation retains device arrays while cached, and evicted inputs reload from their verified artifacts. CUDA producers must make inputs ready on the runtime's current stream before calling. Registration waits for its owned copy before returning. Noise-control delays for external arrays depend on producer identity and kind, never the content hash of the full future series.

Remote sessions upload NumPy bundles automatically. The `research.register_batch` request carries `{context, bytes}`, followed by exactly that many binary bytes after its JSON newline, up to 1 GiB per upload. Loading uses `allow_pickle=False`; no server path is accepted. Responses share the typed candidate/report contract. `python -m foundry schema OUTPUT` exports request, result and parameter schemas.

## Crypto execution

`crypto_spot` uses UTC daily bars, a continuous seven-day calendar and 365-day annualization. Historical Quantiacs quotes and eligibility are retained, with missing publication/revision provenance exposed in the data contract. Missing quotes freeze existing positions; a future last price is never used to fill the past.

`crypto_perpetual` uses USDT linear contracts and isolated collateral. Daily close signals rebalance at the following UTC open. Hourly execution processes mark-price gaps, funding on existing positions, scheduled rebalance, adverse mark high/low liquidation touches, then close valuation. Collateral is redistributed only at daily rebalance; idle cash does not top up a losing isolated position. Signed `target_exposure` values are notional/equity ratios with gross exposure capped by the configured leverage. Formula and `signals` inputs are clipped to [-1,1] and divided by eligible asset count; they do not implicitly request maximum leverage.

Configure `market="crypto_perpetual"` and use the importer listed by `python -m foundry --help` to obtain checksum-verified Binance trade candles, mark candles and settled funding. The initial fixed BTC/ETH/SOL universe uses 2025 data. Historical exchange risk tiers, VIP fees, order-book fills, ADL and tick ordering are not reconstructed. Constant maintenance/fee assumptions and hourly adverse-touch liquidation define this research model. Funding timestamps are retained, but execution buckets them to their hour and uses hourly mark open because the archive lacks settlement marks. Discovery exposes these limitations; this is not an exact exchange simulation or certified vendor PIT dataset.

Typed simulation results retain daily funding, trading/liquidation fees, isolated loss-cap adjustments, liquidation phase bitmaps, bankruptcy and final equity. Reports aggregate cashflows/counts over the research window and name the account model. Loss-cap adjustments are accounting in this model, not claimed exchange insurance payments. Actual maximum gross leverage may exceed the entry limit after adverse prices.

Perpetual CUDA assigns one warp to each account and lanes to assets. Hourly funding, liquidation stages and daily collateral changes remain chronological. Rebalance reductions run across assets. The account uses native arithmetic because the paired account experiment changed decisions immediately beside liquidation thresholds; its production use must satisfy the measured query-plus-commit cost limit described below. CUDA statistics and admission moments use paired FP32.

Small perpetual accounts are planned for CPU execution before launch using `CUDA.perpetual_host_cells` and the complete `B*T*S` workload. The result returns to the runtime's device arrays. This is a declared size-based dispatch, with no runtime-error fallback or result-dependent rerun. Discovery publishes the threshold. Formula operators retain chronological rolling std/slope/correlation and population-standardization semantics; unstable compositions can still amplify tiny backend math differences, so sampled parity is not a guarantee for arbitrary formulas.

The 64-contract experiment found no end-to-end benefit from the BF16 Tensor Core funding-prefix prototype. It is outside production. The replaced prefix kernels and scalar precision-refinement branch are deleted. Legitimate changes in funding intervals are accepted only with schedule evidence; missing funding events remain errors.

Candidate measurements are cached within the immutable study/precision/noise context, bounded by `candidate_batch`; returned device buffers are borrowed read-only. Each caller receives a separate result row. Admission always uses the current portfolio version. CUDA commit trials batch candidates at the same version and consume results in input order. Requests fitting one device batch retain full parallelism. For larger requests, the lookahead grows after a rejected batch and contracts to the observed acceptance distance after a successful candidate; this avoids computing a large suffix against an obsolete portfolio. The successor is still published once, atomically. CPU uses sequential trials.

## PIT and precision

Common `turnover`, `exposure`, `max_single_weight` and the `margin_bps` denominator describe daily target books, not executed derivative notional. Use the separate risk report for realized fees and actual account leverage. Intrabar liquidation timestamps identify the hourly bar; the exact touch time is unknown.

Sources are truncated before feature derivation. Close-of-session decisions first affect the following open. Cross-sectional operators use the eligible universe at that decision time. Quote history and membership are separate. Late publications and revisions cannot rewrite earlier decision rows, and forward labels never enter the feature panel.

The A-share artifact retains full quotes for 365 historical constituents and applies membership separately. Signals use raw OHLCV and execution uses adjusted prices. Its manifest exposes missing revision vintages and original membership announcement timestamps. US data limitations are also published by discovery. Prefix and future-poisoning tests establish computational causality, not vendor PIT certification or out-of-sample evidence.

Execution specifications contain open eligibility, costs, directional freezes and carrying costs. Kernels do not infer market rules from stock codes. CPU/CUDA retain the same full asset universe and shared metric semantics.

The precision acceptance budget is **query + ordered commit ≤ 1.5× the single-FP32 comparison**, on the same device, input candidates, order and market rules. Data loading, registration and JIT are reported separately and cannot dilute this ratio. A kernel-only speedup does not establish compliance. Native arithmetic is an explicit measured exception, not a second execution of a low-precision result. Qualification applies to the measured workload; the earlier 2× results below do not establish this tighter budget.

The numerical policy is fixed for the study and shared by query, weight trials and commit. `mixed` and `paired` select cash-account arithmetic; perpetual accounts and CPU use the explicit `native` policy. The API exposes the effective choice and no longer accepts a per-query precision override. Reports do not contain refinement counters. Wide arrays at interchange boundaries do not imply that all arithmetic is wide, nor that the whole pipeline is FP32.

CUDA reductions, weighted moments and log growth use two FP32 components. Integer exponents preserve the wealth range. Leading-term/correction normalization uses Quick-Two-Sum; arbitrary leading terms still use Two-Sum, and FMA retains product residuals. The error-transform basis is [Hida, Li and Bailey](https://www.davidhbailey.com/dhbpapers/arith15.pdf). Centered weighted variance avoids spurious variance for constant returns; zero variance has Sharpe zero on both backends. Feasible utility ties use the configured metric tolerance, then CAGR and the smallest tied weight. These are shared semantics, not backend-specific admission thresholds. Accuracy checks cover returns, cashflows, liquidation events, chosen weights and both sides of decision thresholds. Historical bitwise baselines are not silently rerecorded when the protocol or numerical semantics change.

The operator contract also makes finite constant-window covariance exactly zero and its correlation undefined. Tiny tanh values use the correctly rounded identity on CPU, CUDA and portable backends. A candidate relying on cancellation noise from a constant covariance now has no valid signal; this is an intentional bug fix, not a widened acceptance tolerance.

Two-observation windows use their closed forms: nonconstant Pearson correlation is exactly ±1, skew is zero, excess kurtosis is −2, and an identifiable two-point regression has zero residual and reproduces its last observation. Constant inputs and missing values retain their defined invalidity. This removes cancellation artifacts across all three backends. CUDA `ts_av_diff` now includes available nonmissing observations during warmup, and `ts_scale` and regression use the CPU's exact zero-denominator rule. These are explicit numerical corrections; historical artifacts are not rerecorded.

MAD uses a directly centered window mean, WMA sums the current window oldest first, and cross-sectional reductions share one logical 256-lane tree across all three backends. Departed observations cannot leave a spurious MAD/WMA position, and identical cross-sectional inputs use the same rounding order. The fixed reduction order is a numerical contract, independent of host thread count.

The [66-operator audit and 4,096-formula workflow](benchmarks/README.md) includes individual timing/error results, precision trials and the scope of the current 1.5× qualification.

## Development and validation

Python owns research, interfaces and orchestration; the existing C++ and CUDA kernels handle numerical work. A single Bash script builds the three Python extensions. Linux is the deployment and backtest platform, including WSL2 and Colab. WSL2 uses Linux Python and produces native `.so` files; Windows Python/MSYS2 cross-builds are removed.

Use Python 3.12 or 3.13, its development headers, and GCC with C++17/OpenMP support. From the repository root, activate the Linux virtual environment used to run Foundry, then install and build:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash build.sh
```

`bash build.sh backtest` rebuilds only that extension; omitting targets builds `ops`, `backtest` and `search`. The script derives Python/NumPy headers and the ABI suffix from the active `python3`, accepts paths with spaces, and uses no machine-specific interpreter or compiler paths. Operator, backtest and policy compilation retain their existing numerical flags. Native builds use Linux Python, including WSL2; Windows remote clients do not build these extensions.

Install CuPy for the runtime CUDA major; CPU execution needs no CuPy or GPU. Pass tunable values through the public API and inspect foundry.schema() for defaults. Architecture and pure-rule checks need no private data, portfolios or credentials.

The pytest validation suite covers architecture, expressions, native ABI, search policy, publication semantics and request/storage protocols. Cloud integration tests use explicit US/A-share, crypto-spot and isolated-perpetual studies and isolated temporary portfolio stores. They do not use the private production portfolio as an undisclosed seed. test_production separately checks an explicitly provisioned baseline and never silently records a replacement.

Use the unified entry point in the authorized cloud environment. GitHub Actions displays and archives these saved reports; it does not run the numerical suite:

```bash
python -m pip install -r requirements-benchmark.txt
python -m foundry benchmark var/runs/benchmark-new
```

Performance reports distinguish loading, transfer/JIT, resident batches and complete use cases, and state GPU, full universe, precision and observed error.

Historical precision experiments are archived [separately](benchmarks/archive/precision-history.md). Use the [current benchmark suite](benchmarks/README.md) for public-API timings and correctness reports.
