# Foundry research and interfaces

[Project overview](../README.md) · [简体中文](research.zh-CN.md)

Point-in-time research for formulas, model signals and target exposures. Markets include US equities, A-shares, daily crypto spot and USDT linear isolated perpetuals. A-shares retain price-limit, suspension, sell-tax and next-session execution coverage. Linux CPU and CUDA execute explicit market specifications; cash markets support paired FP32 and perpetual accounts currently require FP64.

## Architecture

The modular monolith follows the ownership contract in AGENTS.md. tests/test_architecture.py checks dependency cycles, forbidden imports and import-time side effects.

| Owner | Responsibility |
| --- | --- |
| contracts, candidates, research_types | Studies, snapshots, backend capabilities, typed sources/results and storage ports |
| core, expr, market, portfolio | Panel/metric semantics, formulas, trading rules and admission |
| ops, backtest, engine | CPU/portable/CUDA computation over explicit arrays and execution specifications |
| research | Shared evaluation, family exploration and explicit versioned commit use cases |
| search, CLI/RPC | Proposals and transport using the same research decisions |
| adapters | Data loading, fingerprints, atomic storage and serialization |
| runtime | Immutable settings, backend preparation and resource ownership |
| monitor | Published status only; no data, engine or GPU ownership |

Configuration is loaded explicitly by load_config(). Imports do not load market data, inspect GPUs, compile kernels or allocate device memory. open_runtime(settings, study) owns these operations and exposes close(). Domain dependencies are passed as arguments, with no global service locator.

Study fixes the research interval. Its snapshot also binds data, execution specifications, numerical policy and implementation identity. Query results carry the portfolio version. Evaluation never writes a portfolio; commit is a separate compare-and-swap command that atomically publishes one replacement. Device-resident batch evaluation and CLI/RPC serialization share typed use cases. The exported JSON Schema is the machine contract; python -m foundry --help is the command list.

Old wire, portfolio and checkpoint protocols are rejected explicitly. There are no compatibility shims or automatic baseline rewrites. Obsolete Quantiacs source, submission notebooks, configuration and platform-specific tests are removed. Historical Quantiacs data, including data/qnt_crypto, source manifests and private production artifacts, remain preserved; preservation does not imply complete PIT provenance.

## Candidate interface

`research-v10` uses `Candidate(id, source)`. The tagged source is `Formula(expression)`, `ArraySource(kind, digest, snapshot, producer)`, or the explicit `EqualWeight()` portfolio seed. Formula miners and family exploration construct `Formula` objects. External model outputs enter through `Research.register_batch(batch, context_id)` and then use the same evaluation, weight trials, admission and commit rules.

Construct `Research(engine, library_store, array_store, null_shift=False)` with explicit stores. `evaluate(candidates, context_id)` returns scalar reports; `evaluate_device(sources, context_id, report=False)` retains device books and daily returns. Queries do not publish artifacts or change the portfolio. `commit(candidates, context_id)` publishes one atomic successor. Registration is a separate command that stores immutable content-addressed artifacts and takes an owned copy, so the producer may reuse its input buffers after it returns. Reloading a committed array member verifies its artifact and declared cutoffs before execution.

`ArrayBatch` in `foundry.candidates` carries `kind`, `ids`, `values`, `snapshot`, `producer`, `dates`, `assets`, `observed_through` and `fit_through`. Values are NumPy/CuPy `float32` or `float64` with shape `(B,T,S)` on the complete ordered study grid. For cash markets, `signals` use the same `clip(0,1)/eligible_count` conversion as formulas; NaN represents missing input. `target_weights` are finite nonnegative portfolio fractions, with zero weight on ineligible assets and total daily exposure at most 1. An excess within `[portfolio] weight_sum_atol` is normalized to 1; a larger excess is rejected. These weights still pass through market execution rules and next-open timing.

Both cutoff arrays are `int64 (B,T)` inclusive row indices, from `-1` through the decision row. `-1` means no observed input or no fitted model. The engine rejects forward cutoffs, wrong axes and a mismatched snapshot. These are producer declarations: they do not prove that external training or feature generation avoided leakage. `producer` must identify the model/program version. `research.describe` publishes the axes, snapshot and these limits; use `context.snapshot.id` in the batch and `context.id` for query/command concurrency.

Registration transfers the payload to host storage once for hashing and publication. The recent input cache holds at most `candidate_batch` candidates; repeated evaluation retains device arrays while cached, and evicted inputs reload from their verified artifacts. CUDA producers must make inputs ready on the runtime's current stream before calling. Registration waits for its owned copy before returning. Noise-control delays for external arrays depend on producer identity and kind, never the content hash of the full future series.

For RPC, export a host batch with `foundry.adapters.arrays.write_bundle(path, batch)`, then call `research.register_batch` with `{context, path}`. The path is local to the service host; loading uses `allow_pickle=False`. Its response contains typed candidates ready for `research.evaluate` or `research.commit`. Formula requests use `{"id":"alpha-1","source":{"kind":"formula","expression":"gt(ma_dist_20,0)"}}`. Export the request and result schemas with `python -m foundry schema OUTPUT`; the same dataclasses define Python results and JSON serialization.

## Crypto execution

`crypto_spot` uses UTC daily bars, a continuous seven-day calendar and 365-day annualization. Historical Quantiacs quotes and eligibility are retained, with missing publication/revision provenance exposed in the data contract. Missing quotes freeze existing positions; a future last price is never used to fill the past.

`crypto_perpetual` uses USDT linear contracts and isolated collateral. Daily close signals rebalance at the following UTC open. Hourly execution processes mark-price gaps, funding on existing positions, scheduled rebalance, adverse mark high/low liquidation touches, then close valuation. Collateral is redistributed only at daily rebalance; idle cash does not top up a losing isolated position. Signed `target_exposure` values are notional/equity ratios with gross exposure capped by the configured leverage. Formula and `signals` inputs are clipped to [-1,1] and divided by eligible asset count; they do not implicitly request maximum leverage.

Configure `[crypto_perpetual]` and use the importer listed by `python -m foundry --help` to obtain checksum-verified Binance trade candles, mark candles and settled funding. The initial fixed BTC/ETH/SOL universe uses 2025 data. Historical exchange risk tiers, VIP fees, order-book fills, ADL and tick ordering are not reconstructed. Constant maintenance/fee assumptions and hourly adverse-touch liquidation define this research model. Funding timestamps are retained, but execution buckets them to their hour and uses hourly mark open because the archive lacks settlement marks. Discovery exposes these limitations; this is not an exact exchange simulation or certified vendor PIT dataset.

Typed simulation results retain daily funding, trading/liquidation fees, isolated loss-cap adjustments, liquidation phase bitmaps, bankruptcy and final equity. Reports aggregate cashflows/counts over the research window and name the account model. Loss-cap adjustments are accounting in this model, not claimed exchange insurance payments. Actual maximum gross leverage may exceed the entry limit after adverse prices.

Perpetual CUDA simulation supports FP64, FP32 (`mixed`) and double-FP32 (`paired`). A warp evaluates the hourly events of each daily collateral interval in parallel, with resident funding prefixes and an affine Newton solve for rebalance fees. Risk-boundary, exhausted-exponent and ill-conditioned-statistics rows receive explicit scalar FP64 refinement on the same device. Low-precision modes retain FP64 reporting/statistics and are experimental; their guard is not a general error proof. Signed admission is explicitly planned as FP64 (`risk_screening_requires_f64`) until a leveraged screening policy is calibrated.

Candidate measurements are cached within the immutable study/precision/noise context, bounded by `candidate_batch`; returned device buffers are borrowed read-only. Each caller receives a separate result row. Admission always uses the current portfolio version. CUDA commit trials batch candidates at the same version, consume results in input order, and recompute the suffix after each acceptance before one atomic publication. CPU uses sequential trials.

## PIT and precision

Common `turnover`, `exposure`, `max_single_weight` and the `margin_bps` denominator describe daily target books, not executed derivative notional. Use the separate risk report for realized fees and actual account leverage. Intrabar liquidation timestamps identify the hourly bar; the exact touch time is unknown.

Sources are truncated before feature derivation. Close-of-session decisions first affect the following open. Cross-sectional operators use the eligible universe at that decision time. Quote history and membership are separate. Late publications and revisions cannot rewrite earlier decision rows, and forward labels never enter the feature panel.

The A-share artifact retains full quotes for 365 historical constituents and applies membership separately. Signals use raw OHLCV and execution uses adjusted prices. Its manifest exposes missing revision vintages and original membership announcement timestamps. US data limitations are also published by discovery. Prefix and future-poisoning tests establish computational causality, not vendor PIT certification or out-of-sample evidence.

Execution specifications contain open eligibility, costs, directional freezes and carrying costs. Kernels do not infer market rules from stock codes. CPU/CUDA retain the same full asset universe and shared metric semantics.

Paired precision uses two FP32 components for recurrence, blending and statistics, with scaled wealth exponents. Conversion, selected reporting metrics and final admission refinement still use FP64. Acceptance compares numerical errors, trades, selected weights and threshold decisions; paired FP32 does not promise bitwise FP64 equivalence.

Historical resident-kernel measurements, 600 days × 299 stocks × 256 books, CUDA-event milliseconds:

| GPU | FP64 execution + statistics | Paired execution + statistics |
| --- | ---: | ---: |
| L4 | 43.356 | 10.904 |
| RTX PRO 6000 Blackwell Server Edition | 17.593 | 7.035 |
| A100 SXM4 40GB | 7.055 | 12.682 |

These exclude packing/allocation, initial transfer and JIT. They are not end-to-end measurements of this refactor. The prior paired-statistics experiment preserved daily returns and measured maximum statistics error of 1.43e-13, including blended trials. Paired arithmetic is not universally faster than native FP64.

## Development and validation

Python owns research, interfaces and orchestration; the existing C++ and CUDA kernels handle numerical work. A single Bash script builds the three Python extensions. Linux is the deployment and backtest platform, including WSL2 and Colab. WSL2 uses Linux Python and produces native `.so` files; Windows Python/MSYS2 cross-builds are removed.

Use Python 3.12 or 3.13, its development headers, and GCC with C++17/OpenMP support. From the repository root, activate the Linux virtual environment used to run Foundry, then install and build:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash build.sh
```

`bash build.sh backtest` rebuilds only that extension; omitting targets builds `ops`, `backtest` and `search`. The script derives Python/NumPy headers and the ABI suffix from the active `python3`, accepts paths with spaces, and uses no machine-specific interpreter or compiler paths. Operator, backtest and policy compilation retain their existing numerical flags. Linux CI builds on Python 3.12/3.13 and checks ASan/UBSan; it does not maintain a separate Windows toolchain.

Install CuPy separately for the runtime's CUDA major; the validated Colab CUDA 13 environment uses `python -m pip install cupy-cuda13x`. CPU execution needs no CuPy or GPU. All tunable values live in `config/config.ini`. Architecture and pure-rule tests do not require private data, portfolios or credentials.

CPU CI covers architecture, expressions, native ABI, search policy, publication semantics and request/storage protocols. Cloud integration tests use explicit US/A-share, crypto-spot and isolated-perpetual studies and isolated temporary portfolio stores. They do not use the private production portfolio as an undisclosed seed. test_production separately checks an explicitly provisioned baseline and never silently records a replacement.

Run CPU smoke before data/GPU integration in the environment authorized for the work:

```bash
PYTHONPATH=tests python -m unittest test_architecture test_expr test_native test_policy test_monitor test_pit test_api test_service test_runtime test_candidates test_crypto_spot test_perpetual test_perpetual_data
PYTHONPATH=tests python -m unittest test_cuda_backtest test_cuda_statistics test_cuda_markets test_cuda_pit test_cuda_candidates test_cuda_perpetual test_markets test_portfolio test_search
```

Performance reports distinguish loading, transfer/JIT, resident batches and complete use cases, and state GPU, full universe, precision and observed error.
