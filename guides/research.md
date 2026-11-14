# Foundry research and interfaces

[Project overview](../README.md) · [简体中文](research.zh-CN.md)

Point-in-time research for formulas, model signals and target exposures. Markets include US equities, A-shares, daily crypto spot and USDT linear isolated perpetuals. A-shares retain price-limit, suspension, sell-tax and next-session execution coverage. Linux CPU and CUDA execute explicit market specifications; cash accounts and CUDA statistics use paired FP32. Numerical policies are fixed before evaluation; no automatic precision replay is performed.

## Architecture

The modular monolith separates the responsibilities below. [Architecture checks](../tests/test_architecture.py) reject dependency cycles, forbidden imports and import-time side effects.

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

`research-v11` uses `Candidate(id, source)`. The tagged source is `Formula(expression)`, `ArraySource(kind, digest, snapshot, producer)`, or the explicit `EqualWeight()` portfolio seed. Formula miners and family exploration construct `Formula` objects. External model outputs enter through `Research.register_batch(batch, context_id)` and then use the same evaluation, weight trials, admission and commit rules.

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

Perpetual CUDA assigns one warp to each account and lanes to assets. Hourly funding, liquidation stages and daily collateral changes remain chronological. Rebalance reductions run across assets. The account uses native arithmetic because the paired account experiment changed decisions immediately beside liquidation thresholds; its production use must satisfy the measured query-plus-commit cost limit described below. CUDA statistics and admission moments use paired FP32.

Small perpetual accounts are planned for CPU execution before launch using `[cuda] perpetual_host_cells` and the complete `B*T*S` workload. The result returns to the runtime's device arrays. This is a declared size-based dispatch, with no runtime-error fallback or result-dependent rerun. Discovery publishes the threshold. Formula operators retain chronological rolling std/slope/correlation and population-standardization semantics; unstable compositions can still amplify tiny backend math differences, so sampled parity is not a guarantee for arbitrary formulas.

The 64-contract experiment found no end-to-end benefit from the BF16 Tensor Core funding-prefix prototype. It is outside production. The replaced prefix kernels and scalar precision-refinement branch are deleted. Legitimate changes in funding intervals are accepted only with schedule evidence; missing funding events remain errors.

Candidate measurements are cached within the immutable study/precision/noise context, bounded by `candidate_batch`; returned device buffers are borrowed read-only. Each caller receives a separate result row. Admission always uses the current portfolio version. CUDA commit trials batch candidates at the same version and consume results in input order. Requests fitting one device batch retain full parallelism. For larger requests, the lookahead grows after a rejected batch and contracts to the observed acceptance distance after a successful candidate; this avoids computing a large suffix against an obsolete portfolio. The successor is still published once, atomically. CPU uses sequential trials.

## PIT and precision

Common `turnover`, `exposure`, `max_single_weight` and the `margin_bps` denominator describe daily target books, not executed derivative notional. Use the separate risk report for realized fees and actual account leverage. Intrabar liquidation timestamps identify the hourly bar; the exact touch time is unknown.

Sources are truncated before feature derivation. Close-of-session decisions first affect the following open. Cross-sectional operators use the eligible universe at that decision time. Quote history and membership are separate. Late publications and revisions cannot rewrite earlier decision rows, and forward labels never enter the feature panel.

The A-share artifact retains full quotes for 365 historical constituents and applies membership separately. Signals use raw OHLCV and execution uses adjusted prices. Its manifest exposes missing revision vintages and original membership announcement timestamps. US data limitations are also published by discovery. Prefix and future-poisoning tests establish computational causality, not vendor PIT certification or out-of-sample evidence.

Execution specifications contain open eligibility, costs, directional freezes and carrying costs. Kernels do not infer market rules from stock codes. CPU/CUDA retain the same full asset universe and shared metric semantics.

The precision acceptance budget is **query + ordered commit ≤ 2× the single-FP32 comparison**, on the same device, input candidates, order and market rules. Data loading, registration and JIT are reported separately and cannot dilute this ratio. A kernel-only speedup does not establish compliance. Native arithmetic is an explicit measured exception, not a second execution of a low-precision result.

The numerical policy is fixed for the study and shared by query, weight trials and commit. `mixed` and `paired` select cash-account arithmetic; perpetual accounts and CPU use the explicit `native` policy. The API exposes the effective choice and no longer accepts a per-query precision override. Reports do not contain refinement counters. Wide arrays at interchange boundaries do not imply that all arithmetic is wide, nor that the whole pipeline is FP32.

CUDA reductions, weighted moments and log growth use two FP32 components. Integer exponents preserve the wealth range. Leading-term/correction normalization uses Quick-Two-Sum; arbitrary leading terms still use Two-Sum, and FMA retains product residuals. The error-transform basis is [Hida, Li and Bailey](https://www.davidhbailey.com/dhbpapers/arith15.pdf). Centered weighted variance avoids spurious variance for constant returns; zero variance has Sharpe zero on both backends. Feasible utility ties use the configured metric tolerance, then CAGR and the smallest tied weight. These are shared semantics, not backend-specific admission thresholds. Accuracy checks cover returns, cashflows, liquidation events, chosen weights and both sides of decision thresholds. Historical bitwise baselines are not silently rerecorded when the protocol or numerical semantics change.

The operator contract also makes finite constant-window covariance exactly zero and its correlation undefined. Tiny tanh values use the correctly rounded identity on CPU, CUDA and portable backends. A candidate relying on cancellation noise from a constant covariance now has no valid signal; this is an intentional bug fix, not a widened acceptance tolerance.

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
PYTHONPATH=tests python -m unittest test_cuda_backtest test_cuda_statistics test_cuda_moments test_cuda_operator_contracts test_cuda_markets test_cuda_pit test_cuda_candidates test_cuda_perpetual test_markets test_portfolio test_search
```

Performance reports distinguish loading, transfer/JIT, resident batches and complete use cases, and state GPU, full universe, precision and observed error.

### Numerical validation: PRO 6000

The fixed-policy implementation passed 182 tests on Colab RTX PRO 6000 Blackwell Server Edition (97,887 MiB, driver 580.82.07; Python 3.13.16, NumPy 2.4.6, CuPy 14.0.1). Builds, smoke, CPU references, GPU tests and endpoint timing all ran there. The legacy private `production/regression.npz` lacks the current protocol/study identity: `test_production` was excluded and that artifact was not rewritten. This is not a claim of bitwise reproduction of the legacy portfolio.

The comparison runs one warmed query followed by an ordered commit from the same initial portfolio, with identical candidates, order and market rules. Formula caches are cleared after warm-up. Loading, registration and JIT are outside the ratio. Each condition is measured once; the results qualify these workloads, not every device or input size.

The **single-FP32 probe** changes GPU formula arithmetic and account arithmetic to single components. Both sides retain the same paired statistics, wide interchange arrays and preplanned small-perpetual CPU dispatch. It is an arithmetic-cost comparison, not an all-FP32 production implementation. The production formula operators and perpetual account recurrence retain native double arithmetic; cash accounts and CUDA statistics use paired FP32. Queries and commits never trigger precision reruns.

| Workload | Single probe, seconds | Production, seconds | Ratio | Accepted |
| --- | ---: | ---: | ---: | ---: |
| US: 500 assets × 269 days, 64 formulas | 0.05081 | 0.08930 | 1.758× | 6 |
| A-share: 365 assets × 601 days, 64 formulas | 0.10562 | 0.20479 | 1.939× | 8 |
| Crypto spot: 73 assets × 4,018 days, 64 formulas | 0.22727 | 0.42082 | 1.852× | 5 |
| Perpetual: 64 contracts, 64 formulas | 0.45509 | 0.83460 | 1.834× | 7 |
| Perpetual: 64 contracts, 64 target books | 0.91915 | 1.56882 | 1.707× | 17 |
| Perpetual: 64 contracts, 4,096 target books | 15.25573 | 26.91782 | 1.764× | 39 |

All times in the table are **query + commit**. Perpetual inputs retain 365 days and all 8,760 hourly events, with a 184-day evaluation window. The large target-book case spends 10.52891 seconds on query and 16.38891 on commit; prepare and registration add 0.84255 and 1.24893 seconds separately. Device memory at completion is 11,666 MiB; pool reservation is 10,958 MiB. These are end-state readings, not peak measurements.

Against the independent CPU reference, all six conditions have identical admission/rejection decisions, reason codes, constraint booleans, liquidation counts/first timestamps and bankruptcy flags. Maximum absolute differences across query and commit reports are: Sharpe `7.32e-11`, CAGR `2.18e-12`, drawdown `3.48e-13`, and selected weight `5.03e-12`. The 4,096-book funding difference is below `1.00e-15` and trading-fee difference below `2.03e-15`, in initial-equity units. Other report quantities have different scales: the largest normalized scalar error is `1.71e-9`, using `abs(error)/max(1, abs(reference))`; this is not a universal absolute-error bound. Direct CUDA tests additionally check event-stage bitmaps and adjacent liquidation thresholds. Array correctness reuses the unchanged CPU reference collected on the preceding authorized PRO 6000 worker; no cross-worker CPU speedup is claimed.

The single probe changes some selected weights and constraint checks; in spot it even exchanges accepted candidate identities while leaving the accepted count at five. It therefore serves only as a speed denominator. Accuracy is judged against the independent CPU implementation, never from equal accepted counts alone.

Paired arithmetic uses Quick-Two-Sum only for ordered normalization and FMA product residuals. Division forms its residual directly with FMA and preserves round-to-nearest division; an exact unit denominator returns the original pair. This removed redundant pair construction and reduced the previously over-budget cash workloads without lowering precision or changing admission tolerances. Tensor-core funding-prefix experiments are not enabled in production: they did not establish an end-to-end improvement for the chronological account model.
