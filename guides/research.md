# Research and integration

[Project overview](../README.md) · [简体中文](research.zh-CN.md)

Foundry evaluates formulas, model signals and target allocations through one session. The same market rules and portfolio admission decisions apply to manual research, AI agents, formula miners and external model searches.

## Public Python API

```python
import foundry

with foundry.open(
    "var/runs/my_portfolio.json",
    market="us_equity", backend="cuda",
    start="2025-07-01", end="2026-06-30",
    cuda=foundry.CUDA(candidate_batch=64),
    admission=foundry.Admission(min_sharpe_gain=0.02),
) as session:
    report = session.evaluate({"trend": "gt(ma_dist_20,0)"})
    for row in report.rows:
        print(row.id, row.metrics, row.decision)
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

`evaluate` accepts a formula string, a sequence of formulas, an ID-to-formula mapping, a `Candidate`, a candidate sequence or a registered batch. Reports carry metrics, account risks and structured admission decisions. A rejected candidate is a result, not a failure of the whole batch.

| Operation | Purpose | Portfolio effect |
| --- | --- | --- |
| `evaluate(inputs)` | Evaluate candidates and admission proposals | Read-only |
| `diagnose()` | Describe portfolio gaps and discovery context | Read-only |
| `judge_family(inputs, hypothesis="")` | Test neighbourhoods or explicit parameter grids | Read-only |
| `portfolio()` | Get sources, weights, metrics and baseline status | Read-only |
| `trial_weights(matrix)` | Compare component-weight trials | Read-only |
| `commit(report)` | Submit candidates in report order | Atomic publication if admitted |
| `commit_weights(trials, index)` | Submit one reviewed weight trial | Atomic publication if admitted |
| `reweight(iterations=3, count=32, seed=1)` | Search for and submit admissible weight improvements | Can publish a new portfolio |

Weight trials have shape `(trials, members)` in `portfolio()` order. Weights must be finite and nonnegative; all-zero weights mean cash. These are weights of portfolio members, not individual asset targets.

Omit the portfolio path to keep the library in memory. Registered arrays stay on the compute device; admitted, checkpoint-referenced or explicitly retained outputs use `Admission.artifact_dir`, whose default is `var/candidates`. A portfolio path enables durable, atomic publication. `with` releases candidate leases and local compute resources; closing a remote client leaves its shared service running.

Reports bind their study snapshot and portfolio version. Another writer changing that context raises `ContextConflict`; call `session.refresh()` and evaluate again explicitly. There is no silent retry of a stale decision.

Admission reports expose each constraint's raw `delta` and `required` threshold; a null threshold means the requirement is unattainable. Search normalizes these margins independently. Tuning the four `Search.violation_scale_*` parameters changes discovery rewards without changing the study snapshot, registered-array identity or hard admission rules. Rejected diagnostic trial weights use portfolio-owned scales. Correcting this snapshot definition changes old snapshot IDs once; explicitly register old snapshot-bound arrays again instead of reusing their old handles.

### Parameters and discovery

Pass `market`, `backend`, `data_dir`, `start`, `end` and market costs directly. Advanced groups are `foundry.CPU`, `foundry.CUDA`, `Admission`, `Operators`, `Search`, `Single` and `Monitor`. All settings are fixed when a session opens; open another session to compare them.

`foundry.schema()` returns the parameter schema, market defaults and RPC contract without opening a study. `session.describe()` supplies available fields, the 66 operators and resolved parameters. `foundry.mine(...)` and `foundry.serve(...)` accept the same parameter groups. A supplied mining session already owns its parameters.

The CLI is `python -m foundry`. For example:

```bash
python -m foundry --parameters '{"market":"a_share","backend":"cuda"}' fields
python -m foundry schema var/runs/schema.json
```

No INI file or environment override is needed. Parameter paths resolve relative to the repository root unless absolute.

## Bring your own data

The bundled snapshots are replaceable fixtures. Convert CSV, database or API exports into the files below, then select their directory with `foundry.open(data_dir=..., market=..., start=..., end=...)`.

### Daily files

Each `panel.parquet` uses long-form rows with `date`, `symbol`, `open`, `high`, `low`, `close` and `volume`. Use midnight, timezone-naive datetimes to identify exchange sessions, or UTC calendar dates for spot. Store symbols as strings; preserve A-share leading zeros.

| Market | Additional panel columns | Other required files |
| --- | --- | --- |
| `us_equity` | `adj_factor` | `membership.parquet` |
| `a_share` | `published`, `exec_open`, `exec_close` | `membership.parquet`, `manifest.json` |
| `crypto_spot` | Boolean `eligible` | `manifest.json` |

- **US equities:** OHLC is multiplied by `adj_factor`; volume is unchanged. Supply raw prices with factors, or adjusted prices with factors of 1. Record adjustment methods and missing historical vintages.
- **A-shares:** raw OHLCV produces signals; `exec_open` and `exec_close` are historically adjusted execution prices. For each bar the reader freezes the latest version published by that bar's close. `published` is a session date, not an intraday publication clock; later revisions do not replace an earlier decision's bar.
- **Equity eligibility:** `membership.parquet` contains `symbol`, `start`, `end`: inclusive start, exclusive end, null for an open-ended interval. These intervals are assumed known at the relevant closes. Keep quotes outside membership so existing positions can still be valued.
- **Spot eligibility:** supplied per daily row; missing eligibility is false.
- **History:** retain data before the research start for rolling features. Apart from explicit A-share publication versions, use one row per date/symbol. Leave unavailable quotes missing; do not infer historical membership from today's constituents.

A CSV conversion example:

```python
from pathlib import Path
import pandas as pd

folder = Path("/datasets/my_us_equity")
folder.mkdir(parents=True)
panel = pd.read_csv("ohlcv.csv", parse_dates=["date"], dtype={"symbol": str})
members = pd.read_csv("membership.csv", parse_dates=["start", "end"],
                      dtype={"symbol": str})
panel.to_parquet(folder / "panel.parquet", index=False)
members.to_parquet(folder / "membership.parquet", index=False)
```

Then open that directory, choosing dates inside its coverage:

```python
with foundry.open(market="us_equity", data_dir="/datasets/my_us_equity",
                  start="2025-07-01", end="2025-12-31") as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

### Provenance and additional features

A-share and spot manifests contain the `DataContract` fields `schema`, `source`, `retrieved_at`, `features`, `execution_prices`, `publication` and `membership`. Bundled manifests illustrate the format; record your actual source and availability limits. Missing announcement timestamps or revision vintages must remain disclosed.

The current US reader reports a fixed legacy provenance contract. Adding a manifest alone does not replace it; keep your source record alongside the dataset. A custom US provenance contract requires extending the reader. Snapshot identity still binds the loaded numerical data and execution specification.

Extra Parquet columns do not automatically become formula fields. Register model outputs through the public API, or extend the reading/field boundary in [`market/data.py`](../foundry/market/data.py) and the market compiler. Backtest kernels need no provider-specific changes. Historical Quantiacs data remains usable; its retired platform integration is not required.

### Perpetual data

The hourly contract consists of `trade.parquet`, `mark.parquet`, `funding.parquet` and `manifest.json`. The manifest records ordered symbols, artifact hashes and funding-interval transition evidence where required. Daily OHLC alone cannot reconstruct funding or liquidation.

```python
foundry.import_perpetual(
    "/datasets/my_perpetual",
    symbols="BTCUSDT ETHUSDT SOLUSDT",
    history_start="2025-01-01",
    start="2025-07-01", end="2025-12-31",
)
```

The importer acquires public USD-M archives, verifies checksums and retains source records. Open the output with `market="crypto_perpetual"` and matching dates. Existing datasets get asset order from their manifests; `symbols` selects new acquisition. A different provider must satisfy the explicit event contract in [`market/perpetual.py`](../foundry/market/perpetual.py).

## External models and target positions

`session.grid` gives ordered dates, assets and snapshot identity. Arrays use shape `(B,T,S)`: candidates, dates, assets. This example registers three zero-signal candidates to demonstrate the interface; replace them with your actual outputs and information cutoffs.

```python
import numpy as np
import foundry

with foundry.open() as session:
    grid = session.grid
    days, assets = len(grid.dates), len(grid.assets)
    values = np.zeros((3, days, assets), dtype=np.float32)
    unused = np.full((3, days), -1, dtype=np.int64)
    registered = session.register(
        values, kind="signals", ids=("model-small", "model-base", "model-large"),
        grid=grid, producer="external-program-v1",
        observed_through=unused, fit_through=unused,
        complexity=(0.2, 0.5, 0.8),
    )
    report = session.evaluate(registered)
```

| Kind | Meaning | Constraints |
| --- | --- | --- |
| `signals` | Scores converted to positions by Foundry | Floating values; NaN may mark missing signals |
| `target_weights` | Equity or spot long-only asset weights | Finite, nonnegative, total exposure at most 1, zero outside eligibility |
| `target_exposure` | Perpetual signed notional/equity targets | Finite, gross exposure within configured leverage, zero outside eligibility |

Values must be floating arrays. Observation and fitting cutoffs are integer arrays `(B,T)`, using study row indices no later than decision row `t`; `-1` means no observations or fitting were used. Do not use `-1` for a trained model merely to satisfy validation. Daily close decisions execute no earlier than the next open.

`producer` identifies your implementation. Optional per-candidate `complexity` lies in `[0,1]` and influences discovery's soft score only; it cannot change hard admission. Registration owns a bounded resident copy and returns content-bound candidates plus a lease. Local CuPy payloads stay on the GPU; only compact content digests return to the host. Remote registration uploads NumPy arrays without server paths.

Call `session.release(registered)` when finished, normally in `finally` in a repeated loop. Session exit releases its remaining leases. Across active batches, the total row count cannot exceed `candidate_batch`; release a batch before registering more. Released resident references cannot be evaluated, even if an internal measurement remains cached.

To keep selected outputs, use `saved = session.retain(registered.candidates[:1])` before release and save `saved.candidates` for future use. Only this explicit request, accepted portfolio members, or discovery checkpoint references write artifacts to `Admission.artifact_dir`. `retain` returns durable references; it does not release the original batch. Commit and discovery handle their own required retention automatically. Portfolio/RPC, array and discovery formats are `research-v14`, `candidate-array-v3` and `candidate-discovery-v3`; old formats require an explicit rerun, with no automatic migration.

## Strategy discovery

### Score candidates for an agent or custom loop

```python
with foundry.open() as session:
    candidates = (
        foundry.Candidate("trend", foundry.Formula("gt(ma_dist_20,0)")),
        foundry.Candidate("momentum", foundry.Formula("cs_rank(ret_20)")),
    )
    print(session.diagnose())
    discovery = foundry.Discovery(session)
    report = discovery.evaluate(candidates)
    for row in report.rows:
        print(row.candidate.id, row.log_reward, row.decision)
    decision = session.commit(report)
```

A registered batch's `.candidates` works in the same place. Discovery combines portfolio gaps, behavior fingerprints, correlation and novelty with its objective. Rewards guide which candidates to propose next; the portfolio's hard admission decision remains authoritative. Recreate `Discovery` after a portfolio change so its reference context is current.

`diagnose()` is read-only, but its cost depends on the objective. With `Search(objective="single")`, it evaluates the shared deterministic calibration population and returns the calibrated gate; `Single.calibration_sample` defaults to 8192. The portfolio objective does not perform that calibration.

Noise-control discovery reports carry `delayed=True`. They describe delayed candidate behavior and cannot be passed to `session.commit`. Evaluate the actual candidates without delay through `session.evaluate(...)`, then review and submit that new report.

### Plug in an external proposer

```python
with foundry.open(search=foundry.Search(epochs=2, accept=False)) as session:
    batch = (
        foundry.Candidate("trend", foundry.Formula("gt(ma_dist_20,0)")),
        foundry.Candidate("momentum", foundry.Formula("cs_rank(ret_20)")),
    )
    proposer = foundry.ExternalProposer([batch])
    status = foundry.mine("var/runs/external-search", session=session,
                          proposers=(proposer,))
```

`ExternalProposer` accepts finite candidate or registered batches. `foundry.mine(..., session=session, proposers=[proposer])` opens, restores and closes producer resources automatically while leaving the supplied session open. Its optional callback receives one `LearningFeedback`: `feedback.batch` preserves candidate ordering and the research context; `feedback.rewards` is a same-device FP32 vector. CPU programs explicitly call `to_host()`; CUDA programs use `borrow(stream_pointer)` on their own stream and import the borrowed array through zero-copy DLPack. Do not mutate feedback or retain aliases beyond the borrow scope. Feedback owns no market/portfolio buffers, so learning can overlap commit and reweight.

Custom proposers implement an `open(device)` context manager yielding a producer with `propose(request)`, `learn(feedback)`, `release(batch)`, `dump()` and `load()`. `ProposalRequest` contains only count, research context and diagnostic features; model-specific learning settings stay in each constructor. Each producer permits one unreleased batch, and feedback must reference its exact original handle, preserving instance, order and context. The driver joins learning before releasing feedback and trajectories, including on failure. Ordinary and auxiliary evaluations use actual counts; random proposers no longer append implicit GFN probes. Checkpoints use `candidate-discovery-v3` and reject older formats. Additional feedback storage is `O(B)`, with no history/replay cache.


Without a supplied proposer, `foundry.mine(output, ...)` uses the configured built-in GFN or random formula proposer. The default `Search(accept=False)` still evolves an in-memory research portfolio through ordered admission and scheduled reweighting; it prevents persistent library publication. To publish, use `Search(accept=True, seed_portfolio="current")`. A supplied persistent session requires that publication setting; it is rejected with `accept=False`. `Discovery`, mining and walk-forward need an in-process session; their device-resident work is not a remote callback protocol.

### Non-GFN GPU models and non-formula factors

This executable validation client can be adapted to your own models or tensor programs. It uses neither GFN nor formula conversion.

```python
from validation.benchmarks.tensor_programs import TensorProposer

with foundry.open(
    backend="cuda",
    search=foundry.Search(epochs=4, batch_size=64),
    cuda=foundry.CUDA(candidate_batch=64, measurement_cache_mib=32),
) as session:
    proposer = TensorProposer(session, mode="mlp")
    foundry.mine("var/runs/mlp-example", session=session, proposers=[proposer])
```

`mode="mlp"` fits a batch of 8→16→1 FP32 networks on historical prefixes every 64 days. The 20-day return labels end strictly before each prediction segment. Predictors emit `target_weights`; a separate GPU reward surrogate learns from resident feedback to guide the next hyperparameter batch. Research feedback never updates a predictor that is then backfilled into past decisions.

`mode="factor"` uses GPU tensors, feature interactions and cross-sectional sorting to emit `signals`. GPU CEM updates its parameter distribution from the same reward vector. The example also evaluates a nine-point neighborhood and delayed twins through `CandidateGrid`. Both clients share registration, archives, hard admission and ordered commit; model parameters, causal fitting and learning rules remain client-owned.

Array registration copies GPU outputs into a bounded resident lease and hashes chunks on the GPU. Only compact digests return to the host; accepted, checkpoint-referenced and explicitly retained candidates transfer their payloads for persistence. Outputs and fitting cutoffs are declared through `session.register`. Causal prediction fitting does not establish unbiased out-of-sample model selection: hyperparameter/factor search uses research-window feedback, and reported returns are in-sample research results.

`candidate_batch` limits active resident input rows and persistence staging. Producers release their registered batch in `release(batch)`, after learning completes and discovery retains final checkpoint references. Unselected payloads are released without writing files. Explicit retention reuses registration-computed content identities, borrows contiguous input slices, and keeps host PIT cutoffs on the host. It uses bounded staging and atomic files; artifact reads still verify integrity. It returns durable references with the same content digest and an empty scope; resident scopes cannot appear in a library or checkpoint. A checkpoint publication retains its own references, without a historical GPU cache or implicit archival of every trial. See the [same-L4 retention comparison](benchmarks/retention-l4-20261116/README.md) for registration, required persistence and complete timings.

Unified measurement entry point:

```bash
python -m foundry --parameters '{"search":{"epochs":4,"batch_size":64},"cuda":{"candidate_batch":64,"measurement_cache_mib":32}}' \
  benchmark var/runs/nonformula-example --suite nonformula --backends cuda
```

See the [L4 integration report](benchmarks/nonformula-l4-20261116/README.md) for timings, numerical checks and admission results. Its short data window produces empty delayed twins, so the family result is interface evidence, not a robustness claim.

### Test robustness without a formula restriction

For formulas, `session.judge_family({"trend": "gt(ma_dist_20,0)"}, hypothesis="trend")` derives parameter and field neighbours from the operator registry.

For an external parameter sweep, describe the candidate grid explicitly. Within the same open session, use a registered batch containing three model variants:

```python
variants = registered.candidates
family = foundry.CandidateGrid(
    seed=variants[1], variants=variants,
    parameter_coords=((0,), (1,), (2,)),
    hypothesis="Performance persists across adjacent model sizes",
)
result = session.judge_family(family)
```

The seed must appear exactly once in the variants. Coordinates are unique integer ladder positions with the same dimension; use a complete neighbourhood to make plateau and twin diagnostics informative. Family results describe robustness, not portfolio publication. An observation of a stable family is not a guarantee of out-of-sample returns.

## Causal walk-forward models

`foundry.walk_forward` runs your fitting and prediction callbacks on owned feature prefixes. Fit never receives rows after its fold's fitting cutoff. Prediction receives only the prefix ending at that decision row, together with the corresponding eligibility mask. This example uses the bundled US study and fits a per-asset reference level:

```python
with foundry.open(start="2025-07-01", end="2026-06-30") as session:
    xp = session.resident.xp
    dates = session.grid.dates
    first = next(i for i, date in enumerate(dates) if date >= "2025-07-01")
    middle = (first + len(dates)) // 2
    folds = (
        foundry.Fold(0, first - 1, first, middle),
        foundry.Fold(0, middle - 1, middle, len(dates)),
    )

    def fit(view):
        return xp.nanmean(view.features["ma_dist_20"], axis=0)

    def predict(model, view):
        score = view.features["ma_dist_20"][-1] - model
        return xp.where(view.eligibility[-1], score, xp.nan)[None, :]

    result = foundry.walk_forward(
        session, fit, predict, folds=folds, ids=("fold-model",),
        kind="signals", producer="prefix-model-v1", features=("ma_dist_20",),
    )
    report = session.evaluate(result.registered)
    print(result.sign_consistency)
```

`Fold(fit_start, fit_through, predict_start, predict_stop)` uses absolute study row indices. Training includes `fit_through`; prediction excludes `predict_stop`. Fit must precede prediction; prediction folds are ordered and nonoverlapping and each must contain an executable decision. Keep `fit_start` fixed for expanding windows or advance it for rolling windows. Retain enough pre-start history for training.

Each prediction returns `(B,S)` in study asset order, using the backend array library. `FoldView` contains `start`, exclusive `stop`, `dates`, `assets`, `features` and `eligibility`. Outputs undergo the same registration checks as external arrays; invalid exposures are rejected rather than silently masked. The executor records observed row `t` and the fold's fit cutoff. It cannot police a callback that separately reads future data or uses a model pre-trained on it.

Diagnostics attribute a fold's net returns from its first executable decision until the next model first executes, or study end for the last fold. Exit costs, funding and blocked-position carry during gaps remain attributed to the preceding model. `sign_consistency` is the larger of the positive-fold and negative-fold counts divided by all folds; zero/invalid folds count in neither numerator. A consistently losing model can score 1, so inspect the fold signs and returns as well. The operation registers and evaluates predictions but does not commit them. Release `result.registered` after use, or explicitly retain selected candidates first.

## Remote sessions

Run `python -m foundry serve` on the Linux/WSL2 compute host; `python -m foundry status` reports its address. Connect native Windows or Linux Python to the service address, or to a forwarded local port:

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    diagnostics = session.diagnose()
    family = session.judge_family("gt(ma_dist_20,0)")
```

Formula evaluation, registration, family judging, diagnosis, weight trials and portfolio commands use the same public session methods. Fit external models in your own process and register their outputs for remote evaluation. `session.resident` and `evaluate_device` are in-process facilities.

[`rpc/schema.py`](../foundry/rpc/schema.py) derives requests and responses from the typed contracts; [`rpc/codec.py`](../foundry/rpc/codec.py) handles encoding, with [`rpc/server.py`](../foundry/rpc/server.py) and [`rpc/client.py`](../foundry/rpc/client.py) owning transport. Array uploads use a binary bundle with 1 GiB encoded and decoded limits and no pickle. Portfolio/RPC protocol `research-v14` rejects old formats explicitly. Commands run through one bounded worker queue; disconnecting a client does not cancel an accepted command.

Input-contract errors raise `ValueError` or `TypeError`; stale contexts raise `ContextConflict`; resource/implementation failures propagate. Candidate business rejection remains a structured report. Consult `foundry.schema()` rather than constructing undocumented wire messages.

## Crypto execution

Spot uses UTC daily bars and 365-day annualization. Eligibility and quotes remain separate; missing trade prices can block execution and leave an existing position carried. Configured fees apply to executed changes.

Perpetuals model USDT linear contracts with isolated collateral. Daily targets are signed notional/account equity; gross exposure is bounded by configured leverage. Signals do not automatically allocate maximum leverage. Hourly events process mark gaps, existing-position funding, a daily rebalance, adverse high/low liquidation checks and closing marks. Funding follows actual event timestamps mapped to the hourly grid. If the public funding archive lacks settlement mark prices, the importer explicitly records its mark-open proxy.

The current model uses fixed maintenance/fee parameters. It does not reproduce historical tier schedules, VIP fees, ADL, an order book or tick-by-tick intrabar fills. `risk` reports funding, trading fees, liquidation fees, liquidation counts, bankruptcy and account equity with units documented by the schema.

## Numerical behavior and performance

`foundry.CUDA(measurement_cache_mib=32)` retains only candidate returns and compact reports. Candidate books are regenerated in batches bounded by `candidate_batch`; accepted members own individual copies rather than retaining their original batch allocations. With budget `C`, batch size `b`, member bound `M`, days `T` and assets `S`, persistent caches occupy `O(C + (b + M)TS)`, excluding fixed input panels and caller-owned outputs. Temporary expression and weight-trial workspaces also depend on the bounded expression cost and `trial_batch`, not the total candidate count. The cache has no candidate-count × days × assets term. Set its budget to `0` to disable reuse. This performance setting does not change study identity. Commit still checks the expected context and judges each candidate against the evolving portfolio.

CUDA batch sizes and precision are explicit. Paired FP32 uses two components to retain residual precision in supported account/statistical paths. The resolved plan is fixed at session startup; evaluation never chooses a higher precision after seeing a result. Small perpetual batches can be planned for CPU before execution using `CUDA.perpetual_host_cells`; there is no runtime-error fallback.

Use `session.describe()` and report precision/backend fields to inspect the actual plan. The [2026-11-15 PRO 6000 report](benchmarks/arch-v2-pro6000-20261115/report.md) measures source `fa2acfd`: 512 public tests passed, one private-baseline test was skipped, and all 124 cases covering 66 operators passed. The 4096-formula workflow admitted 28 candidates on both backends, with no decision mismatches or numerical tolerance exceedances; complete-case CPU/CUDA wall times were 93.724/2.879 s. These are single-round observations. Earlier `current/` reports remain historical.

The separate private production audit failed: its preserved baseline lacks protocol/study metadata, and the saved portfolio contains legacy BTC-feature formulas. The files were not rewritten or re-recorded; this is not a complete private-baseline pass. CUDA account arithmetic used paired FP32, but no same-device single-FP32 control was run, so the separate 1.5× precision budget was not requalified. See the [validation scope](benchmarks/arch-v2-pro6000-20261115/README.md).

For an authorized compute environment, `python -m foundry benchmark OUTPUT` is the unified validation/benchmark entry point. It separates preparation from resident evaluation and complete query/ordered-commit timing, and reports numerical errors and decision consistency. The source is in [`validation/benchmarks`](../validation/benchmarks); architecture checks are in [`validation/test_architecture.py`](../validation/test_architecture.py).

Use `--backends cpu` or `foundry.benchmark(output, backends=("cpu",))` to measure an independent CPU baseline without initializing CUDA. This selects benchmark contracts and measurements, not the full multi-device regression suite. Set OpenMP/BLAS threads to one and set `cpu.workers` to the allocated CPU capacity and retain that configuration with the result; GPU speedups must name the fixed CPU reference they use.

### CPU parallelism

```python
with foundry.open(backend="cpu", cpu=foundry.CPU(workers=8, candidate_batch=32,
                                               measurement_cache_mib=32)) as session:
    report = session.evaluate(formulas)
    committed = session.commit(report)
```

Independent candidates and weight trials share read-only market panels across workers. Native operators release the GIL; each worker uses one OpenMP lane. At most `workers` tasks are outstanding, results retain input order, and the portfolio has one ordered writer. CPU settings affect execution resources, not snapshot identity or trading semantics.

Expression intermediates expire after each candidate. Weight trials materialize books only for active workers: temporary book space is `O(workers * T * S)`, rather than `O(total_trials * T * S)`. Required return/event outputs occupy their own output-sized storage. Use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for benchmarking and control candidate concurrency through `cpu.workers` to avoid nested thread pools.

### Measure the existing GFlowNet discovery loop

```bash
python -m foundry --parameters '{"search":{"epochs":8,"batch_size":512}}' \
  benchmark var/runs/discovery-new --suite discovery --backends cuda
```

This directly calls the existing `foundry.mine()` workflow: sampling, CUDA candidate evaluation, rewards, GFN learning, ordered admission, CEM reweighting and checkpoints. Scale is controlled by `Search`. The report separates stage sums from wall time because learning overlaps commit/reweight.

GFN model definitions, Trajectory Balance, paired teachers and clipped SGD live in Python/PyTorch. With `backend="cuda"`, prefix states, legal-action masks, random draws, network inference and FP32 training use a dedicated CUDA stream. The host receives probe-chunk outcomes and final formula tokens. Training reuses the current batch's sampled features and masks; the model owns one SGD optimizer for its lifetime. CPU/portable sessions explicitly select the PyTorch CPU device. `GfnProposer(...)` stores configuration only; the session selects the device, and `open(device)` creates the model, stream and optimizer without fallback. Random proposals need no neural model.

The 32-unit tanh architecture, initial field/operator mass, rewards, probe gates and admission criteria are preserved. Device-local PyTorch sampling and FP32 learning change the seeded trajectory from the old native learner and host-sampling implementation, so identical seeds do not imply identical formulas. Checkpoints use `foundry-gfn-torch-v3` with JSON and numeric arrays, without pickle; older checkpoints are rejected. The checkpoint includes device RNG state and requires the same device kind for exact continuation. SGD has no momentum state. Only the current batch's prefixes and legal masks are retained (`O(B * L * (F + A))`), with no replay/history cache. Probe workspaces are bounded to four prefixes and forty common-suffix attempts each; only accepted probes survive the chunk. This suite does not run an independent CPU baseline; in-window adaptive admission is not out-of-sample performance.

See the [L4 batch ownership and resident feedback report](benchmarks/api-boundary-l4-20261116/README.md) for identical-sequence numerical/decision parity, complete timing and resource evidence.
