# Research and integration

[Overview](../README.md) · [简体中文](research.zh-CN.md) · [Data formats](data.md) · [Strategy programs](programs.md)

A Foundry session evaluates formulas, model outputs and stateful programs against one immutable market study. Candidate evaluation and portfolio publication are separate: the same hard admission rules apply to human research, agents and automated discovery.

## Open a study and review results

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

Without a portfolio path, changes remain in memory. A supplied path enables atomic publication of admitted members. Array artifacts use `Admission.artifact_dir`, defaulting to `var/candidates`; durable programs use the caller-supplied artifact catalog.

`evaluate` accepts a formula string, formula sequence, name-to-formula mapping, `Candidate`, candidate sequence, registered batch, local `StrategyProgram`, or deployed `ProgramReference`. Program inputs require `Execution(model="orders")`. Evaluation success is `row.ok`; inspect `row.decision.passed`, `reason`, `stopped_at` and constraint margins for admission. A business rejection is a report row, while malformed input raises `ValueError`/`TypeError` and resource failures propagate.

| Call | Result and effect |
| --- | --- |
| `evaluate(inputs)` | Metrics, risks and admission decisions; no portfolio change |
| `evaluate_device(inputs)` | In-process resident arrays plus the same query's committable report |
| `register(values, ...)` / `register(program)` | Bounded candidate references and an owned lease |
| `retain(inputs)` | Durable references; does not release the original lease |
| `release(batch_or_report)` | End the named temporary lease |
| `plan_program(program_or_reference)` | Device, state bounds, orders and latency before opening a run |
| `simulate(program_or_reference)` | Standalone fresh-account audit; not an admission report |
| `diagnose()` / `judge_family(...)` | Portfolio gaps or neighbourhood diagnostics; read-only |
| `portfolio()` / `trial_weights(matrix)` | Current members or fresh weight comparisons; read-only |
| `commit(report)` | Ordered admission and atomic portfolio publication |
| `commit_weights(trials, index)` / `reweight(...)` | Submit a reviewed or optimized admissible weight change |

A report binds both the study snapshot and exact portfolio version. If another writer changes the portfolio, submission raises `ContextConflict`. Call `refresh()`, evaluate again and review the new result; refresh does not make an old report current. Failed publication leaves no partial library change. Queries do not retain candidates or publish a portfolio.

`foundry.schema()` returns parameter/RPC schemas without opening data or a device. `session.describe()` lists fields, the 66 operators, actual clocks, execution plan, capacity and resolved settings. Pass market costs and groups such as `Clock`, `Sizing`, `Account`, `Execution`, `CPU`, `CUDA`, `Admission` and `Search` directly to `open`/`serve`; open a new session to change them. Relative configured paths resolve from the repository root.

### Admission trial size

`Admission(weight_steps=N)` sets N positive candidate-mixing grid points `1/N, 2/N, ..., 1`, plus an analytic tangency suggestion; duplicate points are skipped. The default is 20. Each order-model trial replays the actual account, so this setting can materially change query cost. For an exploratory small grid:

```python
with foundry.open(admission=foundry.Admission(weight_steps=4)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

This changes which mixture coefficients are examined, not the hard constraints. Smaller grids can miss a feasible improvement. `min_sharpe_gain` defaults to 0.01 and `max_members` to 40. Inspect the report's actual trial count and constraint margins; `trial_weights` separately evaluates the explicit matrix you provide. These parameters are fixed at session open and recorded in the study identity.

## Execution and portfolio composition

Choose the execution model when opening the session:

| Setting | Meaning |
| --- | --- |
| `Execution(model="targets")` | Default four-market target backtests; formulas and equity-fraction arrays |
| `Execution(model="orders")` | Four-market Bar accounts; cash settlement and isolated/cross USDT margin, with common actual execution for formulas, arrays and programs |
| `Account(composition="shared")` | One actual account; member coefficients scale intents once |
| `Account(composition="independent")` | Order model only; coefficients allocate initial cash to separate accounts once |

US and A-share target backtests use daily artifacts. Equity order accounts also accept `market-bars-v2` regular-session intraday bars with independent decision/statistics clocks. Crypto can use daily data, the supported hourly perpetual archive, or publication-aware base bars. Equity order programs use compiled per-session settlement, corporate-action, quantity and tradability rules. Programs see available cash and sellable inventory; perpetual accounts support isolated or cross margin.

For explicit 5-minute crypto data:

```python
with foundry.open(
    market="crypto_perpetual", data_dir="/datasets/crypto_5m",
    start="2025-01-08", end="2025-01-31", backend="cuda",
    clock=foundry.Clock(decision="15min", statistics="1D"),
    account=foundry.Account(margin="isolated", leverage=3., gross_limit=2.,
                            composition="shared"),
    execution=foundry.Execution(model="orders", participation=0.1),
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

A completed 15-minute decision is eligible at the following 5-minute base open, including a shared boundary. It does not wait for the next 15-minute decision. Formula windows count decision bars. Actual positions, pending orders, funding and liquidation continue between decisions. Statistics use their own period ends; crypto annualization is `365 * 86400 / statistics_seconds`.

In a **shared account**, all members see aggregate cash, equity and net positions, plus their own signed `owner_quantity`. Coefficients scale target values and program order quantities; they do not earmark separate cash. Programs emit quantities for an unscaled coefficient of one. To close actual owner inventory, divide its absolute quantity by `context.intent_coefficients` before returning an order, because replay applies that coefficient once. [Program sizing](programs.md#sizing-and-owner-positions) gives the exact convention and reduce-only constraints.

In **independent accounts**, coefficients must sum to at most one, including tiny positive excesses. Each account starts with `Account.initial_cash * coefficient`; unused capital remains cash. Intents are not scaled a second time: context intent coefficients are one. Account-dependent strategies replay against that funded account. A fixed-quantity strategy must still respect its available capital; coefficients do not automatically turn it into equity-based sizing.

Weight matrices have shape `(trials, members)` in `portfolio().sources` order. Coefficients are finite and nonnegative; all-zero means cash. Under the order model, every candidate, changed weight vector and ordered admission opens fresh strategy state and replays the full actual account. Previously realized holdings are not a valid shortcut for an account-dependent strategy.

The Bar model uses declared `bar_path`, participation, tick/quantity steps, reservations and market rules. Missing quotes can prevent fills. It does not model order-book queues or allow callbacks at arbitrary intrabar fill instants. Feedback-based protection can first be submitted at the next configured decision.

## Arrays and model outputs

`session.grid` supplies ordered assets, canonical UTC `decision_times`, int64 UTC nanoseconds `time_ns`, and a study snapshot. Values are float32/float64 `(B,D,S)` arrays: candidates, decisions, assets. Observation and fit cutoffs are int64 UTC nanoseconds `(B,D)` and cannot exceed the corresponding decision. `NO_INFORMATION` means the producer used no observation or fitting information; it does not mean unknown provenance.

This complete example creates a causal numerical score using public resident features:

```python
import numpy as np
import foundry

with foundry.open(backend="cuda") as session:
    port, grid = session.resident, session.grid
    xp = port.xp
    scores = xp.where(port.eligibility, port.features["ma_dist_20"], xp.nan)[None]
    observed = np.broadcast_to(grid.time_ns, (1, len(grid.time_ns))).copy()
    fitted = np.full(observed.shape, foundry.NO_INFORMATION, dtype=np.int64)
    registered = session.register(
        scores, kind="scores", unit="score", ids=("custom-trend",),
        grid=grid, producer="custom-trend-v1", observed_at=observed, fit_at=fitted,
    )
    try:
        report = session.evaluate(registered)
        saved = session.retain(registered.candidates[:1])
        print(report.rows[0].metrics, saved.candidates)
    finally:
        session.release(registered)
```

`session.resident` is a read-only in-process view; remote clients compute their own NumPy arrays and register against the remote grid. Inputs are copied into an owned resident lease, so mutating the caller's buffers after registration cannot change the candidate.

| Input | Semantics |
| --- | --- |
| `scores`, unit `score` | NaN means missing; `Sizing` defines mapping to targets |
| `Sizing(rule="equal_slots")` | Default clipped score divided by eligible asset count |
| `Sizing(rule="independent", scale=...)` | Per-asset mapping without asset-count dilution; excessive gross becomes a candidate rejection |
| `targets`, unit `equity_fraction` | Signed notional/account equity; direct targets bypass score mapping |
| `targets`, unit `quantity` | Asset units; order model only |
| `targets`, unit `notional` | Quote-currency amount; order model only |
| Target `update_mask=False` | Preserve that owner's prior target; order model only |
| Target zero with `update_mask=True` | Explicitly close that asset target |

Targets are finite, obey market direction rules and must be zero for ineligible assets. Equity fractions are checked against the gross budget at registration. Quantity/notional are converted using actual account prices and capital at execution; an unaffordable or over-budget order can be rejected rather than silently clipped. Score inputs do not take masks. The target model requires omitted/all-true masks. Omitted and all-true masks have the same candidate identity.

`producer` identifies the implementation; optional normalized `complexity` affects soft discovery scoring, not hard admission. Fit cutoffs must include all observations used by training labels, preprocessing and calibration. Foundry checks declarations for future references; it cannot prove the provenance of externally computed predictions.

## Resident results without a second query

Use `evaluate_device` when a model or agent needs actual resident returns as well as an admission report:

```python
with foundry.open(backend="cuda") as session:
    result = session.evaluate_device({"trend": "gt(ma_dist_20,0)"})
    print(result.report.rows[0].metrics)
    print(result.batch.rows[0].returns.shape)
    session.commit(result)
```

The frozen `ResidentEvaluation` contains `.report` and `.batch` in matching candidate order. It comes from one evaluation; do not call `evaluate` again or manually assemble report rows. Inputs can also be an array registration, a local program, or a deployed reference. `commit(result)` accepts the wrapper directly. For a direct program query, `release(result)` ends its report-owned lease; for an explicit registration, release that registration yourself. Consume resident arrays within their owning session and do not mutate borrowed buffers. The method requires an in-process session; remote research returns typed reports instead of local device pointers.

CUDA arrays are ordered on the stream enclosing the evaluation call. Use them on that stream, or explicitly hand them to another stream; the returned Python wrapper is not a universal cross-stream readiness token. For example:

```python
with foundry.open(backend="cuda") as session:
    port, xp = session.resident, session.resident.xp
    with port.scope():
        producer = xp.cuda.Stream(non_blocking=True)
        consumer = xp.cuda.Stream(non_blocking=True)
        with producer:
            result = session.evaluate_device("gt(ma_dist_20,0)")
            ready = xp.cuda.Event()
            ready.record()
        with consumer:
            consumer.wait_event(ready)
            mean_return = result.batch.rows[0].returns.mean()
        consumer.synchronize()
        print(float(mean_return))
    session.commit(result)
```

Internal auxiliary formula streams join before their results are used. If PyTorch consumes the batch, establish the same event/stream dependency for its consumer stream. Finish consumption before releasing scoped resources. Proposal reward feedback is a different object: its `borrow(stream_pointer)` API carries production readiness across threads automatically, as described below.

## Candidate ownership and reuse

All active array and program leases together fit `candidate_batch`. Release a completed batch before registering the next one. Registration does not persist every experiment or spill candidates to disk.

- Explicit `register(...)` returns a caller-owned batch. `evaluate(batch)` and `commit(report)` leave that batch lease active; release it yourself.
- Convenient `evaluate(program_or_reference)` returns a report-owned `report.lease`. `release(report)` frees it; successful `commit(report)` consumes it. A failed/conflicting commit leaves the lease available for reevaluation or explicit release.
- `retain(...)` returns durable references with no temporary scope, while the original lease still needs release. Commit and discovery retain the references they need before publication/checkpointing.
- Released resident references are invalid. Durable arrays need their artifact files; durable programs need the same deployed artifact catalog and deterministic loader on reopen.

Reuse a registered program's immutable model across queries in one scope. Every query still calls `open(context)` for independent private state and closes that run. Release drops the session's model references; it does not destroy another reference held by your application. No private account trajectory is reused as a weight-trial result.

## Stateful programs and order audits

A public strategy has a `ProgramDescriptor` and `open(context)` returning a `ProgramRun`. Its `on_event` consumes owned causal feature windows and the actual account, returns bounded `OrderCommands`, then receives actual `on_feedback` after every completed base bar. Each run closes even on failure. `context.batch_size` is the effective batch, and resident `variant_indices` identify the original candidate variants; never size run state from `descriptor.batch_size` alone.

Programs participate in `evaluate`, discovery, family checks, portfolio trials, reweighting and commit. A deployed reference names integrity-checked artifact bytes plus canonical JSON parameters. Supplying `DeployedPrograms` to `open` or `serve` is an explicit application deployment step; research RPC never ships source, installs dependencies or deserializes Python models. See [the complete program guide](programs.md) for runnable code and a remote deployment.

`simulate` remains a standalone audit with resident returns, equity and final account. It does not produce an admission decision and cannot itself be committed. Use `evaluate` for the portfolio workflow.

## Strategy discovery

```python
with foundry.open() as session:
    candidates = (
        foundry.Candidate("trend", foundry.Formula("gt(ma_dist_20,0)")),
        foundry.Candidate("momentum", foundry.Formula("cs_rank(ret_20)")),
    )
    discovery = foundry.Discovery(session)
    report = discovery.evaluate(candidates)
    for row in report.rows:
        print(row.candidate.id, row.log_reward, row.decision)
    session.commit(report)
```

Use registered array/program `.candidates` in the same place. Discovery uses behavior identity, correlations and portfolio gaps to guide proposals; its soft reward cannot overturn hard admission. Recreate `Discovery` after changing the portfolio. Delayed noise-control reports cannot be committed; evaluate the actual candidates without delay first.

For a model/program parameter family, register variants, then pass `CandidateGrid(seed, variants, parameter_coords, hypothesis)`. Coordinates are distinct integer tuples of equal dimension, and the seed occurs exactly once. `judge_family` returns research diagnostics rather than publishing a portfolio or proving out-of-sample robustness.

`ExternalProposer` accepts finite candidate or registered batches:

```python
with foundry.open(search=foundry.Search(epochs=2, accept=False)) as session:
    candidates = (foundry.Candidate("trend", foundry.Formula("gt(ma_dist_20,0)")),)
    proposer = foundry.ExternalProposer([candidates])
    foundry.mine("var/runs/external", session=session, proposers=(proposer,))
```

A custom proposer implements `open(device)` and a scoped producer with `propose`, `learn`, `release`, `dump` and `load`. Feedback preserves the exact proposal batch order; resident rewards support explicit host transfer or stream-ordered CUDA borrowing. Models and fitting remain owned by that proposer. The driver saves needed candidate references before release and keeps no historical GPU batch cache.

### Feedback and custom producers

The fixed-batch adapter's concrete constructor is `ExternalProposer(batches, *, name="external", feedback=None)`. Each element of `batches` is a `Candidate`, `RegisteredBatch`, or candidate sequence. Its optional callback takes one `LearningFeedback` and returns `TrainingResult`, rather than a dictionary or `None`:

```python
def learn(feedback: foundry.LearningFeedback) -> foundry.TrainingResult:
    rewards = feedback.rewards.to_host()
    print(feedback.batch.sequence, [candidate.id for candidate in feedback.batch.candidates])
    return foundry.TrainingResult(metrics={"mean_reward": float(rewards.mean())})


with foundry.open(search=foundry.Search(epochs=1, accept=False)) as session:
    batch = (foundry.Candidate("trend", foundry.Formula("gt(ma_dist_20,0)")),)
    proposer = foundry.ExternalProposer([batch], name="my-model", feedback=learn)
    foundry.mine("var/runs/my-model", session=session, proposers=(proposer,))
```

`feedback.batch` is the exact proposal handle: `owner`, `sequence`, `context`, `ordinary`, and ordered `.candidates`. `.auxiliary` is total candidates minus ordinary candidates. `.rewards` is an owned FP32 vector with the same total row count; inspect `.shape` and `.device`. `to_host()` explicitly returns an owned NumPy copy, including the required CUDA readiness wait.

For GPU learning, allocate your own model/state on the supplied device and consume feedback without a host transfer. Within an in-process CUDA session, this callback copies a summary into caller-owned storage; replace that assignment with your optimizer work:

```python
xp = session.resident.xp
consumer = xp.cuda.Stream(non_blocking=True)
reward_summary = xp.empty((1,), dtype=xp.float32)


def learn_gpu(feedback: foundry.LearningFeedback) -> foundry.TrainingResult:
    with feedback.rewards.borrow(consumer.ptr) as rewards:
        reward_summary[:] = rewards.mean()
    return foundry.TrainingResult()
```

CUDA `borrow` requires the consumer's stream pointer; it waits for the producing event and records completion when the scope exits. CPU borrowing uses `borrow()` with no stream and yields a read-only view. Do not mutate rewards or retain aliases/DLPack tensors beyond the borrow. The driver owns feedback cleanup, waits for learning and consumption, and then releases the proposal. This small reward lease is independent of mutable portfolio state.

For a dynamically generated/trained proposer, implement the public scoped lifecycle:

| Method | Signature and responsibility |
| --- | --- |
| Specification `open` | `open(device: foundry.ComputeDevice)` returns a context manager yielding one producer; initialize/release model, optimizer and streams here |
| Producer `propose` | `propose(request: foundry.ProposalRequest) -> foundry.ProposalBatch \| None`; None ends this producer's proposals |
| Producer `learn` | `learn(feedback: foundry.LearningFeedback) -> foundry.TrainingResult` |
| Producer `release` | `release(batch: foundry.ProposalBatch) -> None`; end the exact active batch and any registrations it owns |
| Producer `dump` | `dump() -> bytes`; explicit versioned, non-pickle proposer state when no batch is outstanding |
| Producer `load` | `load(payload: bytes) -> None`; validate and restore the matching proposer state |

Specifications/producers have a nonempty `name`, unique among proposers in a mining run. `ProposalRequest` contains `.context`, requested `.count`, and a tuple of diagnostic `.features`. Construct `ProposalBatch(owner, sequence, context, ordinary, candidates)` with a unique owner per opened producer and stable batch sequence; retain that exact handle until release. At most one batch is outstanding per producer, and feedback must match its order/context/identity. An `ExternalProposer` given already registered inputs borrows their references; keep those registrations active for the run and release the caller-owned batches afterward. A custom producer that registers new outputs owns their release. Candidate retention needed by the library/checkpoint occurs before release.

Built-in mining uses the selected GFlowNet or random formula proposer. CUDA GFlowNet inference/training uses PyTorch on device. `Search(accept=False)` evolves the in-memory research portfolio without publishing a persistent library; durable publication requires `accept=True`. Discovery, mining and walk-forward need an in-process session. Remote sessions support candidate research calls, rather than transmitting Python learning callbacks.

Mining checkpoints restore discovery/proposer state and durable candidate references, then freshly replay accounts. A separate explicit [program audit checkpoint](programs.md#standalone-audits-and-resumable-state) saves declared private tensors and actual account/order state at a completed base-bar boundary; `simulate(..., until=..., checkpoint=...)`, `export` and `load_program_checkpoint` provide that in-process workflow.

## Causal walk-forward models

`foundry.walk_forward(session, fit, predict, folds=..., ids=..., kind=..., unit=..., producer=..., features=...)` gives `fit` an owned historical feature interval and `predict` an owned causal prefix through the current decision. Predict returns `(B,S)` arrays; the helper registers the resulting predictions and declares observation/fitting cutoffs.

`Fold(fit_start, fit_stop, predict_start, predict_stop)` uses integer UTC nanoseconds and half-open `[start,stop)` bounds. Fit must finish before prediction; folds are ordered and nonoverlapping. Extend the fit interval to include observations used by forward training labels. Future labels never belong in the feature panel. Later model changes must occur at a statistics-period boundary; use finer statistics if a model changes within a period.

The result contains `.registered` and fold return/sign diagnostics. Evaluate it normally, then release `.registered`. Sign consistency alone is insufficient: consistently losing folds can also agree in sign. Adaptive selection on the same research window remains in-sample research.

## Remote sessions

On the compute host, run an explicit service configuration:

```python
import foundry
foundry.serve("var/runs/library.json", backend="cuda")
```

Use `python -m foundry status` there to inspect the port. The service binds loopback; for another machine forward it with `ssh -N -L LOCAL_PORT:127.0.0.1:REMOTE_PORT user@host`, then connect from a checkout with the same client protocol and the `client` dependency group installed (`python -m pip install --group client`):

```python
import foundry
with foundry.connect(("127.0.0.1", 9000)) as session:
    print(session.describe()["execution_plan"])
    report = session.evaluate("gt(ma_dist_20,0)")
```

Replace `9000` with the local forwarded port. CLI requests can select it with `python -m foundry attach 127.0.0.1 9000`; `detach` clears only that selection. Foundry does not manage SSH credentials, tunnels, deployment or remote process startup. Add the `monitor` group when rendering a dashboard. Native Windows clients need no Linux extensions or local CUDA. Model arrays upload as bounded NumPy binary bundles, with 1 GiB encoded/decoded limits. Program queries use explicitly deployed `ProgramReference` values; [deployment example](programs.md#remote-deployment).

The service serializes work through a bounded queue. Client disconnection does not cancel accepted commands, and closing a client does not stop shared compute. Context conflicts require explicit refresh. Old library/checkpoint/RPC protocols are rejected; rerun deliberately rather than guessing or migrating formats.

## Reading metrics and choosing resources

Statistics labels are period starts; returns and report drawdown are measured at statistics closes. Liquidation checks still run at base bars between those samples. `risk.first_liquidation_ns` is the first affected base-bar start, not a precise intrabar fill time; `-1` means none.

`Metrics.activity_basis` distinguishes target-book turnover/exposure from actual executed activity. Under `executed_fills`, turnover sums per-bar absolute filled notional divided by equity after opening mark/funding/risk and before commands, then divides by research decision count. Independent accounts aggregate notional/equity, including idle cash, before each ratio. `margin_bps` annualizes that reported turnover at the decision frequency. Attendance measures exposure-active statistics periods. Read `risk.gross_sampling`: shared accounts report account-event samples, independently funded portfolios report the exact aggregate maximum at statistics closes. Neither is a tick-level maximum.

`Metrics.fills` is the actual native order fill count for the order model, including partial fills as separate fill events. It is `None` when unavailable under the target model; do not infer a trade count from target turnover.

CUDA batching is most valuable for substantial repeated work. Use `CUDA(candidate_batch=..., trial_batch=..., measurement_cache_mib=...)` or `CPU(workers=..., candidate_batch=...)` to bound resources. The target perpetual path can plan small batches on CPU using `perpetual_host_cells`; the order path stays on its explicitly selected device. CUDA programs must order framework work on `context.stream` and keep declared state resident; unsupported devices fail explicitly.

The numerical plan is fixed before evaluation; no result-triggered wider-precision replay occurs. [Benchmark reports](benchmarks/README.md) record hardware, workload, preparation/JIT, resident work, full workflow, errors and decisions. Use pytest and pytest-benchmark with the public Session API for your own local suite; test and helper code is not distributed with Foundry. Historical formula timings do not establish order-program throughput.
