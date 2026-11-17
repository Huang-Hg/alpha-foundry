# Strategy programs and deployed models

[Overview](../README.md) · [Research guide](research.md) · [Data formats](data.md) · [简体中文](programs.zh-CN.md)

Use a program when the next decision depends on actual cash, inventory, pending orders or previous fills. Formulas and arrays remain convenient for precomputed decisions. All three enter the same order-account evaluation, discovery and portfolio replay when the session opens with `Execution(model="orders")`.

The current order model supports crypto spot and isolated USDT linear perpetuals on an explicitly selected CPU or CUDA backend. Equity settlement and corporate-action order support is not available. The model uses completed bars, a declared OHLC path and volume participation, rather than an order book or tick-level queue model.

## A complete public program

The following code uses only public `foundry` interfaces and the Python standard library. Save it as `my_program.py`. It attempts one market buy of asset index zero once that asset is eligible. The quantity is unscaled: portfolio replay applies the member coefficient. A rejected first attempt is not retried by this intentionally simple strategy.

```python
from dataclasses import dataclass
import json
import foundry


@dataclass(frozen=True)
class BuyOnce:
    descriptor: foundry.ProgramDescriptor
    asset: int
    quantity: float

    def open(self, context: foundry.ProgramContext) -> foundry.CheckpointableRun:
        return BuyOnceRun(context, self.asset, self.quantity)


class BuyOnceRun:
    def __init__(self, context, asset, quantity):
        self.context, self.asset, self.quantity = context, asset, quantity
        self.sent = context.xp.zeros((context.batch_size, 1), dtype=context.xp.bool_)

    def on_event(self, event: foundry.ProgramEvent) -> foundry.OrderCommands:
        context, xp = self.context, self.context.xp
        shape = (context.batch_size, 1)
        integer = lambda value: xp.full(shape, value, dtype=xp.int64)
        number = lambda value: xp.full(shape, value, dtype=xp.float64)
        new = (~self.sent) & event.eligibility[-1, self.asset]
        self.sent |= new
        quote = xp.broadcast_to(event.features["close"][-1, self.asset], shape).copy()
        return foundry.OrderCommands(
            action=xp.where(new, int(foundry.OrderAction.NEW), 0).astype(xp.int64),
            owner=integer(context.owner), id=integer(1), asset=integer(self.asset),
            side=integer(1), quantity=number(self.quantity),
            type=integer(int(foundry.OrderType.MARKET)), price=number(0),
            reservation_price=quote, valid_until=integer(-1),
            reduce_only=xp.zeros(shape, dtype=xp.bool_),
        )

    def on_feedback(self, feedback: foundry.ExecutionFeedback) -> None:
        pass  # Consume or stream actual fills here; orders require a decision.

    def snapshot(self) -> tuple[foundry.StateTensor, ...]:
        return (foundry.StateTensor("sent", self.sent.copy()),)

    def restore(self, state: tuple[foundry.StateTensor, ...]) -> None:
        self.sent[:] = state[0].values

    def close(self) -> None:
        del self.sent


def build_buy_once(payload: bytes, parameters: str, digest: str) -> BuyOnce:
    model, settings = json.loads(payload), json.loads(parameters)
    descriptor = foundry.ProgramDescriptor(
        model["id"], artifact_digest=digest, parameters=parameters,
        state_schema=(foundry.TensorState("sent", "bool", (1,)),),
        features=("close",), history=1, batch_size=1, devices=("cpu", "cuda"),
        checkpointable=True,
    )
    return BuyOnce(descriptor, model["asset"], float(settings["quantity"]))


artifact = foundry.ProgramArtifact(
    payload=b'{"asset":0,"id":"buy-once"}',
    semantic_version="buy-once-v2",
    build=build_buy_once,
)
catalog = foundry.DeployedPrograms((artifact,))
reference = foundry.ProgramReference(artifact.digest, '{"quantity":0.001}')
```

The artifact digest binds the actual immutable payload bytes and the explicit implementation semantic version. The trusted, preinstalled factory interprets those bytes; it does not execute supplied Python text or deserialize a Python object. Change the semantic version when changing behavior implemented in the factory/run code. Parameters are a finite JSON object with sorted keys, compact separators and unescaped Unicode, at most 64 KiB UTF-8. `json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` constructs that representation. `parameter_digest` is derived, not a caller-invented identity.

Keep the first block as the importable module's deployment definitions. Run the following in a separate script, using a dataset prepared according to [the data guide](data.md) and dates inside its coverage:

```python
import foundry
from my_program import catalog, reference

with foundry.open(
    "var/runs/program-portfolio.json", programs=catalog,
    market="crypto_spot", data_dir="/datasets/crypto_5m",
    start="2025-01-08", end="2025-01-31", backend="cuda",
    clock=foundry.Clock(decision="5min", statistics="1D"),
    account=foundry.Account(initial_cash=1000., composition="shared"),
    execution=foundry.Execution(model="orders", participation=0.1,
                                order_capacity=32, command_capacity=32),
) as session:
    print(session.plan_program(reference))
    program = catalog.resolve(reference)
    report = session.evaluate(program)
    for row in report.rows:
        print(row.id, row.metrics, row.decision)
    session.commit(report)
```

`evaluate(program)` creates a bounded report-owned lease; successful `commit(report)` consumes it. To inspect without committing, call `session.release(report)`. A failed/conflicting commit leaves the report lease active. The surrounding session closes remaining leases on exit.

For repeated queries, use an explicit scope:

```python
registered = session.register(reference)
try:
    report = session.evaluate(registered)
    saved = session.retain(registered)
    session.commit(report)
finally:
    session.release(registered)
```

This snippet belongs inside the preceding session block. Registration reuses an immutable model instance while each query opens fresh private run state. Explicit registration remains caller-owned after commit. Retain returns durable references, not serialized Python models or the private state of a previous run. Reopening a library containing programs requires the same artifact catalog and loader capabilities.

For a local model that consumes returns, replace `session.evaluate(program)` with `result = session.evaluate_device(program)`. Read resident returns from `result.batch.rows[i].returns`, metrics/admission from `result.report.rows[i]`, and actual fill count from `.metrics.fills`. Use `session.commit(result)` or `session.release(result)` directly: one query produces both views, and the wrapper preserves the report lease. An explicit registration remains caller-owned as above. See [resident results](research.md#resident-results-without-a-second-query).

A local program instance can be registered/evaluated without a catalog. Retain or an accepted commit requires reconstructing it from a matching deployed artifact and canonical parameters; descriptor names alone are insufficient.

## The causal run contract

`ProgramDescriptor` declares the immutable artifact/parameters, private `state_schema`, requested feature names, bounded history, original variant count, supported device kinds, deterministic seed and fitted-information cutoff. `fit_at` must be no later than the first executable research decision. Declare information used by any pretrained model accurately.

`open(context)` creates a fresh `ProgramRun`. It must not mutate shared model parameters or carry private state across queries. `context` contains:

| Field | Contract |
| --- | --- |
| `batch_size` | Effective account/trial rows for this run |
| `variant_indices` | Resident int64 `(B,)` indices into the original descriptor batch |
| `initial_cash` | Resident `(B,)` funded initial cash |
| `intent_coefficients` | Resident `(B,)` scale that replay applies to emitted quantities |
| `owner` | Assigned order namespace for this member |
| `xp`, `stream` | Selected array module and CUDA stream pointer, or CPU/no stream |
| `assets`, `plan`, `descriptor` | Immutable coordinates and declared capabilities |

Always allocate run state using `context.batch_size`. A descriptor can declare many original variants while a query runs one variant, several variants, or many portfolio weight trials. Use `variant_indices` to select the original variant's immutable parameters.

`on_event(event)` runs at each executable decision and receives copied, owned `(H_visible,S)` feature/eligibility windows, int64 causal history timestamps, the actual account and signed `owner_quantity` `(B,S)`. Neither window array bases nor metadata expose future observations. Account snapshots include aggregate positions, equity, cash, reservations, margin and active orders.

Return `OrderCommands` columns of shape `(B,K)` on the selected device. Signed integer columns identify action/owner/order/asset/side/type/expiry; quantities/prices are floating; `reduce_only` is boolean. `K` must fit the configured command bound. Action zero pads unused slots.

Commands become eligible at the next compiled base-bar open. Positive new IDs increase strictly within each assigned owner namespace; `(owner,id)` identifies an order. Amend replaces remaining quantity/conditions and resets priority. Internal negative target-child IDs cannot be cancelled by programs. `valid_until` is inclusive `-1` for good-until-cancelled, or an actual compiled base-open timestamp; arbitrary between-bar expiry is unsupported.

`on_feedback` runs after each completed base bar and returns `None`. It can stream a user-owned audit but cannot submit orders. Feedback includes actual fills, partial fills, cancellations, rejections and liquidations. Identify owner namespaces when reading it; aggregate accounts can contain other members' activity. Fill quantities are unsigned, while owner-attributed liquidation quantities are signed contributions. Global liquidation rows use owner `-1` and include the global fee; owner rows attribute that same fee by absolute contribution. Do not sum both fee representations.

Runs begin at the first executable research decision and close once, including failure cleanup. The framework retains bounded commands/orders/current feedback and measured return/equity arrays, not a complete fill-history log. CUDA callbacks must establish stream dependencies for PyTorch or other frameworks using `context.stream`; device support is declared explicitly, not inferred from arbitrary Python callbacks.

### Reading execution feedback

Feedback is a typed set of resident columns, not a Python list named `fills`. B is the effective run batch, F the bounded feedback capacity, S assets, and Q active-order capacity. Only the first `count[b]` feedback entries are valid in row b; all columns share their row/slot ordering.

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `count` | `(B,)` integer | Valid feedback slots per account |
| `status`, `reason`, `stage` | `(B,F)` integers | Values of `OrderStatus`, `OrderReason`, `OrderStage` |
| `sequence` | `(B,F)` integer | Stable increasing audit sequence within each account |
| `owner`, `id`, `asset` | `(B,F)` integers | Member namespace, order ID and asset index |
| `time` | `(B,F)` int64 UTC ns | Event's model timestamp; not a precise tick time |
| `quantity`, `price`, `fee`, `remaining` | `(B,F)` floating | Executed quantity, quote/asset price, quote fee and reported remaining order quantity |
| `event_ns`, `available_ns` | Host integer UTC ns | Base-bar open and completed-bar availability/close |
| `account.cash`, `.equity`, `.reserved_cash` | `(B,)` floating | Actual close-state cash, total equity and reserved funds |
| `account.quantity`, `.margin`, `.entry_price` | `(B,S)` floating | Net quantities, isolated margin and entry price |
| `account.orders` | Columns `(B,Q)` | Detached active-order snapshot; action zero is an empty slot |

For example, this callback computes own-asset fill quantities for the current bar without transferring a trace to the host:

```python
def on_feedback(self, feedback: foundry.ExecutionFeedback) -> None:
    xp, owner = self.context.xp, self.context.owner
    valid = xp.arange(feedback.status.shape[1])[None, :] < feedback.count[:, None]
    own = (feedback.owner == owner) & (feedback.asset == self.asset)
    filled = ((feedback.status == int(foundry.OrderStatus.FILLED))
              | (feedback.status == int(foundry.OrderStatus.PARTIAL)))
    filled_quantity = xp.where(valid & own & filled, feedback.quantity, 0.).sum(axis=1)
    # Consume here, or copy into explicitly declared checkpoint state.
```

`filled_quantity` is `(B,)` in that asset's units. It counts actual fills, not accepted orders. Persisting it for later decisions would add private mutable state and must be included in a checkpointable program's declared schema/snapshot. Rejection/cancellation rows are not fills. The global/owner liquidation conventions described above still apply.

## Sizing and owner positions

Shared and independent compositions have different capital meanings:

- **Shared:** one account, common cash and aggregate risk; coefficients scale each member's target and emitted order quantities once. A constant `quantity=0.001` means the member's unscaled unit-coefficient intent. Do not multiply it by `intent_coefficients` yourself.
- **Independent:** each member receives its coefficient of initial cash once; unused allocation stays cash. The run sees that funded account and intent coefficient one. A fixed absolute quantity may be unaffordable at a small allocation; use funded cash/equity for an equity-aware strategy.

In a shared account, actual owner inventory has already been scaled. To request a complete owner close, convert back to the unscaled command convention:

```python
actual = event.owner_quantity[:, asset]
unscaled_close_quantity = abs(actual) / context.intent_coefficients
```

Only positive-coefficient rows open a run, so this denominator is positive. Choose sell for positive inventory and buy for negative inventory, and put the resulting `(B,1)` quantities into command columns. Using the raw actual quantity would apply the coefficient twice and under-close the position.

Owner contributions sum to net account quantity. Spot sells can consume only that owner's inventory. In isolated accounts, `reduce_only` must reduce both the owner's contribution and aggregate net exposure; closing a hedged owner's contribution can increase net exposure and therefore is not automatically a valid reduce-only order. Funding, reservations, fees and aggregate liquidation remain actual-account operations; owner attribution does not create separate shared cash balances.

## Discovery, families and replay

`registered.candidates` supplies original variant identities directly to `Discovery`, `ExternalProposer` and `CandidateGrid`. The same hard admission rules apply to programs, formulas and arrays. Under the order model, changed weights or an accepted earlier candidate require fresh private states and actual execution replay. A program's previously realized target or position path is not reused as if account state had stayed constant.

An immediate `commit(report)` can reuse the session's own recent evaluation and admission reports while the research context is unchanged. It does not trust metrics or decisions supplied by the caller. At most one candidate batch of host metadata is retained, with no returns, positions or live accounts. Once a candidate is accepted, subsequent candidates are judged again against the updated portfolio. Refresh, registration changes, release and close invalidate this reuse; a zero `measurement_cache_mib` disables it. This works through both local sessions and RPC without an extra public call.

The descriptor and canonical parameters identify the replayable program; a candidate source additionally binds the study and original variant. Temporary registration has a lease scope, while durable sources have none. Shared-model reuse is scoped to active registrations; run state and account feedback do not become a historical model cache.

## Remote deployment

Deployment belongs to the compute application. The server process imports your trusted loader module and supplies its catalog once:

```python
import foundry
from my_program import catalog

foundry.serve(
    "var/runs/program-portfolio.json", programs=catalog,
    market="crypto_spot", data_dir="/datasets/crypto_5m", backend="cuda",
    start="2025-01-08", end="2025-01-31",
    clock=foundry.Clock(decision="5min", statistics="1D"),
    account=foundry.Account(initial_cash=1000.),
    execution=foundry.Execution(model="orders"),
)
```

The client needs the deployed digest and canonical parameter string, not the server's loader code or artifact path:

```python
import foundry

reference = foundry.ProgramReference(deployed_digest, '{"quantity":0.001}')
with foundry.connect(("127.0.0.1", forwarded_port)) as session:
    print(session.plan_program(reference))
    report = session.evaluate(reference)
    session.commit(report)
```

Replace `deployed_digest` with `artifact.digest` provided by deployment, and `forwarded_port` with the service/tunnel port. Registration and evaluation do not install dependencies, alter server source or send pickle/source strings over RPC. Missing deployments, changed descriptors or mismatched parameters fail explicitly. The same catalog must be supplied after server restart for durable library replay.

## Standalone audits and resumable state

`session.simulate(program_or_reference)` opens a flat account with fresh private state and returns owned resident `ProgramSimulation` arrays plus the final actual account. It is useful for inspecting execution; it is not an admission report and cannot itself be submitted with `commit`.

Mining/discovery checkpoints save durable candidates and proposer state, then replay fresh accounts. They are separate from a paused private program/account checkpoint.

The example declares `checkpointable=True` and implements `CheckpointableRun.snapshot()`/`restore(state)`. Its complete private mutable state is the named boolean `sent` tensor. A snapshot is a tuple of `StateTensor` values in exactly `state_schema` order, each with shape `(descriptor.batch_size, *declared_shape)` and the declared dtype. Include every mutable model/RNG component needed for exact continuation; hidden Python state is not automatically inspected or pickled. Programs without this capability can still run full fresh simulations.

Inside the same configured session, pause at an exact UTC-nanosecond base-bar close and export its detached host checkpoint:

```python
from datetime import datetime, timezone

pause_ns = int(datetime(2025, 1, 15, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
paused = session.simulate(reference, until=pause_ns)
foundry.export(paused.checkpoint, "var/runs/buy-once.npz")
checkpoint = foundry.load_program_checkpoint("var/runs/buy-once.npz")
resumed = session.simulate(reference, checkpoint=checkpoint)
print(resumed.returns.shape, resumed.final_account.equity)
```

This boundary is valid for the preceding 5-minute dataset example; choose an actual covered base close for your data. `until` is exclusive of subsequent bars, and a paused result exposes only completed statistics columns. If the pause falls within a daily statistics period, that unfinished day's state lives in the checkpoint rather than a partially reported daily return. A resumed result includes the already completed prefix, so do not concatenate it with the paused result's returns. Pass another `until` to resume only as far as a later boundary and obtain another checkpoint.

The checkpoint binds the complete `ResearchContext`, descriptor, plan and next-event cursor, including account balances, owner/order state, reservations, high-water IDs, statistics state and declared private tensors. Reopen with the same data, research interval, unchanged portfolio version, program artifact/parameters, device kind/index and precision plan. A portfolio change causes `ContextConflict`; refreshing cannot make an old checkpoint current. A different program, malformed state or incompatible format is rejected explicitly.

`foundry.export(checkpoint, path)` writes an atomic versioned NumPy bundle; `load_program_checkpoint` verifies its metadata, array shapes/dtypes and SHA-256 integrity without opening compute. Both encoded and decoded payloads are bounded to 1 GiB, and object arrays/pickle are forbidden. Only owned bounded account/private state and completed measurements are saved, not market panels, source code or a full event log. The current format is `program-checkpoint-v1`.

This is an in-process standalone audit API. It does not pause the shared RPC worker, checkpoint a portfolio weight trial, or resume at an intrabar instruction/fill position. The pause boundary is after the completed bar's feedback; a resumed run begins with the next event and does not deliver prior feedback twice. Every partial/resumed invocation still opens and closes its own run resources.
