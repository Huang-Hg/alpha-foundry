# Foundry

Quantitative backtesting, bar-level order simulation, and portfolio construction on Linux CPU and CUDA, supporting US equities, China A-shares, and cryptocurrencies.

[简体中文](README.zh-CN.md) · [Research guide](guides/research.md) · [Data formats](guides/data.md) · [Strategy programs](guides/programs.md) · [Benchmarks](guides/benchmarks/README.md)

Foundry evaluates formula expressions, external model arrays, and stateful order programs in a single session, applying the same market calendars, historical eligibility masks, fee and margin rules, and portfolio admission checks to each candidate.

```python
import foundry

with foundry.open(backend="cuda") as session:
    report = session.evaluate({
        "trend": "gt(ma_dist_20,0)",
        "momentum": "cs_rank(ret_20)",
    })
    for row in report.rows:
        print(row.id, row.ok, row.metrics.sharpe, row.decision.passed, row.decision.reason)
```

Calling `foundry.open()` without a file path loads the built-in US equity sample (`./data/us_equity_daily`, `2025-07-01` to `2026-06-30`) into an in-memory portfolio. Pass `backend="cpu"` when running without an NVIDIA GPU, pass a `.json` path as the first argument to save a portfolio on disk, or set `market`, `data_dir`, `start`, and `end` for your own datasets.

## Installation

Local computation runs on Linux, Windows WSL2, or cloud Linux notebooks. Remote clients run on Windows, macOS, or Linux without compiling C++ extensions or installing market-data libraries.

Compute hosts need Python 3.12 or 3.13 with development headers (`python3-dev`), `g++` with C++17 and OpenMP (`libgomp`), and `pip>=25.1` for [pyproject.toml](pyproject.toml) dependency groups:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group runtime
```

| Environment | Install command | Packages |
| --- | --- | --- |
| Remote client (Windows / macOS / Linux) | `python -m pip install --group client` | `numpy`, `jsonschema` |
| Local CPU sessions | `python -m pip install --group runtime` | `client` + `pandas`, `pyarrow`, `exchange_calendars` |
| Terminal dashboard (`foundry monitor`) | `python -m pip install --group client --group monitor` | `client` + `rich` (add `--group monitor` to a compute install if needed) |
| Local CUDA sessions | `python -m pip install --group runtime --group cuda13` | `runtime` + `cupy-cuda13x` (use `--group cuda12` on CUDA 12) |
| Built-in GFlowNet mining on CUDA | `python -m pip install --group mining --group cuda13` | `runtime` + `torch` + `cupy-cuda13x` |

Install either `cuda12` or `cuda13` to match the host CUDA major version, not both. PyTorch (`--group mining`) is only used by the built-in GFlowNet formula proposer; evaluating formulas, arrays, stateful programs, walk-forward models, or external proposers does not require PyTorch.

### Build native C++ extensions

On Linux / WSL2 / Colab hosts that run local CPU or CUDA sessions, build the C++ extensions once from the repository root. CUDA kernels are compiled by CuPy when a CUDA session opens.

<details open>
<summary>Compile C++ extensions</summary>

<!-- native-build:start -->
```bash
python - <<'PY'
import subprocess
import sysconfig
import numpy

suffix = sysconfig.get_config_var("EXT_SUFFIX")
common = ["g++", "-std=c++17", "-fPIC", "-shared", "-Wall", "-Wextra",
          "-Wno-unused-parameter", "-Wno-missing-field-initializers",
          "-I" + sysconfig.get_paths()["include"], "-I" + numpy.get_include(), "-Ifoundry/include"]
targets = [
    ("ops/_ops", "ops", ["-O3", "-march=native", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing", "-fopenmp"],
     ["ops_ts.cpp", "ops_xs.cpp", "ops_inject_ts.cpp", "module.cpp"]),
    ("backtest/_bt", "backtest", ["-O3", "-march=native", "-ffast-math", "-fno-finite-math-only", "-fno-strict-aliasing", "-fopenmp"],
     ["kernel.cpp", "perpetual.cpp", "module.cpp"]),
    ("backtest/_weights", "backtest", ["-O3"], ["weights.cpp"]),
    ("backtest/_orders", "backtest", ["-O3", "-fno-fast-math", "-ffp-contract=off", "-fopenmp"], ["orders_module.cpp"]),
]
for name, module, flags, sources in targets:
    directory = "foundry/" + module + "/src/"
    subprocess.run(common + flags + ["-I" + directory] + [directory + source for source in sources]
                   + ["-o", "foundry/" + name + suffix, "-lgomp", "-lm"], check=True)
PY
```
<!-- native-build:end -->

</details>

## Markets and execution modes

`Execution(model="targets")` is the default target-weight backtester: it maps scores or target equity fractions to positions and rebalances at the next bar open. `Execution(model="orders")` runs a bar-level account engine supporting market, limit, and stop orders, resting orders across bars, cash and margin reservations, partial fills under volume participation caps, per-strategy inventory tracking, corporate actions, funding, and liquidation.

| Market (`market=`) | Target backtests (`model="targets"`) | Order accounts (`model="orders"`) |
| --- | --- | --- |
| `us_equity` | Daily bars, historical universe membership, split & dividend adjustments, trading fees & short borrow | Daily or regular-session intraday cash orders, T+1 sale-proceeds settlement, splits & cash dividends |
| `a_share` | Daily bars, publication dates, suspensions, board price limits, commission, transfer fee & sell stamp tax | Daily or regular-session intraday cash orders, T+1 buy lock, board lot/step rules, suspensions & price limits |
| `crypto_spot` | Daily or custom base bars, historical listing eligibility & maker/taker fees | Market/limit/stop orders, quote-cash reservations & per-strategy spot inventory |
| `crypto_perpetual` | Daily/hourly archive or custom base bars, mark-price valuation, funding & isolated margin | Signed positions, isolated or cross margin (`Account(margin=...)`), funding, margin reservations & liquidation |

### Multi-frequency clocks and account settings

On intraday datasets (such as 5-minute bars), decision intervals, base-bar execution, and return measurement periods are configured separately:

```python
import foundry

with foundry.open(
    market="crypto_perpetual",
    data_dir="/datasets/crypto_5m",
    start="2025-01-08",
    end="2025-01-31",
    backend="cuda",
    clock=foundry.Clock(decision="15min", statistics="1D"),
    account=foundry.Account(
        margin="isolated",       # "cash", "isolated", or "cross"
        leverage=3.0,
        gross_limit=2.0,
        initial_cash=100_000.0,
        composition="shared",    # "shared" or "independent"
    ),
    execution=foundry.Execution(
        model="orders",
        bar_path="open_low_high_close",
        participation=0.10,
    ),
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

A 15-minute decision takes effect at the next 5-minute base-bar open; resting orders, funding, and liquidation continue updating on every 5-minute bar between decisions, while Sharpe and drawdown are computed on daily (`1D`) closes. Equity intraday bars must fall within regular exchange hours.

Under `composition="shared"`, portfolio members trade inside one account and member weights scale order quantities or target positions once. Under `composition="independent"` (order mode), each member receives `initial_cash * weight` in its own sub-account; see [Execution and portfolio composition](guides/research.md#execution-and-portfolio-composition). Both execution modes fill against completed bar OHLC paths and volume participation caps rather than L2/L3 order-book queues.

## Writing strategies

### Formulas

Pass a single expression, a list of expressions, or a `{name: expr}` mapping to `session.evaluate`:

```python
with foundry.open(
    market="us_equity",
    backend="cuda",
    sizing=foundry.Sizing(rule="equal_slots", scale=1.0),
) as session:
    info = session.describe()
    print("Fields:", [f["name"] for f in info["fields"]])

    report = session.evaluate({
        "ma_breakout": "gt(ma_dist_20,0)",
        "vol_adj_mom": "cs_zscore(div(ret_20,ts_std(ret_1,20)))",
    })
```

Run `python -m foundry ops` to list the 66 built-in rolling and cross-sectional operators, and `python -m foundry fields` to inspect fields for the selected market. `Sizing(rule="equal_slots")` normalizes weights across active assets at each decision; `Sizing(rule="independent", scale=...)` scales each asset independently without dividing by active asset count.

### External model arrays and walk-forward fitting

Predictions or target positions from external Python/ML libraries (`NumPy`, `CuPy`, `scikit-learn`, `LightGBM`, `PyTorch`) are registered as `(candidates, decisions, assets)` arrays aligned to `session.grid`, along with per-decision observation and fit timestamps (`observed_at`, `fit_at` in UTC nanoseconds, or `foundry.NO_INFORMATION` when no model fitting was used):

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
        scores,
        kind="scores",           # "scores" or "targets"
        unit="score",            # "score", "equity_fraction", "quantity", or "notional"
        ids=("external-score",),
        grid=grid,
        producer="ridge-v1",
        observed_at=observed,
        fit_at=fitted,
    )
    try:
        report = session.evaluate(registered)
        print(report.rows[0].metrics, report.rows[0].decision)
    finally:
        session.release(registered)
```

For rolling out-of-sample training, `foundry.walk_forward(session, fit, predict, folds=..., ...)` splits non-overlapping `Fold(fit_start, fit_stop, predict_start, predict_stop)` windows, passes historical slices to `fit` and causal prefixes to `predict`, records `observed_at` and `fit_at`, and registers the out-of-fold predictions; see [Causal walk-forward models](guides/research.md#causal-walk-forward-models).

### Stateful order programs

When order placement, stop updates, or position exits depend on live cash, inventory, resting orders, or prior fills, implement `StrategyProgram` and open the session with `Execution(model="orders")`:

```python
from my_program import catalog, reference
import foundry

with foundry.open(
    "var/runs/program-portfolio.json",
    programs=catalog,
    market="crypto_spot",
    data_dir="/datasets/crypto_5m",
    start="2025-01-08",
    end="2025-01-31",
    backend="cuda",
    clock=foundry.Clock(decision="5min", statistics="1D"),
    account=foundry.Account(initial_cash=10_000.0, composition="shared"),
    execution=foundry.Execution(model="orders", participation=0.1),
) as session:
    print(session.plan_program(reference))
    report = session.evaluate(catalog.resolve(reference))
    session.commit(report)
```

Each evaluation or portfolio weight trial opens fresh program state and replays the account from the start of the study. For a standalone single-account run with equity curves, fills, and optional pause/resume checkpoints, call `session.simulate(program)`. Complete runnable examples are in [guides/programs.md](guides/programs.md).

## Custom market data

Foundry reads local Parquet directories specified by `market`, `data_dir`, `start`, and `end`. Daily equity and spot datasets use `panel.parquet` (plus `membership.parquet` for equity universes and `manifest.json`). Intraday datasets use `bars.parquet` with `[open_ns, close_ns)` and `available_ns` timestamps, `manifest.json`, and market-specific funding or corporate-action tables.

```python
with foundry.open(
    market="a_share",
    data_dir="/datasets/my_a_share_daily",
    start="2024-03-01",
    end="2026-06-30",
    backend="cuda",
    commission=0.00025,
    stamp_tax_sell=0.0005,
) as session:
    report = session.evaluate("cs_rank(ret_20)")
```

Include rows earlier than `start` in your Parquet files so rolling lookback windows are warm at the start of the evaluation range. Public Binance USD-M archives can also be imported with `python -m foundry import-perpetual <output_dir>`. Column schemas and CSV conversion scripts are in [guides/data.md](guides/data.md); built-in offline sample datasets are listed in [data/README.md](data/README.md).

## Portfolio workflow and search

`session.evaluate(...)` is read-only: it tests candidate weight mixtures (`Admission(weight_steps=20)` by default plus an analytic tangency step) against the current portfolio and checks whether the candidate improves portfolio utility within drawdown, correlation, and member-count limits. Call `session.commit(report)` to write passing candidates to your portfolio file:

```python
with foundry.open(
    "var/runs/my_portfolio.json",
    backend="cuda",
    admission=foundry.Admission(weight_steps=20, min_sharpe_gain=0.01, max_members=40),
) as session:
    report = session.evaluate({"trend": "gt(ma_dist_20,0)", "mom": "cs_rank(ret_20)"})
    for row in report.rows:
        if not row.decision.passed:
            print(row.id, "rejected:", row.decision.reason, row.decision.stopped_at)

    decision = session.commit(report)
    foundry.export(decision, "var/runs/commit.json")

    print("Active members:", len(session.portfolio().sources))
    gaps = session.diagnose()
```

In each row, `row.ok` indicates whether evaluation completed without syntax, missing-data, or bankruptcy rejection, while `row.decision.passed` indicates whether the candidate passed portfolio admission. To adjust existing member weights, use `session.trial_weights(weight_matrix)` followed by `session.commit_weights(trials, index)`, or call `session.reweight()`. In-process sessions also provide `session.evaluate_device(inputs)` to retrieve device/host return series (`.batch`) alongside the committable report (`.report`) in one pass.

For automated search, `foundry.Discovery(session)` computes dense search rewards across formulas, arrays, or programs, and `foundry.mine` runs a checkpointed search loop:

```python
import foundry

foundry.mine(
    "var/runs/us_search",
    market="us_equity",
    backend="cuda",
    search=foundry.Search(mode="gfn", epochs=10, batch_size=128, accept=False),
)
```

Set `accept=True` in `Search(...)` to commit admitted candidates to the persistent library during mining. Custom generators (LLMs, agents, evolutionary loops) plug into `foundry.mine(..., proposers=...)` via `ExternalProposer` or the `Proposer` interface; see [Strategy discovery](guides/research.md#strategy-discovery). Run `python -m foundry monitor` in another terminal to watch progress live.

## CLI and parameters

All settings are passed as typed Python arguments or as JSON via `python -m foundry --parameters '<json>'`. `foundry.schema()` and `python -m foundry schema schema.json` export the parameter and JSON-RPC schemas without loading market data or GPUs.

| Command (`python -m foundry ...`) | Purpose |
| --- | --- |
| `ops` / `fields` | List the 66 operators or the active market's fields |
| `schema <output.json>` | Write parameter and JSON-RPC schemas |
| `serve` / `status` / `stop` | Start, query, or stop the local JSON-RPC service |
| `attach <addr> <port>` / `detach` | Bind CLI commands to a local or SSH-forwarded port, or clear the binding |
| `request <in.jsonl> <out.jsonl>` | Send batch JSON-RPC requests to the attached service |
| `mine <output_dir>` | Run checkpointed search (`--parameters '{"backend":"cuda",...}'`) |
| `monitor` / `preview` | Open the live terminal dashboard (`--group monitor`) or render saved status files |
| `calibrate <run_dir>` | Print reward calibration stats for a search run |
| `import-perpetual <output_dir>` | Verify and convert public USD-M perpetual archives to Parquet |

| Parameter / Class | Options and defaults |
| --- | --- |
| `market` & market kwargs | `"us_equity"` (default), `"a_share"`, `"crypto_spot"`, `"crypto_perpetual"`, plus `data_dir`, `start`, `end`, and fee/margin rates |
| `backend` | `"cpu"` or `"cuda"` |
| `foundry.Clock` | `decision="1D"`, `statistics="1D"` (`"<N>s"`, `"<N>min"`, `"<N>h"`, `"<N>D"`) |
| `foundry.Sizing` | `rule="equal_slots"` (default) or `"independent"`, `scale=1.0` |
| `foundry.Account` | `margin="cash" \| "isolated" \| "cross"`, `leverage=1.0`, `gross_limit=1.0`, `initial_cash=1.0`, `composition="shared" \| "independent"` |
| `foundry.Execution` | `model="targets" \| "orders"`, `bar_path="open_low_high_close" \| "open_high_low_close"`, `participation=1.0`, `order_capacity=32`, `command_capacity=32`, `quantity_step=0.0`, `price_tick=0.0` |
| `foundry.Admission` | `weight_steps=20`, `min_sharpe_gain=0.01`, `max_members=40`, `artifact_dir="./var/candidates"` |
| `foundry.Search` | `mode="gfn" \| "random"`, `epochs=5`, `batch_size=128`, `accept=False`, `objective="portfolio" \| "single"` |
| `foundry.CPU` / `foundry.CUDA` | `CPU(workers=8, candidate_batch=32)`, `CUDA(streams=8, candidate_batch=256, trial_batch=16384)` |

## Remote server and client

To keep data and GPUs resident on a Linux server while working from another machine (such as Windows Python with `--group client`), start the loopback service on the Linux host:

```python
import foundry

foundry.serve("var/runs/library.json", backend="cuda")
```

Check the listening port on that host with `python -m foundry status`, forward it over SSH from your local machine, and connect:

```sh
ssh -N -L 58731:127.0.0.1:REMOTE_PORT user@compute-host
```

```python
import foundry

with foundry.connect(("127.0.0.1", 58731)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    print(report.rows[0].metrics)
```

For CLI batch requests on the client, run `python -m foundry attach 127.0.0.1 58731` (`detach` clears the local selection). Disconnecting a client leaves the server running; stop it explicitly with `python -m foundry stop`. Remote sessions support formulas, registered arrays, deployed `ProgramReference` evaluation, diagnostics, weight trials, and commits; `evaluate_device`, `simulate`, `walk_forward`, and `mine` run in-process on the compute host.

Further details are covered in the [Research guide](guides/research.md), [Data formats](guides/data.md), [Strategy programs](guides/programs.md), and [Benchmarks](guides/benchmarks/README.md) ([framework comparison method](guides/framework-benchmark.md)).
