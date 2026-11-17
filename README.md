# Foundry

Point-in-time quantitative strategy backtesting, bar-level order simulation, portfolio construction, and automated strategy search on Linux CPU and NVIDIA CUDA.

[简体中文](README.zh-CN.md) · [Research guide](guides/research.md) · [Data formats](guides/data.md) · [Strategy programs](guides/programs.md) · [Benchmarks](guides/benchmarks/README.md)

Foundry provides a single Python and CLI interface for evaluating cross-sectional/time-series formulas, external ML model predictions, and stateful order-placement programs across **US equities**, **China A-shares**, and **cryptocurrencies**. Interactive human research, external Python models, and automated search loops run against the same market calendars, point-in-time eligibility masks, fee/margin rules, and portfolio admission checks.

## Quickstart

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

Calling `foundry.open()` with no portfolio path loads the bundled US equity sample dataset (`./data/us_equity_daily`, `2025-07-01` to `2026-06-30`) into an in-memory portfolio. Pass `backend="cpu"` when running without an NVIDIA GPU, pass a `.json` path as the first argument to persist a portfolio on disk, or pass `market`, `data_dir`, `start`, and `end` to evaluate against your own historical datasets.

---

## Installation

Local computation runs on **Linux** (including **Windows WSL2** and cloud notebooks such as Google Colab). Remote clients can run on native Windows, macOS, or Linux without compiling C++ extensions or installing market-data libraries.

### Prerequisites (compute hosts)

- Python `3.12` or `3.13` with development headers (`python3-dev`)
- `g++` with C++17 and OpenMP support (`libgomp`)
- `pip>=25.1` (for PEP 735 [dependency groups](https://pip.pypa.io/en/stable/user_guide/#dependency-groups) declared in [pyproject.toml](pyproject.toml))

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group runtime
```

### Dependency groups

| Scenario | Install command | Installed third-party packages |
| --- | --- | --- |
| Remote RPC client (Windows / macOS / Linux) | `python -m pip install --group client` | `numpy`, `jsonschema` |
| Local CPU backtesting & research | `python -m pip install --group runtime` | `client` + `pandas`, `pyarrow`, `exchange_calendars` |
| Terminal live monitor (`foundry monitor`) | `python -m pip install --group client --group monitor` | `client` + `rich` (combine with `runtime` on a compute host) |
| Local CUDA backtesting & research | `python -m pip install --group runtime --group cuda13` | `runtime` + `cupy-cuda13x` (or `--group cuda12` for CUDA 12) |
| Built-in GFlowNet formula mining on CUDA | `python -m pip install --group mining --group cuda13` | `runtime` + `torch` + `cupy-cuda13x` |

Install either `cuda12` or `cuda13` to match your host CUDA major version—never both in the same environment. PyTorch (`--group mining`) is required only when training the built-in GFlowNet policy proposer; formula evaluation, custom arrays, stateful programs, walk-forward fitting, and external proposers do not require PyTorch.

### Build native C++ extensions

On any Linux / WSL2 / Colab machine that runs local CPU or CUDA sessions, compile the four C++ extensions once from the repository root using the target Python environment. CUDA kernels are compiled automatically by CuPy when a CUDA session opens.

<details open>
<summary>Compile C++ extensions (Linux / WSL2 / Colab)</summary>

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

---

## Supported markets and execution modes

Foundry supports two execution models selected via `foundry.Execution(model=...)`:

1. **Target-position backtesting (`model="targets"`, default):** maps cross-sectional scores or target equity fractions to desired weights and rebalances at the next bar open under market-specific transaction costs, suspensions, price limits, borrow rates, or perpetual funding.
2. **Bar-level order simulation (`model="orders"`):** maintains explicit cash or margin accounts across market, limit, and stop orders—tracking resting orders across bars, cash/margin reservations, partial fills under volume participation caps, per-member inventory attribution, corporate actions, funding settlements, and liquidations.

| Market (`market=`) | Target backtests (`model="targets"`) | Order accounts (`model="orders"`) |
| --- | --- | --- |
| **US equities** (`"us_equity"`) | Daily bars, historical index/universe membership, split & dividend adjustments, trading costs & short borrow | Daily or regular-session intraday cash orders, T+1 sale-proceeds settlement, stock splits & payable cash dividends |
| **China A-shares** (`"a_share"`) | Daily bars, publication-date vintages, trading suspensions, board price limits, commission, transfer fee & sell stamp tax | Daily or regular-session intraday cash orders, T+1 buy-inventory lock, board lot/quantity rules, suspensions & price limits |
| **Crypto spot** (`"crypto_spot"`) | Daily or custom base bars, historical listing/eligibility masks & maker/taker fees | Market/limit/stop orders, quote-cash reservations & per-strategy spot inventory |
| **USDT linear perpetuals** (`"crypto_perpetual"`) | Daily/hourly archive or custom base bars, mark-price valuation, funding settlements & isolated margin | Signed long/short positions, **isolated or cross margin** (`Account(margin=...)`), funding settlements, margin reservations & liquidation |

### Multi-frequency clocks and account composition

When working with intraday base bars (for example, 5-minute bars), you can configure decision timestamps, order execution, and performance-measurement intervals independently:

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

- **Clock separation:** a decision completed at a 15-minute close becomes eligible to trade at the very next 5-minute base-bar open; resting orders, funding, and liquidation continue evolving on every 5-minute base bar between decisions, while Sharpe and drawdown are measured on daily (`1D`) closes. Equity intraday bars must fall within the exchange's regular trading session.
- **Shared vs. independent portfolio accounts:** `composition="shared"` runs one combined account where portfolio weights scale each member's order quantities or targets once; `composition="independent"` (order model) allocates `initial_cash * weight` into separate sub-accounts per member.
- **Scope boundary:** both execution modes fill against completed bar OHLC paths (`open_low_high_close` or `open_high_low_close`) and volume participation caps. Foundry does not simulate L2/L3 order-book queue priority or sub-bar tick callbacks.

---

## Three ways to express a strategy

All three strategy representations can be evaluated, compared, weighted, and committed within the same session.

### 1. Formula expressions (66 built-in operators)

Write cross-sectional and rolling time-series expressions directly as strings, lists, or `{name: expr}` dictionaries:

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

Run `python -m foundry ops` to list all 66 operators (rolling statistics, regressions, decays, ranks, quantile transforms, logical conditionals, and cross-sectional normalizations) and `python -m foundry fields` to inspect the active market's feature columns. `Sizing(rule="equal_slots")` normalizes active positions across selected assets; `Sizing(rule="independent", scale=...)` maps scores per asset without cross-sectional count dilution.

### 2. External model arrays and causal walk-forward fitting

Bring predictions or target weights from any external library (`NumPy`, `CuPy`, `scikit-learn`, `LightGBM`, `PyTorch`) by aligning a `(candidates, decisions, assets)` array to `session.grid` and declaring per-decision observation and fit timestamps (`observed_at`, `fit_at` in UTC nanoseconds, or `foundry.NO_INFORMATION` when no fitting occurred):

```python
import numpy as np
import foundry

with foundry.open(backend="cuda") as session:
    port, grid = session.resident, session.grid
    xp = port.xp

    # Shape: (B=1, D=len(grid.time_ns), S=len(grid.assets))
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

For rolling out-of-sample ML training, `foundry.walk_forward(session, fit, predict, folds=..., ...)` slices non-overlapping `Fold(fit_start, fit_stop, predict_start, predict_stop)` windows, passes copied historical slices to `fit` and causal prefixes to `predict`, automatically records `observed_at` / `fit_at`, and registers the concatenated out-of-fold predictions; see [Causal walk-forward models](guides/research.md#causal-walk-forward-models).

### 3. Stateful order programs (`StrategyProgram`)

When order submission, stop management, or inventory unwinding depends on live cash, current holdings, resting orders, or fill notifications, implement `StrategyProgram` and open the session with `Execution(model="orders")`:

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

Every evaluation or portfolio weight trial opens fresh private program state and replays the account from scratch so state-dependent feedback remains exact. Use `session.simulate(program)` when you want a standalone single-account audit (returns, equity curve, fills, and optional pause/resume checkpoints) outside the portfolio admission workflow. A complete runnable program and artifact deployment walkthrough is in [Strategy programs](guides/programs.md).

---

## Using your own market data

Foundry reads local Parquet datasets selected by `market`, `data_dir`, `start`, and `end`, truncating source data at the study cutoff before deriving rolling features or execution schedules:

1. **Daily equity and crypto spot tables:** `panel.parquet` (long-form `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`, plus market-specific adjustment/vintage/eligibility columns), `membership.parquet` for historical equity universes, and `manifest.json` recording data provenance.
2. **Intraday and base-bar datasets:** `bars.parquet` with half-open `[open_ns, close_ns)` intervals and `available_ns` publication timestamps, `manifest.json`, plus `funding.parquet` for perpetuals or corporate-action tables for intraday equities.

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

Keep earlier rows prior to `start` inside `panel.parquet` / `bars.parquet` so initial rolling lookback windows are populated at `start`. You can also import verified public Binance USD-M archives via `python -m foundry import-perpetual <output_dir>`. Exact Parquet column schemas and CSV-to-Parquet scripts are in [Data formats](guides/data.md); offline test snapshots bundled in this repository are summarized in [data/README.md](data/README.md).

---

## Portfolio construction, diagnostics, and automated search

### Evaluate, commit, and reweight a portfolio

`session.evaluate(...)` never mutates your portfolio. It tests candidate weight mixtures (`Admission(weight_steps=20)` by default plus an analytic tangency step) against the current portfolio and reports whether each candidate improves portfolio utility and satisfies risk/correlation constraints:

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

    decision = session.commit(report)                 # Atomically saves admitted candidates
    foundry.export(decision, "var/runs/commit.json")  # Exports JSON report to disk

    print("Active members:", len(session.portfolio().sources))
    gaps = session.diagnose()                         # Identify drawdown / regime gaps
```

- **Report inspection:** `row.ok` means the candidate evaluated cleanly (no syntax, bankruptcy, or missing-data rejection); `row.decision.passed` indicates whether it passed portfolio admission.
- **Weight management:** call `session.trial_weights(weight_matrix)` to evaluate custom `(trials, members)` nonnegative weight matrices without changing the portfolio, `session.commit_weights(trials, index)` to publish a chosen weight vector, or `session.reweight()` to optimize existing member weights.
- **In-process resident returns:** `session.evaluate_device(inputs)` returns both the committable report (`.report`) and zero-copy device/host return series (`.batch`) from a single pass.

### Automated search and live monitoring

Use `foundry.Discovery(session)` to score batches of formulas, arrays, or programs with dense search rewards (combining performance, novelty, and current portfolio gaps), or run an end-to-end checkpointed mining loop with `foundry.mine`:

```python
import foundry

# Run built-in GFlowNet or random formula search into var/runs/us_search/
foundry.mine(
    "var/runs/us_search",
    market="us_equity",
    backend="cuda",
    search=foundry.Search(mode="gfn", epochs=10, batch_size=128, accept=False),
)
```

Set `accept=True` in `foundry.Search(...)` when admitted candidates should also be published to the persistent portfolio library. To plug in your own LLM/agent/evolutionary generator, pass `proposers=(foundry.ExternalProposer(...),)` or a custom `Proposer` implementation to `foundry.mine(...)`; see [Strategy discovery](guides/research.md#strategy-discovery). While a mining run or server is active, render the terminal dashboard with:

```bash
python -m foundry monitor
```

---

## CLI and configuration reference

All parameters are passed directly as typed Python arguments or via `--parameters '<json>'` on the single CLI entry point `python -m foundry` (there are no hidden config files or environment-variable overrides). Call `foundry.schema()` or `python -m foundry schema schema.json` to export the full machine-readable parameter and JSON-RPC schema without loading data or GPUs.

### CLI subcommands (`python -m foundry <command>`)

| Command | Description |
| --- | --- |
| `ops` / `fields` | Print the 66 formula operators or the active market's data fields |
| `schema <output.json>` | Write the parameter and JSON-RPC schema JSON |
| `serve` / `status` / `stop` | Start, inspect, or stop the local JSON-RPC compute service |
| `attach <addr> <port>` / `detach` | Point CLI `request`/`status`/`stop` commands at a local or SSH-forwarded port |
| `request <in.jsonl> <out.jsonl>` | Send batch JSON-RPC requests to the attached service without starting a local GPU process |
| `mine <output_dir>` | Run checkpointed candidate search (`--parameters '{"backend":"cuda",...}'`) |
| `monitor` / `preview` | Launch the live terminal dashboard (`--group monitor`) or render saved status snapshots |
| `calibrate <run_dir>` | Inspect search reward calibration statistics for a completed run |
| `import-perpetual <output_dir>` | Download and verify public USD-M perpetual archives into Parquet format |

### Configuration groups (`foundry.open` / `serve` / `mine`)

| Parameter / Class | Key options and defaults |
| --- | --- |
| `market` & market kwargs | `"us_equity"` (default), `"a_share"`, `"crypto_spot"`, `"crypto_perpetual"`, plus `data_dir`, `start`, `end`, and market fee/margin rates |
| `backend` | `"cpu"` or `"cuda"` |
| `foundry.Clock` | `decision="1D"`, `statistics="1D"` (supports `"<N>s"`, `"<N>min"`, `"<N>h"`, `"<N>D"`) |
| `foundry.Sizing` | `rule="equal_slots"` (default) or `"independent"`, `scale=1.0` |
| `foundry.Account` | `margin="cash" \| "isolated" \| "cross"`, `leverage=1.0`, `gross_limit=1.0`, `initial_cash=1.0`, `composition="shared" \| "independent"` |
| `foundry.Execution` | `model="targets" \| "orders"`, `bar_path="open_low_high_close" \| "open_high_low_close"`, `participation=1.0`, `order_capacity=32`, `command_capacity=32`, `quantity_step=0.0`, `price_tick=0.0` |
| `foundry.Admission` | `weight_steps=20`, `min_sharpe_gain=0.01`, `max_members=40`, `artifact_dir="./var/candidates"` |
| `foundry.Search` | `mode="gfn" \| "random"`, `epochs=5`, `batch_size=128`, `accept=False`, `objective="portfolio" \| "single"` |
| `foundry.CPU` / `foundry.CUDA` | `CPU(workers=8, candidate_batch=32)`, `CUDA(streams=8, candidate_batch=256, trial_batch=16384)` |

---

## Remote server and cross-platform client

To keep market panels and GPUs resident on a Linux server while working from another machine (such as native Windows Python with `python -m pip install --group client`):

**1. Start the service on the Linux compute host:**

```python
import foundry

foundry.serve("var/runs/library.json", backend="cuda")
```

Run `python -m foundry status` on the compute host to check its loopback port.

**2. Forward the loopback port with system SSH** (replacing `REMOTE_PORT` with the port shown by `status`):

```sh
ssh -N -L 58731:127.0.0.1:REMOTE_PORT user@compute-host
```

**3. Connect from your local Python client or CLI:**

```python
import foundry

with foundry.connect(("127.0.0.1", 58731)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    print(report.rows[0].metrics)
```

For CLI batch requests on the client machine, run `python -m foundry attach 127.0.0.1 58731` (`detach` clears the local selection). Closing a client session or SSH tunnel leaves the remote service running; stop it explicitly with `python -m foundry stop`. Remote sessions support formula, array, and deployed `ProgramReference` planning/registration/evaluation, diagnostics, weight trials, and portfolio commits; `evaluate_device`, standalone `simulate` / checkpoint resume, `walk_forward`, and `mine` require an in-process session on the compute host.

---

## Further documentation

- [Research and integration guide](guides/research.md) — full session lifecycle, lease rules, shared vs. independent accounts, walk-forward fitting, custom proposers, and remote sessions.
- [Data formats and conversion examples](guides/data.md) — Parquet schemas, point-in-time publication vintages, corporate actions, and CSV conversion scripts.
- [Strategy programs and deployed models](guides/programs.md) — writing stateful `StrategyProgram` classes, order commands, execution feedback, checkpoints, and remote catalog deployment.
- [Performance and correctness benchmarks](guides/benchmarks/README.md) — measured L4 / A100 / PRO 6000 / Colab CPU latencies, numerical accuracy bounds, and external framework comparisons ([methodology](guides/framework-benchmark.md)).
