# Foundry

**Point-in-time quantitative strategy discovery, execution simulation, and portfolio admission on CUDA and CPU.**

[简体中文](README.zh-CN.md) · [Research guide](guides/research.md) · [Data contracts](guides/data.md) · [Strategy programs](guides/programs.md) · [Benchmarks](guides/benchmarks/README.md)

Foundry evaluates algebraic formulas, externally trained model arrays, and stateful order-emitting programs inside a single causal research session. Human researchers, AI agents, and automated discovery loops share the same market calendars, execution rules, risk accounting, and hard portfolio admission gates.

```python
import foundry

with foundry.open(backend="cuda") as session:
    report = session.evaluate({
        "trend": "gt(ma_dist_20,0)",
        "momentum": "cs_rank(ret_20)",
    })
    for row in report.rows:
        print(row.id, row.metrics, row.decision)
```

By default, this opens the bundled US equity fixture in memory. Supply `market`, `data_dir`, `start`, and `end` to run on your own historical datasets.

## Core capabilities

Foundry separates stateless batch evaluation from atomic portfolio state transitions. Device placement and numerical precision are fixed when a session opens—never switched mid-run based on candidate outcomes.

- **Unified candidate model:** evaluate 66-operator DSL formulas, `(B, D, S)` score or target tensors, and stateful `StrategyProgram` instances through the same `Session` and `Discovery` APIs.
- **Explicit causal clocks:** source bars, decision timestamps, next-base-open execution, and valuation periods advance on distinct clocks; external arrays and programs must declare observation and fitting cutoffs (`observed_at`, `fit_at`).
- **Bar-level order accounts:** under `Execution(model="orders")`, market, limit, stop, cancel, and amend commands execute against OHLC paths, volume participation caps, cash/inventory reservations, funding settlements, and isolated-margin liquidation.
- **Hard judge vs. soft guidance:** dense discovery rewards, behavioral deduplication, and neighbourhood plateau checks guide search without ever overriding hard G1/G2 portfolio admission.
- **Scoped resident leases:** register candidate batches into bounded device memory, evaluate or reweight repeatedly without re-uploading, retain admitted artifacts to disk, and deterministically release temporary leases.

## Supported markets and execution models

| Market | Target backtests (`model="targets"`) | Stateful order accounts (`model="orders"`) |
| --- | --- | --- |
| US equities (`us_equity`) | Daily bars, PIT universe membership, split/dividend-adjusted fills, transaction & borrow costs | Not supported |
| China A-shares (`a_share`) | Daily bars, publication-dated vintages, suspensions, price limits, asymmetric fees & stamp tax | Not supported |
| Crypto spot (`crypto_spot`) | Daily or intraday base bars, historical eligibility, proportional fees | Market/limit/stop orders, cash reservations, per-member inventory attribution |
| USDT linear perpetuals (`crypto_perpetual`) | Hourly archive or intraday base bars, settled funding, isolated margin & liquidation | Signed member positions, funding cashflows, margin reservations & liquidation |

`Execution(model="targets")` is the default for formulas and precomputed equity-fraction books. `Execution(model="orders")` routes formulas, arrays, and stateful programs through the crypto Bar execution engine with full chronological replay whenever weights or portfolio members change (for example, 5-minute base bars, 15-minute decisions, and daily return statistics). See [Execution and portfolio composition](guides/research.md#execution-and-portfolio-composition) for shared versus independently funded accounts.

Both execution modes operate on bar OHLC paths and declared participation limits; neither models order-book queue priority or tick-by-tick microstructure.

## Three strategy inputs, one session

```python
report = session.evaluate("gt(ma_dist_20,0)")
report = session.evaluate(registered_array_batch)
report = session.evaluate(strategy_program)
```

- **Formulas** parse against the 66-operator registry with bounded AST cost and rolling-window depth.
- **Arrays** align to `session.grid` (`(B, D, S)` in `float32` or `float64`) as cross-sectional `scores` mapped via `Sizing` or direct `targets` (`equity_fraction`, plus `quantity` / `notional` in order sessions). See [Arrays and model outputs](guides/research.md#arrays-and-model-outputs).
- **Programs** implement `StrategyProgram` (`descriptor` + `open(context)`), receiving copied causal feature windows and live account snapshots at each decision bar and fill feedback after each base bar. See [Strategy programs and deployed models](guides/programs.md).

## Bring your own data

Point `foundry.open` at local Parquet datasets matching the documented market contracts:

```python
with foundry.open(
    market="us_equity", data_dir="/datasets/my_us_equity",
    start="2025-07-01", end="2025-12-31", backend="cuda",
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

Daily equity datasets supply `panel.parquet` and PIT `membership.parquet` (plus `manifest.json` for A-shares and spot). Intraday crypto datasets use `crypto-bars-v1` (`bars.parquet`, `manifest.json`, and dense `funding.parquet` for perpetuals). Full schemas, unit conventions, and CSV conversion scripts live in [Data contracts](guides/data.md); the [bundled fixtures](data/README.md) work offline out of the box.

## Installation and native build

Compute runs on Linux (including WSL2 on Windows and cloud GPU instances) with Python 3.12 or 3.13, Python headers, and GCC supporting C++17 and OpenMP:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group runtime
python -m foundry --help
```

<details>
<summary>Build native C++ extensions (Linux / WSL2 / Colab)</summary>

Run from the repository root inside the target Linux Python environment. CUDA kernels are compiled on demand by CuPy when a CUDA session opens; native Windows remote clients do not compile C++ extensions.

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

Dependencies are managed in [pyproject.toml](pyproject.toml) via [pip dependency groups](https://pip.pypa.io/en/stable/user_guide/#dependency-groups) (`pip>=25.1`):

| Use case | Command |
| --- | --- |
| CPU research or remote RPC client | `python -m pip install --group runtime` |
| CUDA research | `python -m pip install --group runtime --group cuda13` |
| GFlowNet formula mining on CUDA | `python -m pip install --group mining --group cuda13` |
| Cloud regression and benchmark suite | `python -m pip install --group benchmark --group cuda13` |
| Local static checks (`pre-commit`, Ruff, `actionlint`) | `python -m pip install --group dev` |

Install either `cuda12` or `cuda13` (never both) to match the host CUDA driver/runtime. PyTorch is required only for built-in GFlowNet policy training (`mining` / `benchmark`). Pass `backend="cpu"` when opening a CPU session.

### Remote client / server mode

Start a single-worker JSON-RPC compute service on the Linux GPU host:

```python
import foundry

foundry.serve("var/runs/library.json", backend="cuda")
```

Inspect its loopback port via `python -m foundry status`, forward the port over SSH if connecting across machines, and attach from Linux or native Windows Python:

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

Closing a client session leaves the server running. [Remote sessions](guides/research.md#remote-sessions) transfer array batches as bounded binary NumPy bundles (1 GiB limit, no `pickle`) and invoke pre-deployed `ProgramReference` artifacts without sending executable source code.

## Read-only queries and atomic commits

```python
with foundry.open("var/runs/my_portfolio.json") as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

Queries never mutate portfolio state. Every report binds the exact `ResearchContext` (study snapshot + portfolio version); if concurrent work updates the library, `commit` raises `ContextConflict` until the caller explicitly calls `refresh()` and re-evaluates. Note the distinction between `row.ok` (valid execution) and `row.decision.passed` (passed G1/G2 portfolio admission).

| Workflow | Entry point |
| --- | --- |
| Inspect vocabulary, operators, clocks, and execution plan | `session.describe()` |
| Obtain device-resident returns and a committable report together | `session.evaluate_device(inputs)` |
| Manage candidate leases and durable artifacts | `session.register(...)`, `retain(...)`, `release(...)` |
| Diagnose portfolio gaps or test parameter neighbourhoods | `session.diagnose()`, `judge_family(...)` |
| Evaluate or optimize portfolio member weights | `session.trial_weights(...)`, `commit_weights(...)`, `reweight(...)` |
| Score candidates with dense discovery rewards and archive novelty | `foundry.Discovery(session).evaluate(candidates)` |
| Run autonomous or hybrid proposer loops with checkpointing | `foundry.mine(output, ...)` |
| Fit and predict across strictly causal rolling/expanding folds | `foundry.walk_forward(session, fit, predict, ...)` |

Call `foundry.schema()` (or `python -m foundry schema OUTPUT.json`) to export the machine-readable JSON-RPC and parameter schemas without loading market data or initializing a GPU. See the [Research guide](guides/research.md) for end-to-end patterns and [Benchmarks](guides/benchmarks/README.md) for hardware-specific validation reports.
