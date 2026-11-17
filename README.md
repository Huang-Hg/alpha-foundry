# Foundry

**Strategy evaluation, Bar-level order simulation, and portfolio admission on Linux CPU and CUDA.**

[简体中文](README.zh-CN.md) · [Research guide](guides/research.md) · [Data formats](guides/data.md) · [Strategy programs](guides/programs.md) · [Benchmarks](guides/benchmarks/README.md)

Foundry evaluates formula expressions, model output arrays, and stateful order programs under one session contract. Human researchers, AI agents, and automated search loops use the same market calendars, execution rules, reports, and portfolio admission checks.

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

Without arguments, `foundry.open` loads the bundled US equity fixture into an in-memory portfolio. Pass `market`, `data_dir`, `start`, and `end` to evaluate against your own datasets.

## Design scope

Device placement and numerical policy are fixed when a session opens; candidates never trigger a runtime precision switch or second-pass recomputation. Supported CUDA account and moment kernels use paired FP32 arithmetic to retain residual precision, while CPU runs as a first-class backend with explicit worker and cache bounds. Python callbacks in stateful programs execute as written and are not compiled into GPU kernels.

- **Three candidate forms:** a 66-operator formula DSL, `(B, D, S)` score or target arrays, and stateful order programs.
- **Explicit clocks:** source bars, decision timestamps, next-base-open execution, and valuation periods are configured separately; external inputs declare observation and fit cutoffs (`observed_at`, `fit_at`).
- **Account feedback:** under order execution, resting orders, partial fills, cancellations, fees, cash, inventory, and isolated margin determine the state seen at the next decision.
- **Separated search and admission:** gap diagnostics, archive deduplication, neighbourhood checks, and dense rewards guide candidate search without overriding hard G1/G2 portfolio admission.
- **Bounded leases:** register a candidate batch into device memory once, evaluate or trial weights repeatedly, retain selected artifacts to disk, and release the temporary lease.

## Markets and execution models

| Market | Target backtests (`model="targets"`) | Order accounts (`model="orders"`) |
| --- | --- | --- |
| US equities (`us_equity`) | Daily bars, historical membership, adjusted execution prices, fees & borrow | Not supported |
| China A-shares (`a_share`) | Daily bars, publication vintages, suspensions, price limits & sell stamp tax | Not supported |
| Crypto spot (`crypto_spot`) | Daily or explicit base bars, historical eligibility & fees | Market/limit/stop orders, cash reservations & per-member inventory |
| USDT linear perpetuals (`crypto_perpetual`) | Daily/hourly archive or explicit base bars, funding & isolated margin | Signed per-member positions, funding, margin reservations & liquidation |

`Execution(model="targets")` is the default for formulas and equity-fraction arrays. `Execution(model="orders")` runs formulas, arrays, and programs through the crypto Bar account engine and replays accounts from scratch whenever weights or portfolio members change (for example, 5-minute base bars, 15-minute decisions, and daily statistics). [Execution and portfolio composition](guides/research.md#execution-and-portfolio-composition) covers shared versus independently funded accounts.

Both execution models use completed bar OHLC paths and volume participation caps. Neither models order-book queue position or tick-level fills.

## Three inputs, one session

```python
report = session.evaluate("gt(ma_dist_20,0)")
report = session.evaluate(registered_model_outputs)
report = session.evaluate(strategy_program)
```

The third call requires `Execution(model="orders")` and a program implementing `StrategyProgram`. Programs receive copied causal feature windows and live account snapshots, opening fresh private state on every query; see [Strategy programs and deployed models](guides/programs.md) for deployment and lease rules.

Scores map to positions via `Sizing`; direct targets declare `equity_fraction` (plus `quantity` or `notional` in order sessions). Arrays align to `session.grid` and declare observation and fitting cutoffs; see [Arrays and model outputs](guides/research.md#arrays-and-model-outputs).

## Bring your own data

Convert CSV exports, database tables, or vendor feeds into the documented Parquet layouts:

```python
with foundry.open(
    market="us_equity", data_dir="/datasets/my_us_equity",
    start="2025-07-01", end="2025-12-31", backend="cuda",
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

US equities use `panel.parquet` and historical `membership.parquet`. Intraday crypto datasets provide publication-aware `bars.parquet` and `manifest.json` (plus dense `funding.parquet` for perpetuals). [Data formats and conversion examples](guides/data.md) document columns, units, time bounds, eligibility, and provenance disclosure. The [bundled snapshots](data/README.md) serve as offline test fixtures.

## Linux, WSL2, and remote clients

Computation runs on Linux (including WSL2 on Windows). Install Python 3.12 or 3.13, Python development headers, and GCC with C++17 and OpenMP support, then run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group runtime
python -m foundry --help
```

<details>
<summary>Build native extensions (Linux / WSL2 / Colab)</summary>

Run from the repository root with the Linux Python interpreter used for computation. CuPy compiles CUDA kernels when a CUDA session opens. Native Windows remote clients do not need local compilation.

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

Dependencies are declared in [pyproject.toml](pyproject.toml) as [pip dependency groups](https://pip.pypa.io/en/stable/user_guide/#dependency-groups) (`pip>=25.1`):

| Environment | Install command |
| --- | --- |
| CPU research or remote client | `python -m pip install --group runtime` |
| CUDA research | `python -m pip install --group runtime --group cuda13` |
| CUDA GFlowNet mining | `python -m pip install --group mining --group cuda13` |

Select either `cuda12` or `cuda13` to match the host CUDA version; do not install both. GPU GFlowNet mining requires a CUDA-enabled PyTorch build (`mining` includes `runtime`; standard evaluation does not require PyTorch). Pass `backend="cpu"` for CPU sessions.

To use a remote client (including native Windows Python), start a service on the Linux compute host with an explicit portfolio path:

```python
import foundry

foundry.serve("var/runs/library.json", backend="cuda")
```

Query its loopback port on that host with `python -m foundry status`. Forward the port over SSH when connecting from another machine:

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

Closing a client session leaves the service running. [Remote sessions](guides/research.md#remote-sessions) upload arrays as bounded NumPy binary bundles (1 GiB limit, no `pickle`) and reference deployed programs by digest rather than sending source code.

## Portfolio review and submission

```python
with foundry.open("var/runs/my_portfolio.json") as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

Evaluation is read-only. Each report binds the study snapshot and portfolio version; if the portfolio has changed, `commit` raises `ContextConflict` and requires an explicit `refresh()` and re-evaluation. `row.ok` indicates that evaluation succeeded; `row.decision.passed` is the portfolio admission decision.

| Task | Call |
| --- | --- |
| Inspect fields, operators, clocks, and execution plan | `session.describe()` |
| Read resident returns and a committable report in one query | `session.evaluate_device(inputs)` |
| Register, retain, or release candidates | `session.register(...)`, `retain(...)`, `release(...)` |
| Diagnose portfolio gaps or evaluate parameter grids | `session.diagnose()`, `judge_family(...)` |
| Trial or update portfolio member weights | `session.trial_weights(...)`, `commit_weights(...)`, `reweight(...)` |
| Score candidate batches for search | `foundry.Discovery(session).evaluate(candidates)` |
| Run built-in or external proposer loops | `foundry.mine(output, ...)` |
| Fit and register causal walk-forward predictions | `foundry.walk_forward(session, fit, predict, ...)` |

`foundry.schema()` exports the machine-readable parameter and JSON-RPC schemas without opening data or devices. See the [Research guide](guides/research.md) for contract details and [Benchmarks](guides/benchmarks/README.md) for measured hardware, workloads, and numerical bounds.
