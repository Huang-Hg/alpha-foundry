# Foundry

**CUDA strategy research, from an idea to an actual account and a reviewed portfolio.**

[简体中文](README.zh-CN.md) · [Research guide](guides/research.md) · [Your own data](guides/data.md) · [Strategy programs](guides/programs.md)

Bring a formula, an array from your model, or a program that reacts to real simulated fills. Foundry gives people, AI agents and discovery loops the same session, market rules, reports and portfolio admission decisions.

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

This runs against the included US equity snapshot. Replace `data_dir` and dates to research your own market history.

## Research at GPU scale

CUDA batches formula operators, account simulations, statistics and portfolio weight trials while keeping working arrays resident. It is useful when exploring many candidates over the same market panel. Stateful CUDA programs can keep their private tensors on the selected device; Python callbacks are not automatically compiled into GPU kernels.

A session fixes its device and numerical plan before work begins. Paired FP32 arithmetic retains residual precision in supported account and statistical paths. CPU is a supported backend, with explicit worker and memory limits.

- **Three strategy representations:** a 66-operator formula vocabulary, numerical scores or targets, and stateful order programs.
- **Causal clocks:** separate source bars, decision times, next-base-open execution and statistics periods; observation and fitting cutoffs remain explicit.
- **Actual account feedback:** orders, partial fills, cancellations, fees, cash, positions and isolated margin drive the next strategy decision.
- **One discovery workflow:** diagnose gaps, score proposals, examine parameter families, replay portfolio trials and submit candidates through hard admission rules.
- **Bounded reuse:** register a model or array batch once, evaluate it repeatedly, retain selected references and release the lease.

## Choose the execution you need

| Market | Target backtests | Actual order accounts |
| --- | --- | --- |
| US equities | Daily, historical membership, adjusted execution prices and costs | Not currently supported |
| China A-shares | Daily, membership, suspensions, price limits and sell tax | Not currently supported |
| Crypto spot | Daily or explicit base bars, historical eligibility and fees | Market/limit/stop orders, cash and owner inventory |
| USDT linear perpetuals | Daily/hourly archive or explicit base bars, funding and isolated margin | Signed owner positions, funding, reservations and liquidation |

`Execution(model="targets")` is the default for formulas and precomputed equity-fraction targets. `Execution(model="orders")` puts formulas, arrays and programs through the crypto Bar account model and fresh portfolio replay. For example, use 5-minute source bars, 15-minute decisions and daily statistics. [Execution and portfolio semantics](guides/research.md#execution-and-portfolio-composition) explain shared accounts and independently funded accounts.

These are Bar models with a declared OHLC path and volume participation. They do not model order-book queue position or tick-by-tick execution.

## Three inputs, one session

```python
report = session.evaluate("gt(ma_dist_20,0)")
report = session.evaluate(registered_model_outputs)
report = session.evaluate(strategy_program)
```

The last call requires an order session and a program implementing the public protocol. Programs receive causal feature windows and actual accounts, and create fresh private state for every query. [A complete public program example](guides/programs.md) includes artifact deployment, remote references and ownership rules.

Scores use explicit sizing; direct targets declare `equity_fraction`, or `quantity`/`notional` in order sessions. Arrays align with `session.grid` and declare when their observations and fitting information became available. See [array registration](guides/research.md#arrays-and-model-outputs).

## Use your own data

Convert a CSV, database or vendor export to the documented Parquet contracts:

```python
with foundry.open(
    market="us_equity", data_dir="/datasets/my_us_equity",
    start="2025-07-01", end="2025-12-31", backend="cuda",
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

US equities need `panel.parquet` and historical `membership.parquet`. Crypto intraday datasets provide publication-aware base bars and a manifest; perpetuals also provide mark prices and explicit funding records. [Data formats and conversion examples](guides/data.md) cover fields, units, time boundaries, eligibility and provenance. The [included snapshots](data/README.md) are useful examples and require no live market download.

## Run on Linux, WSL2 or a remote GPU

Compute runs on Linux, including WSL2 on Windows. Install Python 3.12 or 3.13, its development headers and a C++17/OpenMP compiler, then run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group runtime
bash build.sh
python -m foundry --help
```

All dependencies are declared once in [pyproject.toml](pyproject.toml), using [pip dependency groups](https://pip.pypa.io/en/stable/user_guide/#dependency-groups) (pip 25.1+). Select only the capabilities you need:

| Environment | Install from the repository root |
| --- | --- |
| CPU research or a remote client | `python -m pip install --group runtime` |
| CUDA research | `python -m pip install --group runtime --group cuda13` |
| GFlowNet mining on CUDA | `python -m pip install --group mining --group cuda13` |
| Cloud regression and benchmarks | `python -m pip install --group benchmark --group cuda13` |
| Static development tools | `python -m pip install --group dev` |

CUDA requires an appropriate NVIDIA driver. Choose `cuda12` or `cuda13` to match the compute host's CUDA major; install only one CuPy group. GPU GFlowNet mining needs a CUDA-enabled PyTorch wheel. The `mining` and `benchmark` groups include runtime dependencies; ordinary research does not require PyTorch. Use `backend="cpu"` when opening a CPU session.

Native Windows Python can be the client without compiling the numerical extensions. On the compute host, start a service with an explicit portfolio path:

```python
import foundry
foundry.serve("var/runs/library.json", backend="cuda")
```

Find its port with `python -m foundry status` on that host. Forward the loopback port over SSH when using another machine, then connect from Windows or Linux:

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

The service owns computation. Closing a client does not stop it. [Remote sessions](guides/research.md#remote-sessions) also support deployed program references without shipping Python source or pickle objects.

## Review before changing a portfolio

```python
with foundry.open("var/runs/my_portfolio.json") as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

Evaluation is read-only. A report records the exact study and portfolio version; a stale submission requires an explicit refresh and reevaluation. `ok=True` means evaluation succeeded, while `row.decision.passed` is the admission conclusion.

| Research task | Public call |
| --- | --- |
| Inspect data, parameters and execution plan | `session.describe()` |
| Get resident returns and a committable report in one query | `session.evaluate_device(inputs)` |
| Register, keep or release candidates | `register(...)`, `retain(...)`, `release(...)` |
| Diagnose gaps or check neighbouring variants | `diagnose()`, `judge_family(...)` |
| Compare or submit portfolio weights | `trial_weights(...)`, `commit_weights(...)`, `reweight(...)` |
| Score candidate proposals | `foundry.Discovery(session).evaluate(candidates)` |
| Run built-in or external proposal loops | `foundry.mine(output, ...)` |
| Generate causal model-fold predictions | `foundry.walk_forward(session, fit, predict, ...)` |

`foundry.schema()` provides the machine-readable API. [The research guide](guides/research.md) covers these workflows; [benchmark reports](guides/benchmarks/README.md) distinguish measured workloads, hardware and numerical scope.
