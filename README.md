# Foundry

**CUDA-accelerated strategy discovery and backtesting — for people, AI agents and mining programs.**

[简体中文](README.zh-CN.md) · [Usage guide](guides/research.md) · [Data](data/README.md)

Turn a formula, an ML signal or a target portfolio into a comparable backtest and an explicit portfolio admission decision. Foundry handles the data context, market rules, batch execution and portfolio state through one Python interface.

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

The example uses the bundled US equity snapshot. You can replace it with your own data and keep the same calls.

## What Foundry gives you

- **CUDA batching across the research workflow.** Evaluate candidate batches, simulate accounts and compare portfolio weights with resident arrays. Batch size and precision are explicit parameters; CPU is also a supported backend.
- **FP32 with residual precision where it matters.** Paired FP32 arithmetic supports account and statistical calculations without a result-triggered precision replay. A session fixes its numerical plan before evaluation.
- **More than formula mining.** Use the 66-operator formula vocabulary, register outputs from Python/CUDA or ML models, or plug an external candidate generator into discovery. Agent-led research and manual experiments use the same candidates and reports.
- **Point-in-time research.** Historical eligibility, declared observation and fitting cutoffs, next-bar execution and causal walk-forward folds help keep future information out of decisions. Data provenance records disclose missing publication history.
- **A complete path from idea to portfolio.** Diagnose the current portfolio, test a candidate family, evaluate weight changes and submit improvements through shared admission rules. Queries and portfolio changes are separate operations.

## Four markets, one interface

| Market | Execution model |
| --- | --- |
| US equities | Historical membership, adjusted prices, next-session execution and costs |
| China A-shares | Historical membership, suspensions, price limits, sell tax and separate signal/execution prices |
| Crypto spot | Continuous UTC calendar, independent bar clocks, eligibility and trading fees |
| USDT linear perpetuals | Long/short exposure, isolated margin, base-bar funding events and liquidation |

Crypto supports explicit base bars with independent decision and statistics frequencies, including 5-minute execution, 15-minute decisions and daily statistics. Existing daily presets and hourly perpetual datasets retain their conventions. Stateful strategy callbacks and orders are later delivery phases. The [guide](guides/research.md#crypto-execution) describes the model's price, funding and liquidation assumptions.

## Bring your own data

Your CSV, database or vendor export can supply the market tables. The bundled snapshots are examples, not a required provider.

```python
with foundry.open(
    market="us_equity",
    data_dir="/datasets/my_us_equity",
    start="2025-07-01",
    end="2025-12-31",
    backend="cuda",
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

For US equities, prepare `panel.parquet` with OHLCV and adjustment factors, plus `membership.parquet` with historical eligibility intervals. Retain earlier history for rolling features. Other markets have explicit data contracts; intraday crypto accepts publication-aware base bars, with mark and funding records for perpetuals. See [file formats and conversion examples](guides/research.md#bring-your-own-data).

If your program already produces signals or positions, use `session.register(...)` instead. It accepts batches of model scores, long-only target weights or signed perpetual exposures with UTC decision/asset coordinates, explicit units and timestamp information cutoffs. Scores use `Sizing(rule="equal_slots")` by default; `Sizing(rule="independent", scale=...)` avoids asset-count dilution. Targets use `kind="targets", unit="equity_fraction"`. Temporary GPU candidates stay resident until `session.release(batch)`; only admitted, checkpoint-referenced or explicitly retained outputs are saved.

## Start on Linux or Windows

Local computation runs on Linux or Windows through WSL2. Native Windows Python can use the remote client without building the numerical extensions.

In Linux/WSL2, install Python 3.12 or 3.13, its development headers, and GCC with C++17/OpenMP support, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash build.sh
python -m foundry --help
```

CUDA also needs an NVIDIA driver and the CuPy package matching your CUDA major version. Use `backend="cpu"` for CPU computation.

For a service already running on a compute host, a Windows or Linux client uses:

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

Here `port` is the service port, or the local end of your tunnel. See [remote sessions](guides/research.md#remote-sessions).

## A small API for the whole workflow

| Task | Call |
| --- | --- |
| Evaluate formulas or registered model outputs | `session.evaluate(inputs)` |
| Register signal or target batches | `session.register(values, ...)` |
| Save selected outputs / release a temporary batch | `session.retain(inputs)` / `session.release(batch)` |
| Inspect portfolio weaknesses | `session.diagnose()` |
| Check formula neighbours or a model parameter grid | `session.judge_family(inputs)` |
| Inspect holdings or compare component weights | `session.portfolio()` / `session.trial_weights(matrix)` |
| Submit candidates or a reviewed weight trial | `session.commit(report)` / `session.commit_weights(trials, index)` |
| Optimize and submit admissible portfolio weights | `session.reweight(...)` |
| Score candidate proposals for discovery | `foundry.Discovery(session).evaluate(candidates)` |
| Run built-in or external proposers | `foundry.mine(output, ...)` |
| Fit causal model folds and register predictions | `foundry.walk_forward(session, fit, predict, ...)` |

`foundry.Candidate`, `ExternalProposer` and `CandidateGrid` connect your own programs to discovery without requiring a formula representation. The [integration guide](guides/research.md#strategy-discovery) includes examples for each workflow.

The same `foundry.mine` call owns producer lifetimes and batch feedback; external models can consume GPU-resident scores without knowing the backtest internals.

Built-in GFlowNet mining learns which formulas to propose using a Python/PyTorch model. With `backend="cuda"`, policy inference and training run on the GPU alongside accelerated strategy evaluation; every proposed strategy still passes the same portfolio admission rules.

Evaluation leaves the portfolio unchanged. To submit a reviewed batch:

```python
with foundry.open("var/runs/my_portfolio.json") as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

## Simple calls, adjustable parameters

Tune market costs, leverage, CUDA batching, admission and mining through arguments. There is no configuration file to maintain.

```python
with foundry.open(
    market="crypto_perpetual",
    account=foundry.Account(margin="isolated", leverage=3.0, gross_limit=3.0),
    cost_rate=0.0004,
    cuda=foundry.CUDA(candidate_batch=64, precision="paired"),
    admission=foundry.Admission(min_sharpe_gain=0.02),
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

`foundry.schema()` exposes the parameter and RPC schemas for agents. `session.describe()` lists the current study's fields, operators and resolved parameters. Open another session to compare settings.

## Performance evidence

Historical measurement from before the R1 protocol migration; this is not R1 validation evidence. Evaluation + ordered commit for 4096 formulas over 269 days × 500 US equities. The CPU reference is one fixed, independent Colab CPU runtime (EPYC 7B12).

| Backend | Time |
| --- | ---: |
| CPU 1 worker | 215.660 s |
| CPU 8 worker | 106.645 s |
| A100 | 6.691 s |
| L4 | 7.809 s |

Actual CPU candidate/trial parallelism improves throughput 2.02× with identical worker reports; all four runs admit 28 candidates. CPU was measured in this run; GPU measurements reuse the same data and rules. One warm-up and one measured round per condition. See the [benchmark report](guides/benchmarks/parallel-cpu-20261116/README.md) for stages, errors, memory bounds and scope.

[Research and integration](guides/research.md) · [Available data](data/README.md) · [Benchmark reports](guides/benchmarks/README.md)
