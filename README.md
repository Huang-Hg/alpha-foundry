# Foundry

**CUDA-accelerated quantitative research and backtesting.**

[简体中文](README.zh-CN.md) · **English** · [Research guide](guides/research.md)

Use the GPU to evaluate formulas, backtest strategies and select portfolios in batches. Foundry helps researchers, AI agents and miners explore many candidates using their own market data, ML signals or target positions. A CPU backend is also available. Run computation on Linux or Windows through WSL2, or connect from native Windows Python to a remote compute service.

```python
import foundry

with foundry.open() as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    print(report.rows[0].metrics)
```

The defaults use bundled US equity data and CUDA. Set data, dates and computation parameters in the call; the session loads data, prepares computation and releases resources.

## Why Foundry?

- **CUDA acceleration across the research workflow.** Batch formula operators, account simulation, metrics and portfolio weight trials on the GPU, reusing data and resident results. Evaluate many candidates together instead of waiting for separate backtests.
- **Formulas, models and custom programs.** Use 66 operators for arithmetic, rolling statistics, ranking and regression, or submit signals, target weights and perpetual exposures from an existing model or miner.
- **Your data and research universe.** Convert daily market data to the documented format and choose your assets and period. Bundled data makes it easy to start; changing providers does not require rewriting numerical kernels.
- **From individual strategies to portfolio selection.** Compare net returns, risk and weights after adding a candidate to the portfolio. Explicit admission reasons reduce the work of building a separate selection pipeline.
- **Quick access for people and agents.** One session handles evaluation, neighbourhood exploration, model registration, weight trials and submission. Local and remote sessions share these operations and expose machine-readable capabilities and results.

### Measured on a complete research workflow

**500 US equities × 269 days, 4096 formulas: a complete call takes 97.69 seconds on CPU and 2.95 seconds on CUDA on the same Colab PRO 6000, about 33.1× faster.** This includes session startup, formula evaluation, backtest metrics, portfolio admission and resource release. Both admit the same 28 candidates.

A single formula takes about 0.09 seconds on either backend; at 64 formulas CUDA is about 6.8× faster. Benefits depend on batch size. These are single warmed measurements with an in-memory portfolio, excluding installation, native builds, RPC and disk persistence. CPU uses 2 OpenMP threads. See the [performance, error and decision report](guides/benchmarks/current/report.md).

## Bring your own market data

**Foundry is not tied to the bundled data or a particular vendor.** For daily equities and crypto spot, convert data from your CSV files, database or API into the supported Parquet tables. Supply the data directory and study period when opening a session:

```python
import foundry

with foundry.open(
    market="us_equity",
    data_dir="./data/my_us_equity",
    start="2025-01-01",
    end="2025-12-31",
    backend="cuda",
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

Your US equity directory supplies `panel.parquet` with daily prices/volume and adjustment factors, plus `membership.parquet` with historical eligibility intervals. A-shares additionally separate signal and execution prices; crypto spot records daily eligibility. Preserve history before the research start for rolling features.

**[Custom data guide: files, columns, CSV conversion and provenance](guides/research.md#bring-your-own-data)**

The repository also includes [ready-to-use test snapshots](data/README.md) for all four markets. They provide a starting point for exploring the engine; you can choose your own assets and research period within your dataset's coverage.

For perpetuals, the dataset must also provide hourly trade/mark prices, settled funding and the manifest/evidence required by the adapter. The built-in Binance importer creates this format; see the [perpetual data instructions](guides/research.md#perpetual-data).

## Markets

| Market | What the engine models |
| --- | --- |
| US equities | Daily signals, historical eligibility, adjusted prices and trading costs |
| A-shares | Historical eligibility, suspensions, price limits, next-day execution and sell-side tax |
| Crypto spot | UTC daily bars, continuous calendar, 365-day annualization and trading costs |
| USDT linear perpetuals | Long/short exposure, isolated collateral, settled funding, leverage and hourly mark-price liquidation |

Daily close decisions execute no earlier than the next open. Perpetuals use fixed fee/maintenance parameters and hourly adverse-price touches; historical risk tiers and intrabar exchange execution are outside the current model. Detailed rules and data limits are in the [research guide](guides/research.md).

## Get started

Run computation on Linux, Colab or Windows through WSL2. Inside that environment, use Python 3.12/3.13, Python development headers and GCC with C++17/OpenMP:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash build.sh
python -m foundry --help
```

Use `foundry.open(backend="cuda")` for GPU execution or `backend="cpu"` for CPU execution. CUDA also requires CuPy matching the installed CUDA major. Inspect the active fields, operators and resolved parameters with `session.describe()`.

Native Windows Python can also connect to an existing service without compiling numerical kernels or installing CUDA on the client. Install the Python dependencies and use `foundry.connect(("127.0.0.1", port))` for a local or tunnel-forwarded service. Obtain its port with `python -m foundry status` on the server.

## One interface, from evaluation to a portfolio

Evaluate a batch, then explicitly submit the report for portfolio admission. Supply a portfolio path to save and reopen it; omit the path for an in-memory study:

```python
import foundry

with foundry.open("production/my_portfolio.json") as session:
    report = session.evaluate({
        "trend": "gt(ma_dist_20,0)",
        "longer_trend": "gt(ma_dist_60,0.05)",
    })
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

| Task | Call |
| --- | --- |
| Evaluate one or many formulas | `session.evaluate(formulas)` |
| Register model outputs | `session.register(values, ...)` |
| Explore neighbouring parameters and fields | `session.judge_family(formulas, hypothesis="trend")` |
| Inspect the current portfolio | `session.portfolio()` |
| Compare batches of portfolio weights | `session.trial_weights(weights)` |
| Admit candidates or change weights | `session.commit(report)` / `session.commit_weights(trials, index)` |
| Export a report | `foundry.export(report, path)` |
| Run a mining job | `foundry.mine(output, ...)` |

For an ML model or external miner, register a batch through `session.register` and pass the returned batch to the same `evaluate` and `commit` methods:

| Input | Use it for |
| --- | --- |
| `signals` | Model scores that use Foundry's signal-to-position conversion |
| `target_weights` | Your own long-only equity or spot allocation |
| `target_exposure` | Signed notional/equity targets for perpetuals |

`session.grid` supplies date and asset coordinates. You declare observation/fitting cutoffs; the session manages the remaining research context. Remote registration transfers arrays automatically without exposing server filesystem paths. See the [public Python API](guides/research.md#public-python-api) for complete examples.

Agents can discover the machine protocol through `foundry.schema()` and available fields/operators through `session.describe()`. Replace `foundry.open()` with `foundry.connect(...)` to use the same research operations on a remote service.

## Simple calls, adjustable parameters

Pass common parameters directly and use parameter groups for batching, precision, admission and mining. No configuration file or source edit is needed:

```python
with foundry.open(
    market="crypto_perpetual",
    leverage=3.0,
    cost_rate=0.0004,
    cuda=foundry.CUDA(candidate_batch=64, precision="paired"),
    admission=foundry.Admission(min_sharpe_gain=0.02),
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

Parameters are fixed when a session opens; open another session to compare settings. Discover every parameter and its defaults through `foundry.schema()`. The [usage guide](guides/research.md#public-python-api) covers model inputs, weight adjustment and remote calls.

[Data snapshots](data/README.md) · [Research and integration](guides/research.md) · [Performance and accuracy](guides/benchmarks/README.md)
