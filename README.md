# Foundry

**From an idea or a model output to a reproducible portfolio decision.**

[简体中文](README.zh-CN.md) · **English** · [Research guide](guides/research.md)

Foundry is a Linux research and backtesting engine for people, AI agents and automated miners. Bring a formula, an ML signal or a batch of target positions; evaluate them under explicit market rules, compare portfolio improvements and submit through the same admission process.

## Why Foundry?

- **Your strategy does not have to be a formula.** Python/CUDA programs, ML models and external miners can supply batches of signals or target positions. Formula search is one producer among several.
- **One decision path for every caller.** Python and discoverable JSON-RPC interfaces share evaluation, weight search and portfolio admission. Agents receive structured metrics and reason codes.
- **Batch research on CPU or CUDA.** Keep arrays on the GPU across numerical work, reuse immutable study data and evaluate candidates in batches. Reduced precision is an explicit execution policy, checked against the CPU reference.
- **Time and market rules are part of the model.** Snapshots bind data, execution rules and numerical policy. Features use information available at the decision time; a daily close signal executes no earlier than the next open.
- **Queries and portfolio changes are separate.** Evaluation leaves the portfolio unchanged. A commit checks the expected version and publishes one complete replacement.

## Markets

| Market | Modeled behavior |
| --- | --- |
| US equities | Daily signals, eligibility, transaction and holding costs |
| A shares | Historical eligibility, suspensions, price limits, next-day execution and sell-side tax |
| Crypto spot | UTC daily bars, continuous calendar, 365-day annualization and trading costs |
| USDT linear perpetuals | Signed exposure, isolated collateral, realized funding, leverage and hourly mark-price liquidation |

Perpetuals currently use a research model with fixed fee/maintenance parameters and hourly adverse-price touches. It does not reconstruct historical risk tiers or the exchange's intrabar execution. Data contracts disclose missing publication vintages and other source limits. See the [model and interface guide](guides/research.md).

## Get started

Use Linux, WSL2 or Colab with Python 3.12/3.13, Python development headers and GCC with C++17/OpenMP:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash build.sh
python -m foundry --help
```

Choose the market, data directory, dates and backend in [config/config.ini](config/config.ini). Market datasets and private portfolios are not bundled. CPU execution needs no GPU; CUDA additionally needs CuPy matching the installed CUDA major.

To build a public perpetual dataset, configure `[crypto_perpetual]` and run:

```bash
python -m foundry import-perpetual data/crypto_perpetual
```

The importer preserves Binance source archives, verifies official checksums and publishes an immutable dataset manifest. Set `[market] name = crypto_perpetual` to research that dataset.

## Connect your program or agent

```python
from foundry.config import load_config
from foundry.runtime import configured_study, open_runtime
from foundry.research import Research
from foundry.research_types import Candidate
from foundry.candidates import Formula
from foundry.adapters.arrays import MemoryArrayStore
from foundry.adapters.library import MemoryLibraryStore
from foundry.portfolio.library import Library

settings = load_config()
study = configured_study(settings)
with open_runtime(settings, study) as runtime:
    research = Research(runtime.engine, MemoryLibraryStore(Library((), study)),
                        MemoryArrayStore())
    candidates = (Candidate("trend", Formula("gt(ma_dist_20,0)")),)
    result = research.evaluate(candidates, research.context.id)
```

For model outputs, register an `ArrayBatch` with `Research.register_batch`, then use the same `evaluate` and `commit` calls. Batches carry ordered dates/assets, a snapshot identity and observation/fitting cutoffs. See [batch input and RPC details](guides/research.md#candidate-interface).

Agents can export the machine contract with `python -m foundry schema output/schema.json`; CLI help is the authoritative command list.

## Engineering approach

Python owns research and integration; C++ and CUDA own the numerical kernels. A modular monolith keeps market semantics independent of transport, storage and device choice. Tests cover PIT behavior, market rules, numerical/decision agreement and atomic state changes. Performance claims distinguish setup, resident computation and complete use cases.

Foundry supplies research infrastructure. A passing backtest or admission decision is not evidence of out-of-sample profitability.
