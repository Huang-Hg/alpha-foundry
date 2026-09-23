# alpha-foundry — portfolio-first formulaic alpha research

[![CI](https://github.com/Huang-Hg/alpha-foundry/actions/workflows/ci.yml/badge.svg)](https://github.com/Huang-Hg/alpha-foundry/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English** | [中文](README.zh-CN.md)

Formulaic alphas are written (by people, language models or a search policy), evaluated on the GPU in batches, and
judged only by what they add to a portfolio: every candidate is tried as an addition to the current portfolio on a
grid of weights, with real blended holdings and trading costs. One execution kernel serves every market; a market is
a plugin that declares its data fields and compiles its trading rules into arrays.

## Architecture

```
foundry/
  core/        Panel (fields as components) and the one set of return statistics
  ops/         the operator table (registry.py) + backends: CPU C++ reference, CUDA, portable numpy
  expr/        formula trees, the FastExpr parser/validator, evaluation on any backend
  backtest/    ExecutionSpec + the execution kernel (C++ reference, CUDA batch mirror)
  market/      crypto (Quantiacs daily), us_equity, a_share: fields, execution rules, universe, calendar
  engine.py    a market's research window resident on the GPU: formula -> book -> backtest -> statistics
  portfolio/   library, reference portfolio, admission, hypothesis families
  search/      portfolio-conditioned GFlowNet (C++ extension) and the mining loop
  service/     resident evaluation service (local or over SSH) and its client
  monitor/     terminal dashboard: typed status schema, components, layouts; HTML preview from the same frames
  __main__.py  python -m foundry <command>
config/config.ini   the single source of tunable parameters (one section per dataclass in foundry/config.py)
tests/              CPU tests run in CI; data and GPU tests run locally
```

Dependencies point one way: `core ← ops ← expr ← backtest/market ← engine ← portfolio ← search/service ← CLI`;
the dashboard reads status files only.

### One kernel, many markets

The execution kernel trades yesterday's target weights at today's open and marks to today's close (the Quantiacs
`calc_relative_return` recursion). Markets differ only in the arrays it reads:

| rule | crypto (Quantiacs) | US equities | A-shares |
|---|---|---|---|
| cost per share (buy / sell) | ATR14 × 0.04 / same | fee × open / same | (commission + transfer) × open / + stamp tax |
| blocked direction | – | – | limit-up open: no buy; limit-down open: no sell; suspension: neither |
| financing (`carry`) | – | borrow on shorts | – |
| Sharpe | arithmetic, 365 days | geometric, 252 | geometric, 242 |

With symmetric costs and no blocks or carry, the kernel reduces exactly to the Quantiacs recursion; the CUDA kernel
runs that case in batches (up to 128 assets), the CPU kernel runs every case.

### Panel components

A `Field` declares a name, a category, whether formulas may read it, and — if derived — its dependencies and how
to compute it with the panel's operator backend (numpy or cupy). A market composes its source fields with shared
derived ones (price/volume primitives, trend); new data is a new field definition.

## Usage

```bash
pip install -r requirements.txt
bash build.sh                        # native extensions: ops/_ops, backtest/_bt, search/_gfn (Linux .so, WSL/msys2 .pyd)

python -m foundry fields             # the active market's formula vocabulary
python -m foundry eval "gt(ma_dist_20, 0)" --file ideas.txt     # admission against the current portfolio
python -m foundry family hypotheses.txt                          # judge each formula's neighbourhood
python -m foundry mine runs/gfn_01                               # machine mining (portfolio objective)
python -m foundry monitor                                        # dashboard
```

The market in use is `[market] name` in `config/config.ini`. Market data (parquet panels) is not part of this
repository.

## Tests

```bash
python -m unittest tests.test_expr tests.test_native tests.test_policy tests.test_monitor   # CPU (CI)
python -m unittest tests.test_markets tests.test_portfolio tests.test_production tests.test_search   # data + GPU
```

## References

- Malkin et al., *Trajectory balance: improved credit assignment in GFlowNets* (NeurIPS 2022) — the policy's loss.
- Treynor & Black (1973) — the appraisal-ratio frontier behind the single-alpha objective.
- Heidari & Nikolopoulos, **FACT** (arXiv:2604.26666) — subgraph identification and verified composition, applied
  to the fused element-wise operators and the fused portfolio blending in the backtest kernel.
- Quantiacs — the crypto daily execution and scoring conventions the kernel reproduces.
