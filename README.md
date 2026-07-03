# alpha-foundry — Formulaic Alpha Mining + Backtest Kernels

[![CI](https://github.com/Huang-Hg/alpha-foundry/actions/workflows/ci.yml/badge.svg)](https://github.com/Huang-Hg/alpha-foundry/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English** | [中文](README.zh-CN.md)

Search formulaic alphas with strongly-typed genetic programming (GP), evaluate them through pure C++ / OpenMP backtest kernels, and fuse them into portfolio weights via deterministic sizing. Three markets are supported: crypto perpetuals, US daily equities, and China A-share daily equities.

---

## How it works

End to end, the system is one pipeline — *formula search → backtest evaluation → pool building → sizing fusion* — and every layer is search-engine agnostic:

1. **DSL (`evaluation/`)** — Formulaic alphas are represented as strongly-typed ASTs. A context-free grammar (CFG + α-Sem-k cost budget) constrains the search space and guarantees syntactic/semantic validity. The operand vocabulary is **built automatically from parquet column names** (auto-classified into price / volume / feature), so switching markets just means switching columns. The evaluator compiles whole token trees to C (and optionally CUDA) for batch evaluation, with a subtree-hash cache reusing common subtrees.

2. **Search (`search/`)** — `gp/`: strongly-typed GP on DEAP with two NSGA-II objectives (net Sharpe + behavioral diversity), mining formulaic alphas on the train split. `alphasage/`: structure-aware GFlowNet mining (sampling P(α)∝reward; TB loss + entropy regularization, RGCN structural encoder, composite R_IC/R_SA/R_NOV reward). `tail_diversity/`: a prototype for tail-conditional diversity search (true vs. spurious diversity, implemented in isolation).

3. **Evaluation + pooling (`rl/`)** — The evaluator computes rank-IC and portfolio backtest returns; candidates pass a series of admission gates (|IC| / orthogonal contribution Δ / OOS holdout) into a `LinearAlphaPool`. The pool holds no fixed weights.

4. **Backtest kernels (`backtest/`)** — Pure C++ / OpenMP: `engine/` runs portfolio-level (T,S) continuous target-position backtests (fees / funding / trailing stop / liquidation); `ops/` is the C++ factor-operator library, `ops_cuda/` holds CUDA kernels (compiled at runtime via cupy NVRTC). The order book is not modeled.

5. **Sizing (`rl/sizing.py`)** — Deterministic fusion: z-score pool members → AFF causal rolling least-squares fusion → top-K long-short construction → portfolio-level vol-target scalar leverage.

6. **Market abstraction (`markets/`)** — `MarketProfile` isolates calendar / annualization / price-adjustment conventions from the engine; crypto / US equity / A-share are three profiles, with vocabulary and profile auto-detected from the data.

Module overview:

| Module | Responsibility |
|---|---|
| `evaluation/` | Formula DSL (AST / CFG / C+CUDA evaluation) + panel adapter + subtree-hash cache |
| `search/` | `gp` (strongly-typed GP on DEAP) + `alphasage` (GFlowNet) + `tail_diversity` (tail-diversity prototype) |
| `rl/` | Evaluator + LinearAlphaPool + sizing fusion |
| `backtest/` | C++/OpenMP backtest kernels (engine) + factor operators (ops C++ / ops_cuda CUDA) |
| `markets/` | MarketProfile abstraction (calendar / annualization / adjustment / vocabulary) |
| `config/` | Single source of tunable parameters (INI ↔ dataclass) |
| `main.py` | Research-side CLI |

---

## Usage

### Install + build extensions

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirement.txt   # numpy/pandas/pyarrow/torch/deap

bash backtest/build.sh           # build C++ extensions: backtest._bt (engine) + backtest.ops._ops (operators); Linux→.so / WSL→Windows .pyd
bash backtest/ops_cuda/build.sh  # CUDA operators: cupy NVRTC runtime compilation + verification against numpy golden reference (needs cupy + GPU)
```

The GPU research path (`[evaluator] device=cuda`) additionally requires `cupy`; for CPU set `device=cpu`.

### Mine alphas / materialize

`main.py` takes positional arguments only (output paths); everything else is read from `config/config.ini`:

```bash
python main.py gp-baseline output/pool.json                       # strongly-typed GP on DEAP (NSGA-II) — crypto perpetual mining
python main.py alphasage   output/pool.json                       # AlphaSAGE GFlowNet — crypto perpetual mining (independent baseline)
python main.py materialize output/pool.json output/materialized/  # alpha JSON → monthly per-alpha parquet
```

Mining output = a deploy-bundle JSON (`{"pool": [{tree, ...}, ...]}`) plus snapshots under `logs/`.

> This repository is a public snapshot: mining requires your own parquet panel matching the format read by `evaluation/crypto_adapter.py` (the data-collection pipeline is not included). `main.py` also has `gp-equity` / `gp-ashare` subcommands, which depend on equity data loaders not included in this snapshot.

---

## References / Acknowledgments

This project builds on the following work, gratefully acknowledged:

- **Alpha Discovery via Grammar-Guided Learning and Search** — the CFG grammar and α-Sem-k cost budget in `evaluation/grammar.py` are inspired by it.
- **FACT** (arXiv:2604.26666) — the operator fusion in `backtest/ops_cuda/` adopts its compositional kernel-synthesis paradigm.
- **NSGA-II** (Deb et al., 2002) + **DEAP** (Fortin et al., 2012) — the multi-objective strongly-typed GP in `search/gp`.
- **AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets** — the method behind the GFlowNet mining implementation in `search/alphasage`.
- **torchgfn** (Lahlou et al.) — the GFlowNet foundation library (TB loss / DiscreteEnv / Sampler).

References for the `search/tail_diversity/` prototype:
- Mallela & Leonelli, 2026 (arXiv:2606.16840) — lower-tail dependence graphs under crashes.
- Engelke & Taeb, JMLR 2025 (arXiv:2403.09604) — eglatent for removing common factors.
- Dominated Novelty Search, GECCO 2025 (arXiv:2502.00593) — unbounded-archive diversity.
- Campbell et al., 2026 (arXiv:2603.15963) — external coordinates for return tails.
