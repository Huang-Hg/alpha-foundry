# Foundry performance and numerical benchmarks

[简体中文](README.zh-CN.md) · [Overview](../../README.md) · [Research guide](../research.md)

Each published directory captures a concrete hardware, dataset, and numerical-accuracy milestone. Latest end-to-end validation: [L4 consecutive admissions and reallocation](program-transitions-l4-20261117/README.md).

| Benchmark report | Measured scope |
| --- | --- |
| [L4 consecutive admissions & reallocation](program-transitions-l4-20261117/README.md) | Stateful program admissions and two-member weight reallocation with evolving portfolio context |
| [L4 review reuse on commit](program-commit-l4-20261117/README.md) | 256 stateful programs when all candidates are rejected: query + commit `42.301 s → 20.851 s` (`2.03×`) |
| [L4 temporal strategies](timed-l4-20261116/README.md) | 4096 intraday array candidates, 256 order programs, and preserved single-FP32 failure controls |
| [Fixed Colab CPU baseline](parallel-cpu-20261116/README.md) | Shared-memory 8-worker C++ candidate and weight-trial parallelism (`215.660 s → 106.645 s`) |
| [External backtest frameworks](frameworks-pro6000-20261115/README.md) | Foundry vs. Spectre, vectorbt, and Freqtrade under an identical spot SMA contract |
| [66-operator CUDA optimization](ops-pro6000-20261115/README.md) | Per-operator latency/accuracy across all 66 operators and 124 rolling-window cases |

## Running benchmarks on authorized compute

Regression suites and benchmark drivers stay local in untracked `tests/` and `scripts/`, executed via `pytest`, `pytest-benchmark`, and `pytest-html`. On an authorized Linux GPU host:

1. Install `--group benchmark` plus the matching CuPy group (`--group cuda12` or `--group cuda13`).
2. Compile the native C++ extensions using the command block in the root [README](../../README.md#installation-and-native-build).
3. Run the local `pytest` / `pytest-benchmark` suite into `var/runs/<run>/`.

Public directories under `guides/benchmarks/` keep only concise bilingual Markdown reports, CSV tables, and `summary.json` files. Raw logs, JUnit/HTML reports, `.npz` outputs, and source archives stay in local `var/runs/`. GitHub Actions validates published summaries and bundled data checksums without launching GPU instances. Note that historical reports may cite command lines from earlier revisions (such as the retired `foundry benchmark` CLI wrapper).
