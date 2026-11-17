# Benchmarks

[简体中文](README.zh-CN.md) · [Overview](../../README.md) · [Research guide](../research.md)

Saved hardware timings and numerical parity reports across CUDA and CPU backends. Each directory records its hardware, dataset, precision policy, stage timings, and error bounds in `summary.json`.

| Report | Workload |
| --- | --- |
| [Order accounts (L4)](order-accounts-l4-20261117/README.md) | Per-fill maker/taker fees, USDT cross margin, and US/A-share intraday cash orders |
| [Consecutive admissions & reallocation (L4)](program-transitions-l4-20261117/README.md) | End-to-end stateful program evaluation, consecutive commits, and two-member reweighting |
| [Program commit reuse (L4)](program-commit-l4-20261117/README.md) | 256 stateful programs when `ResearchContext` is unchanged (`42.301 → 20.851 s` query + commit) |
| [Timed strategies (L4)](timed-l4-20261116/README.md) | 4,096 arrays and 256 order programs on 5-minute base bars, plus single-FP32 failure controls |
| [Parallel CPU (Colab CPU)](parallel-cpu-20261116/README.md) | 1-worker vs. 8-worker candidate evaluation and weight trials on 4,096 US equity formulas |
| [Framework comparison (PRO 6000)](frameworks-pro6000-20261115/README.md) | Foundry vs. Spectre, vectorbt, and Freqtrade on a daily crypto-spot SMA pulse ([method](../framework-benchmark.md)) |
| [66 operators (PRO 6000)](ops-pro6000-20261115/README.md) | Per-operator CPU/CUDA timings across 124 window cases and 4,096-formula workflows |
