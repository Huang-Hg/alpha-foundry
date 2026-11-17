# Foundry performance and correctness

[简体中文](README.zh-CN.md) · [Overview](../../README.md) · [Research guide](../research.md)

The [L4 consecutive admissions and reallocation report](program-transitions-l4-20261117/README.md) measures the end-to-end workflow when portfolio state changes, separating consecutive admissions from explicit two-member historical reallocation fixtures.

| Historical measurement | Scope |
| --- | --- |
| [Review reuse](program-commit-l4-20261117/README.md) | All candidates rejected: 256 programs, query + commit `42.301 → 20.851 s`; does not extrapolate to consecutive admissions |
| [Temporal strategies](timed-l4-20261116/README.md) | 4096 arrays, 256 order programs, and preserved failed single-FP32 controls |
| [Fixed CPU](parallel-cpu-20261116/README.md) | Fixed Colab CPU runtime with 8-worker candidate and weight-trial parallelism |
| [External frameworks](frameworks-pro6000-20261115/README.md) | Foundry, Spectre, vectorbt, and Freqtrade on a restricted spot SMA contract |
| [66 operators](ops-pro6000-20261115/README.md) | Per-operator timing/error across 124 window cases and 4096-formula workflows |

Tests and benchmark drivers stay local in untracked `tests/` and `scripts/`. On an authorized Linux GPU host, install `--group mining` plus the matching CuPy group (`cuda12` or `cuda13`), install `pytest`, `pytest-benchmark`, and `pytest-html` separately in the test environment, compile the native extensions using the root [README](../../README.md#linux-wsl2-and-remote-clients) commands, and run `python -m pytest -c tests/pytest.ini`.

GitHub Actions verifies saved summaries and bundled data checksums without running numerical code. Each report records hardware, scale, precision, stage timings, errors, and decision parity; raw logs and archives stay in local `var/runs/`. Command lines shown in historical reports reflect their recorded revisions (`foundry benchmark` is no longer a public CLI subcommand).
