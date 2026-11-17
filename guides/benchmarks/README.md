# Foundry performance and correctness

[简体中文](README.zh-CN.md)

The [L4 admission and reallocation report](program-transitions-l4-20261117/README.md) measures the complete workflow when portfolio state changes and distinguishes actual consecutive admissions from explicit two-member historical fixtures.

| Historical measurement | Scope |
| --- | --- |
| [Review reuse](program-commit-l4-20261117/README.md) | All rejected: 256 programs, query/commit 42.301 → 20.851 s; not a consecutive-admission claim |
| [Temporal strategies](timed-l4-20261116/README.md) | 4096 arrays, 256 programs and preserved failed precision controls |
| [Fixed CPU](parallel-cpu-20261116/README.md) | Fixed Colab CPU, parallel candidate and weight trials |
| [External frameworks](frameworks-pro6000-20261115/README.md) | Spectre, vectorbt and Freqtrade under a shared trading contract |
| [66 operators](ops-pro6000-20261115/README.md) | Per-operator timing/error and complete formula workflows |

Tests and drivers stay local. Execution, timing and test reports use pytest, pytest-benchmark and pytest-html. Install `--group benchmark` plus the CuPy group matching the compute host, then use the root README build commands.

GitHub Actions verifies and displays saved results without numerical execution. New reports record hardware, scale, precision, complete workflow time, errors and decisions. Full raw artifacts and archives containing test/job source stay local. Historical command lines belong to their recorded revisions; the removed `foundry benchmark` is not a current public API.
