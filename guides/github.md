# GitHub checks and publication boundary

[简体中文](github.zh-CN.md) · [Contributing](../.github/CONTRIBUTING.md)

The public repository tracks only the `foundry/` runtime package, bilingual user guides (`guides/`), reviewed test-data fixtures (`data/`), dependency metadata (`pyproject.toml`), and GitHub quality-gate configuration (`.github/`, `.pre-commit-config.yaml`). Local regression suites (`tests/`) and benchmark drivers (`scripts/`) stay untracked on disk, without renamed directories or embedded source archives. Native extension build commands live in the root [README](../README.md).

| Gate | Tooling |
| --- | --- |
| Tracked-file allowlist & static quality | Allowlist check, `pre-commit`, Ruff, `actionlint`, C++ build check |
| Dependency and security analysis | GitHub Dependency Review, CodeQL (`python`, `c-cpp`, `actions`), Dependabot |
| Published data & benchmark summaries | `sha256sum` on `data/SHA256SUMS`, `summary.json` schema/status validation, Actions Job Summary |

Numerical regression and CPU/CUDA performance benchmarks execute on explicitly authorized cloud instances with `pytest`, `pytest-benchmark`, and `pytest-html`. Freeze code changes first, reuse a single cloud session across adjacent checks, and measure each condition once. GitHub Actions never provisions paid GPUs or substitutes for market-rule and numerical acceptance.

Under `guides/benchmarks/`, publish only concise Markdown reports, CSV tables, and `summary.json` files; keep raw outputs and archives in local `var/runs/`. Saved reports document the exact revision and hardware measured at the time—they do not certify subsequent source changes.

For local static verification, install `--group dev` and run `python -m pre_commit run --all-files`. All workflow actions and pre-commit hooks are pinned to full commit SHAs.
