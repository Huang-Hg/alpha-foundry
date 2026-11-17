# Contributing

Work on a feature branch (`codex/<topic>`) and submit changes through a pull request. Write commit messages, code comments, and docstrings in English; keep public documentation bilingual (`README.md` / `README.zh-CN.md` and `guides/*.md` / `guides/*.zh-CN.md`) whenever user-facing behavior changes.

Pass configuration through Foundry's typed public parameters rather than introducing configuration files, environment overrides, or compatibility shims. See the [Research guide](../guides/research.md) for architecture and interface contracts.

## Local static checks

Run static verification locally without loading market data or initializing a GPU:

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group dev
python -m pre_commit run --all-files
```

This checks syntax, undefined names, YAML/JSON/TOML schemas, private-key patterns, and GitHub Actions workflows.

## Cloud validation and publication boundary

Native compilation, numerical regression suites, and CPU/CUDA benchmarks run on explicitly authorized Linux cloud compute using the root README build snippet, `pytest`, `pytest-benchmark`, and `pytest-html`. Finish and freeze code edits before running the acceptance suite.

Keep the tracked repository clean:
- **Never commit** credentials, `.env*` files, private portfolios (`production/`), runtime state (`var/`, `tmp/`, `logs/`, `output/`), internal notes (`docs/`), local instructions (`AGENTS.md`, `.gitignore`, `.gitattributes`), or local test/driver trees (`tests/`, `scripts/`, `validation/`).
- **Published benchmarks** under `guides/benchmarks/` contain only compact bilingual Markdown reports, CSV tables, and `summary.json` files—never raw archives (`*.tar.gz`), JUnit XML, HTML reports, or cloud job scripts. A saved benchmark documents its recorded source and hardware; it does not automatically certify later revisions.
- **GitHub Actions** enforces the tracked-file allowlist, static checks, dependency review, CodeQL analysis, and published data/benchmark integrity without launching paid GPU compute or auto-merging pull requests.
