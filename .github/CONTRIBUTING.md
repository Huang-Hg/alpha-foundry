# Contributing

Start from a feature branch and open a pull request. Keep commit messages, code comments and docstrings in English. Preserve the English and Chinese user guides when changing public behavior.

Use public Foundry parameters rather than adding a configuration file or another compatibility entry point. See the [research guide](../guides/research.md) for market and interface contracts.

Run static checks locally:

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group dev
python -m pre_commit run --all-files
```

This checks source syntax, undefined names, tracked configuration and GitHub workflow definitions. It does not import Foundry or run numerical code.

Builds, numerical tests and benchmarks run on the explicitly authorized cloud worker. Use the README build commands and pytest with pytest-benchmark in a local suite. Publish concise [benchmark results](../guides/benchmarks/README.md), including source digests, decision differences and timing. A saved benchmark is evidence for its recorded inputs and source, not automatic validation of later code.

Do not publish private portfolios, credentials or unselected data. Keep `AGENTS.md`, `.gitignore` and `.gitattributes` local. Maintainer tests and helpers stay in untracked `tests/` and `scripts/`; do not publish renamed copies or source archives. Public benchmark files are limited to compact Markdown, CSV and summary.json. GitHub Actions runs repository, dependency and security checks using dedicated tools.

Required GitHub checks enforce static quality and saved-evidence integrity. No workflow starts paid compute, merges a PR or publishes a package automatically.
