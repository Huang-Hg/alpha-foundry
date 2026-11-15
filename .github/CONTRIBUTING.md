# Contributing

Start from a feature branch and open a pull request. Keep commit messages, code comments and docstrings in English. Preserve the English and Chinese user guides when changing public behavior.

Use public Foundry parameters rather than adding a configuration file or another compatibility entry point. See the [research guide](../guides/research.md) for market and interface contracts.

Run static checks locally:

```bash
python -m pip install -r requirements-dev.txt
python -m pre_commit run --all-files
```

This checks source syntax, undefined names, tracked configuration and GitHub workflow definitions. It does not import Foundry or run numerical code.

Builds, numerical tests and benchmarks run on the explicitly authorized cloud worker. Use the [single benchmark entry](../guides/benchmarks/README.md), attach the result and source bundles, and report decision differences as well as timing. A saved benchmark is evidence for its recorded inputs and source, not automatic validation of later code.

Do not publish private portfolios, credentials or unselected data. Keep `AGENTS.md`, `.gitignore` and `.gitattributes` local. `tests/` and `scripts/` are retired; maintained checks live in `validation/`.

Required GitHub checks enforce static quality and saved-evidence integrity. No workflow starts paid compute, merges a PR or publishes a package automatically.
