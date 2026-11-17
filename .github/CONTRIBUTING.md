# Contributing

Submit changes on a feature branch via pull request. Keep code comments, docstrings, and commit messages in English; update both English and Chinese documentation (`README.md` / `README.zh-CN.md` and `guides/*.md` / `guides/*.zh-CN.md`) when public interfaces or behavior change.

Public configuration goes through Foundry's typed Python parameters rather than config files or environment variables. See [guides/research.md](../guides/research.md) for API conventions.

## Static checks

Run the repository linter and schema checks before opening a pull request:

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install 'pre-commit>=4.3,<5'
python -m pre_commit run --all-files
```

CI also compiles the C++ extensions from the root `README.md` build block and verifies `data/SHA256SUMS` and `guides/benchmarks/**/summary.json`.

## Repository hygiene

Do not commit credentials, `.env*` files, local runtime directories (`var/`, `tmp/`, `logs/`, `output/`), build artifacts (`*.so`, `__pycache__/`), or raw benchmark dumps (`*.tar.gz`, `*.npz`, JUnit XML, HTML reports). Published benchmark directories under `guides/benchmarks/` keep only bilingual Markdown summaries, CSV tables, and `summary.json`.
