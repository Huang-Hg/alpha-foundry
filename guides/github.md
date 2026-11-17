# GitHub checks and publication boundary

[简体中文](github.zh-CN.md)

Publish the runtime library, bilingual documentation, dependencies and necessary GitHub Actions configuration. Tests and helper scripts remain local in `tests/` and `scripts/`, without renamed copies or source archives. Native build commands live in the README.

| Check | Infrastructure |
| --- | --- |
| Static quality | pre-commit, Ruff, actionlint |
| Dependency and security | GitHub Dependency Review, CodeQL, Dependabot |
| Measured evidence | sha256sum for selected data, source digests embedded in summary.json, Actions Job Summary, upload-artifact |

Run numerical acceptance on explicitly authorized cloud compute using pytest, pytest-benchmark and pytest-html. Complete and freeze edits first, reuse one GPU across adjacent tasks and run each condition once. CI does not start paid GPUs or replace market-rule and numerical checks.

Publish compact Markdown, CSV tables and summary.json with embedded source digests. Full raw results and local test code stay outside Git. Historical reports do not establish acceptance of current source.

For local static checks, install `--group dev` and run `python -m pre_commit run --all-files`. Actions and hooks use fixed commits. GitHub continues to own branch protection, security analysis and dependency review. This change is not pushed or merged, so its workflows have not run on GitHub.
