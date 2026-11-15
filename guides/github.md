# GitHub checks and review

[中文](github.zh-CN.md)

Open changes on a feature branch and review them in a pull request. Nothing in this setup merges changes, publishes releases or starts a paid GPU automatically.

| Required check | What it proves |
| --- | --- |
| Repository checks | pre-commit configuration/file checks, Ruff syntax and undefined-name checks, actionlint, dependency review on PRs, and CodeQL analysis complete successfully. |
| Benchmark evidence | Published evidence and selected data pass their checksums; recorded operator and flow comparisons pass; the measured source, current report renderer and formula corpus match the proposed revision. |

CodeQL scans Python, C/C++ and Actions definitions without building the project. It is a security analysis, not a CUDA compiler or a numerical correctness check. Dependency review blocks newly introduced high/critical vulnerabilities. CodeQL findings appear in the repository Security tab; successful analysis does not mean there are no findings.

Numerical validation stays on the explicitly authorized cloud worker. Use the [benchmark guide](benchmarks/README.md) to produce fresh evidence when changing computation, validation, runtime dependencies or the measured corpus. The evidence check accepts line-ending and trailing-EOF-whitespace normalization, but does not treat a matching AST or an older successful run as validation of changed code. Report rendering has its own recorded hash because the final report can be rendered after measurement.

The Actions summary contains timings, errors, decisions and links to the checked-in reports. Each successful evidence job also retains a downloadable artifact for 30 days. Raw measurement and measured-source bundles remain versioned in `guides/benchmarks/current/`.

## Local static checks

```bash
python -m pip install -r requirements-dev.txt
python -m pre_commit run --all-files --show-diff-on-failure
```

This installs isolated tooling and checks tracked files. It does not import Foundry, compile kernels or run tests. The hooks do not rewrite source files.

Actions and pre-commit hooks use immutable commit references. Dependabot proposes weekly grouped Actions and Python dependency updates. Hook updates are reviewed separately with `python -m pre_commit autoupdate --freeze`, followed by the same static checks. Never merge update PRs automatically.

## Repository settings

The `main` ruleset requires a PR, resolved review conversations and both required checks from GitHub Actions against the current base. It blocks deletion and force pushes. The sole-maintainer repository does not require another person's approval; the draft PR and explicit review still control when a change is ready to merge.

Actions tokens default to read access. Only CodeQL receives security-event write access. The repository requires full commit SHAs for Actions, keeps secret scanning and push protection enabled, and enables Dependabot security alerts/updates and private vulnerability reporting. The legacy numerical CI workflow is disabled.

Branch rules and security settings apply immediately. Dependabot's version-update schedule, issue forms and default contribution links become active when their files reach the default branch. A draft PR can exercise the new PR checks before that happens. Keeping the PR unmerged intentionally leaves those default-branch features pending.

See [Contributing](../.github/CONTRIBUTING.md) and [Security reporting](../.github/SECURITY.md).
