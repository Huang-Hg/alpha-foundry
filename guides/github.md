# GitHub checks and review

[中文](github.zh-CN.md)

Open changes on a feature branch and review them in a pull request. Nothing in this setup merges changes, publishes releases or starts a paid GPU automatically.

| Required check | What it proves |
| --- | --- |
| Repository checks | pre-commit configuration/file checks, Ruff syntax and undefined-name checks, actionlint, dependency review on PRs, and CodeQL analysis complete successfully. |
| Benchmark evidence | Historical evidence and selected data retain their checksums; R5 JUnit results, four-market daily parity, qualified full-scale measurements and exact reviewed source hashes pass the static verifier. |

CodeQL scans Python, C/C++ and Actions definitions without building the project. It is a security analysis, not a CUDA compiler or a numerical correctness check. Dependency review blocks newly introduced high/critical vulnerabilities. CodeQL findings appear in the repository Security tab; successful analysis does not mean there are no findings.

Numerical validation stays on the explicitly authorized cloud worker. Use the [benchmark guide](benchmarks/README.md) to produce fresh evidence when changing computation, regression contracts, numerical dependency constraints or the measured corpus. The current gate binds every numerical source file to the recorded R5 bytes. The dependency-file consolidation and installation-text changes are separately listed with exact before/after hashes in [source-review.json](benchmarks/timed-l4-20261116/source-review.json); the verifier also proves that the numerical dependency constraints and pytest settings stayed unchanged. New files and further edits fail the binding instead of inheriting an open-ended exception.

The Actions summary presents the [R5 L4 report](benchmarks/timed-l4-20261116/README.md), including failed mixed-precision controls and the skipped private baseline. The gate checks 508 passing checks plus that explicit skip, the targeted smoke repairs, exact four-market daily comparisons, and the full 4096-array/256-program CPU correctness audits. It never converts the mixed control into a pass or treats a GPU host's CPU as a speed baseline.

Each successful evidence job retains the public acceptance bundle for 30 days. Its validated source, original JUnit files, daily comparisons and numerical audit summaries are versioned in `guides/benchmarks/timed-l4-20261116/`. The complete raw archive stays local. Earlier report bundles remain historical evidence, with integrity checks; they do not claim to validate the current architecture.

## Local static checks

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group dev
python -m pre_commit run --all-files --show-diff-on-failure
```

This installs isolated tooling and checks tracked files. It does not import Foundry, compile kernels or run tests. The hooks do not rewrite source files.

Actions and pre-commit hooks use immutable commit references. Dependabot proposes weekly grouped Actions and Python dependency updates. Hook updates are reviewed separately with `python -m pre_commit autoupdate --freeze`, followed by the same static checks. Never merge update PRs automatically.

## Repository settings

The `main` ruleset requires a PR, resolved review conversations and both required checks from GitHub Actions against the current base. It blocks deletion and force pushes. The sole-maintainer repository does not require another person's approval; the draft PR and explicit review still control when a change is ready to merge.

Actions tokens default to read access. Only CodeQL receives security-event write access. The repository requires full commit SHAs for Actions, keeps secret scanning and push protection enabled, and enables Dependabot security alerts/updates and private vulnerability reporting. The legacy numerical CI workflow is disabled.

Branch rules and security settings apply immediately. Dependabot's version-update schedule, issue forms and default contribution links become active when their files reach the default branch. A draft PR can exercise the new PR checks before that happens. Keeping the PR unmerged intentionally leaves those default-branch features pending.

See [Contributing](../.github/CONTRIBUTING.md) and [Security reporting](../.github/SECURITY.md).
