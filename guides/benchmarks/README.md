# Foundry benchmarks

[简体中文](README.zh-CN.md) · [Current CPU/CUDA report](current/report.md) · [Historical PRO 6000 result](archive/pro6000-operator66/operator66.md)

The current suite answers three questions: **How long does a real research call take? How much numerical error does CUDA introduce? Does it admit the same candidates as CPU?**

Run one command on the authorized GPU:

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=2 python -m foundry benchmark output/benchmark-new
```

Install CuPy for the runtime's CUDA version first. Numerical runs happen on the selected cloud instance; GitHub Actions only displays and archives saved evidence.

The runner uses [pytest](https://docs.pytest.org/en/stable/how-to/unittest.html), [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/pedantic.html) and pytest-html. Existing regression checks live in `validation/`; the old `tests/` and `scripts/` directories are removed.

| Measurement | What is compared |
| --- | --- |
| Public research API | 1, 64 and 4096 formulas; open, evaluate, commit and close |
| Operators | All 66 operators on the same real inputs; rolling windows 2, 20 and 120 |
| Correctness | Exact admission/reason/constraint decisions; per-metric errors and weight tolerances |
| Regression | PIT, A-share rules, spot, perpetual funding/liquidation, resources and RPC parity |

Each performance condition has one warm-up and one measured round. Warm-up uses a fresh portfolio; timed rounds never reuse its query cache or mutated portfolio. Both CPU and CUDA use host wall time, with explicit CUDA synchronization. Small timings are observations, not statistical rankings.

The output directory contains bilingual Markdown summaries, detailed CSVs, the standard pytest-benchmark JSON, JUnit XML, a standalone HTML check report, raw formula reports, source/data/corpus identities and the run log. A failing correctness check or incomplete benchmark fails the command while preserving its report.

For custom data or fees, pass the same public parameters as `foundry.open`; the CPU and CUDA runs share them. The fixed corpus must be valid for that dataset's vocabulary. `foundry.benchmark(output, sizes=(1, 64), market="us_equity", data_dir="...")` can limit candidate counts. Timing thresholds from another GPU, dataset or implementation are not inherited.

The old PRO 6000 measurements used an internal API and experimental precision paths. They remain a historical record and are not blended with the current public-API measurements. This suite does not requalify the separate 1.5× precision budget.

The [complete evidence archive](current/evidence.tar.gz) contains the raw per-formula reports and setup logs. Extract it to recover the original result directory. Source-file hashes identify the measured working tree; pytest commit metadata names its base checkout. JUnit case counts include unittest subtests.

The [measured source archive](current/source.tar.gz) preserves the exact source bytes and corpus, including original line endings. Published text reports use LF for reproducible GitHub checksum verification. The report renderer has its own hash because its final wording was improved after measurement without rerunning timings.
