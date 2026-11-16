# Foundry benchmarks

[简体中文](README.zh-CN.md) · [2026-11-15 PRO 6000 report](arch-v2-pro6000-20261115/report.md) · [Pre-refactor public API report](current/report.md) · [Earlier operator experiments](archive/pro6000-operator66/operator66.md)

The [resident GPU sampling report](gfn-resident-l4-20261116/README.md) measures device-side states, legality and draws, trace reuse and a persistent optimizer: the 4096-proposal loop falls from 7.889 to 6.168 s; preparation plus session lifecycle falls from 8.921 to 8.586 s. Numerical errors, increased peak memory and candidate-trajectory differences are disclosed. The [PyTorch migration report](gfn-torch-l4-20261116/README.md) remains the previous baseline.

The current suite answers three questions: **How long does a real research call take? How much numerical error does CUDA introduce? Does it admit the same candidates as CPU?**

Use the [fixed Colab CPU baseline](parallel-cpu-20261116/README.md) for A100/L4 comparisons. The comparison fixes one CPU-only runtime and measures 1/8 workers actually executing candidates and trials. The earlier fixed-cpu report only enabled inner native thread pools and remains historical.

The [compact-cache GPU comparison](compact-multigpu-20261116/README.md) replaces the earlier full-book cache with a 32 MiB return/report budget, bounds member storage independently of candidate count, and compares GPU latency, memory and decisions on identical inputs.

The [2026-11-16 TMA and measurement-reuse report](history-pro6000-20261116/README.md) measures faster `ts_slope` and `ts_av_diff`, bounded query-to-commit reuse, and unchanged 4096-formula decisions. It includes the memory tradeoff and actual Blackwell bulk-copy instruction evidence.

The [CUDA operator optimization](ops-pro6000-20261115/README.md) compares before/after measurements on the same PRO 6000. Recurrence, backfill and shift operators improve while the 4096-formula reports, weights and decisions remain identical. The report also identifies cases where CUDA still trails CPU.

The [native framework comparison](frameworks-pro6000-20261115/README.md) separately compares Foundry with Spectre, vectorbt Numba/Rust and Freqtrade on identical daily spot inputs and trading rules. The shared entry is `python -m foundry benchmark OUTPUT --suite frameworks`; see the [setup and method](../framework-benchmark.md). This restricted comparison measures execution, not portfolio admission.

The [architecture-refactor run](arch-v2-pro6000-20261115/README.md) measured source `fa2acfd` on RTX PRO 6000 Blackwell Server (97887 MiB). Public regression results were 512 passed, 1 skipped and 7061 passed subtests in 247.05 s; the 7574 JUnit entries include subtests. All 124 window cases across 66 operators passed. The 4096-formula complete case took 2.879063 s on CUDA versus 93.723696 s on CPU (32.55×); 28 candidates were admitted on both, with zero decision mismatches or numerical tolerance exceedances. The private production audit failed separately on an unchanged legacy artifact, not on a newly recorded baseline.

Run one command on the authorized GPU:

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m foundry benchmark var/runs/benchmark-new --backends cpu
```

This example uses the eight-vCPU CPU runtime. Match `cpu.workers` to actual CPU affinity and quota; keep inner OpenMP and BLAS teams at one. `--backends cpu` needs neither CuPy nor a GPU and runs benchmark contracts plus measurements. Omit `--backends` to run the full regression suite and both CPU/CUDA measurements on an authorized GPU, with matching CuPy installed. Numerical runs happen on the selected cloud instance; GitHub Actions only displays and archives saved evidence.

The runner uses [pytest](https://docs.pytest.org/en/stable/how-to/unittest.html), [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/pedantic.html) and pytest-html. Existing regression checks live in `validation/`; the old `tests/` and `scripts/` directories are removed.

| Measurement | What is compared |
| --- | --- |
| Public research API | 1, 64 and 4096 formulas; open, evaluate, commit and close |
| Operators | All 66 operators on the same real inputs; rolling windows 2, 20 and 120 |
| Correctness | Exact admission/reason/constraint decisions; per-metric errors and weight tolerances |
| Regression | PIT, A-share rules, spot, perpetual funding/liquidation, resources and RPC parity |

Each performance condition has one warm-up and one measured round. Warm-up uses a fresh portfolio; timed rounds never reuse its query cache or mutated portfolio. Both CPU and CUDA use host wall time, with explicit CUDA synchronization. Small timings are observations, not statistical rankings.

The output directory contains bilingual Markdown summaries, detailed CSVs, the standard pytest-benchmark JSON, JUnit XML, a standalone HTML check report, raw formula reports, source/data/corpus identities and the run log. A failing correctness check or incomplete benchmark fails the command while preserving its report.

For custom data or fees, pass the same public parameters as `foundry.open`; the CPU and CUDA runs share them. Select benchmark devices with `backends=("cpu",)` or CLI `--backends cpu`, not the research `backend` parameter. The default is both backends. The fixed corpus must be valid for that dataset's vocabulary. `foundry.benchmark(output, backends=("cpu",), sizes=(1, 64), market="us_equity", data_dir="...")` can limit candidate counts. Timing thresholds from another GPU, dataset or implementation are not inherited. Benchmark schema v2 records selected backends and validation scope; standalone measurements make no cross-device correctness or speedup claim by themselves.

The new public bundle contains reports, detailed CSVs, pytest evidence, run metadata and the measured source archive; full raw per-formula evidence is retained locally. CUDA account arithmetic used paired FP32. No same-device single-FP32 control was measured, so this run does not requalify the separate 1.5× precision budget.

The `current/` directory is an unchanged pre-refactor archive, despite its historical name. Its [complete evidence archive](current/evidence.tar.gz) contains that earlier run's raw per-formula reports and setup logs. Source-file hashes identify its measured working tree; pytest commit metadata names its base checkout. The still older operator experiments used an internal API and experimental precision paths. Neither archive is blended into the architecture-refactor results.

The historical [measured source archive](current/source.tar.gz) preserves that earlier source and corpus, including original line endings. Its published text reports use LF for reproducible GitHub checksum verification. Its renderer has a separate hash because wording was improved after measurement without rerunning timings.
