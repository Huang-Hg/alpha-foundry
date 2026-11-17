# Architecture v2 acceptance on PRO 6000

[简体中文](README.zh-CN.md) · [Performance and errors](report.md) · [Stage timings](flows.csv) · [All operator conditions](operators.csv)

Measured on 2026-11-15 with an NVIDIA RTX PRO 6000 Blackwell Server Edition. The source bytes were checked against local commit `fa2acfd`; the source archive (local archive) preserves those exact bytes and the formula corpus. The benchmark's Git metadata names the remote base checkout, not the overlaid measured source. Use its `foundry.source_sha256` manifest to identify the actual implementation.

The public suite passed: **512 tests and 7,061 subtests**, with one explicit private-baseline skip. All 66 operators covered their 124 declared window conditions. The 4,096-formula workload produced 4,058 valid candidates and 28 admissions on both CPU and CUDA, with zero decision differences or tolerance violations, including final portfolio state. Its complete public call took 93.724 seconds on CPU and 2.879 seconds on CUDA, a 32.55× observed speedup.

**Private production acceptance did not pass.** A separate explicit audit of the unchanged private `production/regression.npz` failed because it lacks a protocol and study declaration. The saved portfolio also belongs to the retired crypto strategy stack. No production artifacts were rewritten or translated, and the public suite's PASS does not certify that old baseline.

CUDA used the fixed paired account-execution plan; CPU used its native path. This run did not compare precision plans against a same-device single-FP32 control and therefore does not requalify the 1.5× precision budget. Each performance condition has one warm-up and one measured round.

The complete raw per-formula reports, smoke output and separate production audit are retained in the local acceptance evidence archive. This directory publishes the compact public summaries and standard pytest artifacts; it contains no private production library or baseline. The paid runtime was released after evidence retrieval and source verification.
