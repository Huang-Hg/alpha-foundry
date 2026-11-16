# L4: existing GFlowNet discovery end-to-end speed

[English](README.md) · [简体中文](README.zh-CN.md)

**Eight epochs / 4096 ordinary proposals take 5.032 s for the mining loop and 6.186 s including session open/close.** With 192 paired probes, 4288 proposals produce 2959 within-epoch unique evaluations and eight ordered admissions.

This calls the existing `foundry.mine()`: the 32-hidden-unit neural GFN, Trajectory Balance, paired teachers, portfolio rewards, archive, commit and CEM reweighting remain intact. Relative to `8045a92`, only benchmark routing and reporting changed; search and numerical kernels did not.

**Execution placement:** candidate evaluation, accounts and portfolio trials use L4. Existing GFN sampling/training and CEM proposal generation remain native C++ CPU work. Only a GPU runtime was used, with no independent CPU baseline; this is not all-GPU policy training.

| Stage | Accumulated time | Execution |
| --- | ---: | --- |
| GFN sampling | 0.160 s | C++ CPU |
| Candidate evaluation / rewards | 2.943 s | CUDA + host orchestration |
| GFN training | 1.107 s | C++ CPU |
| Ordered commit | 0.151 s | CUDA + host CAS |
| CEM reweight | 0.226 s | C++ proposals + CUDA trials |

Training overlaps commit/reweight, so these sums are not wall time. Mining wall time includes diagnostics, archive, status and checkpoint IO. Session open takes another 1.025 s; policy initialization takes 0.004 s.

## Learning and admission evidence

Each epoch uses 512 ordinary proposals, two probe groups and four gradient updates: 32 updates total, seed 25, learning rate 0.002, OPD strength 0.1. Within-batch loss decreases in all eight epochs. Parameter digests exclude RNG state and confirm weight changes. Epochs 2 and 7 each select one reliable teacher; six reweight operations change weights. Admission executes 43950 weight trials, plus 903 reweight evaluations.

| Epoch | Loss before | Loss after | Reliable teachers |
| --- | ---: | ---: | ---: |
| 1 | 423.674 | 403.335 | 0 |
| 2 | 366.980 | 349.293 | 1 |
| 3 | 368.812 | 346.984 | 0 |
| 4 | 374.096 | 352.422 | 0 |
| 5 | 347.031 | 327.904 | 0 |
| 6 | 307.461 | 290.328 | 0 |
| 7 | 296.489 | 282.195 | 1 |
| 8 | 349.206 | 329.543 | 0 |

Candidates and rewards change between epochs; this is not a fixed validation-loss curve. All eight admissions have matching hard-admission pass records. Per-candidate decisions and final weights are retained.

## Conditions and scope

- NVIDIA L4 23 GB, 269 rows × 500 US equities, 249 execution/statistic days, 0.0005 costs, paired FP32 accounts/statistics. Sampled device peak: 5010 MiB.
- 32 MiB compact cache, 256 resident candidates, 32 archive slots; no enlarged cache.
- 40 contract checks and 6657 subtests pass in 39.94 s, followed by a 32-proposal one-epoch smoke and one 512-proposal eight-epoch run. No repeated seeds or CPU baseline.
- GPU/CuPy caches were warmed by contract checks and smoke. Loading, device setup and learning are recorded separately; raw logs are retained.
- This includes policy sampling and learning; the proposal distribution and duplicate rate differ from the fixed 4096-formula corpus. The two times are not a direct speedup comparison.
- This is adaptive in-window strategy discovery, not supervised out-of-sample walk-forward validation. The independent judge_family robustness review is not invoked automatically by mine.

## Reproduce

```bash
python -m foundry --parameters '{"search":{"epochs":8,"batch_size":512}}' \
  benchmark var/runs/discovery-new --suite discovery --backends cuda
```

[Epoch table](epochs.csv) · [Measured summary](measured/summary.json) · [Plan](measured/plan.json) · [Raw reports/checkpoints/logs](raw.tar.gz) · [Source audit](source-audit.json) · [Source](source.tar.gz) · [Checksums](SHA256SUMS)
