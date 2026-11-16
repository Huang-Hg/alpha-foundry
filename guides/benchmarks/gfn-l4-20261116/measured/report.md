# GFlowNet discovery end-to-end timing

NVIDIA L4, 23034 MiB, 580.82.07

The existing GFN sampler and learner execute in native C++ on the host; evaluation and accounts use CUDA. No CPU backtest comparison was run.

| Stage | Accumulated seconds |
| --- | ---: |
| sampling_seconds | 0.160146 |
| evaluation_seconds | 2.942735 |
| training_seconds | 1.107435 |
| commit_seconds | 0.150943 |
| reweight_seconds | 0.225563 |

Mining wall time: 5.032071 s; open through close: 6.186181 s.
Open/data/device setup: 1.025217 s; policy initialization: 0.004243 s.

Training overlaps commit/reweight, so stage sums are not end-to-end wall time. First-use JIT remains in the first run; the run plan records preparation and warm-up conditions.

Epochs: 8; ordinary proposals: 4096; including probes: 4288; accepted: 8.
GPU sampled peak: 5010.0 MiB. Policy changed: True.

This is online policy learning and admission on one historical research window, not supervised out-of-sample walk-forward evidence. It executes the actual mine workflow; separate judge_family robustness review is not automatically included.

[Summary](summary.json) · [Plan](plan.json) · [Epoch reports](../raw.tar.gz)
