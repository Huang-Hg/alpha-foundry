# GFlowNet discovery end-to-end timing

NVIDIA L4, 23034 MiB, 580.82.07

The existing GFN sampler and learner execute in native C++ on the host; evaluation and accounts use CUDA. No CPU backtest comparison was run.

| Stage | Accumulated seconds |
| --- | ---: |
| sampling_seconds | 0.002183 |
| evaluation_seconds | 0.514501 |
| training_seconds | 0.009142 |
| commit_seconds | 0.024904 |
| reweight_seconds | 0.036764 |

Mining wall time: 0.629872 s; open through close: 1.701766 s.
Open/data/device setup: 1.045322 s; policy initialization: 0.004356 s.

Training overlaps commit/reweight, so stage sums are not end-to-end wall time. First-use JIT remains in the first run; the run plan records preparation and warm-up conditions.

Epochs: 1; ordinary proposals: 32; including probes: 56; accepted: 1.
GPU sampled peak: 864.0 MiB. Policy changed: True.

This is online policy learning and admission on one historical research window, not supervised out-of-sample walk-forward evidence. It executes the actual mine workflow; separate judge_family robustness review is not automatically included.

[Summary](summary.json) · [Plan](plan.json) · [Epoch reports](../raw.tar.gz)
