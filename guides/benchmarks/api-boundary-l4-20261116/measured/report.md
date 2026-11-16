# GFlowNet discovery end-to-end timing

NVIDIA L4, 23034 MiB, 580.82.07

GFN states, legality, draws, inference and training stay on CUDA; host code handles probe chunk outcomes and final formula text.

| Stage | Accumulated seconds |
| --- | ---: |
| sampling_seconds | 1.051227 |
| evaluation_seconds | 3.046680 |
| training_seconds | 0.772594 |
| commit_seconds | 0.176705 |
| reweight_seconds | 0.241834 |

Mining wall time: 5.780996 s; open through close: 8.144510 s.
Open/data/device setup: 0.909925 s; policy initialization: 1.317211 s.

Training overlaps commit/reweight, so stage sums are not end-to-end wall time. First-use JIT remains in the first run; the run plan records preparation and warm-up conditions.

Epochs: 8; ordinary proposals: 4096; including probes: 4288; accepted: 8.
GPU sampled peak: 5772.0 MiB. Policy changed: True.

This is online policy learning and admission on one historical research window, not supervised out-of-sample walk-forward evidence. It executes the actual mine workflow; separate judge_family robustness review is not automatically included.

[Summary](summary.json) · [Plan](plan.json) · [Epoch reports](../raw.tar.gz)
