# GFlowNet discovery end-to-end timing

NVIDIA L4, 23034 MiB, 580.82.07

GFN inference, losses, autograd and SGD run in PyTorch CUDA; host code handles grammar and formula text. Evaluation and accounts use CUDA.

| Stage | Accumulated seconds |
| --- | ---: |
| sampling_seconds | 1.587836 |
| evaluation_seconds | 3.030296 |
| training_seconds | 2.205516 |
| commit_seconds | 0.232981 |
| reweight_seconds | 0.246478 |

Mining wall time: 7.888872 s; open through close: 8.921073 s.
Open/data/device setup: 0.912927 s; policy initialization: 0.009034 s.

Training overlaps commit/reweight, so stage sums are not end-to-end wall time. First-use JIT remains in the first run; the run plan records preparation and warm-up conditions.

Epochs: 8; ordinary proposals: 4096; including probes: 4288; accepted: 8.
GPU sampled peak: 4340.0 MiB. Policy changed: True.

This is online policy learning and admission on one historical research window, not supervised out-of-sample walk-forward evidence. It executes the actual mine workflow; separate judge_family robustness review is not automatically included.

[Summary](summary.json) · Plan (local artifact) · Epoch reports (local artifact)
