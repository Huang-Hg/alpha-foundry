# GFlowNet discovery end-to-end timing

NVIDIA L4, 23034 MiB, 580.82.07

GFN states, legality, draws, inference and training stay on CUDA; host code handles probe chunk outcomes and final formula text.

| Stage | Accumulated seconds |
| --- | ---: |
| sampling_seconds | 0.853638 |
| evaluation_seconds | 8.406256 |
| training_seconds | 0.296960 |
| commit_seconds | 0.862073 |
| reweight_seconds | 0.633781 |

Mining wall time: 13.786098 s; open through close: 35.561581 s.
Open/data/device setup: 15.658801 s; policy initialization: 6.096897 s.

Training overlaps commit/reweight, so stage sums are not end-to-end wall time. First-use JIT remains in the first run; the run plan records preparation and warm-up conditions.

Epochs: 1; ordinary proposals: 32; including probes: 56; accepted: 1.
GPU sampled peak: 946.0 MiB. Policy changed: True.

This is online policy learning and admission on one historical research window, not supervised out-of-sample walk-forward evidence. It executes the actual mine workflow; separate judge_family robustness review is not automatically included.

[Summary](summary.json) · Plan (local artifact) · Epoch reports (local artifact)
