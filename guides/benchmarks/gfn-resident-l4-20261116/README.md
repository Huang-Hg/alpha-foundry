# L4: resident sampling and trajectory reuse

[English](README.md) · [简体中文](README.zh-CN.md)

**The 4096-ordinary-proposal mining loop falls from 7.889 s to 6.168 s (21.8% less time). Including preparation, session open and close, total time falls from 8.921 s to 8.586 s (3.8% less).** Initialization remains included in the total. Sampling in epochs 2–8 averages 0.0866 s versus 0.1744 s previously.

Prefix states, legal-action masks and random draws stay on the device, with no host read in the token loop. Only probe-chunk success flags and final expression tokens cross to the host. Training consumes the sampled trace directly. SGD is initialized once; each train call applies the caller's explicit learning rate.

| Item | Previous host sampling | Resident GPU sampling |
| --- | ---: | ---: |
| 采样 / Sampling | 1.588 s | 1.066 s |
| 评价及奖励 / Evaluation and rewards | 3.030 s | 3.202 s |
| 训练调用 / Training calls | 2.206 s | 0.808 s |
| 有序提交 / Ordered commit | 0.233 s | 0.190 s |
| 重配 / Reweight | 0.246 s | 0.252 s |
| 模型与优化器准备 / Policy and optimizer preparation | 0.009 s | 1.355 s |
| 挖掘墙钟 / Mining wall time | 7.889 s | 6.168 s |
| 打开至关闭 / Open through close | 8.921 s | 8.586 s |

First optimizer preparation moved to initialization. Preparation plus training totals 2.215 → 2.162 s; the entire reduction inside train calls is not compute acceleration. Stages overlap, so their sums are not end-to-end wall time.

The L4 model, 269-row × 500-equity data, research snapshot, parameters and initial model weights match. The previous result is saved evidence, not a new run. Device RNG changes the formula sequence: this run has 4288 proposals (192 probes), 2968 within-epoch unique evaluations and eight hard admissions. This is a workflow comparison, not an identical-formula kernel comparison or a claim of identical selected strategies.

## Correctness and resources

- Smoke, then 47 targeted checks and 6785 subtests pass in 96.76 s; one 8×512 workload follows. No repeated seeds or independent CPU baseline.
- Legal-action masks have zero mismatches against host grammar rules; maximum absolute prefix-feature error is 6.42e-8. TB gradient error is 1.11e-6; one-step teacher/scaling/clipping update error is 1.19e-7.
- Three sibling actions, common-support distributions, four distinct shared suffixes, the 40-suffix-attempt cap, 24×requested prefix-attempt budget and teacher vote/KL/confidence gates remain. Reward, archive, admission, account and operator source is unchanged.
- Optimizer reuse, release of previous-batch traces and exact checkpoint continuation pass. Protocol foundry-gfn-torch-v2 stores device RNG state and rejects older protocols or different device kinds.
- Trace storage is current-batch O(B×L×(F+A)); probe workspaces are bounded to four prefixes × 40 suffixes × three branches and released by chunk. Research measurement cache remains 32 MiB, without historical replay.
- Peak PyTorch tensor allocation is 30.08 → 36.17 MiB; sampled whole-device peak is 4340 → 5774 MiB. The total increase has not been attributed per allocation; overall memory reduction is not claimed.
- Smoke includes initial compilation/preparation. Measurement reuses disk compilation caches but runs in a separate process. The GPU was released; no production baseline was rerecorded.

The historical native C++ learner workflow took 5.032 s for mining and 6.186 s overall; this version remains slower than that historical run. Different random candidate workloads limit cross-implementation comparisons.

```bash
python -m foundry --parameters '{"search":{"epochs":8,"batch_size":512}}' \
  benchmark var/runs/discovery-new --suite discovery --backends cuda
```

Comparison (local artifact) · [Epochs](epochs.csv) · [Summary](measured/summary.json) · Plan (local artifact) · Source audit (local artifact) · Source (local archive) · Raw reports (local artifact) · Checks (local artifact) · Checksums (local artifact)
