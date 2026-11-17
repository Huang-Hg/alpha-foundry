# L4: PyTorch GPU GFlowNet acceptance

[English](README.md) · [简体中文](README.zh-CN.md)

**Migration is complete, but end-to-end time increased: 4096 ordinary proposals take 7.889 s versus 5.032 s previously; session open through close takes 8.921 s.** Eight epochs generate 4288 proposals including 192 probes, 2953 within-epoch unique evaluations and eight hard admissions.

The model, TB loss, paired teachers, autograd and SGD now live in Python/PyTorch and execute FP32 on L4. The old C++ learner and build target are deleted. Host code handles grammar and formula text; the existing native CEM primitive still proposes weights. Evaluation, accounts and weight trials use CUDA.

| Stage | Previous native learner workflow | PyTorch GPU workflow |
| --- | ---: | ---: |
| 采样 / Sampling | 0.160 s | 1.588 s |
| 评价与奖励 / Evaluation and rewards | 2.943 s | 3.030 s |
| 训练 / Training | 1.107 s | 2.206 s |
| 有序提交 / Ordered commit | 0.151 s | 0.233 s |
| 重配 / Reweight | 0.226 s | 0.246 s |
| Mining wall time | 5.032 s | 7.889 s |
| Open through close | 6.186 s | 8.921 s |

The first training epoch takes 1.636 s, including first-use training preparation in this process. Later epochs average 0.0814 s, versus 0.1398 s for the previous final seven epochs. Sampling regresses from 0.160 s to 1.588 s and includes Python grammar handling and host/device interaction; no kernel-level profile isolates their individual contributions. Training overlaps commit/reweight; stage sums are not wall time.

This compares workflows with the same L4 model, data, rules and parameters, reusing saved historical evidence. It is not an identical-formula pure-kernel speedup. The new RNG and FP32 learning change candidate trajectories; eight admissions on each side do not mean the same strategies were admitted.

## Numerical and contract acceptance

- Smoke passes, then 46 targeted checks and 6756 subtests pass in 94.44 s, followed by one 8×512 measured run.
- Fixed-trajectory independent reference: absolute loss error 6.79e-06, maximum absolute gradient error 1.16e-06, maximum absolute one-step teacher/scaling/clipping update error 1.17e-07.
- Teacher votes, KL/confidence gates, four distinct shared suffixes, grammar budgets, checkpoint replay and GPU parameter residency pass. Within-batch loss falls in every epoch; every admission has a matching hard pass.
- Reward, archive, robustness, admission, account and operator source is unchanged, with the same research snapshot. This does not claim matching metrics or weights across different candidate sets.

## Resources and reproduction

- L4 23034 MiB; 269 rows × 500 US equities, 249 statistic days. Paired FP32 accounts/statistics, FP32 policy learning. Sampled GPU peak 4340 MiB; PyTorch peak tensor allocation 30.08 MiB.
- Cache budget remains 32 MiB; only current-batch policy trajectories are retained. No independent CPU baseline or repeated seeds.
- Smoke includes first-use CuPy compilation and PyTorch training preparation. The measured workload reuses disk compilation caches but runs in a separate process, so its first training preparation remains timed.
- Native checkpoints are rejected. New checkpoints contain JSON/numeric arrays without pickle. No production baseline was rerecorded.
- Adaptive in-window discovery is not out-of-sample return validation; mine does not automatically execute separate judge_family review.

```bash
python -m foundry --parameters '{"search":{"epochs":8,"batch_size":512}}' \
  benchmark var/runs/discovery-new --suite discovery --backends cuda
```

Comparison and errors (local artifact) · [Epochs](epochs.csv) · [Summary](measured/summary.json) · Plan (local artifact) · Source audit (local artifact) · Source (local archive) · Raw reports (local artifact) · Checks (local artifact) · Checksums (local artifact)
