# L4: non-GFN models and non-formula factor discovery

[English](README.md) · [简体中文](README.zh-CN.md)

**Both GPU programs use the existing `mine` interface for batched outputs, registration, evaluation, resident feedback and learning. The generic driver is unchanged; GFN and formula parser calls are forbidden during acceptance.** One 4×64 run per case takes 3.489 s for MLP mining and 1.201 s for tensor factors.

The MLP is an 8→16→1 network fitted on causal prefixes to 20-day return labels, producing target weights. A separate GPU reward surrogate guides the next hyperparameter batch. The factor client uses GPU tensors, interactions and cross-sectional sorting to produce signals, with GPU CEM updating its parameter distribution. Both share the same archive and hard judge.

| Stage | Batched MLP | Tensor factors + CEM |
| --- | ---: | ---: |
| 历史前缀拟合 / Causal fitting | 2.0591 s | 0.0000 s |
| 推理/因子生成 / Inference or factor generation | 0.0414 s | 0.0331 s |
| 数组注册（含传输/落盘） / Registration | 0.8307 s | 0.8057 s |
| 评价和奖励 / Evaluation and rewards | 0.2735 s | 0.2416 s |
| 驻留反馈学习 / Feedback learning | 0.0981 s | 0.0078 s |
| 搜索提交检查 / Search commit checks | 0.0001 s | 0.0001 s |
| 数据和会话打开 / Session open | 0.7937 s | 0.2110 s |
| 模型及优化器准备 / Model preparation | 1.4548 s | 0.0067 s |
| 完整挖掘循环 / Mining loop | 3.4887 s | 1.2007 s |
| 额外九点邻域/双生子 / Extra family review | 0.0000 s | 0.0693 s |
| 打开至关闭 / Open through close | 5.7497 s | 1.5008 s |

Stages overlap or contain sub-stages and must not be summed into wall time. Mining includes generation, registration, evaluation, learning, commit checks, reweight calls, telemetry and checkpoints. There are no search acceptances or actual reweight trials. MLP preparation includes first PyTorch optimizer startup; cases run sequentially, so their timing difference is not a pure algorithm comparison.

## Separate hard admission from search selection

| Result | MLP | Tensor factors |
| --- | ---: | ---: |
| Candidates / G1 valid | 256 / 256 | 256 / 256 |
| G2 hard passes | 47 | 74 |
| Final search acceptances | 0 | 0 |
| Reference-return correlation range | 0.90071–0.92467 | 0.90872–0.99918 |

Every candidate exceeds the existing rho_max=0.9 search limit. Even G2-passing candidates are therefore not selected; other candidates also fail hard constraints. No thresholds were relaxed and G2 passes are not counted as final selections.

Separate in-memory API contracts submit one previously G2-passing array per mode through evaluate → commit. Queries remain read-only, commits atomically advance state, and reused reports raise CAS conflicts. Standalone versus original batched admission metrics have zero difference. These direct API checks do not count as mining acceptances or modify the search correlation filter.

## Non-formula neighborhoods

All nine explicit CandidateGrid neighbors pass hard admission. Delayed twins pass 0/9 and the protocol reports a held hypothesis. However, this dataset has only 269 rows and the minimum delay is 365 rows: every twin is an empty position series. This validates interface execution, **not an informative negative-control robustness claim**. Data and probe rules were not changed to hide this limitation.

## Correctness, space and bottlenecks

- Smoke runs each case at 2×4; 21 checks and 6844 subtests pass; then each case runs once at 4×64. No repeated seeds or CPU performance baseline.
- Batched versus per-candidate MLP error is 2.98e-8; factor error is zero; resident reward/report error is zero. Future feature/label poisoning leaves prefix predictions unchanged, with zero cutoff violations.
- Reward-regression loss decreases each epoch; factor CEM updates its distribution from GPU feedback. Checks cover output kinds, DLPack pointer identity, parameter changes, checkpoint replay and cleanup.
- Peak PyTorch tensor allocations are 147.2 / 268.1 MiB; sampled whole-GPU peaks are 924 / 1214 MiB. Current output batches peak at 32.84 MiB, input cache capacity is 64 and measurement cache is 32 MiB. No historical replay cache.
- Registration preserves content addressing: GPU→host hashing and NPZ persistence, followed by an owned device copy. It consumes 23.8% of MLP mining and 67.1% of factor mining, dominating the lighter factor workload. Persistent artifacts total about 146 / 151 decimal MB, separate from GPU caches.
- Same L4 and 269-day × 500-equity snapshot. Smoke includes first compilation; measurement uses disk compilation caches. One run per condition does not establish statistical significance.
- Predictive fits are causal, but research hyperparameter/factor selection is adaptive in-sample. This is not an out-of-sample selection or return claim. Source publication/revision limitations remain disclosed in DataContract.
- L4 released; production baselines unchanged.

```bash
python -m foundry --parameters '{"search":{"epochs":4,"batch_size":64},"cuda":{"candidate_batch":64,"measurement_cache_mib":32}}' \
  benchmark var/runs/nonformula-new --suite nonformula --backends cuda
```

[Analysis](analysis.json) · [Epochs](epochs.csv) · [Summary](measured/summary.json) · [Plan](measured/plan.json) · [Submission checks](submission.json) · [Family](measured/factor/family.json) · [Source audit](source-audit.json) · [Source](source.tar.gz) · [Raw reports](raw.tar.gz) · [Checks](contracts.xml)
