# L4: owned proposal batches and resident learning feedback

[English](README.md) · [简体中文](README.zh-CN.md)

**The same 4096 ordinary proposals complete in 5.781 s versus 6.168 s previously. Open, preparation and close bring the total to 8.145 s versus 8.586 s. All 4288 reports including probes have zero numeric difference; admissions and reweight results are identical.**

This change establishes ownership: constructing a proposer allocates no GPU resources; `mine` opens and closes its producer. Batches bind the producer instance, research context and candidate order. CUDA events and DLPack deliver feedback directly to learning. The generic driver no longer passes GFN learning/distillation parameters or assumes twelve candidates per probe.

| Stage | Previous L4 | Current L4 |
| --- | ---: | ---: |
| 采样 / Sampling | 1.066 s | 1.051 s |
| 评价及奖励 / Evaluation and rewards | 3.202 s | 3.047 s |
| 学习调用 / Learning calls | 0.808 s | 0.773 s |
| 有序提交 / Ordered commit | 0.190 s | 0.177 s |
| 重配 / Reweight | 0.252 s | 0.242 s |
| 数据及设备打开 / Session open | 0.926 s | 0.910 s |
| 策略准备 / Policy preparation | 1.355 s | 1.317 s |
| 完整挖掘循环 / Mining wall time | 6.168 s | 5.781 s |
| 打开至关闭 / Open through close | 8.586 s | 8.145 s |

Mining wall time is 6.3% lower; full lifecycle time is 5.1% lower. Learning overlaps commit/reweight, so stage times do not sum to wall time. Each version has one measurement; the previous result is reused saved evidence. No statistical significance or isolated causal speedup is claimed.

## Comparison contract

- NVIDIA L4 23034 MiB; 269 days × 500 US equities, 249 statistical days; identical snapshot, trading/admission rules, parameters and initial model.
- Eight epochs × 512 ordinary proposals plus 192 probe candidates; 2968 within-epoch unique evaluations, 44376 admission weight trials and 903 reweight trials.
- The loop includes sampling, evaluation/rewards, learning, ordered acceptance, reweighting, telemetry and checkpoints. Data/session startup and policy preparation are reported separately and included in full lifecycle time, as is deterministic model/optimizer closure.
- Smoke includes initial compilation; the measured workload runs in a separate process using disk compilation caches. No CPU baseline or cache expansion.

## Correctness and resources

- Smoke and 63 checks plus 6954 subtests pass; contracts take 17.69 s. One complete workflow follows.
- Ordered candidates, every report/reward, eight admissions, eight reweights and training metrics match the previous run. Maximum absolute difference is zero; final parameter digests match.
- Reward equations and TB/teacher/SGD mathematical statements are AST-identical. Probe sampling, account, operator and hard-admission source is unchanged. Independent reference gradient error is 1.11e-6; the scaled/clipped teacher update error is 1.19e-7.
- Checks cover foreign/reordered/stale batches, duplicate learning, cross-thread CUDA stream ordering, DLPack pointer identity and failure cleanup. GPU learning feedback avoids host-scalar reconstruction; report materialization still reads host values.
- One outstanding batch per producer; additional feedback storage is O(B), with a maximum final vector of 536 × 4 bytes in this run. Traces remain current-batch only, with no replay cache; measurement cache stays 32 MiB.
- Peak PyTorch allocation changes from 36.17 to 35.67 MiB; sampled whole-device peak changes from 5774 to 5772 MiB, essentially unchanged.
- `foundry.mine` stays simple; custom producers adopt open/propose/learn/release. Older checkpoints are explicitly rejected without forwarding layers.
- The L4 was released; production baselines were not rerecorded.

```bash
python -m foundry --parameters '{"search":{"epochs":8,"batch_size":512},"cuda":{"measurement_cache_mib":32}}' \
  benchmark var/runs/discovery-new --suite discovery --backends cuda
```

[Comparison](comparison.json) · [Parity](parity.json) · [Epochs](epochs.csv) · [Summary](measured/summary.json) · [Plan](measured/plan.json) · [Source audit](source-audit.json) · [Source](source.tar.gz) · [Raw reports](raw.tar.gz) · [Checks](contracts.xml) · [Checksums](SHA256SUMS)
