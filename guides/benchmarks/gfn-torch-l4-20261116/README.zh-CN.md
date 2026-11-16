# L4：PyTorch GPU GFlowNet 迁移验收

[English](README.md) · [简体中文](README.zh-CN.md)

**迁移完成，但这轮端到端没有加速：4096 个普通提案耗时 7.889 秒，旧实现为 5.032 秒；含会话打开和关闭为 8.921 秒。** 8 轮共 4288 次提案（含 192 个探针），轮内去重后评价 2953 次，完成 8 次硬准入接纳。

模型、TB 损失、配对教师、反向传播和 SGD 已迁到 Python/PyTorch，在 L4 上使用 FP32；旧 C++ 学习器和构建目标已删除。CPU 仍负责语法状态和公式文本，CEM 提案仍使用原有原生小核，行情评价、回测及权重试算使用 CUDA。

| 阶段 | 旧 C++ 学习器流程 | PyTorch GPU 流程 |
| --- | ---: | ---: |
| 采样 / Sampling | 0.160 s | 1.588 s |
| 评价与奖励 / Evaluation and rewards | 2.943 s | 3.030 s |
| 训练 / Training | 1.107 s | 2.206 s |
| 有序提交 / Ordered commit | 0.151 s | 0.233 s |
| 重配 / Reweight | 0.226 s | 0.246 s |
| Mining wall time | 5.032 s | 7.889 s |
| Open through close | 6.186 s | 8.921 s |

训练首轮耗时 1.636 秒，包含该进程首次训练准备；后续每轮平均 0.0814 秒，旧流程后七轮平均 0.1398 秒。采样从 0.160 秒增至 1.588 秒，是明确的回退项，涉及 Python 语法处理与 CPU/GPU 交互；未做内核级归因，不能把其中某一项视为已单独测得的瓶颈。阶段存在重叠，不能相加代替墙钟时间。

这是相同 L4 型号、行情、规则和参数下的流程对照，旧结果复用已保存证据，不是相同公式序列的纯训练核加速比。随机数实现与 FP32 训练改变了候选轨迹；不能把两边都接纳 8 条解释为接纳了相同策略。

## 数值和契约验收

- Smoke 通过后，46 项针对性检查和 6756 个子用例通过，用时 94.44 秒；随后仅测一次 8×512 流程。
- 固定轨迹对独立参考算式：loss 绝对误差 6.79e-06，梯度最大绝对误差 1.16e-06；含教师梯度缩放和裁剪的单步更新最大误差 1.17e-07。
- 教师票数、KL/置信门控、四个不同的共享后缀、语法预算、检查点恢复及 GPU 参数驻留检查通过。每轮批内 loss 下降；每次接纳均有匹配的硬准入通过记录。
- 奖励、Archive、稳健性、准入、账户及算子源码不变，研究快照一致；不是宣称不同候选集的指标或最终权重相同。

## 资源与复现

- L4 23034 MiB，269 行×500 美股，249 个统计交易日；账户/统计双 FP32，策略学习单 FP32。GPU 采样峰值 4340 MiB，PyTorch 张量分配峰值 30.08 MiB。
- 缓存仍为 32 MiB；只保留当前批次的策略轨迹，不增加历史回放缓存。无独立 CPU 对照，无重复种子。
- Smoke 包含首次 CuPy 编译和 PyTorch 训练准备；正式用例复用磁盘编译缓存，但在独立进程执行，因此首轮训练准备仍计入。
- 旧原生检查点明确拒绝。新检查点使用 JSON 和数值数组，不使用 pickle。此次不重录 production 基线。
- 历史窗口内的自适应发现不是样本外收益验证；默认 mine 不自动执行独立 judge_family 审查。

```bash
python -m foundry --parameters '{"search":{"epochs":8,"batch_size":512}}' \
  benchmark var/runs/discovery-new --suite discovery --backends cuda
```

[Comparison and errors](comparison.json) · [Epochs](epochs.csv) · [Summary](measured/summary.json) · [Plan](measured/plan.json) · [Source audit](source-audit.json) · [Source](source.tar.gz) · [Raw reports](raw.tar.gz) · [Checks](contracts.xml) · [Checksums](SHA256SUMS)
