# L4：消除保留路径的重复工作

[English](README.md) · [简体中文](README.zh-CN.md)

**相同保存内容下，完整 MLP 挖掘循环提速 1.14×，非公式因子循环提速 1.38×。** 从打开到关闭分别提速 1.08× / 1.26×；512 个候选的指标、奖励、学习和准入决策完全一致。

| 程序 | 阶段 | 旧版 | 当前 | 加速比 |
| --- | --- | ---: | ---: | ---: |
| mlp | 保留与落盘 | 0.5560 s | 0.2038 s | 2.73× |
| mlp | 完整挖掘循环 | 3.2526 s | 2.8479 s | 1.14× |
| mlp | 打开至关闭 | 5.5941 s | 5.1692 s | 1.08× |
| factor | 保留与落盘 | 0.4122 s | 0.1773 s | 2.33× |
| factor | 完整挖掘循环 | 0.8761 s | 0.6368 s | 1.38× |
| factor | 打开至关闭 | 1.1505 s | 0.9119 s | 1.26× |

对照版本为 `0f8d249`。两版在本轮同一 NVIDIA L4 上各跑一次：269 日 × 500 只美股，每种程序 4×64 候选。注册、生成、评价、学习、有序准入、检查点必要保存、checkpoint 发布与资源关闭均在完整计时内。MLP 与因子按固定顺序运行；不同程序的启动成本不宜横向比较。阶段会重叠，不能直接相加。

## 实际删除的工作

- 新文件写入直接使用注册时确定的内容身份，删除 CPU 端对每个 4 KiB 分块的重复哈希；读取已有工件仍验证内容和来源。
- 按原租约分组，连续候选直接借用原 GPU 切片，只有不连续选择才收集数据。
- 主机上的 PIT 截止数组保持在主机，删除先上传 GPU、拼接、再下载的往返。
- `ArrayStore.put(batch, sources)` 显式接收已验证批次及对应身份。调用方和验证套件同步迁移，不保留旧签名。高层 Session API、工件格式及候选摘要不变。

## 保存承诺与空间不变

检查点仍逐批发布，归档容量与保存候选相同。MLP 和因子均保存 128 个工件，分别为 73,032,192 / 73,029,632 字节，与对照逐项相同。必要保存仍在返回前完成，没有写回积压队列。批量上限仍为 64，测量缓存仍为 32 MiB，不增加历史缓存。

采样显存峰值：MLP 928 → 928 MiB，因子 1220 → 1146 MiB。结束时活动候选数为零。空间仍受批量和已有缓存预算约束，不随累计候选数量增长。

## 验证与局限

Smoke 后，131 项用例和 7108 个子检查全部通过，无需补跑。覆盖首次保存不重哈希、连续切片复用、PIT、四市场、RPC、资金费/强平、损坏工件拒绝、释放失效、恢复与原子提交。完整轨迹比较的最大数值误差为 0；内容摘要、上下文/组合版本也一致，仅运行时随机 scope 不参与比较。

G2 通过数仍为 MLP 47、因子 74；后续相关性筛选后最终挖掘入选为零。成功入选由独立提交用例覆盖，因此本轮测速不代表大量入选重配场景。短数据窗口没有有效延迟双生子，不作为稳健性或样本外收益证据。每条件仅一次，未估计性能置信区间。原有缺失价格除零 warning 保留在原始日志中。

仅在 Colab 数值验收；L4 已释放。私有 production 基线未改写，未推送或合并。

## 复现

```bash
python -m foundry --parameters '{"market":"us_equity","search":{"epochs":4,"batch_size":64},"cuda":{"candidate_batch":64,"measurement_cache_mib":32}}' \
  benchmark var/runs/retention --suite nonformula --backends cuda
```

[Timing and exact parity](parity.json) · [Measured plan](measured/plan.json) · [Previous plan](previous/plan.json) · [Contracts](contracts.xml) · [Source audit](source-audit.json) · [Measured source](source.tar.gz) · [Full trajectories and logs](raw.tar.gz)
