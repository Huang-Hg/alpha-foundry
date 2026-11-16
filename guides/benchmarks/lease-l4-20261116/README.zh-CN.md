# L4：驻留注册与选择性保存

[English](README.md) · [简体中文](README.zh-CN.md)

**注册提速 5.20× / 6.61×，落盘量约减半，完整挖掘速度基本持平。** 512 个候选的指标、奖励、学习结果和准入决策完全一致，最大数值误差为 0。

| 工作负载 | 阶段 | 旧版 | 驻留版 |
| --- | --- | ---: | ---: |
| mlp | 注册 | 0.4983 s | 0.0958 s |
| mlp | 完整挖掘循环 | 3.1989 s | 3.2169 s |
| mlp | 打开至关闭 | 5.6175 s | 5.6180 s |
| factor | 注册 | 0.4194 s | 0.0635 s |
| factor | 完整挖掘循环 | 0.8810 s | 0.8827 s |
| factor | 打开至关闭 | 1.1842 s | 1.1703 s |

旧版同样在本轮同一张 NVIDIA L4 上测量；每条件一次。使用 269 日 × 500 只美股，每种程序 4 批 × 64 候选，measurement cache 为 32 MiB，活动候选容量 64。旧版为 `2f691a9` 的批量持久注册实现。

## 生命周期

- `register` 校验 PIT，拥有有界 GPU 副本，只传回分块摘要，不传完整数据、不写 NPZ。
- `release` 结束租约，释放该批次输入；越界、外来或已释放引用明确拒绝。退出会话释放其剩余租约。
- `retain` 显式保存选定输出并返回持久引用，原租约仍须释放。组合提交先保存入选成员，发现流程先保存检查点最终引用，再释放批次。
- 不保存全部试探，不添加历史 GPU 缓存，不自动溢写。已发布检查点曾引用的工件仍保留，避免破坏恢复能力。

## 时间与空间的实际变化

注册本身大幅缩短，但检查点选择性保存分别花费 0.6156 s / 0.4134 s，包含选取、暂存传输、主机摘要及文件 IO，仍计入完整循环。它消耗了注册节省的时间，不能把注册加速写成端到端加速。阶段存在异步重叠，不能直接相加。MLP 冷启动模型准备和因子家族检验均包含在“打开至关闭”中；两种模式固定按 MLP → 因子运行，因此两者启动成本不宜横向比较。

| 工作负载 | 工件数：旧 → 新 | 落盘 MB：旧 → 新 | 采样峰值 MiB：旧 → 新 |
| --- | ---: | ---: | ---: |
| mlp | 256 → 128 | 146.06 → 73.03 | 962 → 928 |
| factor | 264 → 128 | 150.62 → 73.03 | 1254 → 1146 |

单批输出最大 32.84 MiB，运行结束活动候选为 0。输入所有权空间为 `O(candidate_batch × T × S)`，独立于历史候选总数；保留暂存受同一批量上限约束。每个 4096 字节 GPU 叶块只传回 32 字节摘要，完整数据仅在必要保存时传输。

## 一致性与范围

- Smoke 先通过，再完成 150 项用例、7206 个子检查。最后的旧永续重开用例改为显式保留后仅定点补跑；初始失败和修复日志均保留。
- 检查覆盖 GPU 摘要与 hashlib 的填充/分块边界、PIT 未来截止点、拥有副本、容量、释放失效、RPC、本地重开、四市场、资金费/强平及发布失败原子性。
- 同卡 512 个候选及 9 点因子家族的数值字段逐项相等；唯一排除的是明确升级后的内容摘要、驻留 scope 与组合/上下文版本 ID。
- MLP / 因子的 G2 通过数仍为 47 / 74；全部被后续相关性筛选挡住，挖掘最终入选仍为 0。独立驻留提交用例覆盖成功入选；本轮循环时间不代表大量成功入选重配的耗时。
- 短窗口没有有效延迟双生子，不据此宣称稳健性或样本外收益。原始数据的历史发布/修订缺口仍在 DataContract 中披露。
- 格式升级为 `research-v14` / `candidate-array-v3` / `candidate-discovery-v3`，明确拒绝旧工件，无兼容层。未改写私有 production 基线。
- 首次 smoke 的 NVRTC 头文件依赖已修复；其后生产源码未变，仅调整新契约下的测试断言。L4 已释放，未做本地数值测试。

## 复现

```bash
python -m foundry --parameters '{"market":"us_equity","search":{"epochs":4,"batch_size":64},"cuda":{"candidate_batch":64,"measurement_cache_mib":32}}' \
  benchmark var/runs/resident-inputs --suite nonformula --backends cuda
```

[Parity and timing](parity.json) · [Measured plan](measured/plan.json) · [Previous plan](previous/plan.json) · [Checks](contracts.xml) · [Targeted repair](contracts-repair.xml) · [Source audit](source-audit.json) · [Measured source](source.tar.gz) · [Raw evidence](raw.tar.gz)
