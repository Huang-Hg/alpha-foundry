# L4：非 GFN 模型与非公式因子完整接入

[English](README.md) · [简体中文](README.zh-CN.md)

**两类 GPU 程序均通过原有 `mine` 接口完成批量输出、注册、评价、驻留反馈和学习，没有改动通用驱动器，也没有调用 GFN 或公式解析器。** 各跑一次 4×64 候选流程：MLP 挖掘循环 3.489 秒，张量因子 1.201 秒。

MLP 使用 8→16→1 网络，逐段用历史前缀拟合 20 日收益标签，输出目标持仓。一个独立的 GPU 奖励代理网络根据评价反馈指导下一批超参。因子路径直接用 GPU 张量、交互特征和横截面排序产生信号，GPU CEM 更新参数分布。它们都使用相同的归档和硬准入规则。

| 阶段 | 批量 MLP | 张量因子 + CEM |
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

阶段存在包含和重叠关系，不能相加替代墙钟时间。完整循环包含生成、注册、评价、学习、提交检查、重配调用、遥测和检查点；本轮没有搜索接纳，也没有实际重配试算。MLP 准备包含首个 PyTorch 优化器启动成本，两类案例顺序运行，不能将时间差全部解释为算法优劣。

## 准入结果要分两层看

| 结果 | MLP | 张量因子 |
| --- | ---: | ---: |
| 候选数 / G1 有效数 | 256 / 256 | 256 / 256 |
| G2 硬准入通过 | 47 | 74 |
| 搜索最终接纳 | 0 | 0 |
| 与参考收益的相关性范围 | 0.90071–0.92467 | 0.90872–0.99918 |

现有搜索上限为 `rho_max=0.9`，所有候选均超限，因此硬准入通过的候选也没有被搜索选中。其余候选未满足硬准入指标。没有放宽任何门槛，也没有把通过 G2 等同于最终入选。

另在独立内存会话中，各选择一个已经 G2 通过的数组候选，通过公开 `evaluate → commit` 验证成功提交：查询保持只读、提交原子更新、重复提交旧报告触发 CAS 冲突。单独评价与原批量评价的准入数值差为 0。这两次是直接 API 契约检查，不计入挖掘接纳数，不改变搜索相关性筛选。

## 非公式因子邻域

显式 `CandidateGrid` 的九个参数邻点全部通过硬准入。默认延迟双生子返回 0/9，协议给出的结论为“成立”；但本数据只有 269 行，而最短延迟为 365 行，双生子全部变成空持仓。这只证明接口与判据执行成功，**不能作为有效的负对照稳健性证据**。没有另拉数据或更改探针规则掩盖这个限制。

## 正确性、空间和瓶颈

- 先 smoke（两类各 2×4），再 21 项检查和 6844 个子用例，随后两类各运行一次 4×64。无额外种子重复或 CPU 性能对照。
- MLP 批量与逐候选计算的最大绝对误差 2.98e-8；张量因子误差 0；驻留奖励与报告误差 0。污染未来特征和标签后，历史预测不变，截止点违例为 0。
- MLP 奖励代理网络每轮损失都下降；因子 CEM 参数分布根据 GPU 奖励更新。检查了输出类型、DLPack 指针、模型参数变化、检查点精确续跑及资源关闭。
- 峰值 PyTorch 张量占用分别约 147.2 / 268.1 MiB；全 GPU 采样峰值 924 / 1214 MiB。最大当前输出批次约 32.84 MiB，候选驻留容量 64，测量缓存 32 MiB，无历史轨迹回放缓存。
- 注册保留现有内容寻址契约：GPU→主机哈希和 NPZ 落盘，再持有设备副本。注册占 MLP 循环约 23.8%，占因子循环约 67.1%；对轻量因子，这是主要瓶颈。持久工件约 146 / 151 MB（十进制），不属于显存缓存。
- 同一 L4、269 日×500 美股。Smoke 含首次编译，正式任务复用磁盘编译缓存；每条件只有一次测量，不作统计显著性结论。
- 预测模型按因果折叠训练，但研究期超参/因子选择是样本内自适应过程；这不是样本外选模或收益证明。原始数据公告/修订时间不足的限制沿用 DataContract。
- L4 已释放；生产组合基线未修改。

```bash
python -m foundry --parameters '{"search":{"epochs":4,"batch_size":64},"cuda":{"candidate_batch":64,"measurement_cache_mib":32}}' \
  benchmark var/runs/nonformula-new --suite nonformula --backends cuda
```

[Analysis](analysis.json) · [Epochs](epochs.csv) · [Summary](measured/summary.json) · [Plan](measured/plan.json) · [Submission checks](submission.json) · [Family](measured/factor/family.json) · [Source audit](source-audit.json) · [Source](source.tar.gz) · [Raw reports](raw.tar.gz) · [Checks](contracts.xml)
