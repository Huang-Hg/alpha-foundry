# L4：明确批次归属，GPU 反馈直接交给学习器

[English](README.md) · [简体中文](README.zh-CN.md)

**层间接口重构后，同一批 4096 个普通提案的完整挖掘循环为 5.781 秒，前版为 6.168 秒；包含打开、准备和关闭为 8.145 秒，前版为 8.586 秒。4288 次含探针评价的报告数值最大差为 0，接纳和重配结果完全相同。**

这次解决的是接口所有权：提案器构造不再分配 GPU 资源，`mine` 负责打开和关闭；批次绑定具体生产者、研究上下文和候选顺序；反馈通过 CUDA 事件和 DLPack 直接交给学习器。通用驱动器不再传 GFN 学习率/蒸馏参数或假定每组探针有 12 个候选。

| 阶段 | 前版 L4 | 本版 L4 |
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

挖掘墙钟减少 6.3%，总耗时减少 5.1%。学习与提交/重配重叠，阶段耗时不能直接相加。每个版本只有一次测量；前版复用已保存证据，未额外重跑，不作统计显著性或单项改动因果加速结论。

## 对比口径

- NVIDIA L4 23034 MiB；269 日×500 美股，249 个统计日；同一数据快照、交易/准入规则、参数及初始模型。
- 8 轮×512 个普通提案，另有 192 个探针候选；实际轮内去重评价 2968 个，权重试算 44376 次，重配试算 903 次。
- 完整循环包含采样、评价/奖励、学习、有序接纳、重配、遥测与检查点。数据加载和策略准备单独列示，并纳入打开至关闭的总时间；模型/优化器的确定性关闭也纳入总时间。
- Smoke 包含首次编译；正式测量在独立进程复用编译缓存。没有 CPU 对照，也没有扩大缓存。

## 一致性和资源

- Smoke、63 项检查及 6954 个子用例通过，契约检查耗时 17.69 秒；随后仅运行一次完整流程。
- 候选顺序、全部报告和奖励、8 次接纳、8 次重配及训练指标与前版逐项一致；最大绝对差 0，最终模型参数摘要相同。
- 奖励公式、TB/教师/SGD 数学语句经 AST 对比相同；探针采样、账户、算子和硬准入源码不变。独立参考梯度最大误差 1.11e-6，教师缩放/裁剪后的单步更新误差 1.19e-7。
- 检查了跨生产者、重排/过期批次和重复学习拒绝，以及跨线程 CUDA 流顺序、DLPack 指针一致性和异常关闭。GPU 反馈不经过 Python 标量回传；面向报告的主机读取仍保留。
- 每个生产者最多一个活动批次；新增反馈空间为 O(B)，本次最终反馈向量最大 536×4 字节，轨迹仍只保留当前批次，没有历史回放缓存。测量缓存保持 32 MiB。
- PyTorch 张量峰值 36.17 → 35.67 MiB；全 GPU 采样峰值 5774 → 5772 MiB，整体显存基本持平。
- `foundry.mine` 调用方式保持简洁；自定义提案器需采用新的 `open/propose/learn/release` 协议。旧检查点明确拒绝，没有兼容转发层。
- L4 已释放；未重录 production 基线。

```bash
python -m foundry --parameters '{"search":{"epochs":8,"batch_size":512},"cuda":{"measurement_cache_mib":32}}' \
  benchmark var/runs/discovery-new --suite discovery --backends cuda
```

[Comparison](comparison.json) · [Parity](parity.json) · [Epochs](epochs.csv) · [Summary](measured/summary.json) · [Plan](measured/plan.json) · [Source audit](source-audit.json) · [Source](source.tar.gz) · [Raw reports](raw.tar.gz) · [Checks](contracts.xml) · [Checksums](SHA256SUMS)
