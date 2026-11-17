# L4：减少程序评价到提交的重复重放

[English](README.md)

256 个有状态程序的查询＋有序提交从 **42.301 秒降至 20.851 秒（2.03×）**；包含打开会话到关闭的完整生命周期从 **44.496 秒降至 23.116 秒（1.92×）**。收益、指标、成交/风险摘要、权重与准入结论无差异。

## 同卡前后对比

| Workload | Query before → after (s) | Commit before → after (s) | Query + commit before → after (s) | Speedup | Open through close before → after (s) | Peak device MiB before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| orders-16-shared | 7.4374 → 6.8756 | 6.8356 → 0.0012 | 14.2729 → 6.8768 | 2.076× | 16.0970 → 8.6815 | 218 → 216 |
| orders-16-independent | 8.5230 → 8.5982 | 8.5235 → 0.0013 | 17.0465 → 8.5995 | 1.982× | 18.6756 → 10.2327 | 222 → 222 |
| orders-256-shared | 21.1883 → 20.8361 | 21.1126 → 0.0152 | 42.3009 → 20.8513 | 2.029× | 44.4965 → 23.1158 | 250 → 250 |

**适用范围：这三组压力候选均未入选。** 约 2× 的收益来自消除相同上下文下的第二次独立评价和权重试探。遇到入选导致组合变化，后续候选仍必须重新重放；不能把这张表外推为所有挖掘流程都快一倍。该分支、RPC、调用方修改报告、租约释放与原子写入失败均已通过回归验证。

## 方法与正确性

同一台 NVIDIA L4（23034 MiB，驱动 580.82.07），每条件一次；同一基准驱动、原生核、数据与参数。8 个合成 crypto 现货资产，2 天 5 分钟 Bar、192 个 15 分钟决策、576 个事件、48 个小时统计周期。双 FP32，权重试探显式为 4，生产默认值未改。无新增 CPU 性能基线。

- 收益数组逐位相同；所有报告数值最大绝对误差为 0，决策差异为 0。
- 521 项测试通过、1 项旧私有生产基线跳过；另有 8778 个子测试通过。未重录生产基线。
- 美股、A 股、crypto 现货和永续各 4 个日线公式，指标、风险、准入与有序权重完全一致。
- 仅保留最近一批主机端报告；不保留收益数组、持仓或账户状态，不积累历史批次。
- 完整生命周期包含加载/计划/JIT、生成、注册、评价、提交、保留、导出、报告写盘、释放与关闭。峰值显存为 20ms 采样下界。
- 全部相邻测试使用同一 L4 会话；结果下载校验后已释放。单次测量不代表统计置信区间。

## 复现

```bash
python -m foundry benchmark BEFORE --suite programs --sizes 16 256 --backends cuda
# Run the first command with the baseline study implementation, then use the updated checkout.
python -m foundry benchmark AFTER --suite programs --sizes 16 256 --backends cuda --reference BEFORE
```

本次将 16 和 256 拆成连续命令，以便先验收小规模再运行大规模；没有重复测量。基线恢复 `cc0f2b4` 的两个 study 文件，基准驱动与其他文件共用当前版本。

[Full results](summary.json) · [Every stage](stages.csv) · Original cloud evidence (local archive) · Source manifest（本地工件） · Measured source (local archive) · Baseline delta (local archive) · Static report binding review（本地工件）
