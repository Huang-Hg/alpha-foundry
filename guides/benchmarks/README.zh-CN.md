# 基准测试报告

[English](README.md) · [项目概览](../../README.zh-CN.md) · [研究指南](../research.zh-CN.md)

各硬件环境下的实测耗时与 CPU/CUDA 数值一致性记录。每个目录在 `summary.json` 中记录了对应的设备、数据集、精度策略、分段耗时与误差范围。

| 报告 | 测试范围 |
| --- | --- |
| [订单账户（L4）](order-accounts-l4-20261117/README.zh-CN.md) | 逐笔 Maker/Taker 费率、USDT 全仓保证金及美股/A 股日内现金订单 |
| [连续入选与调权（L4）](program-transitions-l4-20261117/README.zh-CN.md) | 组合状态变化时的有状态程序连续入选与双成员历史组合调权 |
| [程序提交复用（L4）](program-commit-l4-20261117/README.zh-CN.md) | 上下文未变时的 256 个程序查询＋提交（`42.301 → 20.851 秒`） |
| [多周期与订单策略（L4）](timed-l4-20261116/README.zh-CN.md) | 5 分钟基础 Bar 上的 4,096 数组与 256 订单程序，及未达标的单 FP32 对照 |
| [CPU 多线程并行（Colab CPU）](parallel-cpu-20261116/README.zh-CN.md) | 固定 CPU 实例上 4,096 条美股公式的 1 worker 与 8 worker 并行对比 |
| [外部回测框架对比（PRO 6000）](frameworks-pro6000-20261115/README.zh-CN.md) | 同一日线现货 SMA 规则下对比 Foundry、Spectre、vectorbt 与 Freqtrade（[方法说明](../framework-benchmark.zh-CN.md)） |
| [66 算子基准（PRO 6000）](ops-pro6000-20261115/README.zh-CN.md) | 66 个算子、124 个窗口条件的 CPU/CUDA 单算子耗时与 4,096 公式流程 |
