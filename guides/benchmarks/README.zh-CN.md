# Foundry 性能与正确性

[English](README.md) · [项目概览](../../README.zh-CN.md) · [研究指南](../research.zh-CN.md)

[连续入选与调权 L4 报告](program-transitions-l4-20261117/README.zh-CN.md)测量组合状态变化时的完整流程，区分真实连续入选与双成员历史组合调权。

| 历史测量 | 范围 |
| --- | --- |
| [报告复用](program-commit-l4-20261117/README.zh-CN.md) | 全部拒绝场景：256 程序查询＋提交 `42.301 → 20.851 秒`，不外推至连续入选 |
| [时间策略](timed-l4-20261116/README.zh-CN.md) | 4096 数组、256 订单程序，保留未达标的单 FP32 对照 |
| [固定 CPU](parallel-cpu-20261116/README.zh-CN.md) | 固定 Colab CPU 实例上的 8 worker 候选求值与权重试算并行 |
| [外部框架](frameworks-pro6000-20261115/README.zh-CN.md) | 同一受限日线现货 SMA 规则下对比 Foundry、Spectre、vectorbt 与 Freqtrade |
| [66 算子](ops-pro6000-20261115/README.zh-CN.md) | 66 个算子、124 个窗口条件的逐项耗时、误差与 4096 公式流程 |

测试代码与基准驱动保留在本地未跟踪的 `tests/` 和 `scripts/` 中。在授权 Linux GPU 环境安装 `--group mining` 与匹配的 CuPy 依赖组（`cuda12` 或 `cuda13`），在测试环境单独安装 `pytest`、`pytest-benchmark`、`pytest-html`，按根目录 [README](../../README.zh-CN.md#安装与构建) 编译原生扩展后执行 `python -m pytest -c tests/pytest.ini`。

GitHub Actions 仅核验保存的摘要与测试数据校验和，不执行数值计算。公开目录保留精简 Markdown、CSV 与 `summary.json`；完整原始日志与归档包留在本地 `var/runs/`。早期报告中出现的命令属于当时记录的版本（原 `foundry benchmark` 子命令已移除）。
