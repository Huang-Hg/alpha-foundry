# Foundry 性能与数值基准

[English](README.md) · [项目概览](../../README.zh-CN.md) · [研究指南](../research.zh-CN.md)

每个公开基准目录对应一次明确的硬件、数据集与数值精度验收。最新端到端状态演进报告见 [L4 连续入选与组合调权](program-transitions-l4-20261117/README.zh-CN.md)。

| 基准报告 | 测量范围与核心结论 |
| --- | --- |
| [L4 连续入选与组合调权](program-transitions-l4-20261117/README.zh-CN.md) | 组合状态发生真实变化时的连续入选与双成员历史组合调权全流程 |
| [L4 评审结果复用](program-commit-l4-20261117/README.zh-CN.md) | 全部拒绝场景下 256 个有状态程序查询＋提交 `42.301 s → 20.851 s`（`2.03×`） |
| [L4 多周期与订单策略](timed-l4-20261116/README.zh-CN.md) | 4096 个数组候选、256 个订单程序，并完整保留未达标的单 FP32 负向对照 |
| [固定 Colab CPU 并行基线](parallel-cpu-20261116/README.zh-CN.md) | 固定 Colab CPU 实例上 8 worker 候选与权重试算并行（`215.660 s → 106.645 s`） |
| [外部回测框架横评](frameworks-pro6000-20261115/README.zh-CN.md) | 同一日线现货 SMA 交易契约下对比 Foundry、Spectre、vectorbt 与 Freqtrade |
| [66 算子 CUDA 优化](ops-pro6000-20261115/README.zh-CN.md) | 全部 66 个算子、124 个窗口条件的逐项耗时、数值误差与 4096 公式端到端流程 |

## 在授权云端环境复现

回归测试与基准驱动仅保留在本地未跟踪的 `tests/` 与 `scripts/` 中，统一通过 `pytest`、`pytest-benchmark` 与 `pytest-html` 调度。在授权 Linux GPU 实例上：

1. 安装 `--group mining` 及匹配主机 CUDA 主版本的 CuPy 依赖组（`--group cuda12` 或 `--group cuda13`）。
2. 按根目录 [README](../../README.zh-CN.md#环境安装与原生扩展构建) 编译 C++ 原生扩展。
3. 运行本地 `pytest` / `pytest-benchmark` 套件并将完整输出写入 `var/runs/<run>/`。

`guides/benchmarks/` 公开目录仅保留双语 Markdown 报告、精简 CSV 表格与 `summary.json`；原始日志、JUnit/HTML 报告、`.npz` 数组及源码归档统一留在本地 `var/runs/`。GitHub Actions 仅核验已发布摘要与测试数据校验和，不启动计费算力。部分早期报告中记录的命令行（如已移除的 `foundry benchmark` 包装子命令）仅反映当时版本的调用方式。

验证工具 `pytest`、`pytest-benchmark`、`pytest-html` 在测试环境单独安装；本地测试使用 `python -m pytest -c tests/pytest.ini`。
