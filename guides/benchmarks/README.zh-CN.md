# Foundry 性能与正确性

[English](README.md)

[连续入选与调权 L4 报告](program-transitions-l4-20261117/README.zh-CN.md)对比组合状态变化时的完整流程，明确区分真实连续入选与双成员历史组合调权，并在报告内说明测试口径。

| 历史测量 | 内容 |
| --- | --- |
| [报告复用](program-commit-l4-20261117/README.zh-CN.md) | 全部拒绝时，256 程序查询＋提交 42.301 → 20.851 秒；不外推到连续入选 |
| [时间策略](timed-l4-20261116/README.zh-CN.md) | 4096 数组、256 程序；保留不合格精度对照 |
| [固定 CPU](parallel-cpu-20261116/README.md) | 固定 Colab CPU、真实候选与试探并行 |
| [外部框架](frameworks-pro6000-20261115/README.zh-CN.md) | Spectre、vectorbt、Freqtrade 的共同交易契约 |
| [66 算子](ops-pro6000-20261115/README.zh-CN.md) | 逐项耗时、误差与完整公式流程 |

测试代码和驱动留在本地，不随仓库发布；执行、计时与测试报告使用 pytest、pytest-benchmark、pytest-html。依赖通过 `python -m pip install --group benchmark --group cuda13` 安装，构建按根目录 README 执行。GPU 组应匹配实际 CUDA 版本。

GitHub Actions 只校验和展示保存的结果，不跑数值任务。新报告记录硬件、规模、精度、完整用例耗时、误差与决策；完整原始工件和含测试／任务源码的压缩包保留本地。历史报告中的命令属于当时版本，已移除的 `foundry benchmark` 不是当前公开接口。
