# Foundry 性能基准

[English](README.md) · [2026-11-15 PRO 6000 报告](arch-v2-pro6000-20261115/report.zh-CN.md) · [重构前公开接口报告](current/report.zh-CN.md) · [更早的算子实验](archive/pro6000-operator66/operator66.md)

[时间策略 L4 报告](timed-l4-20261116/README.zh-CN.md)覆盖 4096 个数组候选、256 个有状态程序和独立 Agent 接入。查询＋有序提交分别耗时 3.718 秒、41.886 秒；完整规模双 FP32 的 CPU/CUDA 数值误差在容差内，判定无差异。轻量 FP32 对照不合格，失败结果完整保留。统一入口的 `--suite timed` 支持先做小规模精度资格检查，再跑完整规模。

[GPU 驻留采样报告](gfn-resident-l4-20261116/README.zh-CN.md) 测量采样状态、合法动作和抽样留在 GPU、复用轨迹、优化器只初始化一次后的效果：4096 提案循环 7.889 → 6.168 秒；含准备和会话生命周期为 8.921 → 8.586 秒。报告保留数值误差、显存增长和随机候选差异，前版记录见 [PyTorch 迁移验收](gfn-torch-l4-20261116/README.zh-CN.md)。

这套基准回答三个问题：**一次实际研究调用要多久？CUDA 带来多大数值误差？接纳的候选是否与 CPU 相同？**

A100/L4 的对比统一使用[固定 Colab CPU 基线](parallel-cpu-20261116/README.zh-CN.md)：固定同一台独立 CPU 类型运行时，比较 1/8 个实际执行候选和试算的 worker；不使用某张 GPU 附带的主机作为 CPU 基准。此前 fixed-cpu 报告只有算子内部线程，保留为历史记录。

[小缓存与多 GPU 对比](compact-multigpu-20261116/README.zh-CN.md) 将此前的完整持仓缓存改为 32 MiB 收益/报告预算，让成员持仓空间独立于候选总数，并以相同输入比较各 GPU 的耗时、显存和决策。

[2026-11-16 TMA 与测量复用报告](history-pro6000-20261116/README.zh-CN.md) 比较 `ts_slope`、`ts_av_diff` 和查询到提交的提速，验证 4096 条公式决策一致，并披露显存成本和实际 Blackwell 批量搬运指令。

[CUDA 慢算子优化](ops-pro6000-20261115/README.zh-CN.md) 对同一台 PRO 6000 上的修改前后结果逐项比较：递推、回填和移位等算子提速，4096 条公式的报告数值、权重和决策一致，同时列出仍慢于 CPU 的用例。

[原生框架对比](frameworks-pro6000-20261115/README.zh-CN.md) 使用相同日线现货输入和交易规则，单独比较 Foundry、Spectre、vectorbt Numba/Rust 与 Freqtrade。共用入口为 `python -m foundry benchmark OUTPUT --suite frameworks`，安装与口径见[方法说明](../framework-benchmark.zh-CN.md)。该受限对比测量账户执行，不包含组合准入。

[架构重构验收](arch-v2-pro6000-20261115/README.zh-CN.md) 在 RTX PRO 6000 Blackwell Server（97887 MiB）上测量源码 `fa2acfd`。公开回归为 512 项通过、1 项跳过、7061 个子用例通过，耗时 247.05 秒；JUnit 的 7574 个条目包含子用例。66 个算子的 124 个窗口用例全部通过。4096 条公式完整用例 CUDA 耗时 2.879063 秒、CPU 耗时 93.723696 秒，加速 32.55 倍；两端均接纳 28 个候选，无决策不一致或数值误差超限。独立私有生产审计在未改动的旧工件上失败，没有通过重录基线掩盖。

在指定运行时使用统一入口：

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m foundry benchmark var/runs/benchmark-new --backends cpu
```

示例使用 8 vCPU 的 CPU 类型运行时，`cpu.workers` 应匹配实际 CPU affinity 与配额，OpenMP/BLAS 内层线程设为 1。`--backends cpu` 不需要 CuPy 或显卡，只运行基准契约检查与测量；指定 GPU 使用 `--backends cuda` 和匹配版本的 CuPy。CPU 速度对照固定使用独立 CPU 类型运行时，GPU 主机上的 CPU 正确性重放不作为性能基线。计算在选定的云端实例执行；GitHub Actions 只展示和归档已测结果。

执行、计时和 HTML 验证报告分别使用 [pytest](https://docs.pytest.org/en/stable/how-to/unittest.html)、[pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/pedantic.html) 与 pytest-html。原有回归检查迁入 `validation/`，删除旧 `tests/` 和 `scripts/` 目录。

| 测什么 | 比较范围 |
| --- | --- |
| 公开研究接口 | 1、64、4096 条公式；打开、评价、接纳、关闭 |
| 算子 | 全部 66 个算子，使用相同真实输入；滚动窗口为 2、20、120 |
| 正确性 | 接纳、原因码、约束判断完全一致；逐指标误差与权重容差 |
| 回归 | PIT、A 股规则、现货、永续资金费/强平、资源释放与 RPC 一致性 |

默认公式/算子套件每个性能条件预热一次、测量一次。预热与计时使用各自的新组合，不复用预热查询缓存或已修改的组合。时间策略套件则每条件测量一次完整生命周期，包含初始化，不额外预热。CPU/CUDA 统一测量墙钟时间，CUDA 显式同步。微小耗时差异只是一次观测，不用于统计排名。

输出目录包含双语摘要、逐项 CSV、标准 pytest-benchmark JSON、JUnit XML、独立 HTML 验证报告、公式原始报告、源码/数据/语料身份和日志。正确性失败或基准覆盖不完整，命令会失败并保留报告。

自有数据、费用等继续使用 `foundry.open` 的公开参数，两端共享同一组参数。通过 `backends=("cpu",)` 或 CLI `--backends cpu` 选择测量设备，不使用研究参数 `backend`；默认测量两端。固定公式语料必须符合所选数据的字段词表。可用 `foundry.benchmark(output, backends=("cpu",), sizes=(1, 64), market="us_equity", data_dir="...")` 限定候选数量。不同 GPU、数据和实现之间不继承速度门槛。基准 schema v2 明确记录后端与验证范围，独立测量本身不宣称跨设备一致性或加速比。

新公开结果包包含报告、逐项 CSV、pytest 证据、运行元数据与实测源码包；完整逐公式原始证据保留在本地。CUDA 账户使用双 FP32，未测同卡单 FP32 对照，因此本次不重新验收独立的 1.5 倍精度开销预算。

`current/` 名称沿用历史，但目录内容是未改动的重构前归档。其中的[完整证据包](current/evidence.tar.gz)包含那次测量的逐公式原始报告与准备日志；源码摘要标识其实际工作区，pytest 提交元数据记录基础 checkout。更早的算子实验使用内部接口和实验精度路径；两类旧记录都不混入本次架构重构结果。

历史[实测源码包](current/source.tar.gz)保留此前执行的完整源码字节及公式语料，包括原始换行。其发布文本统一为 LF，保证 GitHub 校验和可复现；报告生成器另记摘要，因为此前只在计时后改进文字，没有重跑计时。
