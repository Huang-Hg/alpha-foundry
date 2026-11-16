# TMA 历史数据复用与查询到提交加速

[English](README.md) · [中文](README.zh-CN.md) · [66 operators](optimized/operators.csv) · [Full report](optimized/report.md) · [Decision comparison](before-after.json) · [Measured source](source.tar.gz)

同一张 Colab PRO 6000、同一份美股输入（269 日 × 500 资产），4096 条公式评价＋有序提交从 **2.6609 秒降至 2.3515 秒，提速 1.13 倍**。

| 算子（窗口 20） | CPU / μs | 旧 CUDA / μs | 新 CUDA / μs | CUDA 提速 |
| --- | ---: | ---: | ---: | ---: |
| `ts_slope` | 132.87 | 233.17 | 152.52 | 1.53× |
| `ts_av_diff` | 89.07 | 118.29 | 82.74 | 1.43× |

完整 2/20/120 日窗口见 [target-operators.csv](target-operators.csv)。计时包含转置打包、TMA 搬运和主机同步，不仅是内核执行。

`ts_av_diff` 在三个窗口均快于 CPU。`ts_slope` 相比旧 CUDA 提速 1.46–1.66 倍，但仍比 CPU 慢约 8%–33%。这些单次观测不构成统计排名。

| 4096 条公式 CUDA 阶段 | 修改前 / s | 修改后 / s |
| --- | ---: | ---: |
| open_s | 0.074404 | 0.070756 |
| evaluate_s | 1.172996 | 1.064080 |
| commit_s | 1.487911 | 1.287382 |
| evaluate_commit_s | 2.660907 | 2.351462 |
| close_s | 0.035720 | 0.044667 |
| lifecycle_s | 2.771104 | 2.466954 |

设备整体采样峰值显存由 5824 MiB 变为 11886 MiB；这是采样下界，不是分配峰值。新增缓存以显存换取重复计算减少。

采用 TMA 批量 global→shared 异步搬运与事务屏障，4 只资产交错排列，共用一个 warp 的 8 个时间段。历史在共享内存复用，最终除法移到并行逐元素步骤；累加和除法顺序不变。实际 CuPy 编译缓存的机器码含 `UBLKCP.S.G`，见 [gpu-feature.json](gpu-feature.json)。本轮不改变算术精度。

实现依据：[NVIDIA 异步搬运文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。TMA 在计算能力 9.0 及以上可用；更早设备使用同布局的协作加载。超出共享内存预算的长历史使用同一递推定义的全局内存读取路径，启动前确定。其他设备未在本轮测速。

测量缓存从“最多一个计算批次”改为按内存预算保存完整分配批次。公开参数 `CUDA(measurement_cache_mib=8192)` 可调整或设为 0；GPU 打开会话时将预算限制为可用显存的 20%。缓存键仍绑定快照、来源、精度和延迟。提交保持 CAS，并在每次接纳后根据新组合重新判断准入。四市场用例检查查询后的提交不再调用候选测量，同时与冷缓存有序提交完全一致。

4096 条公式修改前后：最大报告数值变化 0，决策、权重及最终组合一致；有效 4058，接纳 28。完整验收：561 passed, 1 skipped, 5 warnings, 7184 subtests passed in 245.72s。66 算子、124 个窗口条件的误差与符号/NaN/无穷大掩码全部通过，1/64/4096 流程的 CPU/CUDA 比较全部通过。

本次会话先跑 smoke 和旧版对照。首个线程布局未通过性能门禁（[记录](target/benchmark.json)）；移出除法后通过针对性复验（[记录](target-normalize/benchmark.json)），再运行一次最终完整回归和统一 benchmark。每次运行每条件一次预热、一次测量，无重复种子。基线为 `e0a281f`，构建脚本只转换为 LF。私有生产基线未重录且明确跳过，独立单 FP32 的 1.5 倍精度预算未重新验收。短历史结果不证明长历史性能；简单算子启动开销和随组合版本变化的准入计算仍存在。
