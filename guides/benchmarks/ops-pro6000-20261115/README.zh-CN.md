# CUDA 慢算子优化

[English](README.md) · [中文](README.zh-CN.md) · [Before](baseline/report.md) · [After](optimized/report.md) · [Before/after decisions](before-after.json) · [Measured source](source.tar.gz)

同一台 Colab PRO 6000、同一份美股输入（269 日 × 500 资产），比较修改前后的 66 个算子、124 个窗口用例。4096 条公式的评价＋有序提交由 **2.8047 秒降至 2.7699 秒**。

| 算子（窗口 20） | CPU / 微秒 | CUDA 修改前 / 微秒 | CUDA 修改后 / 微秒 | CUDA 提速 |
| --- | ---: | ---: | ---: | ---: |
| `ts_std` | 240.05 | 437.27 | 233.44 | 1.87x |
| `ts_slope` | 132.51 | 325.81 | 236.41 | 1.38x |
| `ts_av_diff` | 87.60 | 195.70 | 121.42 | 1.61x |
| `ts_backfill` | 33.13 | 76.33 | 22.51 | 3.39x |
| `ts_ref` | 27.72 | 28.21 | 19.05 | 1.48x |
| `ts_delta` | 37.16 | 34.71 | 20.53 | 1.69x |
| `ts_corr` | 1101.91 | 872.47 | 421.53 | 2.07x |
| `ts_cov` | 701.61 | 448.81 | 318.29 | 1.41x |

上表单位为微秒，CPU 列来自修改后同轮对照；完整窗口 2/20/120 见 [comparison.csv](comparison.csv)。仍有 24 个窗口用例的 CUDA 慢于 CPU：abs, add_const, div, gt, inv, is_nan, left_tail, mul_const, neg, not, not_equal, right_tail, sign, square, sub, ts_av_diff, ts_backfill, ts_skew, ts_slope, ts_std。单次观测不构成统计排名，端到端约 1.2% 的变化尤其不应视为稳定收益。

修改前后 CUDA 的 1/64/4096 候选报告均通过逐项比较：决策、权重和最终组合一致，报告数值最大绝对变化为 0。每轮 CPU/CUDA 的算子误差、符号/NaN/无穷大掩码及准入比较均通过。有效候选 4058，接纳 28。

递推核将输出分为时间块，每块仍按原顺序累计历史，保留极端值退出窗口后的数值行为；回填直接并行查找，移位和差分合为一次启动。未改交易规则、准入参数或算术策略。简单逐元素算子的单次启动成本仍是差距；本轮没有实施公式融合或宣称全部算子快于 CPU。

验收：560 项通过、7159 个子用例通过、1 项私有基线跳过；针对性 smoke 为 15 项通过、74 个子用例通过。基线源于本地提交 `5d4d0e1`，修改后的实际源码保存在 source.tar.gz，逐文件摘要见 benchmark.json。证据取回后已释放 Colab，无活动会话。

每条件一次预热、一次测量；修改前后各一轮完整用例，未作随机种子重复。加载/JIT 与评价＋提交分列。私有生产基线未重录且按约定跳过；独立的同卡单 FP32 1.5 倍对照仍未验收。本轮结果不证明其他设备或更长历史上的性能。
