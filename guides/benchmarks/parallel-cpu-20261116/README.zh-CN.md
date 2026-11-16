# 固定 Colab CPU：真正并行的候选计算与权重试算

[English](README.md) · [简体中文](README.zh-CN.md)

同一台 CPU 类型运行时上，4096 条公式的评价＋有序提交由 **215.660 秒降到 106.645 秒，提升 2.02 倍**。1/8 worker 的报告数值误差为零，候选顺序、准入决策和最终权重一致；有效 4058 条，接纳 28 条。

## 端到端耗时

评价包含公式算子、信号转持仓、账户回测、统计和针对当前组合的权重试算；提交包含必要的重评、随接纳演进的组合试算和有序发布。计时从驻留会话开始，排除打开、加载及预热；最后一列另列打开到关闭的完整耗时。

| 4096 条公式 | 评价 | 有序提交 | 合计 | 打开到关闭 |
| --- | ---: | ---: | ---: | ---: |
| CPU 1 worker | 107.637 s | 108.023 s | 215.660 s | 215.811 s |
| CPU 8 worker | 54.997 s | 51.648 s | 106.645 s | 106.783 s |
| A100 | 2.931 s | 3.760 s | 6.691 s | 6.934 s |
| L4 | 3.710 s | 4.100 s | 7.809 s | 8.098 s |

A100/L4 相对此固定并行 CPU 分别快 **15.94× / 13.66×**。64 条公式 CPU 从 4.113 秒降到 2.098 秒；A100/L4 为 0.421/0.403 秒。

## 并行与内存

CPU 为 **AMD EPYC 7B12，8 vCPU（虚拟机报告 4 核×2 线程）**，affinity 0–7，无 cgroup 时间配额。两种 worker 数使用同一实例、同一源码、同一编译产物和顺序。大批量实际 CPU 时间/墙钟时间从 0.99 核升至 **3.50 核**，不是只设置线程环境变量。

仍是一个进程，通过共享内存工作线程并行运行释放 GIL 的 C++ 算子和账户核。每个 worker 内 OpenMP 为 1，BLAS 为 1；组合提交保持单写者，不能并行改变接纳顺序。

CPU 候选批次为 32，GPU 为 256；双方紧凑收益/报告缓存均为 32 MiB。CPU 最多同时挂起 8 个任务，不复制全量行情到多个进程，不保留公式中间数组或完整试算持仓缓存。试算临时持仓空间为 O(workers×T×S)，不是 O(候选数×权重点数×T×S)；必要的收益/事件输出另按其输出规模占用空间。周期采样读到 CPU 1/8 worker 的 VmHWM 为 612.5/636.4 MiB，这是观测下界，不是完整峰值追踪。

## 验收范围与误差

- CPU smoke：69 项、6693 个子用例通过，9.58 秒；覆盖并发执行、异常排空、四市场账户、资金费及强平。现金账户和永续事件的串/并行对照通过逐位检查。
- 每个 worker 条件：158 项、6660 个子用例通过，包含全部 66 算子的 124 个窗口用例与 64/4096 条公式流程。每个条件一次预热、一次测量，没有多种子重复。
- 1/8 worker 报告最大绝对误差为 0；新旧 CPU 报告及 A100/L4 报告均通过语义比较，准入/原因码/权重选择无不一致。
- GPU 对 CPU 的最大尺度归一化误差为 7.40983567e-07；归一化为 abs(actual-reference)/max(1,abs(reference))。逐字段误差和阈值记录在 comparisons.json。
- 本次不是全量 GPU/RPC/私有 production 回归；未修改或重录私有生产基线。

A100/L4 复用 2026-11-16 的同数据、同公式、同交易规则实测，未再开 GPU。CUDA 数值核和准入规则源码未变；共享公式调度仅将原有流及缓存分配移到资源对象。源码差异名单和 GPU 原始报告哈希随结果保存。这是跨设备的已存实测对照，不声称本次重测 GPU 调度。此前 Xeon 400 秒结果来自不同 CPU，不用来计算本轮提速。

输入为 269 日×500 美股，公式来自固定 4096 条语料；成本 0.0005，收盘决策、次日开盘执行，现金账户含实际费用。CPU 使用原生算术，GPU 使用固定双 FP32，不按结果复算。

## 复现与工件

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m foundry \
  --parameters '{"market":"us_equity","cpu":{"workers":8}}' \
  benchmark var/runs/cpu-parallel-new --backends cpu --sizes 64 4096
```

[Flow timings](flows.csv) · [Semantic comparisons](comparisons.json) · [Summary](summary.json) · [Source](source.tar.gz) · [Raw CPU reports and logs](cpu-details.tar.gz) · [Checksums](SHA256SUMS)

[1 worker](workers-1/report.md) · [8 workers](workers-8/report.md) · [GPU evidence](../compact-multigpu-20261116/README.md)
