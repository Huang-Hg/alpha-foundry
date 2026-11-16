# 固定 Colab CPU 基线：A100 与 L4

[English](README.md) · [中文](README.zh-CN.md) · [All operators](operators.csv) · [Stages](flows.csv) · [Errors](comparisons.json) · [CPU metadata](cpu.json) · [Thread pools](threadpools.json)

CPU 基准固定为独立 **Colab CPU 类型高内存运行时，Intel Xeon 2.20GHz，8 vCPU（4 核、2 线程/核）**。CPU affinity 为 0–7，cgroup 未限流；OpenMP 和 OpenBLAS 均实际加载 8 线程。此处不使用任何 GPU 附带的主机 CPU 作为分母。

4096 条公式，269 日 × 500 美股资产，32 MiB 测量缓存：

| Backend | Evaluate / s | Commit / s | Sum / s | Speedup vs fixed CPU |
| --- | ---: | ---: | ---: | ---: |
| CPU · 8 threads | 218.121768 | 181.853170 | 399.974939 | 1× |
| A100 | 2.931202 | 3.759916 | 6.691118 | 59.78× |
| L4 | 3.709663 | 4.099566 | 7.809228 | 51.22× |

完整生命周期（含打开/关闭）另列于 flows.csv；以上合计只包含评价与有序提交，包含公式算子、回测统计、准入和组合更新。

| GPU | Candidates | Fixed CPU / s | GPU / s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| A100 | 64 | 6.870593 | 0.420813 | 16.33× |
| A100 | 256 | 27.084911 | 1.830468 | 14.80× |
| A100 | 1024 | 100.423059 | 2.013612 | 49.87× |
| A100 | 4096 | 399.974939 | 6.691118 | 59.78× |
| L4 | 64 | 6.870593 | 0.403065 | 17.05× |
| L4 | 256 | 27.084911 | 2.093350 | 12.94× |
| L4 | 1024 | 100.423059 | 2.217054 | 45.30× |
| L4 | 4096 | 399.974939 | 7.809228 | 51.22× |

## 算子对照

固定 CPU 基线，窗口 20，单位 μs；所有 124 个算子/窗口条件见 CSV。

| Operator | CPU · 8 threads | A100 | L4 |
| --- | ---: | ---: | ---: |
| `ts_slope` | 201.30 | 153.57 | 179.61 |
| `ts_av_diff` | 205.49 | 125.01 | 114.46 |

## 结果与边界

所有设备的 4096 条公式均有 4058 个有效候选、28 个接纳。与固定 CPU 基线逐项比较，决策差异 0，数值超限 0；最大尺度化报告误差 7.41e-07。

CPU 先完成 64 条公式 smoke，再测 256/1024/4096 与全部 66 算子；每条件一次预热、一次实测，没有重复种子。GPU 复用上一轮 A100/L4 原始测量和逐公式报告；计算核心、数据/执行快照、参数、公式顺序一致。本次只调整基准入口、CPU 独立执行和报告元数据，没有修改生产数值核。

完整流程的指标、权重与准入直接对照新的 8 线程 CPU 报告。算子表重新计算耗时比；历史 GPU 算子误差仍来自其原同轮 CPU 对照，未保存逐元素输出以重新对照新的 CPU。单独 CPU 测量不代表完整多市场回归，也未重跑私有生产基线或独立精度预算。

8 线程是已启用的原生并行配置，并不表示整个流程始终占满 8 核：公式调度和有序提交仍有串行部分，未做线程数寻优。GPU 墙钟时间同样包含各自 Colab 主机调度。此报告衡量当前实现与所选 Colab 配置，不代表 CPU/GPU 硬件的理论上限。

## 复现与证据

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE python -m foundry --parameters '{"market":"us_equity"}' benchmark var/runs/cpu-reference --backends cpu --sizes 64 256 1024 4096
```

[Measured CPU metadata](measured/benchmark.json) · [Smoke](smoke/junit.xml) · [Checks](measured/junit.xml) · [CPU raw reports](cpu-details.tar.gz) · [Measured source](source.tar.gz) · [Exact cloud job](cloud-job.py) · [GPU evidence](../compact-multigpu-20261116/README.md)
