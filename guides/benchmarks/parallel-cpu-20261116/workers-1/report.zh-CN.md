# 独立后端性能测量

**PASS** · cpu

此报告记录独立基线，不宣称跨设备误差或加速比，也不代表完整回归。

OpenMP threads: 1; OpenBLAS threads: 1.

[Metadata](benchmark.json) · [Summary](summary.json) · [Operators](operators.csv) · [Flows](flows.csv) · [Checks](checks.html)

| Candidates | Evaluate / s | Commit / s | Sum / s | Lifecycle / s | Valid / accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.686726 | 2.426299 | 4.113025 | 4.250920 | 64 / 11 |
| 4096 | 107.636748 | 108.023318 | 215.660066 | 215.810862 | 4058 / 28 |

每个条件一次预热、一次测量；基线结果可与同数据、规则、计算源码的设备结果比较。
