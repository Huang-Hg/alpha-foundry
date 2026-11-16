# 独立后端性能测量

**PASS** · cpu

此报告记录独立基线，不宣称跨设备误差或加速比，也不代表完整回归。

OpenMP threads: 8; OpenBLAS threads: 8.

[Metadata](benchmark.json) · [Summary](summary.json) · [Operators](operators.csv) · [Flows](flows.csv) · [Checks](checks.html)

| Candidates | Evaluate / s | Commit / s | Sum / s | Lifecycle / s | Valid / accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 14.181038 | 12.903873 | 27.084911 | 27.314381 | 254 / 17 |
| 1024 | 56.662372 | 43.760687 | 100.423059 | 100.644347 | 1013 / 21 |
| 4096 | 218.121768 | 181.853170 | 399.974939 | 400.219412 | 4058 / 28 |

每个条件一次预热、一次测量；基线结果可与同数据、规则、计算源码的设备结果比较。
