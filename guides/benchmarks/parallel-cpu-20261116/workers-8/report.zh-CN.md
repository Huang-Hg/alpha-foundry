# 独立后端性能测量

**PASS** · cpu

此报告记录独立基线，不宣称跨设备误差或加速比，也不代表完整回归。

OpenMP threads: 1; OpenBLAS threads: 1.

Metadata（本地工件） · [Summary](summary.json) · [Operators](operators.csv) · [Flows](flows.csv) · Checks（本地工件）

| Candidates | Evaluate / s | Commit / s | Sum / s | Lifecycle / s | Valid / accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.869723 | 1.227880 | 2.097603 | 2.244804 | 64 / 11 |
| 4096 | 54.997264 | 51.647825 | 106.645089 | 106.782930 | 4058 / 28 |

每个条件一次预热、一次测量；基线结果可与同数据、规则、计算源码的设备结果比较。
