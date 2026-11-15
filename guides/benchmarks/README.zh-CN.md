# Foundry 性能基准

[English](README.md) · [最新 CPU/CUDA 报告](current/report.zh-CN.md) · [旧 PRO 6000 记录](archive/pro6000-operator66/operator66.md)

这套基准回答三个问题：**一次实际研究调用要多久？CUDA 带来多大数值误差？接纳的候选是否与 CPU 相同？**

在授权 GPU 环境中只需一个入口：

```bash
python -m pip install -r requirements-benchmark.txt
bash build.sh
OMP_NUM_THREADS=2 python -m foundry benchmark output/benchmark-new
```

先安装与 CUDA 运行时匹配的 CuPy。计算在选定的云端实例执行；GitHub Actions 只展示和归档已测结果。

执行、计时和 HTML 验证报告分别使用 [pytest](https://docs.pytest.org/en/stable/how-to/unittest.html)、[pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/pedantic.html) 与 pytest-html。原有回归检查迁入 `validation/`，删除旧 `tests/` 和 `scripts/` 目录。

| 测什么 | 比较范围 |
| --- | --- |
| 公开研究接口 | 1、64、4096 条公式；打开、评价、接纳、关闭 |
| 算子 | 全部 66 个算子，使用相同真实输入；滚动窗口为 2、20、120 |
| 正确性 | 接纳、原因码、约束判断完全一致；逐指标误差与权重容差 |
| 回归 | PIT、A 股规则、现货、永续资金费/强平、资源释放与 RPC 一致性 |

每个性能条件预热一次、测量一次。预热与计时使用各自的新组合，不复用预热查询缓存或已修改的组合。CPU/CUDA 统一测量墙钟时间，CUDA 显式同步。微小耗时差异只是一次观测，不用于统计排名。

输出目录包含双语摘要、逐项 CSV、标准 pytest-benchmark JSON、JUnit XML、独立 HTML 验证报告、公式原始报告、源码/数据/语料身份和日志。正确性失败或基准覆盖不完整，命令会失败并保留报告。

自有数据、费用等继续使用 `foundry.open` 的公开参数，两端共享同一组参数；固定公式语料必须符合所选数据的字段词表。可用 `foundry.benchmark(output, sizes=(1, 64), market="us_equity", data_dir="...")` 限定候选数量。不同 GPU、数据和实现之间不继承速度门槛。

旧 PRO 6000 数字来自内部接口和精度实验，保留为历史记录，不混入本轮公开接口结果。本套基准不重新认证独立的 1.5× 精度开销预算。

[完整证据包](current/evidence.tar.gz)包含逐公式原始报告与准备日志，解压可恢复原始结果目录。实际执行的工作区由源码文件摘要标识，pytest 提交元数据记录其基础 checkout；JUnit 用例数包含 unittest 子用例。

[实测源码包](current/source.tar.gz)保留执行时的完整源码字节及公式语料，包括原始换行。发布的文本报告统一为 LF，保证 GitHub 校验和可复现。报告生成器另记摘要：计时结束后只改进了报告文字，没有重跑计时。
