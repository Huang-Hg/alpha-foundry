# Foundry

**CUDA 加速的量化研究与回测框架。**

**简体中文** · [English](README.md) · [研究与接口指南](guides/research.zh-CN.md)

用 GPU 批量计算公式、回测策略并筛选组合，让研究者、AI agent 和挖掘程序更快地探索大量候选。支持自己的行情数据、ML 信号和目标持仓，也提供 CPU 后端。可在 Linux、Windows 的 WSL2 环境运行计算，或从 Windows 原生 Python 连接远端计算服务。

```python
import foundry

with foundry.open() as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    print(report.rows[0].metrics)
```

默认使用附带美股数据和 CUDA。数据、日期与计算参数都可以在调用时调整，会话负责加载数据、准备计算和释放资源。

## 为什么用 Foundry？

- **CUDA 加速完整研究流程。** 从公式算子、账户回测到指标和组合权重试探，批量利用 GPU，并复用数据与显存中的计算结果。大量候选可以一起评价，减少逐条回测的等待。
- **公式、模型、自定义程序都能接入。** 66 个算子覆盖算术、滚动统计、排序和回归；已有 ML 模型或挖掘程序可直接提交信号、目标权重或永续敞口。
- **自己的数据，自己的研究范围。** 将日线行情整理为约定格式即可接入，自选资产和研究区间。附带数据方便上手，更换供应商无需重写回测数值核。
- **从单条策略走到组合选择。** 比较候选加入组合后的净收益、风险与权重，明确返回接纳或拒绝的原因，减少另写组合筛选流程的工作。
- **人和 agent 都能快速调用。** 一个会话提供评价、邻域探索、模型注册、权重试探和提交；本机与远程使用相同操作，并提供机器可读的接口说明和结果。

### 完整研究流程的实测速度

**500 只美股 × 269 天、4096 条公式：同一 Colab PRO 6000 上，CPU 完整调用用时 97.69 秒，CUDA 用时 2.95 秒，约快 33.1 倍。** 包含会话打开、公式评价、回测指标、组合接纳和资源释放；两端均接纳相同的 28 条候选。

单条公式两端都约 0.09 秒；64 条公式时 CUDA 约快 6.8 倍。加速收益取决于批量，不把大批量数字套用到每次调用。这是一次预热后测量，使用内存组合；不含安装、原生构建、RPC 和磁盘保存。CPU 使用 2 个 OpenMP 线程，详见[性能、误差与决策一致性报告](guides/benchmarks/current/report.zh-CN.md)。

## 接入自己的行情数据

**可以使用自己的数据，不绑定仓库样本或特定供应商。** 对日线股票和加密货币现货，将 CSV、数据库或 API 数据转换为支持的 Parquet 表结构，打开会话时指定数据和研究区间：

```python
import foundry

with foundry.open(
    market="us_equity",
    data_dir="./data/my_us_equity",
    start="2025-01-01",
    end="2025-12-31",
    backend="cuda",
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

美股目录提供日线价格、成交量与复权因子的 `panel.parquet`，以及历史资格区间 `membership.parquet`。A 股另外区分信号价格与执行价格；加密货币现货记录每日资格。数据中应保留研究起点之前的历史，供滚动特征预热。

**[自有数据接入指南：文件、字段、CSV 转换与来源记录](guides/research.zh-CN.md#接入自己的数据)**

仓库同时提供[四类市场的测试快照](data/README.md)，方便初次体验。接入自己的快照后，可以在数据覆盖范围内选择资产与研究区间。

永续还需要小时成交价、标记价、已结算资金费，以及适配器要求的 manifest 和事件证据；内置 Binance 导入器可以生成这套格式，详见[永续数据接入](guides/research.zh-CN.md#永续数据)。

## 支持哪些市场？

| 市场 | 引擎处理的规则 |
| --- | --- |
| 美股 | 日线信号、历史资格、复权价格与交易费用 |
| A 股 | 历史资格、停牌、涨跌停、次日执行与卖出税费 |
| 加密货币现货 | UTC 日线、连续交易日历、365 日年化与交易费用 |
| USDT 线性永续 | 多空敞口、逐仓抵押、已结算资金费、杠杆与小时标记价强平 |

日线收盘决策最早在次日开盘执行。永续采用固定费用/维持保证金参数和小时不利价格触达模型，当前不重建历史风险档位及交易所逐笔执行。详细规则与数据限制见[研究指南](guides/research.zh-CN.md)。

## 快速开始

Linux、Colab 和 Windows 的 WSL2 环境可运行计算。准备 Python 3.12/3.13、Python 开发头文件，以及支持 C++17/OpenMP 的 GCC，在该环境内安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash build.sh
python -m foundry --help
```

调用 `foundry.open(backend="cuda")` 使用 GPU，`backend="cpu"` 使用 CPU；CUDA 还需安装与环境匹配的 CuPy。通过 `session.describe()` 查看当前字段、算子和实际参数。

Windows 原生 Python 也可以作为客户端连接已有服务，无需在客户端编译数值核或安装 CUDA。安装 Python 依赖后，用 `foundry.connect(("127.0.0.1", port))` 连接本机或隧道转发的服务端口；端口可通过服务端的 `python -m foundry status` 查询。

## 从评价到组合，用同一个接口

批量评价后，显式提交结果进行组合准入。传入组合文件路径即可保存并在下次打开时恢复；不传路径则在内存中研究：

```python
import foundry

with foundry.open("production/my_portfolio.json") as session:
    report = session.evaluate({
        "trend": "gt(ma_dist_20,0)",
        "longer_trend": "gt(ma_dist_60,0.05)",
    })
    decision = session.commit(report)
    foundry.export(decision, "output/decision.json")
```

| 你要做什么 | 调用方式 |
| --- | --- |
| 评价一条或一批公式 | `session.evaluate(formulas)` |
| 注册模型输出 | `session.register(values, ...)` |
| 探索公式附近的参数与字段 | `session.explore(formulas, hypothesis="trend")` |
| 查看当前组合 | `session.portfolio()` |
| 比较一批组合权重 | `session.trial_weights(weights)` |
| 接纳候选或调整权重 | `session.commit(report)` / `session.commit_weights(trials, index)` |
| 导出结果 | `foundry.export(report, path)` |
| 启动挖掘任务 | `foundry.mine(output, ...)` |

接入 ML 模型或外部挖掘器时，用 `session.register` 注册批量输出，然后交给相同的 `evaluate` 与 `commit`：

| 输入 | 用途 |
| --- | --- |
| `signals` | 提交模型分数，由 Foundry 转换为持仓 |
| `target_weights` | 提交自己计算的股票或现货多头权重 |
| `target_exposure` | 提交永续的有符号名义本金/净值目标敞口 |

`session.grid` 提供日期和资产坐标；你声明模型使用数据和拟合的截止点，会话管理其余研究上下文。远程调用会自动传输数组，不需要安排服务器文件路径。完整示例见[公开 Python 接口](guides/research.zh-CN.md#公开-python-接口)。

AI agent 可调用 `foundry.schema()` 发现机器协议、`session.describe()` 查看当前可用字段和算子。将 `foundry.open()` 换成 `foundry.connect(...)`，即可使用远端服务完成同样的研究操作。

## 简单调用，也能细调

常用参数直接传入，批量、精度、准入和挖掘设置使用参数组；无需配置文件，也不必修改源码：

```python
with foundry.open(
    market="crypto_perpetual",
    leverage=3.0,
    cost_rate=0.0004,
    cuda=foundry.CUDA(candidate_batch=64, precision="paired"),
    admission=foundry.Admission(min_sharpe_gain=0.02),
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

参数在会话创建时固定，比较不同设置时打开新的会话。全部参数及默认值可以通过 `foundry.schema()` 查询，[使用指南](guides/research.zh-CN.md#公开-python-接口)提供模型接入、权重调整和远程调用示例。

[测试数据](data/README.md) · [研究与接入指南](guides/research.zh-CN.md) · [性能与精度](guides/benchmarks/README.zh-CN.md)
