# Foundry

**CUDA 策略研究：从一个想法，到实际账户和经过审阅的组合。**

[English](README.md) · [研究指南](guides/research.zh-CN.md) · [自备数据](guides/data.zh-CN.md) · [策略程序](guides/programs.zh-CN.md)

输入公式、模型生成的数组，或根据实际模拟成交作出下一步决策的程序。人工、AI Agent 和自动发现循环共用同一个会话、市场规则、报告与组合准入判据。

```python
import foundry

with foundry.open(backend="cuda") as session:
    report = session.evaluate({
        "trend": "gt(ma_dist_20,0)",
        "momentum": "cs_rank(ret_20)",
    })
    for row in report.rows:
        print(row.id, row.metrics, row.decision)
```

示例使用仓库内的美股快照。更换 `data_dir` 和日期即可研究自己的市场历史。

## 用 GPU 扩展研究规模

CUDA 批量执行公式算子、账户模拟、统计和组合权重试算，并让工作数组驻留设备。它适合在同一行情面板上反复探索大量候选。CUDA 策略程序也可将私有张量保留在设备上；普通 Python 回调不会自动编译成 GPU 内核。

会话启动时固定设备与数值计划。支持的账户和统计路径使用双 FP32 保留残差精度。CPU 同样是正式后端，可显式控制 worker 和内存容量。

- **三种策略表达：**66 算子公式、数值评分或目标，以及有状态订单程序。
- **明确的因果时钟：**源 Bar、决策时刻、下一基础开盘执行和统计周期相互独立，观测与拟合截止时间显式声明。
- **实际账户反馈：**订单、部分成交、撤单、费用、现金、持仓和逐仓保证金共同驱动下一次决策。
- **统一发现流程：**诊断组合缺口、评价提案、检验参数家族、重放权重试算，再通过硬准入提交候选。
- **有界复用：**注册一次模型或数组批次，反复评价，保留选定引用，最后释放租约。

## 选择需要的执行方式

| 市场 | 目标持仓回测 | 实际订单账户 |
| --- | --- | --- |
| 美股 | 日线、历史资格、复权执行价和费用 | 当前不支持 |
| A 股 | 日线、资格、停牌、涨跌停和卖出税 | 当前不支持 |
| 加密现货 | 日线或显式基础 Bar、历史资格和费用 | 市价/限价/止损单、现金和成员持仓归属 |
| USDT 线性永续 | 日线/小时归档或显式基础 Bar、资金费与逐仓保证金 | 有符号成员持仓、资金费、资金预留和强平 |

公式和预计算权益比例目标默认使用 `Execution(model="targets")`。`Execution(model="orders")` 让公式、数组和程序共用 crypto Bar 实际账户与新的完整组合重放。可配置 5 分钟源 Bar、15 分钟决策和日统计。[执行与组合语义](guides/research.zh-CN.md#执行与组合资金语义)说明共享账户和独立资金账户的区别。

这些是具有明确 OHLC 路径与成交量参与率的 Bar 模型，不模拟订单簿排队或逐 tick 成交。

## 三种输入，一个会话

```python
report = session.evaluate("gt(ma_dist_20,0)")
report = session.evaluate(registered_model_outputs)
report = session.evaluate(strategy_program)
```

最后一种需要订单会话与符合公开协议的程序。程序只收到因果特征窗口和实际账户，每次查询创建独立私有状态。[完整程序示例](guides/programs.zh-CN.md)同时说明工件部署、远程引用和租约归属。

评分通过显式仓位规则映射；直接目标声明 `equity_fraction`，订单会话还接受 `quantity`/`notional`。数组对齐 `session.grid`，并声明观测与拟合信息的实际可用时间，见[数组注册](guides/research.zh-CN.md#数组与模型输出)。

## 接入自己的数据

把 CSV、数据库或供应商导出转换为约定的 Parquet 文件：

```python
with foundry.open(
    market="us_equity", data_dir="/datasets/my_us_equity",
    start="2025-07-01", end="2025-12-31", backend="cuda",
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

美股需要 `panel.parquet` 与历史 `membership.parquet`。加密日内数据提供带发布时间的基础 Bar 和 manifest；永续另需标记价及显式资金费记录。[数据格式与转换示例](guides/data.zh-CN.md)覆盖字段、单位、时间端点、资格和来源记录。[仓库内快照](data/README.md)可作参考，加载时无需在线下载行情。

## Linux、WSL2 或远程 GPU

计算运行在 Linux，包括 Windows 上的 WSL2。安装 Python 3.12 或 3.13、对应开发头文件和支持 C++17/OpenMP 的编译器，在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash build.sh
python -m foundry --help
```

CUDA 需要合适的 NVIDIA 驱动，以及匹配计算主机 CUDA 主版本的 CuPy 发行包。GPU GFlowNet 挖掘还使用支持 CUDA 的 PyTorch。CPU 计算在打开会话时选择 `backend="cpu"`。

原生 Windows Python 可直接作为客户端，不必编译数值扩展。计算主机上使用明确的组合路径启动服务：

```python
import foundry
foundry.serve("var/runs/library.json", backend="cuda")
```

在该主机执行 `python -m foundry status` 查询端口。跨机器使用时通过 SSH 转发 loopback 端口，再从 Windows 或 Linux 连接：

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

`port` 是服务端口或隧道本地端口。服务持有计算资源，关闭客户端不会停止服务。[远程会话](guides/research.zh-CN.md#远程会话)也支持已部署程序引用，不通过研究请求传输 Python 源码或 pickle 对象。

## 审阅后再修改组合

```python
with foundry.open("var/runs/my_portfolio.json") as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

评价只读。报告记录准确的研究与组合版本；过期提交需要显式刷新并重新评价。`ok=True` 表示评价成功，`row.decision.passed` 才是准入结论。

| 研究任务 | 公开调用 |
| --- | --- |
| 查看数据、参数和执行计划 | `session.describe()` |
| 一次查询同时取得驻留收益与可提交报告 | `session.evaluate_device(inputs)` |
| 注册、保留或释放候选 | `register(...)`、`retain(...)`、`release(...)` |
| 诊断缺口或检验相邻变体 | `diagnose()`、`judge_family(...)` |
| 比较或提交组合权重 | `trial_weights(...)`、`commit_weights(...)`、`reweight(...)` |
| 为候选提案评分 | `foundry.Discovery(session).evaluate(candidates)` |
| 运行内置或外部提案循环 | `foundry.mine(output, ...)` |
| 生成因果模型分折预测 | `foundry.walk_forward(session, fit, predict, ...)` |

`foundry.schema()` 提供机器可读接口。[研究指南](guides/research.zh-CN.md)介绍完整流程；[基准报告](guides/benchmarks/README.zh-CN.md)区分实际测量的工作负载、硬件和数值适用范围。
