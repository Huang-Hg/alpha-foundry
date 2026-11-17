# Foundry

**Linux CPU 与 CUDA 上的策略评估、Bar 级订单模拟与组合准入。**

[English](README.md) · [研究指南](guides/research.zh-CN.md) · [数据格式](guides/data.zh-CN.md) · [策略程序](guides/programs.zh-CN.md) · [基准报告](guides/benchmarks/README.zh-CN.md)

Foundry 在同一会话契约下评估公式表达式、模型输出数组和有状态订单程序。人工研究、AI Agent 与自动搜索循环共用相同的交易日历、执行规则、报告结构与组合准入检查。

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

不传路径时，`foundry.open` 默认加载仓库内置的美股测试快照并使用内存组合。传入 `market`、`data_dir`、`start` 与 `end` 可切换至自有历史数据。

## 设计范围

会话打开时固定计算设备与数值精度计划，不会根据单次结果在运行期切换精度重算。已接入的 CUDA 账户核与统计归约核采用双 FP32 算术保留残差精度；CPU 作为正式后端提供显式的线程与缓存上限。有状态程序中的普通 Python 回调按原样执行，不会自动编译为 GPU 内核。

- **三种策略输入：**66 算子公式 DSL、`(B, D, S)` 评分或目标数组、以及有状态订单程序。
- **独立因果时钟：**源数据 Bar、决策时刻、下一基础 Bar 开盘成交与统计周期独立配置；外部输入显式声明观测与拟合截止时间（`observed_at`、`fit_at`）。
- **实际账户反馈：**订单模式下，挂单、部分成交、撤单、手续费、现金、持仓与逐仓保证金共同决定下一次决策可见的账户状态。
- **搜索与准入分离：**组合缺口诊断、行为去重、参数邻域检验与稠密奖励仅用于引导候选搜索，不替代 G1/G2 硬准入结论。
- **有界驻留租约：**候选批次注册后驻留设备内存供反复评估与调权，仅将需要保留的引用落盘，用毕显式释放租约。

## 市场与执行模型

| 市场 | 目标持仓回测（`model="targets"`） | 订单账户（`model="orders"`） |
| --- | --- | --- |
| 美股（`us_equity`） | 日线、历史成分资格、复权执行价、交易成本与融券费 | 当前不支持 |
| A 股（`a_share`） | 日线、发布日期版本、停牌、涨跌停限制与卖出印花税 | 当前不支持 |
| 加密现货（`crypto_spot`） | 日线或显式基础 Bar、历史资格与手续费 | 市价/限价/止损单、现金预留与成员库存归属 |
| USDT 线性永续（`crypto_perpetual`） | 日线/小时归档或显式基础 Bar、资金费与逐仓保证金 | 有符号成员持仓、资金费、保证金预留与强平 |

公式与权益比例目标数组默认使用 `Execution(model="targets")`。`Execution(model="orders")` 将公式、数组与程序统一送入加密 Bar 订单账户，并在权重或组合成员变化时从头重放账户（例如 5 分钟基础 Bar、15 分钟决策、日频统计）。共享账户与独立资金账户的差异见[执行与组合资金语义](guides/research.zh-CN.md#执行与组合资金语义)。

两种模式均基于已完成 Bar 的 OHLC 路径与成交量参与率上限建模，不模拟订单簿排队位置或逐笔成交。

## 三种输入，一个会话

```python
report = session.evaluate("gt(ma_dist_20,0)")
report = session.evaluate(registered_model_outputs)
report = session.evaluate(strategy_program)
```

第三种调用要求 `Execution(model="orders")` 且程序实现 `StrategyProgram` 协议。程序只接收独立拷贝的因果特征窗口与实际账户快照，每次查询创建新的私有状态；工件部署与租约规则见[策略程序与模型部署](guides/programs.zh-CN.md)。

评分经 `Sizing` 规则映射为仓位；直接目标声明 `equity_fraction`（订单会话还支持 `quantity` / `notional`）。数组对齐 `session.grid` 并声明观测与拟合截止点，见[数组与模型输出](guides/research.zh-CN.md#数组与模型输出)。

## 接入自有数据

将 CSV、数据库或供应商导出转换为约定的 Parquet 表结构：

```python
with foundry.open(
    market="us_equity", data_dir="/datasets/my_us_equity",
    start="2025-07-01", end="2025-12-31", backend="cuda",
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

美股使用 `panel.parquet` 与历史 `membership.parquet`。加密日内数据提供带发布时间的 `bars.parquet` 与 `manifest.json`（永续另需稠密 `funding.parquet`）。字段、单位、时间边界、资格与来源局限披露见[数据格式指南](guides/data.zh-CN.md)。[仓库内置快照](data/README.md)用作离线测试示例。

## Linux、WSL2 与远程客户端

计算运行在 Linux（含 Windows WSL2）。安装 Python 3.12 或 3.13、对应开发头文件以及支持 C++17 / OpenMP 的 GCC，在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group runtime
python -m foundry --help
```

<details>
<summary>构建原生扩展（Linux / WSL2 / Colab）</summary>

在仓库根目录、执行计算的 Linux Python 环境中运行。CUDA 内核在显式打开 CUDA 会话时由 CuPy 编译；原生 Windows 远程客户端无需本地编译。

<!-- native-build:start -->
```bash
python - <<'PY'
import subprocess
import sysconfig
import numpy

suffix = sysconfig.get_config_var("EXT_SUFFIX")
common = ["g++", "-std=c++17", "-fPIC", "-shared", "-Wall", "-Wextra",
          "-Wno-unused-parameter", "-Wno-missing-field-initializers",
          "-I" + sysconfig.get_paths()["include"], "-I" + numpy.get_include(), "-Ifoundry/include"]
targets = [
    ("ops/_ops", "ops", ["-O3", "-march=native", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing", "-fopenmp"],
     ["ops_ts.cpp", "ops_xs.cpp", "ops_inject_ts.cpp", "module.cpp"]),
    ("backtest/_bt", "backtest", ["-O3", "-march=native", "-ffast-math", "-fno-finite-math-only", "-fno-strict-aliasing", "-fopenmp"],
     ["kernel.cpp", "perpetual.cpp", "module.cpp"]),
    ("backtest/_weights", "backtest", ["-O3"], ["weights.cpp"]),
    ("backtest/_orders", "backtest", ["-O3", "-fno-fast-math", "-ffp-contract=off", "-fopenmp"], ["orders_module.cpp"]),
]
for name, module, flags, sources in targets:
    directory = "foundry/" + module + "/src/"
    subprocess.run(common + flags + ["-I" + directory] + [directory + source for source in sources]
                   + ["-o", "foundry/" + name + suffix, "-lgomp", "-lm"], check=True)
PY
```
<!-- native-build:end -->

</details>

依赖统一声明在 [pyproject.toml](pyproject.toml)，使用 [pip 依赖组](https://pip.pypa.io/en/stable/user_guide/#dependency-groups)（`pip>=25.1`）按需安装：

| 环境 | 安装命令 |
| --- | --- |
| CPU 研究或远程客户端 | `python -m pip install --group runtime` |
| CUDA 研究 | `python -m pip install --group runtime --group cuda13` |
| CUDA GFlowNet 挖掘 | `python -m pip install --group mining --group cuda13` |

按计算主机的 CUDA 主版本选择 `cuda12` 或 `cuda13`，只安装其中一组。GPU GFlowNet 挖掘需要带 CUDA 支持的 PyTorch（`mining` 已包含 `runtime`，常规研究无需安装 PyTorch）。CPU 会话传入 `backend="cpu"`。

原生 Windows Python 可直接作为远程客户端。在计算主机上指定组合路径启动服务：

```python
import foundry

foundry.serve("var/runs/library.json", backend="cuda")
```

在该主机执行 `python -m foundry status` 查看监听端口；跨机器连接时通过 SSH 转发回环端口：

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

关闭客户端不会停止服务端。[远程会话](guides/research.zh-CN.md#远程会话)通过有界 NumPy 二进制包上传数组（上限 1 GiB，禁用 `pickle`），并通过已部署摘要引用策略程序，不传输源码。

## 审阅与组合提交

```python
with foundry.open("var/runs/my_portfolio.json") as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

评估只读。报告绑定具体的研究快照与组合版本；若组合已被更新，`commit` 会抛出 `ContextConflict`，须显式调用 `refresh()` 并重新评估。`row.ok` 仅表示求值与回测完成，`row.decision.passed` 表示是否通过组合准入。

| 任务 | 调用 |
| --- | --- |
| 查看字段、算子、时钟与执行计划 | `session.describe()` |
| 单次查询同时取得驻留收益与可提交报告 | `session.evaluate_device(inputs)` |
| 注册、保留或释放候选 | `session.register(...)`、`retain(...)`、`release(...)` |
| 诊断组合缺口或检验参数邻域 | `session.diagnose()`、`judge_family(...)` |
| 试算或更新组合成员权重 | `session.trial_weights(...)`、`commit_weights(...)`、`reweight(...)` |
| 计算候选搜索奖励 | `foundry.Discovery(session).evaluate(candidates)` |
| 运行内置或外部提案循环 | `foundry.mine(output, ...)` |
| 生成并注册因果分折预测 | `foundry.walk_forward(session, fit, predict, ...)` |

`foundry.schema()` 可在不加载数据或初始化设备的情况下导出参数与 JSON-RPC 协议定义。接口细节见[研究指南](guides/research.zh-CN.md)；实测硬件、工作负载与误差范围见[基准报告](guides/benchmarks/README.zh-CN.md)。
