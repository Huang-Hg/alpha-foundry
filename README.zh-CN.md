# Foundry

**面向 CUDA 与 CPU 的时点一致（PIT）量化策略发现、账户执行模拟与组合准入基础设施。**

[English](README.md) · [研究指南](guides/research.zh-CN.md) · [数据契约](guides/data.zh-CN.md) · [策略程序](guides/programs.zh-CN.md) · [性能基准](guides/benchmarks/README.zh-CN.md)

Foundry 在同一因果研究会话中统一评价代数公式、外部模型张量与有状态发单程序。人工研究员、AI Agent 与自治挖掘循环共享完全一致的交易日历、执行规则、风险核算与组合硬准入判据。

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

默认打开仓库内置的美股测试快照（纯内存组合）。指定 `market`、`data_dir`、`start` 与 `end` 即可切换至自有历史数据。

## 核心能力

Foundry 将无状态批量评估与原子化组合状态演进严格分离。计算设备与数值精度策略在会话打开时一次性固定，绝不在运行期根据单条结果临时切换精度重算。

- **三类候选同构表达：**66 算子公式 DSL、`(B, D, S)` 评分或目标持仓张量、以及有状态 `StrategyProgram` 共享同一套 `Session` 与 `Discovery` 接口。
- **显式因果时钟：**源数据 Bar、决策时刻、下一基础 Bar 开盘成交与统计评估周期彼此独立；外部数组与策略程序须如实声明观测与拟合信息截止点（`observed_at`、`fit_at`）。
- **Bar 级真实订单账户：**在 `Execution(model="orders")` 下，市价、限价、止损、撤单与改单按声明的 OHLC 路径、成交量参与率上限、现金/库存预留、资金费结算与逐仓强平规则推进。
- **硬裁判与软引导分离：**稠密搜索奖励、跨模态行为去重与参数高原双生子检验仅用于引导提案排序与策略学习，绝不绕过 G1/G2 硬准入。
- **有界驻留租约：**模型批次一次注册驻留显存、反复评价与调权试算，仅对入选或显式 `retain` 的候选落盘，用毕通过 `release` 确定性释放。

## 支持的市场与执行模型

| 市场 | 目标持仓回测（`model="targets"`） | 实际订单账户（`model="orders"`） |
| --- | --- | --- |
| 美股（`us_equity`） | 日线、PIT 历史成分资格、复权执行价、双边成本与融券费 | 暂不支持 |
| A 股（`a_share`） | 日线、`published` 发布版本过滤、停牌、涨跌停冻结、非对称佣金与卖出印花税 | 暂不支持 |
| 加密现货（`crypto_spot`） | 日线或显式日内基础 Bar、历史资格掩码、比例手续费 | 市价/限价/止损单、现金预留与按成员归属的现货库存 |
| USDT 线性永续（`crypto_perpetual`） | 小时归档或显式日内基础 Bar、已结算资金费、逐仓保证金与强平 | 有符号成员持仓、资金费现金流、保证金预留与逐仓强平 |

`Execution(model="targets")` 为公式与权益比例目标账本的默认快速路径；`Execution(model="orders")` 则令公式、数组与有状态程序统一走加密货币 Bar 订单账户，并在组合权重或成员变化时从头重放真实账户轨迹（例如 5 分钟基础 Bar、15 分钟决策、日频统计）。共享账户与独立资金账户的资金语义见[执行与组合资金语义](guides/research.zh-CN.md#执行与组合资金语义)。

上述两种模式均基于已完成 Bar 的 OHLC 路径与成交量参与率建模，不模拟盘口订单簿排队位置或逐笔（tick-by-tick）撮合。

## 三种策略输入，同一会话接口

```python
report = session.evaluate("gt(ma_dist_20,0)")
report = session.evaluate(registered_array_batch)
report = session.evaluate(strategy_program)
```

- **公式**基于 66 算子注册表解析求值，静态检查语法树代价上限与时序嵌套深度。
- **外部数组**对齐 `session.grid`（`float32` 或 `float64` 的 `(B, D, S)` 张量），支持经 `Sizing` 映射的横截面 `scores`，或直接声明 `targets`（`equity_fraction`，订单会话还支持 `quantity` / `notional`），详见[数组与模型输出](guides/research.zh-CN.md#数组与模型输出)。
- **策略程序**实现 `StrategyProgram`（`descriptor` + `open(context)`），在每个决策点接收独立拷贝的因果特征窗口与真实账户快照，并在每根基础 Bar 结束后接收成交反馈，详见[策略程序与模型部署](guides/programs.zh-CN.md)。

## 接入自有数据

将 CSV、数据库或供应商数据整理为约定的本地 Parquet 格式即可直接加载：

```python
with foundry.open(
    market="us_equity", data_dir="/datasets/my_us_equity",
    start="2025-07-01", end="2025-12-31", backend="cuda",
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

日线股票需要 `panel.parquet` 与历史成分表 `membership.parquet`（A 股与日线现货另需 `manifest.json`）；加密货币日内数据采用 `crypto-bars-v1` 契约（`bars.parquet`、`manifest.json`，永续另需完整 `funding.parquet`）。字段口径、时区边界与完整转换脚本见[数据契约指南](guides/data.zh-CN.md)；仓库内置的[测试行情快照](data/README.md)可离线直接运行。

## 环境安装与原生扩展构建

计算后端运行在 Linux（含 Windows WSL2 与云端 GPU 实例），需准备 Python 3.12 或 3.13、Python 开发头文件以及支持 C++17 / OpenMP 的 GCC：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group runtime
python -m foundry --help
```

<details>
<summary>编译 C++ 原生扩展（Linux / WSL2 / Colab）</summary>

在仓库根目录、实际执行计算的 Linux Python 环境中运行。CUDA 内核在显式打开 CUDA 会话时由 CuPy 按需编译；Windows 原生远程客户端无需编译原生扩展。

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

全部依赖集中声明在 [pyproject.toml](pyproject.toml)，通过 [pip 依赖组](https://pip.pypa.io/en/stable/user_guide/#dependency-groups)（`pip>=25.1`）按需安装：

| 使用场景 | 安装命令 |
| --- | --- |
| CPU 研究或远程 RPC 客户端 | `python -m pip install --group runtime` |
| CUDA 研究 | `python -m pip install --group runtime --group cuda13` |
| CUDA GFlowNet 公式挖掘 | `python -m pip install --group mining --group cuda13` |

请根据主机 CUDA 主版本二选一安装 `cuda12` 或 `cuda13`。仅内置 GFlowNet 策略训练（`mining`）依赖支持 CUDA 的 PyTorch；日常研究与评估无需安装 PyTorch。使用 CPU 后端时传入 `backend="cpu"`。

### 远程服务与跨平台客户端

在计算主机上指定组合路径启动单 worker 串行服务：

```python
import foundry

foundry.serve("var/runs/library.json", backend="cuda")
```

在计算主机运行 `python -m foundry status` 查看监听端口；跨机器访问时通过 SSH 转发回环端口，即可从原生 Windows 或另一台 Linux 主机连接：

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

服务端独占计算资源，关闭客户端不会停止服务。[远程会话](guides/research.zh-CN.md#远程会话)通过 1 GiB 上限的 NumPy 二进制包上传数组（禁用 `pickle`），并通过预部署的 `ProgramReference` 摘要调用策略程序，不在网络上传输源码字符串或序列化对象。

## 只读评估与原子化组合提交

```python
with foundry.open("var/runs/my_portfolio.json") as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

评估严格只读。每份报告均绑定精确的 `ResearchContext`（数据/执行语义快照 + 组合版本）；若其他写者已更新组合，提交会抛出 `ContextConflict`，须显式调用 `refresh()` 后重新评估。注意区分 `row.ok=True`（语法与回测有效）与 `row.decision.passed=True`（通过组合硬准入）。

| 研究任务 | 公开接口 |
| --- | --- |
| 查看字段词表、算子、时钟与执行计划 | `session.describe()` |
| 一次查询同时获取驻留收益张量与可提交报告 | `session.evaluate_device(inputs)` |
| 注册、持久化保留或释放候选租约 | `session.register(...)`、`retain(...)`、`release(...)` |
| 诊断组合缺口或执行参数邻域稳健性检验 | `session.diagnose()`、`judge_family(...)` |
| 试算、提交或用 CEM 优化组合成员权重 | `session.trial_weights(...)`、`commit_weights(...)`、`reweight(...)` |
| 计算含行为归档去重的稠密发现奖励 | `foundry.Discovery(session).evaluate(candidates)` |
| 运行带断点续跑的内置或外部提案挖掘循环 | `foundry.mine(output, ...)` |
| 按严格因果滚动/扩展折训练并注册模型预测 | `foundry.walk_forward(session, fit, predict, ...)` |

调用 `foundry.schema()`（或 `python -m foundry schema OUTPUT.json`）可在不加载行情、不初始化 GPU 的情况下导出完整的机器可读协议。完整工作流见[研究指南](guides/research.zh-CN.md)，实测性能与误差边界见[性能基准报告](guides/benchmarks/README.zh-CN.md)。
