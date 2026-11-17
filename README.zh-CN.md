# Foundry

面向 Linux CPU 与 NVIDIA CUDA 的点时间（Point-in-Time）量化策略回测、Bar 级订单模拟、组合构建与自动策略挖掘工具。

[English](README.md) · [研究指南](guides/research.zh-CN.md) · [数据格式](guides/data.zh-CN.md) · [策略程序](guides/programs.zh-CN.md) · [基准报告](guides/benchmarks/README.zh-CN.md)

Foundry 提供统一的 Python 与命令行入口，在同一套会话中评估横截面/时序公式表达式、外部机器学习模型输出以及有状态订单策略程序，支持**美股**、**A 股**与**加密货币**市场。手工研究、外部模型验证、AI Agent 调用与自动搜索循环共用同一套交易日历、历史可交易资格掩码、费率/保证金规则及组合准入检查。

## 快速上手

```python
import foundry

with foundry.open(backend="cuda") as session:
    report = session.evaluate({
        "trend": "gt(ma_dist_20,0)",
        "momentum": "cs_rank(ret_20)",
    })
    for row in report.rows:
        print(row.id, row.ok, row.metrics.sharpe, row.decision.passed, row.decision.reason)
```

不传组合文件路径时，`foundry.open()` 默认加载仓库自带的美股离线示例数据（`./data/us_equity_daily`，研究区间 `2025-07-01` 至 `2026-06-30`）并使用内存组合。无 NVIDIA GPU 时传入 `backend="cpu"`；需要持久化保存组合时将 `.json` 路径作为第一个参数传入；接入自有历史数据时传入 `market`、`data_dir`、`start` 与 `end`。

---

## 安装与构建

本地计算运行在 **Linux**（包括 **Windows WSL2** 及 Google Colab 等云端环境）。纯远程客户端可直接运行在原生 Windows、macOS 或 Linux 上，无需编译 C++ 扩展，也无需安装本地行情读取库。

### 计算节点前置依赖

- Python `3.12` 或 `3.13` 及对应开发头文件（如 `python3-dev`）
- 支持 C++17 与 OpenMP（`libgomp`）的 `g++`
- `pip>=25.1`（用于安装 [pyproject.toml](pyproject.toml) 中声明的 PEP 735 [依赖组](https://pip.pypa.io/en/stable/user_guide/#dependency-groups)）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group runtime
```

### 按使用场景选择依赖组

| 使用场景 | 安装命令 | 安装的第三方依赖 |
| --- | --- | --- |
| 跨平台远程客户端（Windows / macOS / Linux） | `python -m pip install --group client` | `numpy`、`jsonschema` |
| 本地 CPU 回测与策略研究 | `python -m pip install --group runtime` | `client` + `pandas`、`pyarrow`、`exchange_calendars` |
| 终端实时监控面板（`foundry monitor`） | `python -m pip install --group client --group monitor` | `client` + `rich`（在计算节点上可与 `runtime` 组合安装） |
| 本地 CUDA 回测与策略研究 | `python -m pip install --group runtime --group cuda13` | `runtime` + `cupy-cuda13x`（CUDA 12 主机改用 `--group cuda12`） |
| CUDA 内置 GFlowNet 公式自动挖掘 | `python -m pip install --group mining --group cuda13` | `runtime` + `torch` + `cupy-cuda13x` |

请根据计算主机 CUDA 主版本在 `cuda12` 与 `cuda13` 中二选一（切勿同时安装）。仅内置 GFlowNet 策略网络训练需要安装 PyTorch（`--group mining`）；常规公式求值、外部模型数组评估、有状态订单程序、滚动样本外拟合（`walk_forward`）及外部候选提案器均不需要安装 PyTorch。

### 编译 C++ 原生扩展

在运行本地 CPU 或 CUDA 会话的 Linux / WSL2 / Colab 主机上，激活目标 Python 环境并在仓库根目录执行一次以下构建脚本。CUDA 内核会在首次打开 CUDA 会话时由 CuPy 自动编译。

<details open>
<summary>编译 C++ 扩展（Linux / WSL2 / Colab）</summary>

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

---

## 支持的市场与撮合模式

Foundry 通过 `foundry.Execution(model=...)` 提供两种撮合执行模型：

1. **目标仓位回测（`model="targets"`，默认）：**将横截面打分或目标权益比例映射为目标持仓，在下一根 Bar 开盘按各市场的双边费率、停牌、涨跌停、融券费或永续资金费完成调仓。
2. **Bar 级订单账户模拟（`model="orders"`）：**维护真实的现金或保证金账户，支持市价单、限价单与止损单，跨 Bar 跟踪未成交挂单、挂单资金/保证金占用、成交量参与率上限下的部分成交、分策略持仓归属、除权派息、资金费结算与强平。

| 市场（`market=`） | 目标仓位回测（`model="targets"`） | 订单账户模拟（`model="orders"`） |
| --- | --- | --- |
| **美股**（`"us_equity"`） | 日线、历史指数/股票池成分进出、拆股与分红复权、交易佣金与融券费 | 日线或常规交易时段日内现金订单、卖出所得现金 T+1 结算、拆股与到期现金股息发放 |
| **A 股**（`"a_share"`） | 日线、数据发布日期版本、停牌、各板块涨跌停限制、佣金、过户费与卖出印花税 | 日线或常规交易时段日内现金订单、买入股份 T+1 解锁可卖、手数/最小变动单位约束、停牌与涨跌停 |
| **加密现货**（`"crypto_spot"`） | 日线或自定义周期基础 Bar、历史上线/可交易资格掩码与 Maker/Taker 手续费 | 市价/限价/止损单、计价币挂单冻结与分策略现货库存归属 |
| **USDT 线性永续**（`"crypto_perpetual"`） | 日线/小时归档或自定义基础 Bar、标记价格估值、资金费结算与逐仓保证金 | 多空有符号持仓、**逐仓或全仓保证金**（`Account(margin=...)`）、资金费结算、挂单保证金占用与强平 |

### 多周期时钟与账户资金组合方式

接入日内基础 Bar（例如 5 分钟行情）时，可以将行情执行粒度、策略决策频率与绩效统计周期独立配置：

```python
import foundry

with foundry.open(
    market="crypto_perpetual",
    data_dir="/datasets/crypto_5m",
    start="2025-01-08",
    end="2025-01-31",
    backend="cuda",
    clock=foundry.Clock(decision="15min", statistics="1D"),
    account=foundry.Account(
        margin="isolated",       # "cash"、"isolated"（逐仓）或 "cross"（全仓）
        leverage=3.0,
        gross_limit=2.0,
        initial_cash=100_000.0,
        composition="shared",    # "shared"（共享总账户）或 "independent"（独立子账户）
    ),
    execution=foundry.Execution(
        model="orders",
        bar_path="open_low_high_close",
        participation=0.10,
    ),
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

- **多周期推进：**15 分钟收盘生成的决策在紧邻的下一个 5 分钟基础 Bar 开盘即可进入撮合；两次决策之间的每个 5 分钟 Bar 上，挂单触发、资金费结算与强平检查照常推进；而 Sharpe 与最大回撤按日频（`1D`）统计点结算。股票日内 Bar 须位于交易所常规连续竞价时段内。
- **共享账户与独立子账户：**`composition="shared"` 使用同一个实际账户，组合权重仅对各成员的目标仓位或下单数量做一次比例缩放；`composition="independent"`（仅订单模式）按 `initial_cash * weight` 为各成员一次性划拨独立初始资金，各自独立运行。
- **模拟边界说明：**两种模式均基于已完成 Bar 的确定性 OHLC 路径（`open_low_high_close` 或 `open_high_low_close`）与成交量参与率上限模拟成交，不模拟 L2/L3 委托簿排队优先级或 Bar 内逐笔回调。

---

## 三种策略编写方式

三种策略形式可在同一个会话中混合评估、对比、配权与提交入库。

### 1. 公式表达式（内置 66 个算子）

直接传入单个公式字符串、公式列表或 `{名称: 表达式}` 字典即可批量回测：

```python
with foundry.open(
    market="us_equity",
    backend="cuda",
    sizing=foundry.Sizing(rule="equal_slots", scale=1.0),
) as session:
    info = session.describe()
    print("可用字段:", [f["name"] for f in info["fields"]])

    report = session.evaluate({
        "ma_breakout": "gt(ma_dist_20,0)",
        "vol_adj_mom": "cs_zscore(div(ret_20,ts_std(ret_1,20)))",
    })
```

执行 `python -m foundry ops` 可查看全部 66 个内置算子（涵盖滚动矩统计、时序回归、指数/线性衰减、分位数变换、条件分支及横截面排序/标准化等），执行 `python -m foundry fields` 可查看当前市场的数据字段。`Sizing(rule="equal_slots")` 在当期有信号的资产间归一化仓位；`Sizing(rule="independent", scale=...)` 按单资产独立映射信号，不随当期持仓标的数稀释单票仓位。

### 2. 外部模型数组与滚动样本外拟合（Walk-Forward）

使用 `NumPy`、`CuPy`、`scikit-learn`、`LightGBM` 或 `PyTorch` 等外部工具生成 `(候选数, 决策步, 资产数)` 预测打分或目标持仓后，对齐 `session.grid` 并标注每个决策点所用的数据观测截止时间与模型拟合时间（UTC 纳秒 `observed_at`、`fit_at`；无训练过程时填 `foundry.NO_INFORMATION`），即可送入同一回测与准入流程：

```python
import numpy as np
import foundry

with foundry.open(backend="cuda") as session:
    port, grid = session.resident, session.grid
    xp = port.xp

    # 维度: (B=1, D=len(grid.time_ns), S=len(grid.assets))
    scores = xp.where(port.eligibility, port.features["ma_dist_20"], xp.nan)[None]
    observed = np.broadcast_to(grid.time_ns, (1, len(grid.time_ns))).copy()
    fitted = np.full(observed.shape, foundry.NO_INFORMATION, dtype=np.int64)

    registered = session.register(
        scores,
        kind="scores",           # "scores"（信号打分）或 "targets"（直接目标）
        unit="score",            # "score"、"equity_fraction"、"quantity" 或 "notional"
        ids=("external-score",),
        grid=grid,
        producer="ridge-v1",
        observed_at=observed,
        fit_at=fitted,
    )
    try:
        report = session.evaluate(registered)
        print(report.rows[0].metrics, report.rows[0].decision)
    finally:
        session.release(registered)
```

需要按时间滚动训练机器学习模型时，可直接调用 `foundry.walk_forward(session, fit, predict, folds=..., ...)`：按互不重叠的 `Fold(fit_start, fit_stop, predict_start, predict_stop)` 区间自动切分只读历史训练窗口与预测前缀，校验无未来函数越界后自动拼接并注册样本外预测结果；详见[因果 Walk-Forward 模型指南](guides/research.zh-CN.md#因果-walk-forward-模型)。

### 3. 有状态订单程序（`StrategyProgram`）

当挂撤单、止盈止损或减仓逻辑取决于实时可用现金、当前持仓库存、在途挂单或上一根 Bar 的实际成交回报时，实现 `StrategyProgram` 协议并在 `Execution(model="orders")` 下运行：

```python
from my_program import catalog, reference
import foundry

with foundry.open(
    "var/runs/program-portfolio.json",
    programs=catalog,
    market="crypto_spot",
    data_dir="/datasets/crypto_5m",
    start="2025-01-08",
    end="2025-01-31",
    backend="cuda",
    clock=foundry.Clock(decision="5min", statistics="1D"),
    account=foundry.Account(initial_cash=10_000.0, composition="shared"),
    execution=foundry.Execution(model="orders", participation=0.1),
) as session:
    print(session.plan_program(reference))
    report = session.evaluate(catalog.resolve(reference))
    session.commit(report)
```

每次评估或试算不同组合权重时，Foundry 都会为程序创建全新的私有状态并从头重放真实账户轨迹，确保路径依赖策略的资金与库存反馈准确无误。若只需单独审计某个程序的净值曲线、成交明细或使用暂停/恢复检查点（无需组合准入试算），可调用 `session.simulate(program)`。完整的可运行代码示例与部署说明见[策略程序指南](guides/programs.zh-CN.md)。

---

## 接入自有行情数据

Foundry 根据 `market`、`data_dir`、`start` 与 `end` 读取本地 Parquet 数据集，并在计算任何滚动特征与撮合日程前先按截止时间截断源数据：

1. **日线股票与加密现货表：**`panel.parquet`（长表格式，含 `date`、`symbol`、`open`、`high`、`low`、`close`、`volume` 及各市场对应的复权因子/发布日期/交易资格字段）、股票历史成分区间表 `membership.parquet`，以及记录数据来源与局限的 `manifest.json`。
2. **日内与自定义基础 Bar 数据集：**包含半开区间 `[open_ns, close_ns)` 与可见时间 `available_ns` 的 `bars.parquet`、`manifest.json`，以及永续合约的 `funding.parquet` 或股票日内公司行动表。

```python
with foundry.open(
    market="a_share",
    data_dir="/datasets/my_a_share_daily",
    start="2024-03-01",
    end="2026-06-30",
    backend="cuda",
    commission=0.00025,
    stamp_tax_sell=0.0005,
) as session:
    report = session.evaluate("cs_rank(ret_20)")
```

准备数据时请在 `panel.parquet` / `bars.parquet` 中保留 `start` 之前的一段历史行情，以便研究起点处的滚动窗口（如 20 日均线）具备完整前置样本。对于 Binance USD-M 公开永续历史归档，可直接用 `python -m foundry import-perpetual <输出目录>` 校验并导入。详细字段定义、单位约定与 CSV 转 Parquet 脚本见[数据格式指南](guides/data.zh-CN.md)；仓库内置离线测试数据说明见 [data/README.md](data/README.md)。

---

## 组合构建、诊断与自动挖掘

### 候选评估、入库提交与权重优化

`session.evaluate(...)` 为纯只读操作：它会按权重网格（默认 `Admission(weight_steps=20)` 外加解析切线权重）将候选策略与当前组合进行混合试算，检验候选是否提升组合表现并满足回撤、相关性与成员数等硬约束：

```python
with foundry.open(
    "var/runs/my_portfolio.json",
    backend="cuda",
    admission=foundry.Admission(weight_steps=20, min_sharpe_gain=0.01, max_members=40),
) as session:
    report = session.evaluate({"trend": "gt(ma_dist_20,0)", "mom": "cs_rank(ret_20)"})
    for row in report.rows:
        if not row.decision.passed:
            print(row.id, "未入选:", row.decision.reason, row.decision.stopped_at)

    decision = session.commit(report)                 # 按顺序复核并将通过准入的候选写入组合文件
    foundry.export(decision, "var/runs/commit.json")  # 导出结构化 JSON 决策记录

    print("当前组合成员数:", len(session.portfolio().sources))
    gaps = session.diagnose()                         # 诊断当前组合的回撤区间与风格缺口
```

- **结果字段区分：**`row.ok` 表示候选语法合法且顺利跑完回测（无破产或全空数据拒绝）；`row.decision.passed` 表示是否通过组合准入判据。
- **组合成员调权：**调用 `session.trial_weights(weight_matrix)` 可在不修改组合的前提下批量试算自定义的 `(试算数, 成员数)` 非负权重矩阵；随后用 `session.commit_weights(trials, index)` 提交指定权重，或调用 `session.reweight()` 自动重优化现有成员权重。
- **单次获取设备端收益序列：**在进程内会话中调用 `session.evaluate_device(inputs)`，可在单次求值中同时拿到可提交报告（`.report`）与驻留收益率数组（`.batch`）。

### 自动策略挖掘与终端监控

使用 `foundry.Discovery(session)` 可为任意公式、模型数组或策略程序计算融合了收益表现、行为去重与组合缺口信息的稠密搜索引导评分；使用 `foundry.mine(...)` 则可启动带断点续跑检查点的自动挖掘循环：

```python
import foundry

# 启动内置 GFlowNet 或随机公式搜索，检查点与日志保存在 var/runs/us_search/
foundry.mine(
    "var/runs/us_search",
    market="us_equity",
    backend="cuda",
    search=foundry.Search(mode="gfn", epochs=10, batch_size=128, accept=False),
)
```

若希望搜索过程中通过准入的策略同步写入持久组合库，可将 `Search(accept=True)` 显式开启。若需接入自定义的 LLM / Agent / 遗传算法生成器，向 `foundry.mine(...)` 传入 `proposers=(foundry.ExternalProposer(...),)` 或实现 `Proposer` 接口即可；详见[策略发现指南](guides/research.zh-CN.md#策略发现)。挖掘或服务运行期间，可在终端打开实时监控面板：

```bash
python -m foundry monitor
```

---

## 命令行工具与配置总览

所有运行参数均通过类型化的公开参数直接传入 Python 函数，或在唯一命令行入口 `python -m foundry` 中通过 `--parameters '<json>'` 传入（不使用隐式配置文件或环境变量覆盖）。调用 `foundry.schema()` 或 `python -m foundry schema schema.json` 可在不加载行情和 GPU 的情况下导出完整的参数与 JSON-RPC 协议定义。

### CLI 子命令（`python -m foundry <command>`）

| 子命令 | 功能说明 |
| --- | --- |
| `ops` / `fields` | 列出全部 66 个公式算子说明或当前市场可用字段 |
| `schema <output.json>` | 导出机器可读的参数与 JSON-RPC 协议 Schema |
| `serve` / `status` / `stop` | 启动、查看或停止本地 JSON-RPC 计算服务 |
| `attach <addr> <port>` / `detach` | 将本地 CLI 命令绑定至指定地址与端口（如 SSH 转发端口）或解除绑定 |
| `request <in.jsonl> <out.jsonl>` | 向已绑定的服务批量发送 JSON-RPC 请求（不会在客户端隐式拉起本地 GPU 进程） |
| `mine <output_dir>` | 启动带检查点的候选搜索任务（配合 `--parameters '{"backend":"cuda",...}'`） |
| `monitor` / `preview` | 启动终端实时监控看板（需安装 `--group monitor`）或渲染已保存的状态快照 |
| `calibrate <run_dir>` | 查看已完成搜索运行的目标奖励标定统计 |
| `import-perpetual <output_dir>` | 校验并导入公开 USD-M 永续合约历史归档至 Parquet 格式 |

### 核心配置组（适用于 `foundry.open` / `serve` / `mine`）

| 参数 / 配置类 | 常用选项与默认值 |
| --- | --- |
| `market` 与市场参数 | `"us_equity"`（默认）、`"a_share"`、`"crypto_spot"`、`"crypto_perpetual"`，及对应的 `data_dir`、`start`、`end` 与费率参数 |
| `backend` | `"cpu"` 或 `"cuda"` |
| `foundry.Clock` | `decision="1D"`、`statistics="1D"`（支持 `"<N>s"`、`"<N>min"`、`"<N>h"`、`"<N>D"`） |
| `foundry.Sizing` | `rule="equal_slots"`（默认）或 `"independent"`，`scale=1.0` |
| `foundry.Account` | `margin="cash" \| "isolated" \| "cross"`，`leverage=1.0`，`gross_limit=1.0`，`initial_cash=1.0`，`composition="shared" \| "independent"` |
| `foundry.Execution` | `model="targets" \| "orders"`，`bar_path="open_low_high_close" \| "open_high_low_close"`，`participation=1.0`，`order_capacity=32`，`command_capacity=32`，`quantity_step=0.0`，`price_tick=0.0` |
| `foundry.Admission` | `weight_steps=20`，`min_sharpe_gain=0.01`，`max_members=40`，`artifact_dir="./var/candidates"` |
| `foundry.Search` | `mode="gfn" \| "random"`，`epochs=5`，`batch_size=128`，`accept=False`，`objective="portfolio" \| "single"` |
| `foundry.CPU` / `foundry.CUDA` | `CPU(workers=8, candidate_batch=32)`，`CUDA(streams=8, candidate_batch=256, trial_batch=16384)` |

---

## 远程服务与跨平台客户端

当行情数据与 GPU 位于远程 Linux 服务器，而日常编写策略在本地电脑（如仅安装 `python -m pip install --group client` 的原生 Windows Python）时，可按以下三步远程连接：

**1. 在 Linux 计算主机上启动回环服务：**

```python
import foundry

foundry.serve("var/runs/library.json", backend="cuda")
```

在计算主机执行 `python -m foundry status` 查看监听端口。

**2. 在本地使用系统 SSH 建立端口转发**（将 `REMOTE_PORT` 替换为 `status` 输出的实际端口）：

```sh
ssh -N -L 58731:127.0.0.1:REMOTE_PORT user@compute-host
```

**3. 从本地 Python 客户端或命令行连接评估：**

```python
import foundry

with foundry.connect(("127.0.0.1", 58731)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    print(report.rows[0].metrics)
```

若使用命令行批量请求，可在客户端先执行 `python -m foundry attach 127.0.0.1 58731` 选择端点（`detach` 清除本地端点记录）。关闭客户端会话或断开 SSH 隧道不会终止远程计算服务；停止远程服务须显式执行 `python -m foundry stop`。远程会话支持公式、模型数组与已部署 `ProgramReference` 的规划、注册、评估、组合诊断、权重试算及入库提交；`evaluate_device`、独立账户审计 `simulate` / 检查点恢复、`walk_forward` 与 `mine` 则需在计算主机进程内会话运行。

---

## 延伸文档

- [研究与程序接入指南](guides/research.zh-CN.md) — 会话完整生命周期、内存租约管理、共享/独立账户细节、滚动样本外拟合、自定义搜索提案器与远程会话说明。
- [数据格式与转换示例](guides/data.zh-CN.md) — 各市场 Parquet 表结构、发布时间版本规则、除权派息与 CSV 转换脚本。
- [策略程序与模型部署](guides/programs.zh-CN.md) — `StrategyProgram` 编写教程、订单指令结构、成交回报处理、检查点保存恢复与远程目录部署。
- [性能与正确性基准报告](guides/benchmarks/README.zh-CN.md) — L4 / A100 / PRO 6000 / Colab CPU 实测耗时、数值误差边界及外部框架对比（[对比方法说明](guides/framework-benchmark.zh-CN.md)）。
