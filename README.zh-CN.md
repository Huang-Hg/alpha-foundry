# Foundry

运行在 Linux CPU 与 CUDA 上的量化回测、Bar 级订单模拟与组合研究库，支持美股、A 股和加密货币。

[English](README.md) · [研究指南](guides/research.zh-CN.md) · [数据格式](guides/data.zh-CN.md) · [策略程序](guides/programs.zh-CN.md) · [基准报告](guides/benchmarks/README.zh-CN.md)

同一个会话下可以直接评估因子公式、外部模型预测数组和带账户状态的下单程序，统一按市场交易日历、历史可交易范围、手续费与保证金规则计算收益并检查组合准入条件。

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

不传组合文件路径时，`foundry.open()` 默认加载仓库自带的美股示例数据（`./data/us_equity_daily`，`2025-07-01` 至 `2026-06-30`）并使用内存组合。无 NVIDIA GPU 时传入 `backend="cpu"`；需要落盘保存组合时将 `.json` 路径作为第一个参数传入；使用自有历史数据时指定 `market`、`data_dir`、`start` 和 `end`。

## 安装与构建

本地回测与计算运行在 Linux、Windows WSL2 或云端 Linux Notebook。仅连接远程服务的客户端可直接运行在原生 Windows、macOS 或 Linux 上，无需编译 C++ 扩展或安装行情读取库。

计算节点需要 Python 3.12 或 3.13 及对应头文件（`python3-dev`）、支持 C++17 与 OpenMP（`libgomp`）的 `g++`，以及 `pip>=25.1`（用于安装 [pyproject.toml](pyproject.toml) 中的依赖组）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group runtime
```

| 环境 | 安装命令 | 安装的第三方包 |
| --- | --- | --- |
| 远程客户端（Windows / macOS / Linux） | `python -m pip install --group client` | `numpy`、`jsonschema` |
| 本地 CPU 回测与研究 | `python -m pip install --group runtime` | `client` + `pandas`、`pyarrow`、`exchange_calendars` |
| 终端监控面板（`foundry monitor`） | `python -m pip install --group client --group monitor` | `client` + `rich`（在计算节点上可与 `runtime` 一并安装） |
| 本地 CUDA 回测与研究 | `python -m pip install --group runtime --group cuda13` | `runtime` + `cupy-cuda13x`（CUDA 12 环境改用 `--group cuda12`） |
| CUDA 内置 GFlowNet 公式挖掘 | `python -m pip install --group mining --group cuda13` | `runtime` + `torch` + `cupy-cuda13x` |

`cuda12` 与 `cuda13` 按主机 CUDA 主版本二选一，不要同时安装。PyTorch（`--group mining`）仅在训练内置 GFlowNet 公式生成网络时用到；常规公式回测、外部模型数组、订单程序、`walk_forward` 滚动拟合与外部提案器均不依赖 PyTorch。

### 编译 C++ 扩展

运行本地 CPU 或 CUDA 会话前，在仓库根目录用目标 Python 环境编译一次 C++ 扩展。CUDA 内核会在首次打开 CUDA 会话时由 CuPy 自动编译。

<details open>
<summary>编译命令</summary>

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

## 市场与撮合模式

默认模式 `Execution(model="targets")` 为目标仓位回测，将横截面打分或目标资金占比换算为仓位并在下一根 Bar 开盘调仓。`Execution(model="orders")` 运行 Bar 级订单账户，支持市价、限价与止损单、跨 Bar 挂单留存、现金与保证金占用、成交量参与率上限下的部分成交、分策略持仓记录、除权派息、资金费结算与强平。

| 市场（`market=`） | 目标仓位回测（`model="targets"`） | 订单账户（`model="orders"`） |
| --- | --- | --- |
| `us_equity`（美股） | 日线、历史成分股范围、拆股分红复权、佣金与融券费 | 日线或常规交易时段日内现金单、卖出资金 T+1 交收、拆股与现金分红到账 |
| `a_share`（A 股） | 日线、数据发布日期版本、停牌、涨跌停、佣金、过户费与卖出印花税 | 日线或常规交易时段日内现金单、买入股份 T+1 解锁、手数/步长约束、停牌与涨跌停 |
| `crypto_spot`（加密现货） | 日线或自定义基础 Bar、历史交易资格与 Maker/Taker 费率 | 市价/限价/止损单、计价币挂单冻结与分策略现货持仓 |
| `crypto_perpetual`（USDT 永续） | 日线/小时归档或自定义基础 Bar、标记价格估值、资金费与逐仓保证金 | 多空持仓、逐仓或全仓保证金（`Account(margin=...)`）、资金费、挂单保证金与强平 |

### 多周期时钟与账户设置

使用日内基础 Bar（如 5 分钟线）时，可以分别指定策略决策周期、基础 Bar 撮合粒度与指标统计周期：

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
        margin="isolated",       # "cash"、"isolated" 或 "cross"
        leverage=3.0,
        gross_limit=2.0,
        initial_cash=100_000.0,
        composition="shared",    # "shared" 或 "independent"
    ),
    execution=foundry.Execution(
        model="orders",
        bar_path="open_low_high_close",
        participation=0.10,
    ),
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

15 分钟收盘产生的决策在下一个 5 分钟基础 Bar 开盘进入撮合；两次决策之间的每个 5 分钟 Bar 上，挂单触发、资金费与强平照常更新；Sharpe 和回撤按日频（`1D`）收盘统计。股票日内 Bar 须处于交易所常规连续竞价时段内。

`composition="shared"` 表示组合成员共用一个账户，成员权重只对下单数量或目标仓位缩放一次；`composition="independent"`（订单模式）按 `initial_cash * weight` 为各成员分配独立子账户，细节见[执行与组合资金语义](guides/research.zh-CN.md#执行与组合资金语义)。两种模式均按已完成 Bar 的 OHLC 路径和成交量参与率上限撮合，不模拟逐笔盘口排队。

## 编写策略

### 公式表达式

`session.evaluate` 接受单条公式字符串、公式列表或 `{名称: 公式}` 字典：

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

命令行运行 `python -m foundry ops` 可列出全部 66 个滚动与横截面算子，`python -m foundry fields` 可查看当前市场字段。`Sizing(rule="equal_slots")` 在当期持仓标的间等权分摊仓位；`Sizing(rule="independent", scale=...)` 按单标的独立缩放，不除以当期持仓个数。

### 外部模型数组与 Walk-Forward 训练

外部统计或机器学习库（`NumPy`、`CuPy`、`scikit-learn`、`LightGBM`、`PyTorch`）生成的预测或目标仓位，按 `(候选数, 决策步, 资产数)` 对齐 `session.grid` 后注册，并注明每一步的数据观测截止时间与模型拟合时间（UTC 纳秒 `observed_at`、`fit_at`；未做模型拟合时填 `foundry.NO_INFORMATION`）：

```python
import numpy as np
import foundry

with foundry.open(backend="cuda") as session:
    port, grid = session.resident, session.grid
    xp = port.xp

    scores = xp.where(port.eligibility, port.features["ma_dist_20"], xp.nan)[None]
    observed = np.broadcast_to(grid.time_ns, (1, len(grid.time_ns))).copy()
    fitted = np.full(observed.shape, foundry.NO_INFORMATION, dtype=np.int64)

    registered = session.register(
        scores,
        kind="scores",           # "scores" 或 "targets"
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

做滚动样本外训练时，可以用 `foundry.walk_forward(session, fit, predict, folds=..., ...)` 按不重叠的 `Fold(fit_start, fit_stop, predict_start, predict_stop)` 切分训练与预测区间，自动记录 `observed_at` / `fit_at` 并注册拼接后的样本外预测，用法见[因果 Walk-Forward 模型](guides/research.zh-CN.md#因果-walk-forward-模型)。

### 有状态订单程序

当报撤单、止损调整或平仓逻辑依赖实时现金、持仓、未成交挂单或上一轮成交回报时，实现 `StrategyProgram` 并在 `Execution(model="orders")` 下运行：

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

每次评估或试算组合权重都会新建程序状态并从头重放账户。只需单独查看某个程序的净值、成交明细或做暂停/恢复检查点时，调用 `session.simulate(program)`。完整示例与部署方式见 [guides/programs.zh-CN.md](guides/programs.zh-CN.md)。

## 使用自有数据

Foundry 按 `market`、`data_dir`、`start` 和 `end` 读取本地 Parquet 目录。股票与现货日线使用 `panel.parquet`（股票另配 `membership.parquet` 历史成分表及 `manifest.json`）；日内数据使用带 `[open_ns, close_ns)` 与 `available_ns` 时间戳的 `bars.parquet`、`manifest.json`，以及对应的资金费或除权派息表。

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

Parquet 表中应保留 `start` 之前的一段历史行，保证起始点处的滚动窗口有足够样本。公开的 Binance USD-M 历史归档也可通过 `python -m foundry import-perpetual <输出目录>` 校验导入。各表字段与 CSV 转换脚本见 [guides/data.zh-CN.md](guides/data.zh-CN.md)，内置示例数据集列表见 [data/README.md](data/README.md)。

## 组合管理与策略搜索

`session.evaluate(...)` 不修改组合文件：它按混合权重网格（默认 `Admission(weight_steps=20)` 加解析切线点）把候选策略与现有组合做试算，检查收益提升、回撤、相关性与成员上限。需要保存通过准入的候选时调用 `session.commit(report)`：

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

    decision = session.commit(report)
    foundry.export(decision, "var/runs/commit.json")

    print("当前成员数:", len(session.portfolio().sources))
    gaps = session.diagnose()
```

报告行中 `row.ok` 表示回测正常跑完（无语法错误、缺失数据或破产拒绝），`row.decision.passed` 表示是否通过组合准入。调整已有成员权重可调用 `session.trial_weights(weight_matrix)` 配合 `session.commit_weights(trials, index)`，或直接调用 `session.reweight()`。进程内会话还可通过 `session.evaluate_device(inputs)` 一次拿到报告（`.report`）和收益率数组（`.batch`）。

自动搜索方面，`foundry.Discovery(session)` 为公式、数组或程序计算搜索引导评分，`foundry.mine` 运行带检查点的挖掘循环：

```python
import foundry

foundry.mine(
    "var/runs/us_search",
    market="us_equity",
    backend="cuda",
    search=foundry.Search(mode="gfn", epochs=10, batch_size=128, accept=False),
)
```

在 `Search(...)` 中设 `accept=True` 可将搜索过程中通过准入的策略同步写入组合库。接入外部生成器（LLM、Agent、进化搜索等）可向 `foundry.mine(..., proposers=...)` 传入 `ExternalProposer` 或自定义 `Proposer`，详见[策略发现](guides/research.zh-CN.md#策略发现)。运行期间可在另一终端执行 `python -m foundry monitor` 查看实时状态。

## 命令行与配置项

所有参数通过 Python 关键字参数传入，或在命令行通过 `python -m foundry --parameters '<json>'` 传入。`foundry.schema()` 与 `python -m foundry schema schema.json` 可在不加载数据和 GPU 的情况下导出参数与 JSON-RPC 定义。

| 命令（`python -m foundry ...`） | 用途 |
| --- | --- |
| `ops` / `fields` | 查看 66 个算子说明或当前市场字段 |
| `schema <output.json>` | 导出参数与 JSON-RPC Schema |
| `serve` / `status` / `stop` | 启动、查询或停止本地 JSON-RPC 计算服务 |
| `attach <addr> <port>` / `detach` | 绑定本地或 SSH 转发的服务端口，或清除绑定 |
| `request <in.jsonl> <out.jsonl>` | 向已绑定的服务批量发送 JSON-RPC 请求 |
| `mine <output_dir>` | 运行带检查点的策略搜索（`--parameters '{"backend":"cuda",...}'`） |
| `monitor` / `preview` | 打开终端监控看板（需 `--group monitor`）或渲染状态快照 |
| `calibrate <run_dir>` | 查看搜索运行的奖励标定统计 |
| `import-perpetual <output_dir>` | 校验并将公开 USD-M 永续归档转为 Parquet |

| 参数 / 配置类 | 常用选项与默认值 |
| --- | --- |
| `market` 与市场参数 | `"us_equity"`（默认）、`"a_share"`、`"crypto_spot"`、`"crypto_perpetual"`，及 `data_dir`、`start`、`end` 与费率参数 |
| `backend` | `"cpu"` 或 `"cuda"` |
| `foundry.Clock` | `decision="1D"`，`statistics="1D"`（支持 `"<N>s"`、`"<N>min"`、`"<N>h"`、`"<N>D"`） |
| `foundry.Sizing` | `rule="equal_slots"`（默认）或 `"independent"`，`scale=1.0` |
| `foundry.Account` | `margin="cash" \| "isolated" \| "cross"`，`leverage=1.0`，`gross_limit=1.0`，`initial_cash=1.0`，`composition="shared" \| "independent"` |
| `foundry.Execution` | `model="targets" \| "orders"`，`bar_path="open_low_high_close" \| "open_high_low_close"`，`participation=1.0`，`order_capacity=32`，`command_capacity=32`，`quantity_step=0.0`，`price_tick=0.0` |
| `foundry.Admission` | `weight_steps=20`，`min_sharpe_gain=0.01`，`max_members=40`，`artifact_dir="./var/candidates"` |
| `foundry.Search` | `mode="gfn" \| "random"`，`epochs=5`，`batch_size=128`，`accept=False`，`objective="portfolio" \| "single"` |
| `foundry.CPU` / `foundry.CUDA` | `CPU(workers=8, candidate_batch=32)`，`CUDA(streams=8, candidate_batch=256, trial_batch=16384)` |

## 远程服务与客户端

把行情和 GPU 留在 Linux 服务器上、在本地机器（例如仅安装 `--group client` 的 Windows Python）编写策略时，先在服务器启动回环服务：

```python
import foundry

foundry.serve("var/runs/library.json", backend="cuda")
```

在服务器执行 `python -m foundry status` 查看端口，在本地用系统 SSH 转发该端口后连接：

```sh
ssh -N -L 58731:127.0.0.1:REMOTE_PORT user@compute-host
```

```python
import foundry

with foundry.connect(("127.0.0.1", 58731)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    print(report.rows[0].metrics)
```

客户端使用命令行请求时先执行 `python -m foundry attach 127.0.0.1 58731`（`detach` 解除绑定）。断开客户端不会停止远程服务，停止服务需执行 `python -m foundry stop`。远程会话支持公式、已注册数组和已部署 `ProgramReference` 的规划、注册、评估、诊断、权重试算与提交；`evaluate_device`、`simulate`、`walk_forward` 与 `mine` 需在计算节点进程内运行。

更多细节见[研究指南](guides/research.zh-CN.md)、[数据格式](guides/data.zh-CN.md)、[策略程序](guides/programs.zh-CN.md)和[基准报告](guides/benchmarks/README.zh-CN.md)（[框架对比方法](guides/framework-benchmark.zh-CN.md)）。
