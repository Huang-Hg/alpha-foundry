# Foundry 研究与接口指南

[项目介绍](../README.zh-CN.md) · [English](research.md)

面向时点一致数据的公式、模型信号与目标敞口研究和回测。支持美股、A 股、加密货币日线现货与 USDT 线性逐仓永续。A 股保留涨跌停、停牌、卖出税费和次日执行回归。Linux CPU/CUDA 执行明确的市场规格；股票、现货账户及 CUDA 统计使用双 FP32；数值策略在评价前固定，不再按结果触发精度复算。

## 接入自己的数据

仓库附带的是可替换的研究样本。向 foundry.open() 传入 market、data_dir、start 和 end 即可选择自己的日线数据。将 CSV、数据库或 API 导出数据转换为下列格式，不要求供应商账号或配置文件。

### 日线文件与字段

各市场的 `panel.parquet` 都采用长表，每行包含 `date`、`symbol`、`open`、`high`、`low`、`close`、`volume`。`date` 存为无时区、午夜的 datetime，表示交易所交易日，现货使用 UTC 日期；`symbol` 存为字符串，数值列使用数值类型。A 股代码保留前导零。

| 市场 | 额外行情字段 | 其他必需文件 |
| --- | --- | --- |
| `us_equity` | `adj_factor` | `membership.parquet` |
| `a_share` | `published`、`exec_open`、`exec_close` | `membership.parquet`、`manifest.json` |
| `crypto_spot` | 布尔类型 `eligible` | `manifest.json` |

- **美股：** 引擎将 OHLC 乘以 `adj_factor`。提供原始 OHLC 与对应因子，或已复权 OHLC 与恒为 1 的因子，避免重复复权；记录复权方法及历史版本缺口。
- **A 股：** OHLCV 为生成信号的原始价格，`exec_open`、`exec_close` 为累计历史复权执行价。`published` 是交易日日期，读取器仅使用该行情日收盘前已发布的最新版本，迟到修订不会替换旧决策的行情；它表达日线可用时间，不是盘中发布时钟。
- **资格：** 股票的 `membership.parquet` 包含 `symbol`、`start`、`end`，区间左闭右开，空 `end` 表示尚未退出。读取器假定这些区间在相应收盘时已知，不会自动重建缺失公告历史。保留资格区间外的报价，以便已有持仓继续估值。现货在每日行情行记录 `eligible`，缺失资格视为不合格。
- **历史与缺口：** 保留研究起点前的历史供滚动窗口使用；除 A 股明确的发布版本外，每个日期/资产只存一条观测。缺失报价保留缺失，不用未来报价填补过去，也不用今天的成分名单反推历史。

### 示例：转换自己的美股 CSV

准备包含上述美股字段的 `ohlcv.csv` 和包含资格字段的 `membership.csv`，下面创建一个新的数据目录：

```python
from pathlib import Path
import pandas as pd

directory = Path("data/my_us_equity")
directory.mkdir(parents=True)
panel = pd.read_csv("ohlcv.csv", parse_dates=["date"], dtype={"symbol": str})
members = pd.read_csv("membership.csv", parse_dates=["start", "end"],
                      dtype={"symbol": str})
panel.to_parquet(directory / "panel.parquet", index=False)
members.to_parquet(directory / "membership.parquet", index=False)
```

使用 foundry.open(market="us_equity", data_dir="./data/my_us_equity", start=..., end=...)，日期设在数据覆盖区间内。session.describe() 列出当前公式字段；自己的数据与附带快照使用相同公开接口。

### 来源记录与自定义特征

A 股和现货的 manifest 必须包含 `DataContract` 的七个字段：`schema`、`source`、`retrieved_at`、`features`、`execution_prices`、`publication`、`membership`。可参考附带清单的格式，但应填写自己的来源、价格口径和真实可用时间限制，不复制 PIT 声明或虚构发布时间。

当前美股文件读取器报告的是固定的遗留数据契约，仅增加自定义 `manifest.json` 不会改变报告中的来源说明。应随数据保留真实来源记录；要将自定义美股来源暴露到研究契约，需要修改数据适配器。数值快照身份仍由已加载面板与执行规格生成。

额外的 Parquet 列不会自动成为公式字段。自己计算的模型输出通过 `ArrayBatch` 接入；需要新增公式字段时，明确扩展数据适配器与市场字段定义。相关边界是 [`foundry/adapters/data.py`](../foundry/adapters/data.py)、[`MarketData`](../foundry/contracts.py) 和市场的 `compile` 方法，更换数据来源无需重写回测数值核。

### 永续数据

当前读取器使用 Binance 风格的小时数据契约：`trade.parquet`、`mark.parquet`、`funding.parquet` 和 `manifest.json`，包含文件摘要、有序合约列表，以及需要时的资金费周期变更证据。附带的三合约和 64 合约快照可作为格式参考；仅有日线 OHLC 不足以模拟资金费与强平。

调用 foundry.import_perpetual(output, symbols="BTCUSDT ETHUSDT SOLUSDT", history_start="2025-01-01", start="2025-07-01", end="2025-12-31") 可明确指定采集参数。下面的 CLI 使用默认合约和日期：

```bash
python -m foundry import-perpetual data/crypto_perpetual_new
```

导入器保留来源记录并核验归档校验和。随后用 foundry.open(market="crypto_perpetual", data_dir="./data/crypto_perpetual_new", start=..., end=...) 研究。已有快照的合约顺序由 manifest 决定，symbols 控制新导入。自有来源须满足经过验证的事件契约，或明确实现适配器和执行规格。详见 [perpetual_data.py](../foundry/adapters/perpetual_data.py) 和[加密货币执行](#加密货币执行)。

## 公开 Python 接口

常用操作只需 `import foundry`。`foundry.open(...)` 创建本机计算会话，`foundry.connect((host, port))` 连接已有服务；两者提供相同的报告、模型注册、邻域探索和组合操作。Windows 原生 Python 可作为远程客户端，本机数值计算使用 WSL2；Linux/Colab 可直接运行。导入包不加载行情或初始化 GPU。

```python
import foundry

with foundry.open(
    "production/my_portfolio.json",
    market="us_equity", backend="cuda",
    start="2025-07-01", end="2026-06-30",
    cuda=foundry.CUDA(candidate_batch=64),
    admission=foundry.Admission(min_sharpe_gain=0.02),
) as session:
    report = session.evaluate({"trend": "gt(ma_dist_20,0)"})
    decision = session.commit(report)
    foundry.export(decision, "output/decision.json")
```

`evaluate` 接收公式字符串、公式序列或 `{候选 ID: 公式}`。`report.rows` 的每行含 `metrics`、`risk`、`decision` 和 ID。`commit(report)` 按报告对应的组合版本重新判断准入；评价本身不修改组合。调用者不必构造存储器、快照或版本令牌。

参数直接传入，没有 INI 或隐式配置文件。常用项为 `market`、`backend`、`data_dir`、`start`、`end`；费用与杠杆使用 `cost_rate`、`commission`、`stamp_tax_sell`、`leverage`、`maintenance_rate` 等对应市场参数。高级参数组为 `foundry.CUDA`、`Admission`、`Operators`、`Search`、`Single`、`Monitor`，默认值只在 `foundry/parameters.py` 维护。`foundry.schema()` 返回参数 schema 与各市场默认值，`session.describe()["parameters"]` 返回实际参数。

参数对象不可变，改变设置需打开新会话。`foundry.mine(output, **parameters)`、`foundry.serve(portfolio, **parameters)` 使用同一套参数。CLI 接受相同 schema 的 JSON，例如 `python -m foundry --parameters '{"market":"a_share","backend":"cuda"}' fields`，无需创建配置文件。

### 外部模型与目标持仓

`session.grid` 给出有序日期、资产及对应数据身份。下面用无拟合程序的零信号演示形状；实际接入时替换为自己的输出和真实信息截止点：

```python
import numpy as np
import foundry

with foundry.open() as session:
    grid = session.grid
    days, assets = len(grid.dates), len(grid.assets)
    values = np.zeros((1, days, assets), dtype=np.float32)
    unused = np.full((1, days), -1, dtype=np.int64)
    candidates = session.register(
        values, kind="signals", ids=("model-v1",), grid=grid,
        producer="external-program-v1",
        observed_through=unused, fit_through=unused,
    )
    report = session.evaluate(candidates)
```

`kind` 可选 `signals`、`target_weights` 或 `target_exposure`，单位与市场约束见[候选接口](#候选接口)。两个截止点是 `(候选数, 日期数)` 的整数行号，必须如实声明使用的信息且不超过决策行；`-1` 表示未使用观测或拟合。远程会话自动上传 NumPy 数组，无需服务器路径；进程内会话还可接收 CuPy 数组。

### 组合操作与生命周期

- `session.explore(formulas, hypothesis="trend")` 探索公式的参数和字段邻域。
- `session.portfolio()` 返回成员、权重与指标，`baseline` 标识新研究使用的等权比较基准。
- `session.trial_weights(matrix)` 评价 `(试探数, 成员数)` 的非负有限构件权重，列顺序与 `portfolio()` 一致。它们不是逐资产目标持仓。
- `session.commit_weights(trials, index)` 提交选定试探，仍执行统一准入。全零权重表示现金，不恢复为等权仓位。
- `foundry.export(report, path)` 导出 JSON 并创建父目录。

省略组合路径时只在内存中研究；提供路径后原子保存组合，外部数组存于 `Admission.artifact_dir`。过时报告明确拒绝，不自动重试；其他会话更新组合后，调用 `session.refresh()` 读取新状态并重新评价。退出 `with` 释放本机资源；关闭远程客户端不停止共享服务。进程内挖掘器可调用 `session.evaluate_device(inputs)` 保留设备数组，远程使用结构化报告。

## 模块边界

采用模块化单体与端口适配器，按下表划分职责。[架构检查](../validation/test_architecture.py) 检查依赖无环、禁止依赖及 import 副作用。

| 所有者 | 职责 |
| --- | --- |
| contracts、candidates、research_types | 研究区间、快照、后端能力、类型化来源/结果与存储端口 |
| core、expr、market、portfolio | 面板与指标语义、公式、交易规则和准入 |
| ops、backtest、engine | 基于明确数组和执行规格的 CPU/portable/CUDA 计算 |
| research | 共用评价、邻域探索和显式版本化提交 |
| search、CLI/RPC | 提案或传输，共用研究决策 |
| adapters | 数据读取、指纹、原子存储和序列化 |
| runtime | 不可变配置、后端准备与资源所有权 |
| monitor | 只读取发布状态，不持有行情、引擎或 GPU |

公开入口将显式参数一次解析为不可变设置。导入不读取行情、探测 GPU 或分配资源；会话拥有并释放计算资源，领域依赖由参数传入。

Study 固定研究区间；快照进一步绑定数据、执行规格、数值策略和实现身份。查询结果携带组合版本。评价不写组合，commit 是独立的 compare-and-swap 命令，一次原子发布替换结果。设备驻留接口与 CLI/RPC 序列化共用类型化用例。可导出 JSON Schema 是机器契约，`python -m foundry --help` 是命令清单。

旧请求、组合和 checkpoint 协议明确拒绝，不保留兼容层，不自动重录基线。Quantiacs 专属源码、提交 notebook、配置与测试已从当前工程移除；历史数据（包括 data/qnt_crypto）、来源 manifest 和私有生产工件继续保留。保留数据不代表其具备完整 PIT 来源证明。

## 候选接口

`research-v12` 使用 `Candidate(id, source)`。来源明确区分 `Formula(expression)`、`ArraySource(kind, digest, snapshot, producer)` 和组合基准 `EqualWeight()`。公式挖掘器与邻域探索构造 `Formula`；模型输出通过 `Research.register_batch(batch, context_id)` 注册，然后共用评价、权重试探、准入和提交规则。

公开会话自动装配运行环境与存储。`evaluate` 返回结构化报告，进程内 `evaluate_device` 保留设备持仓和收益。查询不修改组合，`commit` 原子发布完整后继组合。注册复制输入，返回后生产者可复用缓冲区；重新打开组合时验证数组工件和截止点。

`foundry.candidates.ArrayBatch` 包含 `kind`、`ids`、`values`、`snapshot`、`producer`、`dates`、`assets`、`observed_through`、`fit_through`。`values` 是 NumPy/CuPy 的 `float32` 或 `float64 (B,T,S)`，必须覆盖研究快照完整且有序的日期/资产网格。股票与现货的 `signals` 与公式一样经过 `clip(0,1)/当日可持有资产数`，NaN 表示缺失。`target_weights` 是有限、非负的组合比例；无资格资产权重必须为零，单日总敞口不得超过 1。仅对 `Admission.weight_sum_atol` 内的微小超额归一化到 1，更大超额明确拒绝。目标权重继续服从市场执行规则和次日开盘时序。

两个截止点数组均为 `int64 (B,T)`，使用包含端点的面板行号，范围为 `-1` 到对应决策行；`-1` 表示未使用观测或拟合模型。引擎拒绝未来截止点、错误坐标和旧快照。时间信息来自生产者声明，不能证明外部训练或特征计算没有泄露未来；`producer` 应标识模型/程序版本。`research.describe` 发布完整坐标、快照及这些限制：batch 的 `snapshot` 使用 `context.snapshot.id`，查询/命令的版本令牌使用 `context.id`。

注册时向宿主传输一次数据，用于内容摘要和保存；最近输入的设备缓存最多保留 `candidate_batch` 个候选。缓存命中后评价继续使用设备数组，逐出的输入从已验证工件重载。CUDA 生产者须在提交前让输入就绪于 runtime 当前 stream；注册会等自有复制完成再返回。外部数组的噪声对照延迟只依赖生产者身份与输入类型，不依赖包含未来数值的全序列摘要。

远程会话自动上传 NumPy bundle。`research.register_batch` 请求携带 `{context, bytes}`，JSON 换行后紧跟指定长度的二进制载荷，单次上限 1 GiB；加载使用 `allow_pickle=False`，不接受服务器路径。返回类型与进程内一致。`python -m foundry schema OUTPUT` 导出请求、结果和参数 schema。

## 加密货币执行

`crypto_spot` 使用 UTC 日线、7×24 连续日历与 365 日年化。保留 Quantiacs 历史报价及资格，缺失的发布时间/修订版本通过数据契约公开。缺失报价时冻结已有持仓，不从未来末值填补过去。

`crypto_perpetual` 使用 USDT 线性合约和逐仓保证金。日线收盘信号在下一 UTC 日开盘调仓；每小时依次处理标记价跳空、旧仓资金费、计划调仓、标记价高低点触发强平、收盘估值。仅在每日调仓时重新分配逐仓抵押，空闲现金不自动补仓。`target_exposure` 接收有符号的名义本金/净值比，总绝对敞口上限为配置杠杆；公式与 `signals` 截到 [-1,1] 后除以合资格资产数，不隐式放大到最大杠杆。

`market="crypto_perpetual"` 通过公开参数选择数据和执行规则；唯一 CLI 的导入命令下载并核验 Binance 成交 K 线、标记价 K 线和已结算资金费归档 SHA256。初始固定 BTC/ETH/SOL 集合使用 2025 数据。这是固定维持保证金/费用、小时不利价格触达的研究模型，未重建历史风险档位、VIP 费用、订单簿、ADL 或 tick 路径。保留真实资金费时间戳；执行映射到所属小时，归档缺少结算标记价时明确使用该小时开盘标记价。发现接口公开上述限制，不声称交易所精确仿真或完整供应商 PIT。

类型化模拟结果包含每日资金费、交易/强平费用、逐仓损失封顶调整、强平阶段位图、破产与终值。报告的现金流和次数按研究窗口汇总。损失封顶调整仅为本模型会计量，不冒充交易所保险赔付；行情不利变化后的实际最大杠杆可能高于入场上限。永续账户在执行前明确规划为原生精度，CUDA 统计与准入矩计算使用双 FP32。

永续 CUDA 采用每个 warp 一个账户、lane 分担资产的实现；小时资金费、强平阶段和每日抵押分配保持时间顺序。调仓归约按资产并行。双 FP32 账户在强平阈值相邻浮点数上出现决策变化，因此账户保留原生算术，但生产保留条件必须满足下述查询＋提交的实测开销预算。

小规模永续账户在启动前按 `CUDA.perpetual_host_cells` 与完整 `B*T*S` 规模明确调度到 CPU，再返回 runtime 的设备数组。发现接口公开阈值。这不是运行错误回退，也不会根据结果重新执行另一种精度。提交按输入顺序处理；能装入一个设备批次的请求保持完整并行，多批次请求在连续拒绝时扩大预计算范围，出现接纳时按本次接纳距离缩小范围，避免对旧组合版本多算大段后续候选，最终仍一次原子发布。

64 合约实验中的 BF16 Tensor Core 资金费前缀未带来端到端收益，因此不进入生产。旧前缀内核和标量精度复算分支已删除。资金费结算间隔变化须有日程证据，缺失事件继续明确报错。公式保留滚动标准差、斜率、相关性和总体标准化的统一语义；病态组合仍可能放大微小后端数学差异，样本准入一致不等于任意公式一致性证明。

## PIT 与精度

共享指标中的 `turnover`、`exposure`、`max_single_weight` 和 `margin_bps` 分母描述每日目标持仓，不等同于永续实际成交本金。实际费用和账户杠杆见独立风险报告；日内强平时间标识所在小时，不能当作精确触碰时刻。

先截断源数据，再派生特征。收盘决策最早在次日开盘生效。横截面只包含当时可见且有资格的资产。报价历史与成分区间分离；迟到发布与修订不能覆盖旧决策行；forward 标签不进入特征面板。

A 股数据保留 365 只历史成分的完整报价，资格另按区间应用。信号使用原始 OHLCV，执行使用复权价格。manifest 明示缺少历史修订版本和原始成分公告时间戳。美股数据限制同样通过发现接口暴露。前缀与未来污染测试证明计算因果性，不代表完整供应商 PIT 认证，也不是样本外收益证据。

执行规格包含开盘可交易资格、成本、方向冻结与持仓费用，内核不凭股票代码猜交易规则。CPU/CUDA 保留相同完整资产范围与指标语义。

精度开销按**完整查询＋有序提交 ≤ 单 FP32 对照的 1.5 倍**验收：同一设备、候选、顺序和市场规则。加载、注册、传输/JIT 单独报告，不用它们稀释比例；内核局部加速不能代替完整用例验收。原生算术只能作为经实测的固定策略保留，不再作为低精度结果的第二次复算。资格只覆盖实测条件；下方旧的 2 倍预算结果不能证明满足新预算。

查询、权重试探与提交共用研究的固定数值策略。`mixed` / `paired` 选择股票和现货账户算术；永续账户及 CPU 显式采用 `native`，API 公布有效策略，不再支持逐次查询覆盖精度，也不再报告精度复算计数。数组交换边界仍可使用宽类型，这既不表示全链宽精度，也不能宣称全链 FP32。

CUDA 归约、加权矩和对数增长用两个 FP32 分量计算，以整数指数保存财富范围。高位与残差的归一化使用 Quick-Two-Sum，任意高位相加仍用 Two-Sum，乘积残差继续由 FMA 保留；误差变换依据见 [Hida、Li、Bailey](https://www.davidhbailey.com/dhbpapers/arith15.pdf)。加权方差先中心化，避免常量收益产生伪方差；两端均定义零方差 Sharpe 为零。可行权重效用并列使用统一指标容差，再比较 CAGR，最后选择最小权重。验收同时比较收益、现金流、强平事件、权重与阈值两侧的准入结论；协议或数值语义改变时，不悄悄重录旧逐位基线。

算子契约同时明确：有限常量窗口的协方差严格为零，相关性未定义；微量 tanh 在 CPU、CUDA、portable 三端使用正确舍入的恒等区间。依赖常量协方差抵消噪声的旧候选将不再产生有效信号，这是明确的错误修正，不是放宽准入容差。

## 开发与验证

Python 负责研究、接口与编排，现有 C++/CUDA 内核负责数值计算，一个 Bash 脚本构建三个 Python 扩展。正式部署与回测环境为 Linux，包括 WSL2 和 Colab。WSL2 使用 Linux Python 生成原生 `.so`，已删除 Windows Python/MSYS2 交叉构建分支。

构建需要 Python 3.12 或 3.13、对应开发头文件，以及支持 C++17/OpenMP 的 GCC。在仓库根目录激活实际运行 Foundry 的 Linux 虚拟环境，安装依赖并构建：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash build.sh
```

`bash build.sh backtest` 只重建回测扩展；不指定目标时构建 `ops`、`backtest`、`search`。脚本从当前 `python3` 获取 Python/NumPy 头文件与 ABI 后缀，支持含空格的路径，不绑定旧机器的解释器或编译器路径。算子、回测和策略各自的数值编译选项保持不变。原生构建使用 Linux Python，包括 WSL2；Windows 远程客户端无需构建这些扩展。

GPU 后端安装匹配运行环境 CUDA 主版本的 CuPy；CPU 无需 CuPy 或 GPU。参数直接传给公开 API，通过 foundry.schema() 查询默认值。架构与纯规则检查不依赖私有行情、组合或凭据。

pytest 验证套件覆盖架构、表达式、原生 ABI、搜索策略、发布时点和请求/存储协议。云端测试显式选择美股/A 股/加密现货/逐仓永续，使用隔离的临时组合库，不从私有生产库获取隐藏种子。test_production 单独校验明确提供的生产基线，不悄悄重录。

在授权云端使用统一入口。GitHub Actions 展示和归档已保存报告，不执行数值套件：

```bash
python -m pip install -r requirements-benchmark.txt
python -m foundry benchmark output/benchmark-new
```

性能报告分别记录加载、传输/JIT、驻留批量与完整用例，注明设备、完整规模、精度和误差。

历史精度实验已[单独归档](benchmarks/archive/precision-history.zh-CN.md)。公开接口耗时和正确性请看[当前基准套件](benchmarks/README.zh-CN.md)。
