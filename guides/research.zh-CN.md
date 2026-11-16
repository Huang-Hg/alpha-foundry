# 研究与程序接入

[项目介绍](../README.zh-CN.md) · [English](research.md)

Foundry 用同一个会话评价公式、模型信号和目标持仓。人工研究、AI Agent、公式挖掘器与外部模型搜索共用市场规则和组合准入结论。

候选由谁提出，与它如何表示是两个独立维度。人类、Agent 和机器学习程序都可以提供公式，也都可以提供任意 Python/CUDA 程序生成的数值数组；非公式因子无需改写成 Foundry 的公式语法。

## 公开 Python 接口

```python
import foundry

with foundry.open(
    "var/runs/my_portfolio.json",
    market="us_equity", backend="cuda",
    start="2025-07-01", end="2026-06-30",
    cuda=foundry.CUDA(candidate_batch=64),
    admission=foundry.Admission(min_sharpe_gain=0.02),
) as session:
    report = session.evaluate({"trend": "gt(ma_dist_20,0)"})
    for row in report.rows:
        print(row.id, row.metrics, row.decision)
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

`evaluate` 接受公式字符串、公式序列、ID 到公式的映射、`Candidate`、候选序列或已注册批次。报告包含指标、账户风险和结构化准入结论；某个候选被拒绝不会中断整个批次。

评价成功时，`row.ok` 为 true，`row.metrics` 包含 `sharpe`、`exposure` 等指标。准入要单独查看 `row.decision.passed`、`reason` 和 `stopped_at`；`ok=True` 不代表 `passed=True`。`foundry.export` 用于导出强类型报告；`foundry.schema()` 和 `session.describe()` 已返回字典，使用 `json.dumps` 序列化即可。

| 操作 | 用途 | 对组合的影响 |
| --- | --- | --- |
| `evaluate(inputs)` | 评价候选与准入方案 | 只读 |
| `diagnose()` | 描述组合缺口与发现上下文 | 只读 |
| `judge_family(inputs, hypothesis="")` | 检验邻域或显式参数网格 | 只读 |
| `portfolio()` | 获取成员来源、权重、指标与基线状态 | 只读 |
| `trial_weights(matrix)` | 比较组合成员权重方案 | 只读 |
| `commit(report)` | 按报告顺序提交候选 | 通过准入后原子发布 |
| `commit_weights(trials, index)` | 提交已审阅的权重方案 | 通过准入后原子发布 |
| `reweight(iterations=3, count=32, seed=1)` | 搜索并提交满足准入规则的权重改进 | 可能发布新组合 |

权重试验的形状为 `(试验数, 成员数)`，成员顺序与 `portfolio()` 一致。权重必须有限且非负，全零表示现金。这些是组合成员权重，不是逐资产目标持仓。

省略组合路径时，组合库保存在内存中。注册数组暂存于计算设备，仅入选、检查点引用或明确要求保留的候选写入 `Admission.artifact_dir`，默认是 `var/candidates`。传入组合路径后启用组合持久化和原子发布。退出 `with` 释放候选租约及本地计算资源；关闭远程客户端不会停止共享服务。

报告绑定研究快照和组合版本。其他写入者改变上下文后，旧报告会触发 `ContextConflict`；显式调用 `session.refresh()` 后重新评价，不会自动重试旧决策。

准入报告保留每个约束的原始变化量 `delta` 和阈值 `required`，空阈值表示无法达到的要求。搜索层独立归一化这些裕度；修改四个 `Search.violation_scale_*` 只改变发现奖励，不再改变研究快照、数组身份或硬准入。被拒候选的展示试探点使用组合层自有尺度。本次修正会让旧快照 ID 一次性变化，旧快照绑定的数组需要显式重新注册，不能继续复用旧句柄。

### 参数与自描述

`market`、`backend`、`data_dir`、`start`、`end` 和市场费用直接传入。高级参数组包括 `foundry.Clock`、`Sizing`、`Account`、`Execution`、`CPU`、`CUDA`、`Admission`、`Operators`、`Search`、`Single` 与 `Monitor`。参数在会话打开时固定；比较其他设置时另开会话。

`foundry.schema()` 无需打开研究即可返回参数 schema、市场默认值与 RPC 契约。`session.describe()` 返回当前字段、66 个算子和实际参数。`foundry.mine(...)` 与 `foundry.serve(...)` 使用相同参数组；向挖掘器传入已有会话时，不再重复传参数。

唯一 CLI 是 `python -m foundry`，例如：

```bash
python -m foundry --parameters '{"market":"a_share","backend":"cuda"}' fields
python -m foundry schema var/runs/schema.json
```

不需要 INI 文件或环境变量覆盖。参数中的相对路径基于仓库根目录解析，也可以传绝对路径。

## 接入自己的数据

仓库快照可以替换。把 CSV、数据库或 API 导出转换为以下文件，再通过 `foundry.open(data_dir=..., market=..., start=..., end=...)` 选择数据。

### 日线文件

`panel.parquet` 采用长表，每行包含 `date`、`symbol`、`open`、`high`、`low`、`close` 和 `volume`。日期使用不带时区的午夜时间表示交易所交易日；现货使用 UTC 日历日。证券代码保存为字符串，保留 A 股代码的前导零。

| 市场 | 额外行情列 | 其他必需文件 |
| --- | --- | --- |
| `us_equity` | `adj_factor` | `membership.parquet` |
| `a_share` | `published`、`exec_open`、`exec_close` | `membership.parquet`、`manifest.json` |
| `crypto_spot` | 布尔值 `eligible` | `manifest.json` |

- **美股：** OHLC 乘以 `adj_factor`，成交量不调整。可提供原始价格与因子，或已复权价格与全 1 因子，避免重复复权。记录复权方法及缺失的历史版本。
- **A 股：** 原始 OHLCV 产生信号，`exec_open` 和 `exec_close` 提供历史复权执行价格。读取器按每根行情收盘时已发布的最新版本冻结该根行情；后续修订不替换早期决策使用的记录。`published` 是交易日口径，不是日内公告时钟。
- **股票资格：** `membership.parquet` 包含 `symbol`、`start`、`end`；起点包含、终点不包含，空终点表示仍有效。假定这些区间在相应收盘时已知。资格区间外也应保留报价，以便已有持仓继续估值。
- **现货资格：** 每个日线行声明，缺失资格视为 false。
- **历史长度：** 保留研究起点之前的数据以计算滚动特征。除 A 股显式发布版本外，每个日期/资产只有一行。缺失报价保持缺失，不使用当前成分倒推历史资格。

CSV 转换示例：

```python
from pathlib import Path
import pandas as pd

folder = Path("/datasets/my_us_equity")
folder.mkdir(parents=True)
panel = pd.read_csv("ohlcv.csv", parse_dates=["date"], dtype={"symbol": str})
members = pd.read_csv("membership.csv", parse_dates=["start", "end"],
                      dtype={"symbol": str})
panel.to_parquet(folder / "panel.parquet", index=False)
members.to_parquet(folder / "membership.parquet", index=False)
```

随后在数据覆盖范围内选择研究日期：

```python
with foundry.open(market="us_equity", data_dir="/datasets/my_us_equity",
                  start="2025-07-01", end="2025-12-31") as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

### 来源记录与额外特征

A 股和现货 manifest 包含 `DataContract` 的七个字段：`schema`、`source`、`retrieved_at`、`features`、`execution_prices`、`publication`、`membership`。仓库 manifest 可作为格式参考，但应填写自己的实际来源与可得性限制；缺少公告时间或修订版本必须如实披露。

当前美股读取器使用固定的历史来源契约，仅新增 manifest 不会替换报告中的来源说明。请在数据目录旁保留实际来源记录；接入自定义美股来源契约需要扩展读取器。数值快照仍绑定实际加载的数据和执行规格。

新增 Parquet 列不会自动成为公式字段。模型输出可直接注册；若要扩展行情字段，在 [`market/data.py`](../foundry/market/data.py) 及对应市场编译器中接入，无需改写回测内核。历史 Quantiacs 数据仍可使用，不依赖已经退役的平台集成。

### 永续数据

保留的小时归档契约由 `trade.parquet`、`mark.parquet`、`funding.parquet` 和 `manifest.json` 组成，使用小时执行事件生成日线决策与统计。manifest 记录资产顺序、工件哈希，以及需要时的资金费间隔变更证据。仅有日线 OHLC 无法还原资金费和强平。

```python
foundry.import_perpetual(
    "/datasets/my_perpetual",
    symbols="BTCUSDT ETHUSDT SOLUSDT",
    history_start="2025-01-01",
    start="2025-07-01", end="2025-12-31",
)
```

导入器获取公开 USD-M 归档、验证校验和并保存来源记录。读取时使用 `market="crypto_perpetual"`、对应目录与日期。已有数据的资产顺序由 manifest 决定；`symbols` 用于新数据采集。更细行情通过以下显式契约接入。

### 加密货币日内文件

现货与逐仓永续均可提供 `manifest.json` 和 `bars.parquet`，使用 `schema="crypto-bars-v1"`；永续还须提供 `funding.parquet`。读取器位于 [`market/intraday.py`](../foundry/market/intraday.py)，小时归档导入器不生成此格式。

| 文件 | 必需列 |
| --- | --- |
| `bars.parquet` | 字符串 `symbol`；int64 UTC 纳秒 `open_ns`、`close_ns`、`available_ns`；浮点 `open`、`high`、`low`、`close`、`volume`；布尔 `eligible` |
| 永续 Bar 附加列 | 浮点 `mark_open`、`mark_high`、`mark_low`、`mark_close`，仅供执行 |
| `funding.parquet` | 字符串 `symbol`；int64 UTC 纳秒 `settlement_ns`；浮点 `rate`、`mark` |

Bar 是对齐的左闭右开基础区间 `[open_ns,close_ns)`；`available_ns >= close_ns` 表示包含资格信息的整行何时可见。同一 `(symbol,open_ns)` 可有不同可见时间的版本。读取器先按研究截止点截断 Bar 终点与可见时间，再仅用决策收盘前已可见的版本冻结每根决策 Bar。组成行情缺失或迟到时，该次决策的特征不可用；后来的发布不会改写过去决策。执行采用研究截止前可见的首个发布版本。现货报价缺失时可保留已有持仓；永续执行要求完整且有限的成交价和标记价基础 Bar。

以下是永续 manifest 示意。请用实际 parquet 字节的 `hashlib.sha256(file_bytes).hexdigest()` 替换摘要占位符，并填写真实来源与限制：

```json
{
  "schema": "crypto-bars-v1",
  "market": "crypto_perpetual",
  "source": "Your archived feed and version",
  "retrieved_at": "2026-11-16T00:00:00Z",
  "features": "Completed trade OHLCV; mark and funding excluded",
  "execution_prices": "First published trade and mark base bars",
  "publication": "Per-row available_ns; describe missing historical vintages",
  "membership": "Historical eligible values published with each bar",
  "timezone": "UTC",
  "interval": "5min",
  "history_start": "2025-01-01T00:00:00Z",
  "end": "2025-02-01T00:00:00Z",
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "artifacts": {
    "bars": {"path": "bars.parquet", "sha256": "<bars file digest>"},
    "funding": {"path": "funding.parquet", "sha256": "<funding file digest>"}
  }
}
```

现货将 `market` 设为 `crypto_spot`，省略资金费工件及标记价列。manifest 的历史边界须为显式 ISO UTC 时间戳，`end` 为排他边界。研究 `start`、`end` 可用 ISO UTC 时间戳；仅日期的 `end` 包含完整 UTC 当日。源历史起点、研究起点和截止点必须对齐完整统计周期。决策与统计时长须为基础周期的整数倍；固定时长单位为 `s`、`min`、`h`、`D`。

资金费表须显式覆盖每个基础开盘时点与资产：`settlement_ns` 等于该基础开盘时间，`rate=0` 明确表示该时点无现金流，`mark` 为有限正数结算标记价。缺行会拒绝读取，不默认为零。结算发生于基础开盘之间时，需要更细的基础数据。资金费率与标记价不进入特征面板。

```python
with foundry.open(
    market="crypto_perpetual", data_dir="/datasets/crypto_5m",
    start="2025-01-08", end="2025-01-31", backend="cuda",
    clock=foundry.Clock(decision="15min", statistics="1D"),
    account=foundry.Account(margin="isolated", leverage=3., gross_limit=2.),
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

此处特征及注册的 `(B,D,S)` 数组使用已完成的 15 分钟决策 Bar。`Execution(fill="next_open")` 在决策完成后的下一个 5 分钟基础开盘执行，允许收盘与开盘共用边界，不等待下一次决策。没有新目标的基础 Bar 保持实际数量。公式窗口按决策 Bar 数计数，因此 `ma_dist_20` 使用 20 条观测。

## 数值因子、模型信号与目标持仓

`session.grid` 提供规范 ISO UTC `decision_times`、等价的 int64 UTC 纳秒 `time_ns`、有序资产和快照身份。数组维度为 `(B,D,S)`：候选、决策、资产。四市场日线预设保留原有口径，crypto Bar 工件还支持独立的决策与统计频率。股票决策为交易所时段收盘（包含夏令时和提前收盘），crypto 为已完成决策 Bar 的右端点。日历坐标不补造缺失的历史发布时间。

以下示例注册三个常量零评分候选，没有使用观测或拟合。接入实际策略时替换数值和因果声明：

```python
import numpy as np
import foundry

with foundry.open() as session:
    grid = session.grid
    decisions, assets = len(grid.decision_times), len(grid.assets)
    values = np.zeros((3, decisions, assets), dtype=np.float32)
    unused = np.full((3, decisions), foundry.NO_INFORMATION, dtype=np.int64)
    registered = session.register(
        values, kind="scores", unit="score", ids=("model-small", "model-base", "model-large"),
        grid=grid, producer="external-program-v1",
        observed_at=unused, fit_at=unused,
        complexity=(0.2, 0.5, 0.8),
    )
    report = session.evaluate(registered)
```

| 类型与单位 | 含义 | 约束 |
| --- | --- | --- |
| `scores`, `score` | 由会话 `Sizing` 规则转换的评分 | Float32/float64；NaN 为缺失，映射成零目标 |
| `targets`, `equity_fraction` | 带符号名义金额/账户权益比例 | 有限；现金账户非负；总绝对敞口不超过 `Account.gross_limit`；不具资格资产为零 |

`Sizing(rule="equal_slots")` 保留日线预设：现金市场使用 `clip(score, 0, 1) / N`，可做空市场使用 `clip(score, -1, 1) / N`，`N` 为当日具资格资产数量。`Sizing(rule="independent", scale=0.1)` 使用截断评分乘显式尺度，不除以资产数量；增加零评分资产不会稀释已有资产目标。独立规则超总敞口预算时返回候选拒绝并继续批次，不自动归一化到满仓。

直接 `targets` 跳过评分映射，必须传入 `unit="equity_fraction"`；零表示明确清仓。当前目标契约接受省略 `update_mask` 或全真的布尔 `(B,D,S)` 掩码，两者具有相同规范身份。含 false 的掩码需要后续账户执行能力，当前明确拒绝；评分不接受掩码。数量或名义金额目标尚未支持。

`foundry.Account(margin="isolated", leverage=3., gross_limit=2.)` 将逐仓抵押杠杆与允许的总目标敞口分开。默认日线经济口径保留：现金 1 倍，逐仓永续杠杆与敞口上限均为 5 倍。评分校准若有拟合，必须计入 `fit_at`。



历史[独立 Agent 接入记录](integrations/agent-l4-20261116/README.zh-CN.md) 使用迁移前协议和真实数值因子和 GPU 岭回归模型跑通了这条路径，并根据返回的反馈调整候选。

数值必须为 float32 或 float64。观测和拟合声明为 `(B,D)` int64 UTC 纳秒数组，每项不得晚于对应的 `grid.time_ns`。只有未使用观测或拟合时才能使用 `foundry.NO_INFORMATION`（int64 最小值）。例如未拟合、决策时观测的评分使用 `observed_at=np.broadcast_to(grid.time_ns, (B,D))`，`fit_at` 填 `NO_INFORMATION`。训练模型必须如实声明训练数据和标签使用的最晚信息时点。日线收盘决策最早于下一开盘执行。

`producer` 标识生成程序。可选的逐候选 `complexity` 范围为 `[0,1]`，仅影响发现流程的软评分，不改变硬准入。注册创建有界驻留副本，返回绑定内容身份的候选与租约。本地 CuPy 数据留在 GPU，只传回紧凑摘要；远程注册自动上传 NumPy 数组，无需服务器路径。

用完调用 `session.release(registered)`；循环中通常放在 `finally` 内。退出会话也会释放其剩余租约。所有活动批次的总行数不得超过 `candidate_batch`，满额后须先释放再注册。释放后的驻留引用失效，即使内部仍有该候选的紧凑测量缓存也不能继续使用。

需要保存部分输出时，在释放前调用 `saved = session.retain(registered.candidates[:1])`，以后使用 `saved.candidates`。只有明确保留、组合入选或发现检查点引用的候选才写入 `Admission.artifact_dir`。`retain` 返回持久引用，原批次仍须释放；提交与发现流程自动保存其必需引用。组合/RPC、数组与检查点协议分别为 `research-v16`、`candidate-array-v4`、`candidate-discovery-v5`（二进制批次传输为 `candidate-batch-v3`），旧格式须显式重跑，不自动迁移。

## 策略发现

### 为 Agent 或自定义循环评分

```python
with foundry.open() as session:
    candidates = (
        foundry.Candidate("trend", foundry.Formula("gt(ma_dist_20,0)")),
        foundry.Candidate("momentum", foundry.Formula("cs_rank(ret_20)")),
    )
    print(session.diagnose())
    discovery = foundry.Discovery(session)
    report = discovery.evaluate(candidates)
    for row in report.rows:
        print(row.candidate.id, row.log_reward, row.decision)
    decision = session.commit(report)
```

已注册批次的 `.candidates` 可放在同一位置。Discovery 将组合缺口、行为指纹、相关性和新颖度用于目标评分。奖励指导下一步生成什么候选，组合硬准入仍是最终依据。组合变化后重新创建 `Discovery`，使其参照上下文保持一致。

`diagnose()` 只读，但计算量取决于目标。使用 `Search(objective="single")` 时，它会评价共用的确定性校准候选群，并返回校准门槛；`Single.calibration_sample` 默认是 8192。组合目标不执行这项校准。

噪声控制报告带有 `delayed=True`，描述的是延迟候选行为，不能直接传给 `session.commit`。请先通过 `session.evaluate(...)` 无延迟评价实际候选，再审阅并提交新报告。

### 接入外部提案器

```python
with foundry.open(search=foundry.Search(epochs=2, accept=False)) as session:
    batch = (
        foundry.Candidate("trend", foundry.Formula("gt(ma_dist_20,0)")),
        foundry.Candidate("momentum", foundry.Formula("cs_rank(ret_20)")),
    )
    proposer = foundry.ExternalProposer([batch])
    status = foundry.mine("var/runs/external-search", session=session,
                          proposers=(proposer,))
```

`ExternalProposer` 接受有限的候选或已注册批次。`foundry.mine(..., session=session, proposers=[proposer])` 自动打开、恢复和关闭提案器资源，不关闭调用者传入的会话。可选回调接收一个 `LearningFeedback`：`feedback.batch` 保留候选顺序及研究快照，`feedback.rewards` 是同设备的 FP32 向量。CPU 程序可显式调用 `to_host()`；CUDA 程序在自己的流上使用 `borrow(stream_pointer)`，再用 DLPack 零拷贝接入 PyTorch。借用范围内不得修改反馈或向外保留别名。反馈不借用行情/组合缓冲区，因此可与提交和重配重叠。

自定义提案器只需实现 `open(device)` 上下文管理器，返回具备 `propose(request)`、`learn(feedback)`、`release(batch)`、`dump()`、`load()` 的生产者。`ProposalRequest` 仅含批量规模、研究上下文和诊断特征；模型训练参数留在各自构造器。每个生产者仅允许一个未释放批次，反馈必须匹配原始批次句柄，不能混用实例、顺序或快照。驱动器等待学习完成后释放反馈和轨迹，再采样下一批；异常同样释放。普通候选与辅助候选按实际数量统计，随机提案器不再隐式追加 GFN 探针。检查点协议为 `candidate-discovery-v5`，旧检查点明确拒绝。额外反馈空间为 `O(B)`，不保存历史奖励或轨迹。


不指定提案器时，`foundry.mine(output, ...)` 使用配置的 GFN 或随机公式提案器。默认 `Search(accept=False)` 仍通过有序准入和定期重配演进内存研究组合，只禁止发布到持久组合库。需要发布时使用 `Search(accept=True, seed_portfolio="current")`。传入持久会话也必须启用该设置，`accept=False` 时会明确拒绝。`Discovery`、挖掘和 walk-forward 需要进程内会话，其设备驻留计算不是远程回调协议。

### 非 GFN GPU 模型和非公式因子示例

下面的可执行客户端位于验证套件，可直接替换其中的模型或张量程序；它不使用 GFN，也不把输出转换成公式。

```python
from validation.benchmarks.tensor_programs import TensorProposer

with foundry.open(
    backend="cuda",
    search=foundry.Search(epochs=4, batch_size=64),
    cuda=foundry.CUDA(candidate_batch=64, measurement_cache_mib=32),
) as session:
    proposer = TensorProposer(session, mode="mlp")
    foundry.mine("var/runs/mlp-example", session=session, proposers=[proposer])
```

`mode="mlp"` 每 64 日用历史前缀拟合一批 8→16→1 的 FP32 网络，训练标签为 20 日收益，标签终点严格早于预测段。网络生成 `targets`；另一个 GPU 奖励代理网络从驻留反馈学习，指导下一批超参选择。它不会把研究期反馈更新后的预测器回填到过去。

`mode="factor"` 用 GPU 张量、交互特征和横截面排序生成 `scores`，GPU CEM 根据同一奖励向量更新参数分布。示例还通过 `CandidateGrid` 执行九点邻域和延迟双生子检查。两者共用注册、归档、硬准入和有序提交接口；模型参数、拟合过程和学习规则由客户端持有。

数组注册把 GPU 输出复制到有界驻留租约，并在 GPU 上计算分块摘要，仅将紧凑摘要传回主机。入选、检查点引用及明确保留的候选才传输完整数据并落盘。输出及拟合截止点通过 `session.register` 显式声明。因果预测拟合不等于独立样本外选模：超参和因子搜索使用研究窗口反馈，报告中的收益仅供样本内研究。

`candidate_batch` 限制活动驻留输入总行数及持久化暂存。生产者在 `release(batch)` 中释放自己注册的批次；驱动器先等待学习完成并保存检查点最终引用。未选中数据直接释放，不写文件。明确保留复用注册时计算的内容身份，连续候选借用原切片，主机 PIT 截止数组保持在主机；使用有界暂存及原子文件，读取工件仍校验完整性。保留操作返回内容摘要相同、scope 为空的持久引用。组合和检查点禁止包含驻留 scope；不缓存历史 GPU 批次，也不自动归档每次试探。 注册、必要落盘和完整用时见[同卡保留路径优化对照](benchmarks/retention-l4-20261116/README.zh-CN.md)。

统一测速入口：

```bash
python -m foundry --parameters '{"search":{"epochs":4,"batch_size":64},"cuda":{"candidate_batch":64,"measurement_cache_mib":32}}' \
  benchmark var/runs/nonformula-example --suite nonformula --backends cuda
```

耗时、误差和准入结果见 [L4 接入验收报告](benchmarks/nonformula-l4-20261116/README.zh-CN.md)。该短数据窗口的延迟双生子为空，邻域结果只作为接口验收，不能证明因子稳健。

### 不限于公式的稳健性检验

公式可调用 `session.judge_family({"trend": "gt(ma_dist_20,0)"}, hypothesis="trend")`，由算子注册表派生参数与字段邻域。

外部模型参数扫描使用显式候选网格。假设同一打开的会话内已注册三个模型变体：

```python
variants = registered.candidates
family = foundry.CandidateGrid(
    seed=variants[1], variants=variants,
    parameter_coords=((0,), (1,), (2,)),
    hypothesis="Performance persists across adjacent model sizes",
)
result = session.judge_family(family)
```

种子必须在变体中恰好出现一次。坐标是维度一致、互不重复的整数阶梯位置；完整邻域才能使高原和双生子诊断有意义。家族结果用于研究，不发布组合，也不保证样本外收益。

## 因果 walk-forward 模型

`foundry.walk_forward` 为拟合和预测回调提供独立持有的特征前缀。拟合看不到该折截止点之后的行；预测只收到截至当前决策行的前缀，以及对应资格掩码。下面使用仓库美股研究区间，拟合每个资产的参考水平：

```python
with foundry.open(start="2025-07-01", end="2026-06-30") as session:
    xp = session.resident.xp
    times = session.grid.time_ns
    first = next(i for i, time in enumerate(session.grid.decision_times)
                 if time >= "2025-07-01T00:00:00Z")
    middle = (first + len(times)) // 2
    folds = (
        foundry.Fold(int(times[0]), int(times[first]), int(times[first]), int(times[middle])),
        foundry.Fold(int(times[0]), int(times[middle]), int(times[middle]), int(times[-1]) + 1),
    )

    def fit(view):
        return xp.nanmean(view.features["ma_dist_20"], axis=0)

    def predict(model, view):
        score = view.features["ma_dist_20"][-1] - model
        return xp.where(view.eligibility[-1], score, xp.nan)[None, :]

    result = foundry.walk_forward(
        session, fit, predict, folds=folds, ids=("fold-model",),
        kind="scores", unit="score", producer="prefix-model-v1", features=("ma_dist_20",),
    )
    report = session.evaluate(result.registered)
    print(result.sign_consistency)
```

`Fold(fit_start, fit_stop, predict_start, predict_stop)` 使用 int UTC 纳秒与左闭右开区间 `[start, stop)`。拟合结束后才能开始预测；预测折按序、互不重叠，且每折包含一次可执行决策。扩展窗口保持拟合起点，滚动窗口显式推进起点；最后一个排他边界可用 `int(grid.time_ns[-1]) + 1`。

预测每次返回研究资产顺序下的 `(B,S)` 后端数组。`FoldView` 包含时间戳 `start`、排他时间戳 `stop`、`decision_times`、`assets`、`features` 和 `eligibility`；数组为独立拥有的因果前缀。输出通过同样的注册检查。执行器将当前决策写入 `observed_at`，将最后可见拟合决策写入 `fit_at`；拟合区间需包含训练标签使用的全部观测。外部回调仍须对独立读取的数据及预训练模型来源负责。

折诊断根据模型在评价区间内首次实际基础 Bar 执行归属完整统计周期，直到下一模型首次执行所在周期；最后一折截至研究结束。评价决策偏移之前的目标保持空仓。首折可以归属其起始统计周期内此前的空仓前缀；后续模型切换必须落在统计周期的首个执行事件。周期内部切换或多折共享一个周期时，须使用更细的 `Clock.statistics`。`FoldResult.measurement_start` 和 `measurement_stop` 是统计周期的 UTC 纳秒左闭右开边界，独立于决策时间和成交时间；最终排他边界为最后周期起点加配置的统计周期时长。间隙中的退出费用、资金费和无法平仓的持仓继续归属前一模型。`sign_consistency` 为正折数与负折数的较大值除以总折数；零收益或无效折不进入分子。持续亏损模型也可能为 1，应同时检查符号和收益。操作仅注册并评价预测，不提交组合；使用后释放 `result.registered`，需要保留时先显式 retain。

## 远程会话

在 Linux/WSL2 计算主机运行 `python -m foundry serve`，用 `python -m foundry status` 查看地址。Windows 原生或 Linux Python 均可连接服务地址或隧道本地端口：

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    diagnostics = session.diagnose()
    family = session.judge_family("gt(ma_dist_20,0)")
```

诊断、权重试验和组合命令共用公开会话接口。模型可在自己的程序中拟合，然后注册输出进行远程评价。`session.resident` 与 `evaluate_device` 仅用于进程内计算。

[`rpc/schema.py`](../foundry/rpc/schema.py) 从类型契约派生请求与响应；[`rpc/codec.py`](../foundry/rpc/codec.py) 负责编码，[`rpc/server.py`](../foundry/rpc/server.py) 和 [`rpc/client.py`](../foundry/rpc/client.py) 负责传输。数组上传采用二进制包，编码前后均限制 1 GiB，禁用 pickle。组合与 RPC 使用 `research-v16`，明确拒绝旧格式。命令通过一个有界 worker 队列执行；客户端断开不会取消已经接纳的命令。

输入契约错误抛出 `ValueError` 或 `TypeError`，旧上下文抛出 `ContextConflict`，资源或实现失败直接传播；候选业务拒绝保留为结构化结果。机器调用请读取 `foundry.schema()`，不要拼接未公开的协议字段。

## 加密货币执行

现货使用连续 UTC 日历，支持保留的日线工件或显式基础 Bar。资格与报价分离；交易价格缺失可能阻止成交，使已有持仓继续保留。配置费用作用于实际执行的变化。

永续模拟 USDT 线性合约和逐仓抵押。目标是有符号名义金额/账户权益，总绝对敞口受 `Account.gross_limit` 限制；信号不会自动使用最大杠杆。每个基础事件依次处理标记价跳空、已有持仓资金费、调度到来的新目标、不利高低价强平检查和收盘标记。保留的 Binance 小时归档使用日线目标、小时事件，以及已披露的结算时间分桶与标记开盘价代理；`crypto-bars-v1` 使用所提供的基础频率和精确基础开盘资金费记录，不沿用该代理。

当前使用固定维持保证金和费率，不还原历史档位、VIP 费用、ADL、订单簿或逐 tick 日内成交路径。`risk` 报告资金费、交易费用、强平费用、强平次数、破产状态和账户权益，单位以 schema 描述为准。

账户估值独立于决策频率，复合归入 `Clock.statistics` 周期。crypto 年化系数为 `365 * 86400 / statistics_seconds`，日线预设保留原有口径。收益、Sharpe、报告回撤、`Metrics.n_periods` 和 `YearMetrics.periods` 均使用统计网格；周期标签为 UTC 起点，因此跨年收盘的周期归属其起点年份。报告回撤采样于统计收盘，基础 Bar 强平检查在两次统计收盘之间仍执行。`risk.first_liquidation_ns` 标识首次强平所在基础 Bar 的 UTC 起点，不表示精确 tick 或成交时点；`-1` 表示没有强平。

`turnover` 是每次决策的目标绝对变化均值，`margin_bps` 用决策频率年化这一分母。`exposure` 与逐年出勤率在统计收盘采样最近目标，描述目标而非实际成交持仓；实际费用和账户杠杆见 `risk`。

## 数值与性能

`foundry.CUDA(measurement_cache_mib=32)` 只缓存收益序列和紧凑报告。候选持仓按 `candidate_batch` 限制的小批次重新生成；入选成员独立复制自身持仓，不保留原候选批次的大数组。设预算为 `C`、批量为 `b`、成员上限为 `M`、天数为 `T`、资产数为 `S`，除固定行情面板和调用方持有的输出外，持久缓存空间为 `O(C + (b + M)TS)`。临时表达式与权重试探工作区另受表达式代价上限和 `trial_batch` 限制，不随候选总数增长。缓存不含“候选数 × 天数 × 资产数”项。预算设为 `0` 可禁用复用。该性能参数不改变研究快照身份；提交仍校验预期上下文，并随组合变化重新判断准入。

CUDA 批量和精度显式配置。双 FP32 用两个分量在支持的账户与统计路径保留残差精度。数值计划在会话启动时固定，不根据评价结果提高精度复算。小规模永续批次可通过 `CUDA.perpetual_host_cells` 在执行前规划为 CPU，不会因运行错误静默降级。

通过 `session.describe()` 及报告的精度/后端字段查看实际计划。[2026-11-15 PRO 6000 报告](benchmarks/arch-v2-pro6000-20261115/report.zh-CN.md) 对应源码 `fa2acfd`：512 项公开测试通过、1 项私有基线测试跳过，覆盖 66 个算子的 124 个用例全部通过。4096 条公式流程两端均接纳 28 个候选，没有决策不一致或数值误差超限；完整用例 CPU/CUDA 墙钟耗时为 93.724/2.879 秒。以上均为单次测量，此前 `current/` 报告保留为历史记录。

独立私有生产审计未通过：保留基线缺少 protocol/study 元数据，保存的组合仍含旧 BTC 特征公式。原文件没有改写或重录，不能宣称私有基线完全通过。CUDA 账户使用双 FP32，但未执行同卡单 FP32 对照，因此没有重新验收独立的 1.5 倍精度开销预算。详见[验收范围](benchmarks/arch-v2-pro6000-20261115/README.zh-CN.md)。

在获得授权的计算环境中，`python -m foundry benchmark OUTPUT` 是统一验证和基准入口，分别记录准备、驻留评价与完整查询/有序提交耗时，并报告数值误差和决策一致性。实现位于 [`validation/benchmarks`](../validation/benchmarks)，架构检查位于 [`validation/test_architecture.py`](../validation/test_architecture.py)。

使用 `--backends cpu` 或 `foundry.benchmark(output, backends=("cpu",))` 可独立测量 CPU 基线，不初始化 CUDA。此模式运行基准契约检查与性能测量，不运行完整多设备回归。将 OpenMP/BLAS 内层线程数设为 1，用 `cpu.workers` 匹配实际 CPU 配额并随结果保存；GPU 加速比必须注明所采用的固定 CPU 基线。

### CPU 并行参数

```python
with foundry.open(backend="cpu", cpu=foundry.CPU(workers=8, candidate_batch=32,
                                               measurement_cache_mib=32)) as session:
    report = session.evaluate(formulas)
    committed = session.commit(report)
```

独立候选和权重试算共享只读行情，在工作线程上并行执行；原生算子释放 GIL，每个 worker 内的 OpenMP 使用一个线程。最多同时挂起 `workers` 个任务，结果按输入顺序返回，组合仍由单写者按顺序提交。CPU 参数只影响执行资源，不改变研究快照身份或交易规则。

公式的中间数组在单个候选完成后释放；权重试算只为正在运行的 worker 展开持仓，临时持仓空间为 `O(workers × T × S)`，而非 `O(试算总数 × T × S)`。收益和事件明细是调用所需输出，另按输出规模占用内存。运行基准时设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，通过 `cpu.workers` 控制候选并发，避免线程池叠加。

### 测量现有 GFlowNet 挖掘流程

```bash
python -m foundry --parameters '{"search":{"epochs":8,"batch_size":512}}' \
  benchmark var/runs/discovery-new --suite discovery --backends cuda
```

直接调用现有 `foundry.mine()`，测量采样、CUDA 候选评价、奖励反馈、GFN 训练、有序准入、CEM 重配及 checkpoint 的完整流程。候选规模通过 `Search` 调整；报告同时列出阶段累计时间与端到端墙钟时间，训练与提交/重配的重叠不能重复相加。

GFN 模型定义、Trajectory Balance、配对教师和带梯度裁剪的 SGD 均在 Python/PyTorch 中实现。使用 `backend="cuda"` 时，前缀状态、合法动作掩码、抽样、网络推理和 FP32 训练使用独立 CUDA 流；CPU 仅接收探针小批次结果和最终公式 token。训练复用本批次采样时保存的特征与掩码，SGD 优化器随模型只初始化一次。CPU/portable 会话明确选择 PyTorch CPU 设备。`GfnProposer(...)` 只保存配置；设备由会话显式确定，只有 `open(device)` 才创建模型、流和优化器，不静默回退。随机提案器不需要神经网络。

32 单元 tanh 网络、初始字段/算子概率分配、奖励、探针门控和准入条件保持不变。设备上的 PyTorch 随机采样与 FP32 训练会改变旧原生学习器及主机采样版本的随机轨迹，同一 seed 不保证相同公式。检查点采用 `foundry-gfn-torch-v3`，只包含 JSON 和数值数组，不使用 pickle，明确拒绝旧版检查点；检查点包含设备随机状态，精确续跑要求相同设备类别；SGD 无动量状态。仅保留当前批次的前缀和合法动作掩码，空间为 `O(B × L × (F + A))`，不建立历史回放缓存。探针临时空间上限为四个前缀、每个四十次共同后缀尝试，只保留成功选中的探针。此基准不运行独立 CPU 对照，也不将同一历史窗口的自适应准入视为样本外收益。

最新接口验收见 [L4 批次归属与驻留反馈报告](benchmarks/api-boundary-l4-20261116/README.zh-CN.md)：同一候选序列，数值与决策一致，附完整耗时和资源数据。
