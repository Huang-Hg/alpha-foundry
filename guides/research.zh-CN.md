# 研究与程序接入

[项目介绍](../README.zh-CN.md) · [English](research.md)

Foundry 用同一个会话评价公式、模型信号和目标持仓。人工研究、AI Agent、公式挖掘器与外部模型搜索共用市场规则和组合准入结论。

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

省略组合路径时，组合库保存在内存中。注册数组仍写入 `Admission.artifact_dir`，默认是 `var/candidates`。传入组合路径后启用持久化和原子发布。退出 `with` 释放本地计算资源；关闭远程客户端不会停止共享服务。

报告绑定研究快照和组合版本。其他写入者改变上下文后，旧报告会触发 `ContextConflict`；显式调用 `session.refresh()` 后重新评价，不会自动重试旧决策。

准入报告保留每个约束的原始变化量 `delta` 和阈值 `required`，空阈值表示无法达到的要求。搜索层独立归一化这些裕度；修改四个 `Search.violation_scale_*` 只改变发现奖励，不再改变研究快照、数组身份或硬准入。被拒候选的展示试探点使用组合层自有尺度。本次修正会让旧快照 ID 一次性变化，旧快照绑定的数组需要显式重新注册，不能继续复用旧句柄。

### 参数与自描述

`market`、`backend`、`data_dir`、`start`、`end` 和市场费用直接传入。高级参数组包括 `foundry.CPU`、`foundry.CUDA`、`Admission`、`Operators`、`Search`、`Single` 与 `Monitor`。参数在会话打开时固定；比较其他设置时另开会话。

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

小时数据契约由 `trade.parquet`、`mark.parquet`、`funding.parquet` 和 `manifest.json` 组成。manifest 记录资产顺序、工件哈希，以及需要时的资金费间隔变更证据。仅有日线 OHLC 无法还原资金费和强平。

```python
foundry.import_perpetual(
    "/datasets/my_perpetual",
    symbols="BTCUSDT ETHUSDT SOLUSDT",
    history_start="2025-01-01",
    start="2025-07-01", end="2025-12-31",
)
```

导入器获取公开 USD-M 归档、验证校验和并保存来源记录。读取时使用 `market="crypto_perpetual"`、对应目录与日期。已有数据的资产顺序由 manifest 决定；`symbols` 用于新数据采集。其他供应商需满足 [`market/perpetual.py`](../foundry/market/perpetual.py) 的显式事件契约。

## 外部模型与目标持仓

`session.grid` 提供日期、资产顺序和快照身份。数组形状为 `(B,T,S)`，分别表示候选、日期、资产。下面注册三个零信号候选以演示接口；实际接入时替换为模型输出和真实信息截止点。

```python
import numpy as np
import foundry

with foundry.open() as session:
    grid = session.grid
    days, assets = len(grid.dates), len(grid.assets)
    values = np.zeros((3, days, assets), dtype=np.float32)
    unused = np.full((3, days), -1, dtype=np.int64)
    registered = session.register(
        values, kind="signals", ids=("model-small", "model-base", "model-large"),
        grid=grid, producer="external-program-v1",
        observed_through=unused, fit_through=unused,
        complexity=(0.2, 0.5, 0.8),
    )
    report = session.evaluate(registered)
```

| 类型 | 含义 | 约束 |
| --- | --- | --- |
| `signals` | 由 Foundry 转换为持仓的评分 | 浮点值；允许 NaN 表示缺失信号 |
| `target_weights` | 股票或现货的多头资产权重 | 有限、非负、总敞口不超过 1，不具资格资产为零 |
| `target_exposure` | 永续有符号名义金额/权益比例 | 有限、总绝对敞口不超过配置杠杆，不具资格资产为零 |

数值必须是浮点数组。观测和拟合截止点为 `(B,T)` 整数数组，使用研究网格行号且不晚于决策行 `t`；`-1` 表示未使用观测或拟合。已训练模型不能为了通过检查而伪报 `-1`。日线收盘决策最早在下一开盘执行。

`producer` 标识生成程序。可选的逐候选 `complexity` 范围为 `[0,1]`，仅影响发现流程的软评分，不改变硬准入。注册后返回绑定内容身份的候选并保存工件。远程注册自动传输 NumPy 数组，无需服务器路径；本地也支持 CuPy。

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

自定义提案器只需实现 `open(device)` 上下文管理器，返回具备 `propose(request)`、`learn(feedback)`、`release(batch)`、`dump()`、`load()` 的生产者。`ProposalRequest` 仅含批量规模、研究上下文和诊断特征；模型训练参数留在各自构造器。每个生产者仅允许一个未释放批次，反馈必须匹配原始批次句柄，不能混用实例、顺序或快照。驱动器等待学习完成后释放反馈和轨迹，再采样下一批；异常同样释放。普通候选与辅助候选按实际数量统计，随机提案器不再隐式追加 GFN 探针。检查点协议为 `candidate-discovery-v2`，旧检查点明确拒绝。额外反馈空间为 `O(B)`，不保存历史奖励或轨迹。


不指定提案器时，`foundry.mine(output, ...)` 使用配置的 GFN 或随机公式提案器。默认 `Search(accept=False)` 仍通过有序准入和定期重配演进内存研究组合，只禁止发布到持久组合库。需要发布时使用 `Search(accept=True, seed_portfolio="current")`。传入持久会话也必须启用该设置，`accept=False` 时会明确拒绝。`Discovery`、挖掘和 walk-forward 需要进程内会话，其设备驻留计算不是远程回调协议。

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
    dates = session.grid.dates
    first = next(i for i, date in enumerate(dates) if date >= "2025-07-01")
    middle = (first + len(dates)) // 2
    folds = (
        foundry.Fold(0, first - 1, first, middle),
        foundry.Fold(0, middle - 1, middle, len(dates)),
    )

    def fit(view):
        return xp.nanmean(view.features["ma_dist_20"], axis=0)

    def predict(model, view):
        score = view.features["ma_dist_20"][-1] - model
        return xp.where(view.eligibility[-1], score, xp.nan)[None, :]

    result = foundry.walk_forward(
        session, fit, predict, folds=folds, ids=("fold-model",),
        kind="signals", producer="prefix-model-v1", features=("ma_dist_20",),
    )
    report = session.evaluate(result.registered)
    print(result.sign_consistency)
```

`Fold(fit_start, fit_through, predict_start, predict_stop)` 使用研究网格绝对行号；训练包含 `fit_through`，预测不包含 `predict_stop`。拟合必须早于预测，预测折按顺序排列且不得重叠，每折至少包含一次可执行决策。扩展窗口保持 `fit_start` 不变，滚动窗口则显式推进起点；研究起点前需有足够训练历史。

每次预测用当前后端的数组库返回 `(B,S)`，资产顺序与研究一致。`FoldView` 提供 `start`、不包含的 `stop`、`dates`、`assets`、`features` 和 `eligibility`。输出执行与外部数组相同的注册检查，非法敞口直接拒绝，不自动遮盖。执行器记录观测行 `t` 和该折拟合截止点；无法约束回调私自读取未来数据，或使用已在未来数据上训练的模型。

每折净收益从首个可执行决策归属到下一模型开始执行；最后一折归属到研究结束。空档期的退出成本、资金费和无法退出的持仓延续计入前一模型。`sign_consistency` 是正收益折数与负收益折数中较大者除以总折数；零收益或无效折不计入分子。持续亏损模型也可能得到 1，因此要同时看各折符号和收益。该操作注册并评价预测，不提交候选。

## 远程会话

在 Linux/WSL2 计算主机运行 `python -m foundry serve`，用 `python -m foundry status` 查看地址。Windows 原生 Python 或 Linux 客户端连接服务地址，或隧道转发到本地的端口：

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    diagnostics = session.diagnose()
    family = session.judge_family("gt(ma_dist_20,0)")
```

公式评价、数组注册、家族检验、诊断、权重试验和组合命令共用公开会话接口。模型可在自己的程序中拟合，然后注册输出进行远程评价。`session.resident` 与 `evaluate_device` 仅用于进程内计算。

[`rpc/schema.py`](../foundry/rpc/schema.py) 从类型契约派生请求与响应；[`rpc/codec.py`](../foundry/rpc/codec.py) 负责编码，[`rpc/server.py`](../foundry/rpc/server.py) 和 [`rpc/client.py`](../foundry/rpc/client.py) 负责传输。数组上传采用二进制包，编码前后均限制 1 GiB，禁用 pickle。组合与 RPC 使用 `research-v13`，明确拒绝旧格式。命令通过一个有界 worker 队列执行；客户端断开不会取消已经接纳的命令。

输入契约错误抛出 `ValueError` 或 `TypeError`，旧上下文抛出 `ContextConflict`，资源或实现失败直接传播；候选业务拒绝保留为结构化结果。机器调用请读取 `foundry.schema()`，不要拼接未公开的协议字段。

## 加密货币执行

现货使用 UTC 日线和 365 日年化。资格与报价分离；交易价格缺失可能阻止成交，使已有持仓继续保留。配置费用作用于实际执行的变化。

永续模拟 USDT 线性合约和逐仓抵押。日线目标是有符号名义金额/账户权益，总绝对敞口受配置杠杆限制；信号不会自动使用最大杠杆。小时事件依次处理标记价跳空、已有持仓资金费、日线再平衡、不利高低价强平检查和收盘标记。资金费按实际事件时间映射到小时网格；公开归档缺少结算标记价时，导入器会明确记录标记开盘价代理。

当前使用固定维持保证金和费率，不还原历史档位、VIP 费用、ADL、订单簿或逐 tick 日内成交路径。`risk` 报告资金费、交易费用、强平费用、强平次数、破产状态和账户权益，单位以 schema 描述为准。

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
