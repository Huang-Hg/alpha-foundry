# 研究与程序接入

[项目介绍](../README.zh-CN.md) · [English](research.md) · [数据格式](data.zh-CN.md) · [策略程序](programs.zh-CN.md)

一个 Foundry 会话在固定市场研究上评价公式、模型输出和有状态程序。候选评价与组合发布分开进行：人工、Agent 和自动发现共用同一套硬准入规则。

## 打开研究并审阅结果

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

省略组合路径时，修改只保留在内存。提供路径后，入选成员可原子发布到持久库。数组工件使用 `Admission.artifact_dir`，默认 `var/candidates`；持久程序通过调用方提供的工件目录重建。

`evaluate` 接受公式字符串、公式序列、名称到公式的映射、`Candidate`、候选序列、已注册批次、本地 `StrategyProgram` 或已部署 `ProgramReference`。程序输入要求 `Execution(model="orders")`。`row.ok` 表示评价成功；准入需检查 `row.decision.passed`、`reason`、`stopped_at` 与各约束裕度。候选业务拒绝返回报告行；输入错误抛出 `ValueError`/`TypeError`，资源失败直接传播。

| 调用 | 结果与影响 |
| --- | --- |
| `evaluate(inputs)` | 指标、风险与准入结论，不改组合 |
| `evaluate_device(inputs)` | 进程内驻留数组与同一次查询的可提交报告 |
| `register(values, ...)` / `register(program)` | 有界候选引用和调用方拥有的租约 |
| `retain(inputs)` | 持久引用，不释放原租约 |
| `release(batch_or_report)` | 结束指定临时租约 |
| `plan_program(program_or_reference)` | 运行前查看设备、状态容量、订单能力和时延 |
| `simulate(program_or_reference)` | 独立新账户审计，不是准入报告 |
| `diagnose()` / `judge_family(...)` | 组合缺口或邻域诊断，只读 |
| `portfolio()` / `trial_weights(matrix)` | 当前成员或新的权重比较，只读 |
| `commit(report)` | 有序准入与组合原子发布 |
| `commit_weights(trials, index)` / `reweight(...)` | 提交已审阅或求解得到的可行权重 |

报告同时绑定研究快照与准确的组合版本。其他写入者改变组合后，提交会抛出 `ContextConflict`。此时调用 `refresh()`，重新评价和审阅；刷新不会让旧报告自动变新。发布失败不留下部分组合变更。查询不会自动保留所有候选或发布组合。

`foundry.schema()` 无需打开数据或设备就能返回参数/RPC schema；`session.describe()` 给出字段、66 个算子、实际时钟、执行计划、容量和解析后的设置。市场成本及 `Clock`、`Sizing`、`Account`、`Execution`、`CPU`、`CUDA`、`Admission`、`Search` 等组直接传入 `open`/`serve`；修改设置时打开新的会话。相对配置路径从仓库根目录解析。

### 准入试算规模

`Admission(weight_steps=N)` 设置 N 个正混合系数网格点 `1/N, 2/N, ..., 1`，另加解析切线建议，重复点跳过；默认 20。订单模型每个试算都重放实际账户，因此它会明显影响查询成本。探索时可使用较小网格：

```python
with foundry.open(admission=foundry.Admission(weight_steps=4)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

这只改变检查哪些混合系数，不改变硬约束；较小网格可能漏掉可行改进。`min_sharpe_gain` 默认 0.01，`max_members` 默认 40。应查看报告实际试算数和约束裕度；`trial_weights` 则评价调用方显式给出的矩阵。这些参数在打开会话时固定并进入研究身份。

## 执行与组合资金语义

在打开会话时选择执行模型：

| 设置 | 含义 |
| --- | --- |
| `Execution(model="targets")` | 默认四市场目标回测，接受公式和权益比例数组 |
| `Execution(model="orders")` | 加密现货/USDT 逐仓 Bar 账户，公式、数组与程序共用实际执行 |
| `Account(composition="shared")` | 一个实际账户，成员系数仅缩放一次意图 |
| `Account(composition="independent")` | 仅订单模型；系数一次性分配各自初始现金 |

美股和 A 股当前支持日线。crypto 可使用日线、支持的小时永续归档，或带发布时间的基础 Bar，并独立配置决策/统计频率。股票程序订单所需的结算和公司行动能力尚未提供。

显式 5 分钟 crypto 数据示例：

```python
with foundry.open(
    market="crypto_perpetual", data_dir="/datasets/crypto_5m",
    start="2025-01-08", end="2025-01-31", backend="cuda",
    clock=foundry.Clock(decision="15min", statistics="1D"),
    account=foundry.Account(margin="isolated", leverage=3., gross_limit=2.,
                            composition="shared"),
    execution=foundry.Execution(model="orders", participation=0.1),
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

完成的 15 分钟决策最早在接下来的 5 分钟基础开盘生效，包括相邻周期共用的边界，不必等到下一次 15 分钟决策。公式窗口以决策 Bar 计数。两次决策之间，实际持仓、挂单、资金费和强平仍持续演进。统计使用独立周期收盘；crypto 年化系数为 `365 * 86400 / statistics_seconds`。

**共享账户**中，所有成员看到总现金、权益与净持仓，并另外收到自身有符号 `owner_quantity`。系数缩放目标值和程序订单数量，不划出独立现金。程序返回系数为一时的未缩放数量。若要关闭已实现的成员持仓，先将其绝对数量除以 `context.intent_coefficients`，因为重放会再应用一次系数。[程序仓位语义](programs.zh-CN.md#仓位与成员持仓)给出准确约定和 reduce-only 限制。

**独立账户**中，系数之和不得大于一，即使只超出很小正值也会拒绝。每个账户起始现金为 `Account.initial_cash * coefficient`，未分配资金保持现金。意图不二次缩放，运行上下文中的意图系数为一。账户依赖策略根据实际分配的资金重放；固定数量策略仍须自行满足资本约束，资金系数不会自动把它改成权益比例策略。

权重矩阵为 `(trials, members)`，成员顺序对应 `portfolio().sources`。系数必须有限且非负，全零表示现金。在订单模型中，每个候选、变化的权重向量和有序准入都创建新的程序状态并重放实际账户，不能把曾实现的持仓路径当作账户状态不变时的静态混合捷径。

Bar 模型使用明确的 `bar_path`、参与率、价格/数量步长、预留资金和市场规则；缺失报价可能阻止成交。它不模拟订单簿排队，也不会在任意 Bar 内成交瞬间调用用户程序。根据反馈追加的保护单最早只能在下一次配置的决策时提交。

## 数组与模型输出

`session.grid` 给出有序资产、标准 UTC `decision_times`、对应 int64 UTC 纳秒 `time_ns` 和研究快照。数值数组为 float32/float64 `(B,D,S)`：候选、决策、资产。观测与拟合截止点为 int64 UTC 纳秒 `(B,D)`，不能晚于对应决策。`NO_INFORMATION` 仅表示没有使用相应观测或拟合信息，不表示来源不明。

下面使用公开驻留特征生成一个因果数值评分：

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
        scores, kind="scores", unit="score", ids=("custom-trend",),
        grid=grid, producer="custom-trend-v1", observed_at=observed, fit_at=fitted,
    )
    try:
        report = session.evaluate(registered)
        saved = session.retain(registered.candidates[:1])
        print(report.rows[0].metrics, saved.candidates)
    finally:
        session.release(registered)
```

`session.resident` 是进程内只读视图。远程客户端在自己的程序中生成 NumPy 数组，并对齐远程网格注册。注册拥有独立驻留副本；返回后修改原缓冲区不会改变候选。

| 输入 | 语义 |
| --- | --- |
| `scores`，单位 `score` | NaN 表示缺失，由 `Sizing` 定义目标映射 |
| `Sizing(rule="equal_slots")` | 默认截断评分并除以合格资产数 |
| `Sizing(rule="independent", scale=...)` | 逐资产映射，不受资产数稀释；超总敞口作为候选拒绝 |
| `targets`，单位 `equity_fraction` | 有符号名义金额/账户权益，绕过评分映射 |
| `targets`，单位 `quantity` | 资产数量，仅订单模型 |
| `targets`，单位 `notional` | 报价币金额，仅订单模型 |
| 目标 `update_mask=False` | 保留该成员先前目标，仅订单模型 |
| 目标为零且 `update_mask=True` | 明确关闭该资产目标 |

目标必须有限，符合市场方向规则，并在无资格资产上为零。权益比例在注册时检查总敞口预算；quantity/notional 在执行时按实际价格和账户资金换算，不可支付或超预算的订单可能被拒绝，不偷偷裁剪。评分不接受 mask。目标模型要求省略 mask 或全 True；省略与全 True 对应同一候选身份。

`producer` 标明实现身份；可选归一化 `complexity` 只影响软发现评分，不改变硬准入。拟合截止点须覆盖训练标签、预处理和校准所用的全部信息。Foundry 检查声明是否引用未来，但不能证明外部预测的真实来源。

## 一次查询取得驻留结果

模型或 Agent 同时需要实际驻留收益和准入报告时，使用 `evaluate_device`：

```python
with foundry.open(backend="cuda") as session:
    result = session.evaluate_device({"trend": "gt(ma_dist_20,0)"})
    print(result.report.rows[0].metrics)
    print(result.batch.rows[0].returns.shape)
    session.commit(result)
```

不可变 `ResidentEvaluation` 中的 `.report` 与 `.batch` 按同一候选顺序对应，来自一次评价；无需再次 `evaluate`，也无需手工拼装报告行。输入也可为数组注册、本地程序或已部署引用，`commit(result)` 直接接受该封装。直接程序查询可用 `release(result)` 结束报告租约；显式注册则释放原注册批次。在所属会话内消费驻留数组，不修改借用缓冲区。该方法要求进程内会话，远程研究返回类型化报告，不返回客户端本地设备指针。

CUDA 数组的执行顺序跟随包围评价调用的流。直接在该流使用，或显式转交其他流；返回 Python 封装不等于建立了任意跨流就绪关系。例如：

```python
with foundry.open(backend="cuda") as session:
    port, xp = session.resident, session.resident.xp
    with port.scope():
        producer = xp.cuda.Stream(non_blocking=True)
        consumer = xp.cuda.Stream(non_blocking=True)
        with producer:
            result = session.evaluate_device("gt(ma_dist_20,0)")
            ready = xp.cuda.Event()
            ready.record()
        with consumer:
            consumer.wait_event(ready)
            mean_return = result.batch.rows[0].returns.mean()
        consumer.synchronize()
        print(float(mean_return))
    session.commit(result)
```

内部公式辅助流在使用其结果前已汇合。若由 PyTorch 消费批次，也要为它的流建立相应事件依赖。释放作用域资源前完成消费。提案奖励反馈是另一类对象，其 `borrow(stream_pointer)` 会自动传递跨线程的生产就绪关系，见下文。

## 候选归属与复用

所有活动数组和程序租约合计不超过 `candidate_batch`。注册下一批前先释放已完成批次；注册不会把每次试探都落盘，也不会隐式溢写。

- 显式 `register(...)` 返回调用方拥有的批次。`evaluate(batch)` 和 `commit(report)` 不会释放它，使用后自己释放。
- 便捷 `evaluate(program_or_reference)` 返回带 `report.lease` 的报告。`release(report)` 释放它，成功 `commit(report)` 消费它；失败或冲突保留租约，供重新评价或显式释放。
- `retain(...)` 返回无临时 scope 的持久引用，原租约仍需释放。提交和发现会在发布/检查点前保留它们必需的引用。
- 已释放的驻留引用失效。持久数组需要对应工件文件；持久程序在重新打开时需要相同部署目录和确定性 loader。

在同一注册作用域内复用不可变模型；每次查询仍调用 `open(context)` 创建独立私有状态，并关闭该次运行。释放只移除会话持有的模型引用，不会销毁应用另外持有的引用。实际账户轨迹不会作为权重试算缓存复用。

## 有状态程序与订单审计

公开策略由 `ProgramDescriptor` 与 `open(context)` 构成，后者返回 `ProgramRun`。`on_event` 消费独立拥有的因果特征窗口和实际账户，返回有界 `OrderCommands`；每个基础 Bar 完成后收到实际 `on_feedback`。失败路径也会关闭运行。`context.batch_size` 是本次有效批量，驻留 `variant_indices` 表示原始候选变体；不要仅按 `descriptor.batch_size` 分配运行状态。

程序可参与 `evaluate`、发现、家族检查、权重试算、重配与提交。部署引用包含经完整性校验的工件字节和规范 JSON 参数。向 `open` 或 `serve` 提供 `DeployedPrograms` 属于应用显式部署；研究 RPC 不传源码、不安装依赖、不反序列化 Python 模型。完整代码与远程部署见[程序指南](programs.zh-CN.md)。

`simulate` 保留独立审计职责，返回驻留收益、权益与最终实际账户，不作准入判定，也不能直接提交；组合流程使用 `evaluate`。

## 策略发现

```python
with foundry.open() as session:
    candidates = (
        foundry.Candidate("trend", foundry.Formula("gt(ma_dist_20,0)")),
        foundry.Candidate("momentum", foundry.Formula("cs_rank(ret_20)")),
    )
    discovery = foundry.Discovery(session)
    report = discovery.evaluate(candidates)
    for row in report.rows:
        print(row.candidate.id, row.log_reward, row.decision)
    session.commit(report)
```

已注册数组/程序的 `.candidates` 可放在同一位置。发现流程用行为身份、相关性和组合缺口指导提案，软奖励不能推翻硬准入。组合变化后重新创建 `Discovery`。延迟噪声对照报告不能提交，应先无延迟评价实际候选。

模型/程序参数家族可先注册变体，再调用 `CandidateGrid(seed, variants, parameter_coords, hypothesis)`。坐标为维数相同、互不重复的整数元组，种子恰好出现一次。`judge_family` 提供研究诊断，不发布组合，也不证明样本外稳健性。

`ExternalProposer` 接受有限的候选或已注册批次：

```python
with foundry.open(search=foundry.Search(epochs=2, accept=False)) as session:
    candidates = (foundry.Candidate("trend", foundry.Formula("gt(ma_dist_20,0)")),)
    proposer = foundry.ExternalProposer([candidates])
    foundry.mine("var/runs/external", session=session, proposers=(proposer,))
```

自定义提案器实现 `open(device)`，在作用域中返回具备 `propose`、`learn`、`release`、`dump`、`load` 的生产者。反馈严格保持原提案批次顺序；驻留奖励支持显式传回主机，或遵循流依赖的 CUDA 借用。模型和拟合由提案器持有。驱动器在释放前保存所需引用，不缓存历史 GPU 批次。

### 反馈与自定义生产者

固定批次适配器的准确构造器为 `ExternalProposer(batches, *, name="external", feedback=None)`。`batches` 每项可以是 `Candidate`、`RegisteredBatch` 或候选序列。可选回调接收一个 `LearningFeedback`，返回 `TrainingResult`，不能返回普通字典或 `None`：

```python
def learn(feedback: foundry.LearningFeedback) -> foundry.TrainingResult:
    rewards = feedback.rewards.to_host()
    print(feedback.batch.sequence, [candidate.id for candidate in feedback.batch.candidates])
    return foundry.TrainingResult(metrics={"mean_reward": float(rewards.mean())})


with foundry.open(search=foundry.Search(epochs=1, accept=False)) as session:
    batch = (foundry.Candidate("trend", foundry.Formula("gt(ma_dist_20,0)")),)
    proposer = foundry.ExternalProposer([batch], name="my-model", feedback=learn)
    foundry.mine("var/runs/my-model", session=session, proposers=(proposer,))
```

`feedback.batch` 是准确的提案句柄，包含 `owner`、`sequence`、`context`、`ordinary` 及有序 `.candidates`；`.auxiliary` 为总候选数减普通候选数。`.rewards` 是同样总行数的独立 FP32 向量，提供 `.shape`、`.device`。`to_host()` 显式返回独立 NumPy 副本，并在需要时等待 CUDA 生产完成。

GPU 学习在指定设备初始化自己的模型/状态，无需主机传输即可消费反馈。进程内 CUDA 会话中的以下回调把摘要写入调用方自己的存储，可将该赋值换成优化器计算：

```python
xp = session.resident.xp
consumer = xp.cuda.Stream(non_blocking=True)
reward_summary = xp.empty((1,), dtype=xp.float32)


def learn_gpu(feedback: foundry.LearningFeedback) -> foundry.TrainingResult:
    with feedback.rewards.borrow(consumer.ptr) as rewards:
        reward_summary[:] = rewards.mean()
    return foundry.TrainingResult()
```

CUDA `borrow` 需要消费者流指针，会等待生产事件，并在退出作用域时记录消费完成。CPU 使用不带流的 `borrow()`，得到只读视图。不要修改奖励，也不要把别名/DLPack 张量带出借用范围。驱动器负责关闭反馈，等待学习与消费后再释放提案；这个小奖励租约不依赖可变组合状态。

动态生成/训练的提案器实现以下公开生命周期：

| 方法 | 签名与职责 |
| --- | --- |
| 配置 `open` | `open(device: foundry.ComputeDevice)` 返回上下文管理器，产生一个生产者；在此初始化/释放模型、优化器与流 |
| 生产者 `propose` | `propose(request: foundry.ProposalRequest) -> foundry.ProposalBatch \| None`，None 表示该生产者结束提案 |
| 生产者 `learn` | `learn(feedback: foundry.LearningFeedback) -> foundry.TrainingResult` |
| 生产者 `release` | `release(batch: foundry.ProposalBatch) -> None`，结束准确的活动批次及其拥有的注册 |
| 生产者 `dump` | `dump() -> bytes`，没有活动批次时导出明确版本、非 pickle 的提案器状态 |
| 生产者 `load` | `load(payload: bytes) -> None`，校验并恢复匹配的提案器状态 |

配置和生产者具有非空 `name`，同一挖掘运行内不同提案器名称唯一。`ProposalRequest` 包含 `.context`、请求 `.count` 及诊断 `.features` 元组。构造 `ProposalBatch(owner, sequence, context, ordinary, candidates)` 时，每个打开的生产者使用唯一 owner 和稳定批次序号，并将原句柄保留到 release。每个生产者最多一个未释放批次，反馈必须匹配其顺序、上下文和身份。`ExternalProposer` 接收已注册输入时只借用引用，调用方在挖掘期间保持租约有效，并在结束后释放；自定义生产者自行注册新输出时负责释放。库/检查点必需的候选会在释放前完成保留。

内置挖掘使用选定的 GFlowNet 或随机公式提案器，CUDA GFlowNet 推理和训练在设备上使用 PyTorch。`Search(accept=False)` 只演进内存研究组合，不发布持久库；持久发布需要 `accept=True`。发现、挖掘和 walk-forward 需要进程内会话，远程会话提供候选研究调用，不传输 Python 学习回调。

挖掘 checkpoint 恢复发现/提案器状态及持久候选引用，然后重新重放账户。另一套显式[程序审计 checkpoint](programs.zh-CN.md#独立审计与恢复状态)在完整基础 Bar 边界保存声明的私有张量及实际账户/订单状态，通过进程内 `simulate(..., until=..., checkpoint=...)`、`export` 和 `load_program_checkpoint` 使用。

## 因果 walk-forward 模型

`foundry.walk_forward(session, fit, predict, folds=..., ids=..., kind=..., unit=..., producer=..., features=...)` 为 `fit` 提供独立拥有的历史特征区间，为 `predict` 提供直到当前决策的因果前缀。预测返回 `(B,S)` 数组，辅助器注册全部预测并填写观测/拟合截止点。

`Fold(fit_start, fit_stop, predict_start, predict_stop)` 使用整数 UTC 纳秒和左闭右开 `[start,stop)`。拟合先于预测，折按序且不重叠。若训练标签消耗后续观测，须将它们纳入拟合区间；未来标签不属于特征面板。后续模型切换要落在统计周期边界；周期内切换应选择更细统计频率。

结果包含 `.registered` 及分折收益/符号诊断，可按普通候选评价，最后释放 `.registered`。仅有符号一致性不足以说明有效，持续亏损的折也可能同号。同一研究窗口上的自适应选模仍是样本内研究。

## 远程会话

计算主机运行明确的服务配置：

```python
import foundry
foundry.serve("var/runs/library.json", backend="cuda")
```

在该主机用 `python -m foundry status` 查看端口。服务绑定 loopback；跨机器使用 `ssh -L LOCAL_PORT:127.0.0.1:REMOTE_PORT user@host` 转发。客户端检出匹配协议的代码，安装 Python 依赖后连接：

```python
import foundry
with foundry.connect(("127.0.0.1", 9000)) as session:
    print(session.describe()["execution_plan"])
    report = session.evaluate("gt(ma_dist_20,0)")
```

把 `9000` 换成隧道本地端口。原生 Windows 客户端无需 Linux 扩展或本地 CUDA。模型数组通过有界 NumPy 二进制包上传，编码/解码均限制 1 GiB；程序查询使用已部署 `ProgramReference`，见[部署示例](programs.zh-CN.md#远程部署)。

服务通过有界队列串行处理任务。断开客户端不会取消已接受命令，关闭客户端也不停止共享计算。上下文冲突要求显式刷新。旧组合、checkpoint、RPC 协议会明确拒绝；需要有意重跑，不猜测或自动迁移格式。

## 阅读指标与选择资源

统计标签是周期起点，收益和报告回撤在统计收盘度量。两次统计采样之间，基础 Bar 强平检查仍执行。`risk.first_liquidation_ns` 是首次受影响基础 Bar 的起点，不是精确 Bar 内成交时间；`-1` 表示没有强平。

`Metrics.activity_basis` 区分目标持仓指标与实际成交指标。`executed_fills` 的换手为各基础 Bar 的实际绝对成交名义金额，除以开盘标记/资金费/风险处理后、命令执行前的账户权益，累计后再除以研究决策数。独立账户先合计各自成交和开盘权益，并包含闲置现金，再计算每个比值。`margin_bps` 按决策频率年化这一换手分母。出勤率衡量有实际敞口的统计周期。`risk.gross_sampling` 中，共享账户使用账户事件采样，独立账户使用统计收盘的准确组合最大值，两者都不是 tick 级最大值。

`Metrics.fills` 是订单模型的原生成交事件数，部分成交各计一次。目标模型无法提供该计数时为 `None`，不能从目标换手推断成交笔数。

CUDA 批量适合有规模的重复研究。通过 `CUDA(candidate_batch=..., trial_batch=..., measurement_cache_mib=...)` 或 `CPU(workers=..., candidate_batch=...)` 控制资源。目标永续路径可按 `perpetual_host_cells` 在执行前规划小批次 CPU；订单路径始终使用明确选择的设备。CUDA 程序须通过 `context.stream` 建立框架流依赖，并让声明的私有状态驻留设备；不支持的设备明确失败。

数值计划在评价前固定，不根据结果提高精度重算。[基准报告](benchmarks/README.zh-CN.md)记录硬件、规模、准备/JIT、驻留工作、完整流程、误差与决策。自行测量时，用 pytest、pytest-benchmark 调用公开 Session API；测试和辅助脚本留在本地，不随 Foundry 发布。历史公式耗时不代表订单程序吞吐。
