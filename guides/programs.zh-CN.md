# 策略程序与模型部署

[项目介绍](../README.zh-CN.md) · [研究指南](research.zh-CN.md) · [数据格式](data.zh-CN.md) · [English](programs.md)

当下一次决策依赖实际现金、持仓、挂单或已有成交时，使用策略程序。预计算决策仍可用公式和数组。以 `Execution(model="orders")` 打开会话后，三种表达共用实际账户评价、发现与组合重放。

订单模型支持明确选择的 CPU/CUDA 上的美股、A 股、加密现货与 USDT 线性永续。现金市场遵循编译后的结算、公司行动、可交易性和数量规则；永续账户支持逐仓与全仓保证金。模型基于完成的 Bar、声明的 OHLC 路径和成交量参与率，不模拟订单簿或 tick 级排队。

## 完整公开程序

下面只使用公开 `foundry` 接口及 Python 标准库，保存为 `my_program.py`。资产零取得资格后，它尝试一次市价买入。数量未预先乘成员系数，组合重放负责应用系数。这个简化策略的首次请求若被拒绝，不会重试。

```python
from dataclasses import dataclass
import json
import foundry


@dataclass(frozen=True)
class BuyOnce:
    descriptor: foundry.ProgramDescriptor
    asset: int
    quantity: float

    def open(self, context: foundry.ProgramContext) -> foundry.CheckpointableRun:
        return BuyOnceRun(context, self.asset, self.quantity)


class BuyOnceRun:
    def __init__(self, context, asset, quantity):
        self.context, self.asset, self.quantity = context, asset, quantity
        self.sent = context.xp.zeros((context.batch_size, 1), dtype=context.xp.bool_)

    def on_event(self, event: foundry.ProgramEvent) -> foundry.OrderCommands:
        context, xp = self.context, self.context.xp
        shape = (context.batch_size, 1)
        integer = lambda value: xp.full(shape, value, dtype=xp.int64)
        number = lambda value: xp.full(shape, value, dtype=xp.float64)
        new = (~self.sent) & event.eligibility[-1, self.asset]
        self.sent |= new
        quote = xp.broadcast_to(event.features["close"][-1, self.asset], shape).copy()
        return foundry.OrderCommands(
            action=xp.where(new, int(foundry.OrderAction.NEW), 0).astype(xp.int64),
            owner=integer(context.owner), id=integer(1), asset=integer(self.asset),
            side=integer(1), quantity=number(self.quantity),
            type=integer(int(foundry.OrderType.MARKET)), price=number(0),
            reservation_price=quote, valid_until=integer(-1),
            reduce_only=xp.zeros(shape, dtype=xp.bool_),
        )

    def on_feedback(self, feedback: foundry.ExecutionFeedback) -> None:
        pass  # Consume or stream actual fills here; orders require a decision.

    def snapshot(self) -> tuple[foundry.StateTensor, ...]:
        return (foundry.StateTensor("sent", self.sent.copy()),)

    def restore(self, state: tuple[foundry.StateTensor, ...]) -> None:
        self.sent[:] = state[0].values

    def close(self) -> None:
        del self.sent


def build_buy_once(payload: bytes, parameters: str, digest: str) -> BuyOnce:
    model, settings = json.loads(payload), json.loads(parameters)
    descriptor = foundry.ProgramDescriptor(
        model["id"], artifact_digest=digest, parameters=parameters,
        state_schema=(foundry.TensorState("sent", "bool", (1,)),),
        features=("close",), history=1, batch_size=1, devices=("cpu", "cuda"),
        checkpointable=True,
    )
    return BuyOnce(descriptor, model["asset"], float(settings["quantity"]))


artifact = foundry.ProgramArtifact(
    payload=b'{"asset":0,"id":"buy-once"}',
    semantic_version="buy-once-v2",
    build=build_buy_once,
)
catalog = foundry.DeployedPrograms((artifact,))
reference = foundry.ProgramReference(artifact.digest, '{"quantity":0.001}')
```

工件摘要绑定实际不可变字节与显式实现语义版本。预先安装、受信任的工厂解释这些字节，不执行传入的 Python 文本，也不反序列化 Python 对象。修改工厂/运行代码的行为时同步更新语义版本。参数是有限 JSON 对象，键排序、无多余空格、Unicode 不转义，UTF-8 最多 64 KiB；可用 `json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` 生成。`parameter_digest` 自动派生，不由调用方随意填写。

第一块作为可导入模块的部署定义；下面另存为调用脚本。按[数据指南](data.zh-CN.md)准备数据，日期须落在实际覆盖范围内：

```python
import foundry
from my_program import catalog, reference

with foundry.open(
    "var/runs/program-portfolio.json", programs=catalog,
    market="crypto_spot", data_dir="/datasets/crypto_5m",
    start="2025-01-08", end="2025-01-31", backend="cuda",
    clock=foundry.Clock(decision="5min", statistics="1D"),
    account=foundry.Account(initial_cash=1000., composition="shared"),
    execution=foundry.Execution(model="orders", participation=0.1,
                                order_capacity=32, command_capacity=32),
) as session:
    print(session.plan_program(reference))
    program = catalog.resolve(reference)
    report = session.evaluate(program)
    for row in report.rows:
        print(row.id, row.metrics, row.decision)
    session.commit(report)
```

`evaluate(program)` 创建有界报告租约，成功 `commit(report)` 消费它。如果只观察而不提交，调用 `session.release(report)`；失败/冲突提交保留租约。外层会话退出时释放剩余租约。

反复查询可使用显式作用域：

```python
registered = session.register(reference)
try:
    report = session.evaluate(registered)
    saved = session.retain(registered)
    session.commit(report)
finally:
    session.release(registered)
```

这段应放在上面的会话块内。注册复用不可变模型实例，每次查询仍创建独立私有运行。显式注册在提交后继续归调用方持有。retain 返回持久引用，不序列化 Python 模型，也不保存之前的私有运行状态；重新打开含程序成员的组合时，需要相同目录与 loader 能力。

本地模型需要收益张量时，可将 `session.evaluate(program)` 换成 `result = session.evaluate_device(program)`。通过 `result.batch.rows[i].returns` 读取驻留收益，通过 `result.report.rows[i]` 读取指标/准入，通过 `.metrics.fills` 查看实际成交事件数。直接调用 `session.commit(result)` 或 `session.release(result)`；一次查询同时生成两种视图，封装保留报告租约。显式注册仍按上文由调用方释放，见[驻留结果](research.zh-CN.md#一次查询取得驻留结果)。

本地实例可以在没有部署目录时注册/评价；retain 或被接纳的 commit 必须能通过匹配工件与规范参数重建，仅有描述名称不足以支持持久引用。

## 因果运行契约

`ProgramDescriptor` 声明不可变工件/参数、私有 `state_schema`、请求字段、有界历史、原始变体数、支持设备、确定性 seed 和拟合信息截止点。`fit_at` 不晚于首个可执行研究决策；预训练模型的信息使用也须如实声明。

`open(context)` 创建新的 `ProgramRun`，不得修改共享模型参数或让上次查询的私有状态进入本次运行。`context` 包含：

| 字段 | 契约 |
| --- | --- |
| `batch_size` | 本次有效账户/试算行数 |
| `variant_indices` | 驻留 int64 `(B,)`，指向原始描述批次中的变体 |
| `initial_cash` | 驻留 `(B,)` 实际初始资金 |
| `intent_coefficients` | 驻留 `(B,)`，重放将应用于输出数量的系数 |
| `owner` | 该成员被分配的订单命名空间 |
| `xp`、`stream` | 数组模块与 CUDA 流指针；CPU 时无流 |
| `assets`、`plan`、`descriptor` | 不可变坐标与声明能力 |

始终按 `context.batch_size` 分配状态。描述可包含多个原始变体，本次查询可能只运行一个、若干变体或多个权重试算。通过 `variant_indices` 选取对应的不可变参数。

`on_event(event)` 在每个可执行决策调用，接收独立拥有的 `(H_visible,S)` 特征/资格窗口、int64 因果历史时间、实际账户，以及自身有符号 `owner_quantity` `(B,S)`。窗口的数组基底与元数据都不暴露未来观测。账户快照含总持仓、权益、现金、预留资金、保证金和活动订单。`available_cash` 扣除预留与未结算现金（全仓时反映可用抵押资金）；`sellable_quantity` 扣除现金市场未结算库存；活动卖单预留在接收或成交命令时另行校验。

返回选定设备上的 `(B,K)` `OrderCommands`。action/owner/id/asset/side/type/expiry 使用有符号整数，数量/价格使用浮点，`reduce_only` 使用 bool；K 不超过命令容量。action 为零表示填充空槽。

命令最早在下一编译好的基础开盘生效。正的新订单 ID 在各自 owner 命名空间内严格递增，`(owner,id)` 标识订单。amend 替换剩余数量/条件并重置优先级。程序不能撤销内部负数目标子单 ID。`valid_until=-1` 表示撤销前有效，否则须为实际基础开盘时间戳并包含该端点；不支持任意 Bar 间过期时间。

`on_feedback` 在每根基础 Bar 完成后调用并返回 `None`，可以流式保存审计，但不能发单。反馈包含实际成交、部分成交、撤单、拒绝和强平；共享账户中可能含其他成员活动，读取时按 owner 区分。普通成交数量无符号，成员强平数量为有符号贡献。全局强平行 owner 为 `-1`，带总费用；成员行按绝对贡献分配同一费用，不能把两种费用重复相加。

运行从首个可执行研究决策开始，失败路径也只关闭一次。框架保留有界命令、订单、当前反馈和统计收益/权益，不累计完整成交历史。CUDA 回调要用 `context.stream` 为 PyTorch 等框架建立依赖；设备能力必须显式声明，不会把任意 Python 回调自动变成 CUDA 核。

费用由成交的 maker/taker 费率与方向性市场费用相加。已经挂单等待的限价单按 maker，立即可成交的限价单、市价单和止损单按 taker；负 maker 费率表示返佣。美股现金卖出所得与 A 股新买库存遵循编译后的交易日结算规则；拆股和到期支付股息在执行开盘前生效。公司行动影响的订单会撤销，日订单在交易日边界过期。这些是 Bar 级规则，不重建交易所队列。

### 读取执行反馈

反馈是类型化驻留列，不是名为 `fills` 的 Python 列表。B 为有效运行批量，F 为有界反馈容量，S 为资产数，Q 为活动订单容量。第 b 行仅前 `count[b]` 个槽有效，各列的行/槽顺序一致。

| 字段 | 形状/类型 | 含义 |
| --- | --- | --- |
| `count` | `(B,)` 整数 | 每账户有效反馈槽数 |
| `status`、`reason`、`stage` | `(B,F)` 整数 | 对应 `OrderStatus`、`OrderReason`、`OrderStage` |
| `sequence` | `(B,F)` 整数 | 账户内稳定递增的审计序号 |
| `owner`、`id`、`asset` | `(B,F)` 整数 | 成员命名空间、订单 ID 与资产索引 |
| `liquidity` | `(B,F)` 整数 | `LiquidityRole.NONE=0`、`MAKER=1`、`TAKER=2`；非成交审计行使用 NONE |
| `time` | `(B,F)` int64 UTC 纳秒 | 模型事件时间，不是精确 tick 时间 |
| `quantity`、`price`、`fee`、`remaining` | `(B,F)` 浮点 | 成交数量、报价币/资产价格、报价币费用与报告的订单剩余数量 |
| `event_ns`、`available_ns` | 主机整数 UTC 纳秒 | 基础开盘与完成 Bar 的可用/收盘时间 |
| `account.cash`、`.equity`、`.reserved_cash` | `(B,)` 浮点 | 实际收盘状态现金、总权益与预留 |
| `account.available_cash` | `(B,)` 浮点 | 满足结算与预留约束后的可用资金 |
| `account.sellable_quantity` | `(B,S)` 浮点 | 扣除未结算数量后的现金市场库存，尚未扣除活动卖单预留 |
| `account.quantity`、`.margin`、`.entry_price` | `(B,S)` 浮点 | 净数量、逐仓抵押金额或全仓初始保证金要求，以及入场价 |
| `account.orders` | 各列 `(B,Q)` | 独立活动订单快照，action 为零表示空槽 |

下面在当前 Bar 内计算本成员指定资产的实际成交量，不把完整轨迹传回主机：

```python
def on_feedback(self, feedback: foundry.ExecutionFeedback) -> None:
    xp, owner = self.context.xp, self.context.owner
    valid = xp.arange(feedback.status.shape[1])[None, :] < feedback.count[:, None]
    own = (feedback.owner == owner) & (feedback.asset == self.asset)
    filled = ((feedback.status == int(foundry.OrderStatus.FILLED))
              | (feedback.status == int(foundry.OrderStatus.PARTIAL)))
    filled_quantity = xp.where(valid & own & filled, feedback.quantity, 0.).sum(axis=1)
    # Consume here, or copy into explicitly declared checkpoint state.
```

`filled_quantity` 为 `(B,)`，单位是该资产数量，统计实际成交而非已受理订单。若要保留到之后决策，它就成为私有可变状态，须加入可恢复程序的 schema/snapshot。拒绝/撤单行不属于成交，强平仍遵循前面的全局/成员约定。

## 仓位与成员持仓

两种资金语义不同：

- **共享：**共用现金、总风险和一个账户，系数仅缩放一次成员目标与订单数量。固定 `quantity=0.001` 表示单位系数下的未缩放意图，不要自行再乘 `intent_coefficients`。
- **独立：**每个成员一次性收到按系数分配的初始现金，未分配部分保持现金。运行看到的是实际已分配账户，意图系数为一。小分配可能支付不了固定绝对数量；需要权益自适应时按实际资金/权益计算。

共享账户的实际成员库存已经被缩放。完全关闭时先还原未缩放命令数量：

```python
actual = event.owner_quantity[:, asset]
unscaled_close_quantity = abs(actual) / context.intent_coefficients
```

仅正系数行会打开运行，分母为正。正库存发卖出，负库存发买入，再把 `(B,1)` 数量写入命令。直接返回实际数量会再次乘系数，从而只关闭部分仓位。

各成员贡献之和等于账户净数量。现金市场卖出只能消费本成员已结算且未预留的库存。永续账户的 `reduce_only` 必须同时减少本成员贡献和账户净敞口；关闭对冲成员可能扩大净敞口，不能自动当作 reduce-only。资金费、预留、费用和总账户强平仍按实际账户处理，成员归属不会创造独立共享现金余额。

## 发现、家族与重放

`registered.candidates` 可直接进入 `Discovery`、`ExternalProposer` 和 `CandidateGrid`，保留原始变体身份。程序、公式与数组共用硬准入。订单模型下，权重变化或前面的候选入选后，必须创建新状态并重放实际执行，不把先前持仓路径当作账户不变时的静态结果复用。

紧接着调用 `commit(report)` 时，如果研究上下文没有变化，会话可以复用自己最近生成的评价与准入报告，不信任调用方传入的指标或结论。最多保留一个候选批次的主机端报告，不保存收益数组、持仓或活动账户；一旦某个候选入选，后续候选重新对照更新后的组合判定。刷新、注册变更、释放和关闭会清除复用结果，`measurement_cache_mib=0` 可关闭复用。本地和 RPC 使用相同机制，无需增加调用步骤。

描述与规范参数确定可重放程序；候选来源还绑定研究和原始变体。临时注册带租约 scope，持久引用不带。共享模型只在活动注册范围内复用，私有状态与账户反馈不会变成历史模型缓存。

## 远程部署

部署由计算应用负责。服务进程导入受信任 loader 模块，并在启动时提供目录：

```python
import foundry
from my_program import catalog

foundry.serve(
    "var/runs/program-portfolio.json", programs=catalog,
    market="crypto_spot", data_dir="/datasets/crypto_5m", backend="cuda",
    start="2025-01-08", end="2025-01-31",
    clock=foundry.Clock(decision="5min", statistics="1D"),
    account=foundry.Account(initial_cash=1000.),
    execution=foundry.Execution(model="orders"),
)
```

客户端只需部署摘要与规范参数，无需服务端 loader 代码或工件路径：

```python
import foundry

reference = foundry.ProgramReference(deployed_digest, '{"quantity":0.001}')
with foundry.connect(("127.0.0.1", forwarded_port)) as session:
    print(session.plan_program(reference))
    report = session.evaluate(reference)
    session.commit(report)
```

`deployed_digest` 使用部署时提供的 `artifact.digest`，`forwarded_port` 使用服务或隧道端口。研究注册/评价不安装依赖、不修改服务源码，也不通过 RPC 传输 pickle 或源码字符串。缺少部署、描述变化、参数不匹配都明确失败。服务重启后仍须提供相同目录，才能重放持久组合。

## 独立审计与恢复状态

`session.simulate(program_or_reference)` 创建空仓账户和新的私有状态，返回独立拥有的驻留 `ProgramSimulation` 数组与最终实际账户。它用于执行审计，不是准入报告，不能直接 `commit`。

挖掘/发现 checkpoint 保存持久候选及提案器状态，然后重新重放账户；它与暂停的私有程序/账户 checkpoint 不同。

示例显式声明 `checkpointable=True` 并实现 `CheckpointableRun.snapshot()`/`restore(state)`。其全部可变私有状态只有具名 bool 张量 `sent`。快照是严格按 `state_schema` 顺序返回的 `StateTensor` 元组，每项形状为 `(descriptor.batch_size, *declared_shape)`，dtype 必须匹配。精确续跑所需的全部可变模型/RNG 状态都应声明；不会自动内省或 pickle 隐藏 Python 状态。未声明恢复能力的程序仍可执行新的完整模拟。

在同一配置会话内，选择准确的 UTC 纳秒基础 Bar 收盘边界暂停，再导出独立主机 checkpoint：

```python
from datetime import datetime, timezone

pause_ns = int(datetime(2025, 1, 15, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
paused = session.simulate(reference, until=pause_ns)
foundry.export(paused.checkpoint, "var/runs/buy-once.npz")
checkpoint = foundry.load_program_checkpoint("var/runs/buy-once.npz")
resumed = session.simulate(reference, checkpoint=checkpoint)
print(resumed.returns.shape, resumed.final_account.equity)
```

这个边界适用于前面的 5 分钟数据示例；自己的数据应选择覆盖范围内的实际基础收盘。`until` 不含之后的 Bar，暂停结果只返回已完成统计列。如果在日统计周期中间暂停，未完成当日的状态留在 checkpoint，不报告半天的日收益。恢复结果包含此前已完成的前缀，不能再与暂停收益拼接；恢复时指定另一个 `until`，可运行到更晚边界并获得新的 checkpoint。

checkpoint 绑定完整 `ResearchContext`、描述、计划及下一事件游标，包含账户余额、成员/订单状态、预留、ID 高水位、统计状态和声明的私有张量。重新打开时需要相同数据、研究区间、未变化的组合版本、程序工件/参数、设备类别/索引和精度计划。组合变化会产生 `ContextConflict`，刷新不能让旧 checkpoint 自动变新。程序、状态或格式不匹配也会明确拒绝。

`foundry.export(checkpoint, path)` 原子写入版本化 NumPy 包；`load_program_checkpoint` 不打开计算即可校验元数据、数组形状/dtype 和 SHA-256。编码及解码载荷都限制 1 GiB，禁用对象数组和 pickle。只保存有界账户/私有状态与完成的统计值，不含行情面板、源码或完整事件历史。当前格式为 `program-checkpoint-v2`。

这是进程内独立审计接口。远程 `Session.simulate` 尚不支持，包括已部署引用；远程可使用规划、注册与候选评价。此接口不会暂停共享 RPC worker，不保存组合权重试算，也不在 Bar 内指令/成交位置恢复。边界位于完整 Bar 反馈之后，恢复从下一事件开始，不重复投递旧反馈。每次部分执行/恢复仍创建并关闭自己的运行资源。
