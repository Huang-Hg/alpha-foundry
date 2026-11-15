# Foundry

**面向研究者、AI Agent 和挖掘程序的 CUDA 加速策略发现与回测基础设施。**

[English](README.md) · [使用指南](guides/research.zh-CN.md) · [数据](data/README.md)

输入一个公式、一批模型信号或目标持仓，获得可比较的回测结果和明确的组合准入结论。数据上下文、市场规则、批量计算与组合状态都由同一个 Python 接口管理。

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

示例使用随仓库提供的美股快照。换成自己的数据后，调用方式保持一致。

## Foundry 的优势

- **贯穿研究流程的 CUDA 批处理。** 候选批量评价、账户模拟和组合权重比较可以复用设备驻留数组。批量大小和精度均可调整，也支持 CPU 后端。
- **在需要的计算中保留 FP32 残差精度。** 双 FP32 算术用于账户和统计计算；数值方案在会话启动时确定，不根据结果触发精度复算。
- **不限于公式挖掘。** 可以使用 66 个算子的公式词汇，注册 Python/CUDA 或 ML 模型输出，也可以把外部候选生成器接入发现流程。Agent 闭环和人工想法使用同样的候选与报告。
- **面向时点可得信息的研究。** 历史资格、观测和拟合截止点声明、次根行情执行与因果 walk-forward 折叠共同约束未来数据泄露；数据来源记录明确披露缺失的公告历史。
- **从想法验证到组合接纳。** 诊断组合缺口、检验候选家族、比较权重方案，再通过统一准入规则提交改进。查询与组合变更是明确分开的操作。

## 四类市场，同一套接口

| 市场 | 执行模型 |
| --- | --- |
| 美股 | 历史成分资格、复权价格、下一交易时段执行及成本 |
| 中国 A 股 | 历史资格、停牌、涨跌停、卖出税费，以及分离的信号与执行价格 |
| 加密货币现货 | UTC 日线、365 日年化、交易资格与手续费 |
| USDT 线性永续 | 多空敞口、逐仓保证金、小时资金费事件与强平 |

永续采用日线决策、小时账户事件递推。价格、资金费与强平的建模口径见[使用指南](guides/research.zh-CN.md#加密货币执行)。

## 接入自己的数据

CSV、数据库或数据供应商导出都可以成为行情来源。仓库快照用于示例和验证，并不绑定特定供应商。

```python
with foundry.open(
    market="us_equity",
    data_dir="/datasets/my_us_equity",
    start="2025-07-01",
    end="2025-12-31",
    backend="cuda",
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

美股需要包含 OHLCV 和复权因子的 `panel.parquet`，以及记录历史资格区间的 `membership.parquet`。保留研究起点之前的历史，以便计算滚动特征。其他市场同样有明确的数据契约；永续还需要小时成交价、标记价和资金费记录。详见[文件格式与转换示例](guides/research.zh-CN.md#接入自己的数据)。

如果程序已经生成信号或持仓，直接使用 `session.register(...)`。它支持模型评分、多头目标权重和永续多空敞口，并要求声明日期、资产坐标与信息截止点。

## 在 Linux 或 Windows 上使用

本地计算支持 Linux，以及 Windows 下的 WSL2。Windows 原生 Python 可以直接作为远程客户端，无需编译数值扩展。

在 Linux/WSL2 中，准备 Python 3.12 或 3.13、对应开发头文件，以及支持 C++17/OpenMP 的 GCC，然后执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash build.sh
python -m foundry --help
```

CUDA 还需要 NVIDIA 驱动及匹配 CUDA 主版本的 CuPy。使用 `backend="cpu"` 可选择 CPU 计算。

连接已经运行的计算服务时，Windows 和 Linux 客户端使用同样的调用：

```python
with foundry.connect(("127.0.0.1", port)) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

`port` 是服务端口，或隧道的本地转发端口。详见[远程会话](guides/research.zh-CN.md#远程会话)。

## 少量接口，覆盖完整流程

| 任务 | 调用 |
| --- | --- |
| 评价公式或已注册模型输出 | `session.evaluate(inputs)` |
| 注册信号或持仓批次 | `session.register(values, ...)` |
| 诊断当前组合的不足 | `session.diagnose()` |
| 检验公式邻域或模型参数网格 | `session.judge_family(inputs)` |
| 查看组合或比较成员权重 | `session.portfolio()` / `session.trial_weights(matrix)` |
| 提交候选或已审阅的权重方案 | `session.commit(report)` / `session.commit_weights(trials, index)` |
| 优化并提交满足准入规则的组合权重 | `session.reweight(...)` |
| 为发现流程评价候选提案 | `foundry.Discovery(session).evaluate(candidates)` |
| 运行内置或外部提案器 | `foundry.mine(output, ...)` |
| 因果分折拟合并注册预测 | `foundry.walk_forward(session, fit, predict, ...)` |

`foundry.Candidate`、`ExternalProposer` 和 `CandidateGrid` 让自己的程序直接参与策略发现，无需先转换成公式。完整接入示例见[发现流程指南](guides/research.zh-CN.md#策略发现)。

评价不会修改组合。审阅结果后，可以单独提交：

```python
with foundry.open("var/runs/my_portfolio.json") as session:
    report = session.evaluate("gt(ma_dist_20,0)")
    decision = session.commit(report)
    foundry.export(decision, "var/runs/decision.json")
```

## 调用简单，参数仍可微调

市场成本、杠杆、CUDA 批量、准入和挖掘参数都通过调用参数传入，无需维护配置文件。

```python
with foundry.open(
    market="crypto_perpetual",
    leverage=3.0,
    cost_rate=0.0004,
    cuda=foundry.CUDA(candidate_batch=64, precision="paired"),
    admission=foundry.Admission(min_sharpe_gain=0.02),
) as session:
    report = session.evaluate("gt(ma_dist_20,0)")
```

`foundry.schema()` 为 Agent 提供参数和 RPC schema；`session.describe()` 返回当前研究的字段、算子与实际参数。比较其他设置时，打开新的会话即可。

## 性能依据

在 RTX PRO 6000 Blackwell Server 上，4096 条公式的公开 API 完整用例耗时为 **CUDA 2.879 秒、CPU 93.724 秒，加速 32.55 倍**。面板包含 500 只美股和 269 行日线，CPU 使用两个 OpenMP 线程。两端均接纳 28 个候选，实测流程没有决策不一致或数值误差超限。

| 公式数量 | CPU | CUDA | 加速倍数 |
| --- | --- | --- | --- |
| 1 | 0.0874 秒 | 0.0868 秒 | 1.01× |
| 64 | 1.443 秒 | 0.231 秒 | 6.25× |
| 4096 | 93.724 秒 | 2.879 秒 | 32.55× |

完整用例包含打开、评价、有序提交、组合检查和关闭。以上为源码 `fa2acfd` 预热后单次测量的墙钟时间，不是多次重复的统计估计。[2026-11-15 报告](guides/benchmarks/arch-v2-pro6000-20261115/report.zh-CN.md) 给出全部 66 个算子及误差。公开回归通过；独立私有生产基线因原有旧格式工件未通过，原文件未改动，详见[验收范围](guides/benchmarks/arch-v2-pro6000-20261115/README.zh-CN.md)。此前 `current/` 报告继续作为历史记录保留。

[研究与程序接入](guides/research.zh-CN.md) · [可用数据](data/README.md) · [基准报告](guides/benchmarks/README.md)
