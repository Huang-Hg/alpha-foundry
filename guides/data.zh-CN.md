# 接入自己的市场数据

[项目介绍](../README.zh-CN.md) · [研究指南](research.zh-CN.md) · [English](data.md)

Foundry 读取由 `market`、`data_dir`、`start`、`end` 选择的本地工件。供应商、CSV 导出或数据库都可以作为来源，仓库快照只是可选示例。特征与执行规则在请求截止点内派生，因此需要保留真实可用时间、历史资格及更早的滚动历史。

主要有两种格式：日线股票/现货表，以及显式 crypto 基础 Bar。有状态 crypto 程序通常使用基础 Bar，以明确区分执行、决策和统计频率。

## 日线股票与现货

`panel.parquet` 使用长表，通常每个日期/资产一行；A 股可按明确发布时间保留多个版本。

| 字段/文件 | 契约 |
| --- | --- |
| `date` | 不带时区的午夜交易所会话日期；现货为 UTC 自然日 |
| `symbol` | 字符串资产标识，保留前导零 |
| `open`、`high`、`low`、`close`、`volume` | 数值 OHLCV，缺失观测保持缺失 |
| 美股 `adj_factor` | OHLC 复权因子，不改变成交量 |
| A 股 `published` | 该版本可用的交易会话日期 |
| A 股 `exec_open`、`exec_close` | 历史复权执行价；原始 OHLCV 用于特征 |
| 现货 `eligible` | 收盘时历史资格，bool |
| 股票 `membership.parquet` | `symbol`、`start`、`end`；起点包含，终点不含，空 end 表示未结束 |
| A 股/现货 `manifest.json` | 下文的来源契约 |

美股可以提供原始 OHLC 和复权因子，或已复权 OHLC 配合全一因子；不隐式复权成交量。历史资格独立于报价：资格区间外仍保留报价，以便给已有持仓估值或退出。资格区间应表达当时已知的信息，不能从当前成分反推。

A 股读取器选择 `published <= date` 的最新版本。迟到修订不能替换早期决策 Bar；这是会话日期级的发布处理，不代表具备日内公告时间。不可用价格保持缺失，公告/修订历史的缺口应明确披露。

假设 CSV 已有上表字段，美股转换示例为：

```python
from pathlib import Path
import pandas as pd

folder = Path("/datasets/my_us_equity")
folder.mkdir(parents=True, exist_ok=True)
panel = pd.read_csv("ohlcv.csv", parse_dates=["date"], dtype={"symbol": str})
members = pd.read_csv("membership.csv", parse_dates=["start", "end"], dtype={"symbol": str})
panel.to_parquet(folder / "panel.parquet", index=False)
members.to_parquet(folder / "membership.parquet", index=False)
```

```python
import foundry
with foundry.open(
    market="us_equity", data_dir="/datasets/my_us_equity",
    start="2025-07-01", end="2025-12-31", backend="cuda",
) as session:
    print(session.grid)
    report = session.evaluate("gt(ma_dist_20,0)")
```

面板保留研究开始前的历史，供滚动窗口使用。`start` 选择研究区间，不是要求删掉更早观测。股票日线收盘使用交易所日历，包括 DST 和提前收盘，最早在下一会话开盘执行。纯日期 end 包含该会话/自然日。

### 来源元数据

A 股和日线现货 manifest 包含以下七个字符串字段：

```json
{
  "schema": "my-daily-v1",
  "source": "Provider, export version and acquisition method",
  "retrieved_at": "Actual acquisition timestamp",
  "features": "Raw/adjusted feature-price convention",
  "execution_prices": "Adjustment and execution-price convention",
  "publication": "Publication timestamps and missing vintage limitations",
  "membership": "Historical eligibility source and availability limitations"
}
```

填写数据真实含义。当前美股读取器使用固定的历史来源契约，在表旁添加 manifest 不会覆盖它；请另外保留自己的来源记录。支持其他美股来源契约需要扩展读取器。缺失元数据或外部给出的 cutoff 不能证明完整供应商 PIT。

额外 Parquet 列不会自动成为公式字段，应查看 `session.describe()` 的字段列表。无需扩展读取器的任意特征/模型逻辑，可在自己的应用内生成因果评分或目标，再按 `session.grid` 注册。

## 显式 crypto 基础 Bar

现货和 USDT 逐仓永续都接受 `schema="crypto-bars-v1"`。需要 `bars.parquet` 与 `manifest.json`，永续还需完整 `funding.parquet`。

| 文件 | 必需字段 |
| --- | --- |
| `bars.parquet` | 字符串 `symbol`；int64 UTC 纳秒 `open_ns`、`close_ns`、`available_ns`；浮点 `open`、`high`、`low`、`close`、`volume`；bool `eligible` |
| 永续 Bar 附加字段 | 浮点 `mark_open`、`mark_high`、`mark_low`、`mark_close` |
| `funding.parquet` | 字符串 `symbol`；int64 UTC 纳秒 `settlement_ns`；浮点 `rate`、`mark` |

每根基础 Bar 是左闭右开 `[open_ns,close_ns)`，时间对齐声明的固定 `interval`，且 `close_ns = open_ns + interval`。`available_ns >= close_ns` 表示完整行及资格何时可见。同一 `(symbol,open_ns)` 允许多个发布版本，但 `available_ns` 必须不同。

价格/标记价为每资产单位的报价币金额，成交量和订单数量使用资产单位。资金费 `rate` 是每次结算的名义金额比例。同一账户的资产使用一致报价币，模型不提供跨币种抵押换算。

OHLC 必须全部有限为正，且高低边界一致；缺失现货报价时四项全部缺失。成交量非负，缺失位置与报价一致。永续执行要求完整有限的成交价和标记价 Bar。现货允许缺失报价，缺失时不能成交；订单模型估值沿用最后有效收盘价，首个有效上市报价前为零。

读取器先截断源数据，再派生特征。每个完成的决策 Bar 仅使用截至该决策可见的版本，迟到基础行使该特征 Bar 不可用，而不会用未来数据回填。执行使用截至研究 cutoff 可见的首个发布事实版本。标记价和资金费仅用于执行，不进入特征面板。

### 转换含发布时间的 CSV

假设 `bars.csv` 包含实际 UTC 时间 `open_time`、`close_time`、`available_time`，以及上面的 OHLCV/资格字段。下面的现货示例生成完整 manifest 和真实文件校验和：

```python
from pathlib import Path
import hashlib
import json
import pandas as pd

folder = Path("/datasets/crypto_5m")
folder.mkdir(parents=True, exist_ok=True)
bars = pd.read_csv("bars.csv", dtype={"symbol": str, "eligible": bool})
for source, target in (("open_time", "open_ns"), ("close_time", "close_ns"),
                       ("available_time", "available_ns")):
    stamps = pd.to_datetime(bars.pop(source), utc=True)
    bars[target] = stamps.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")
for name in ("open", "high", "low", "close", "volume"):
    bars[name] = bars[name].astype("float64")
bars.to_parquet(folder / "bars.parquet", index=False)


def iso_ns(value):
    return pd.Timestamp(int(value), tz="UTC").isoformat().replace("+00:00", "Z")


payload = (folder / "bars.parquet").read_bytes()
manifest = {
    "schema": "crypto-bars-v1", "market": "crypto_spot", "timezone": "UTC",
    "interval": "5min", "history_start": iso_ns(bars.open_ns.min()),
    "end": iso_ns(bars.close_ns.max()), "symbols": sorted(bars.symbol.unique().tolist()),
    "source": "My archived feed and export version",
    "retrieved_at": "Replace with actual acquisition timestamp",
    "features": "Trade OHLCV, frozen from published vintages",
    "execution_prices": "First published trade OHLC",
    "publication": "Actual available_time; disclose missing historical vintages",
    "membership": "Historical eligible values published with each complete bar",
    "artifacts": {"bars": {"path": "bars.parquet", "sha256": hashlib.sha256(payload).hexdigest()}},
}
(folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

研究 `start`/`end` 要在存储范围内。源历史起点、研究起点与截止点必须覆盖完整统计周期；日统计需要完整 UTC 自然日。若源数据只有 8 小时，可选择 `1h` 等整除的统计周期，而不能声称有完整日统计。不要虚构发布时间：若只有当前修订价格，应披露这一缺口，不能将其包装为完整历史发布档案。

```python
import foundry
with foundry.open(
    market="crypto_spot", data_dir="/datasets/crypto_5m",
    start="2025-01-08", end="2025-01-31", backend="cuda",
    clock=foundry.Clock(decision="15min", statistics="1D"),
    execution=foundry.Execution(model="orders"),
) as session:
    print(session.describe()["execution_plan"])
    report = session.evaluate("gt(ma_dist_20,0)")
```

固定时长使用 `s`、`min`、`h`、`D`，决策与统计时长都必须是基础 Bar 的整数倍。manifest 范围使用显式 ISO UTC 时间，`end` 不包含；研究 end 若为时间戳也不包含，若仅为日期则包含当天 UTC 全日。研究端点需对齐统计周期，决策窗口按已完成的决策 Bar 计数。

### 添加永续标记价和资金费

使用 `market="crypto_perpetual"` 时：

1. 每个基础开盘/资产组合提供四个标记 OHLC 字段，与成交价网格一致。
2. `funding.parquet` 为每个基础开盘/资产组合提供一行。`settlement_ns` 等于基础开盘；`rate=0` 明确声明该事件没有资金费，`mark` 必须为有限正结算标记价。实际非零结算必须保留，不能把缺失当作零。
3. 在 `artifacts` 添加 `"funding": {"path": "funding.parquet", "sha256": "..."}`，使用实际文件摘要，并修改 manifest 的 market。
4. 以 `Account(margin="isolated", leverage=..., gross_limit=...)` 打开。leverage 控制抵押要求，gross_limit 是独立的敞口预算。

若结算发生在基础开盘之间，需提供更细 Bar，读取器不会移动结算时间。也不会从日线成交 OHLC 推断资金费、标记极值或强平路径。

### 保留的小时归档格式

仓库内永续数据采用另一套归档格式：`trade.parquet`、`mark.parquet`、`funding.parquet`、manifest 和周期变更证据。它在小时执行上提供日线决策/统计，并披露资金费代理口径。可从支持的公开归档获取这种格式：

```python
import foundry
foundry.import_perpetual(
    "/datasets/perpetual_archive", symbols="BTCUSDT ETHUSDT SOLUSDT",
    history_start="2025-01-01", start="2025-07-01", end="2025-12-31",
)
```

导入器保留来源 URL、校验和和必要周期变更证据，不生成 `crypto-bars-v1`。自己的更细频率导出应采用上面的显式格式。现有数据的资产顺序由 manifest 决定；导入参数 `symbols` 选择新采集范围。

## 数据、预测与 PIT

行情 Bar、决策特征、执行报价和模型预测职责不同。日线收盘决策不能在当天更早的开盘成交。资格控制新的可交易横截面，报价仍用于已有持仓估值。模型观测/拟合截止点需要覆盖标签与预处理信息。`foundry.walk_forward` 提供因果前缀，但调用方仍对独立读取的数据及预训练模型来源负责。

打开后，先检查 `session.grid`、`session.describe()["clock"]`、字段和来源，再注册预测。准确的 `(B,D,S)` 坐标与 `(B,D)` cutoff 见[数组注册](research.zh-CN.md#数组与模型输出)。数据契约记录已知与缺失的信息，不承诺样本外收益。
