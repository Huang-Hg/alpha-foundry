# Foundry

**把策略想法、模型信号和目标持仓，变成可复现、可比较的组合决策。**

**简体中文** · [English](README.md) · [研究与接口指南](guides/research.zh-CN.md)

Foundry 是面向研究者、AI agent 和自动挖掘程序的 Linux 回测基础设施。你可以写公式，也可以接入 ML 模型、自定义 Python/CUDA 程序或外部挖掘器；它们共用市场规则、评价指标、权重试探和组合准入流程。

## 为什么用 Foundry？

- **策略表达不受公式限制。** 直接提交批量信号、目标权重或永续目标敞口。公式挖掘只是其中一种接入方式。
- **人、agent 和挖掘器用同一套决策。** Python 批量接口与可发现的 JSON-RPC 接口共用研究用例，返回结构化指标和拒绝原因，方便程序继续探索。
- **为批量研究使用 CPU 与 CUDA。** 复用研究数据，让数值计算保留设备数组。精度策略显式选择，并与 CPU 参考实现比较误差、交易和准入结果。
- **把时间与交易规则算进去。** 快照绑定数据、执行规格和数值策略；特征只能使用决策时已知信息，日线收盘信号最早次日开盘执行。
- **评价与接纳分开。** 查询不会偷偷改组合；提交检查组合版本，一次发布完整结果，便于并行提案和结果追踪。

## 支持哪些市场？

| 市场 | 已建模的行为 |
| --- | --- |
| 美股 | 日线信号、持有资格、交易与持有成本 |
| A 股 | 历史资格、停牌、涨跌停、次日执行、卖出税费 |
| 加密货币现货 | UTC 日线、连续交易日历、365 日年化、交易费用 |
| USDT 线性永续 | 多空敞口、逐仓抵押、真实资金费、杠杆、小时标记价强平 |

永续目前采用固定费用/维持保证金参数与小时不利价格触达模型，未重建历史风险档位或交易所逐笔执行。数据缺少历史发布时间、修订版本等限制，会通过数据契约公开。具体规则见[研究与接口指南](guides/research.zh-CN.md)。

## 快速开始

支持 Linux、WSL2 和 Colab。准备 Python 3.12/3.13、Python 开发头文件，以及支持 C++17/OpenMP 的 GCC：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash build.sh
python -m foundry --help
```

在 [config/config.ini](config/config.ini) 中选择市场、数据目录、研究日期与后端。仓库不附带行情和私有组合；CPU 无需 GPU，CUDA 后端需另外安装与环境匹配的 CuPy。

想先用公开永续数据体验，可配置 `[crypto_perpetual]` 后导入：

```bash
python -m foundry import-perpetual data/crypto_perpetual
```

导入器下载 Binance 官方归档、核验校验和、保留来源并发布不可变数据清单。随后设置 `[market] name = crypto_perpetual`，使用该数据开始研究。

## 接入自己的程序或 agent

```python
from foundry.config import load_config
from foundry.runtime import configured_study, open_runtime
from foundry.research import Research
from foundry.research_types import Candidate
from foundry.candidates import Formula
from foundry.adapters.arrays import MemoryArrayStore
from foundry.adapters.library import MemoryLibraryStore
from foundry.portfolio.library import Library

settings = load_config()
study = configured_study(settings)
with open_runtime(settings, study) as runtime:
    research = Research(runtime.engine, MemoryLibraryStore(Library((), study)),
                        MemoryArrayStore())
    candidates = (Candidate("trend", Formula("gt(ma_dist_20,0)")),)
    result = research.evaluate(candidates, research.context.id)
```

接入模型输出时，用 `Research.register_batch` 注册 `ArrayBatch`，之后继续调用同样的 `evaluate` 与 `commit`。输入带有日期/资产坐标、快照身份和观测/拟合截止点。详见[非公式候选与 RPC 接口](guides/research.zh-CN.md#候选接口)。

AI agent 可通过 `python -m foundry schema output/schema.json` 导出机器协议，通过 CLI 帮助发现命令。

## 工程原则

Python 负责研究与接入，C++/CUDA 负责数值核。模块化单体将市场语义与传输、存储、设备选择分开。测试保护 PIT、交易规则、精度与决策一致性，以及组合提交的原子性；性能报告区分准备开销、驻留计算和完整用例。

Foundry 提供研究工具。回测通过或候选获准接纳，不代表已经证明样本外盈利。
