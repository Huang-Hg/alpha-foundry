# 原生回测框架对比方法

[English](framework-benchmark.md) · [基准指南](benchmarks/README.zh-CN.md)

本基准用同一套有限交易规则比较 Foundry、Spectre、vectorbt 和 Freqtrade，计入各自的指标计算、原生账户执行和结果输出。它不比较策略发现算法、Foundry 组合准入或实盘交易，也不代表这些框架支持相同的市场和交易功能。

本页说明方法与复现环境，不包含性能结论；实际测量以完成后的运行工件为准。

## 共同数据与策略

来源是保留的 `data/crypto_spot_daily/panel.parquet` 快照。用例使用 2024-07-01 至 2025-12-31 的 UTC 日线 OHLCV，交易从 2025-01-01 开始，之前的行用于指标预热。固定资产顺序为 BTC、ETH、SOL、BNB、XRP、ADA、DOGE、LTC，小规模使用其前缀；不根据策略收益筛选资产。缺失观测、非正价格或不一致的 OHLC 边界会直接使准备失败，不填补行情。

候选使用 SMA 窗口 `(5,10,20,30,40,60,90,120)` 和阈值 `(-.02,-.01,0,.01,.02,.03,.04,.05)`，按阈值优先、窗口其次排列。1、16、64 个候选用例取该网格的前缀。所有框架接收相同的 float64 报价数组，因子实际输出精度会记录，不预设所有框架相同。

每个候选/资产对拥有独立账户，初始现金为 1,000,000 计价货币单位；不共享现金，不做跨资产排名或组合优化。

1. 在行 `t` 收盘时，若 `close[t] > SMA(window)[t] * (1 + threshold)`、`t >= trade_start` 且 `t % 3 == 0`，产生入场信号。行号从用例起点算起，包含预热行。
2. 下单数量为 `0.25 * flat_cash[t] / close[t]`，在 `open[t+1]` 买入。这是按前收盘价确定的配置，不是用下一开盘价计算现金的 25%。
3. 在 `open[t+2]` 全部卖出。最后两行禁止入场，因此完成的用例无需期末强制平仓。
4. 数量连续，不使用杠杆、借款费用、滑点、最小订单或交易所数量步长。若开盘跳空使现金不足以支付既定买入及费用，现金余量检查会拒绝该用例。
5. 零费和单边 10 个基点分别运行，保留各框架原生费用行为。

固定的高流动性资产集合和供应商快照适合受控执行比较，但不提供无幸存者偏差的历史成分，也没有完整公告/修订历史。本实验不构成投资表现的证据。

## 各实现实际执行什么

| 实现 | 指标 | 账户执行与证据 |
| --- | --- | --- |
| Foundry CPU | 原生 `ts_mean` | 原生现金核；根据其净值及既定一根持仓目标重建成交 |
| Foundry CUDA | 原生 CUDA `ts_mean` | 原生双 FP32 现金核；重建成交明确标记为非独立证据 |
| Spectre CUDA | 原生 `FactorEngine` 批量计算唯一 SMA 窗口和前收盘价 | CPU 上的原生 `trading.run_backtest` / `SimulationBlotter`；实际交易、持仓、现金和每日账户历史 |
| vectorbt Numba / Rust | 原生 `MA.run` | 原生 `Portfolio.from_orders`、独立账户、原生订单记录与账户数组 |
| Freqtrade | 在原生策略分析中调用 pandas rolling | 原生 `Backtesting`、订单和钱包；从实际成交按收盘价标记持仓/净值，并核对原生期末余额 |

Spectre 每个条件计算一次因子，再以其原生 saved-factor 格式供独立账户复用。缓存创建、读取和清理均计入调用。自动一根延迟只应用一次；比较误差时将指标值映射回决策日期。GPU 因子计算不意味着其 Python 账户循环在 GPU 上运行。

Spectre 的原生 `order_target` 保留分数数量，但 blotter 将每笔费用四舍五入到两位小数。适配器保留该行为，不修改上游账户实现。这种费用差异会改变后续现金和下单量，因此与浮点误差分开报告；零费条件用于更严格的账户等价检查。

Freqtrade 使用显式离线现货场所元数据：连续数量/价格、无最小数量，不联网发现交易所。实际策略仍由它的原生订单、交易和钱包实现执行。此用例不模拟真实交易所数量步长、流动性或撮合引擎。

## 计时边界

每个框架、设备、规模和费率条件使用独立进程。先调用一次以完成导入、JIT 和运行时预热，再以全新账户状态及新建条件缓存测量一次。不进行多次测量后挑选最快值。

实测区间连续覆盖：重新读取共同用例、原生指标与账户执行、结果物化、共同指标计算，以及 `result.npz` 写出。适配器导入、进程启动、用例准备和冷启动预热单列或排除在该区间之外。NVML 采样器启停、已安装依赖的版本枚举、正确性核对及最终元数据 JSON 发布均在计时之外。无效输出会在成功元数据发布前使任务失败。

运行器为 OpenMP、BLAS、Numba 和 Rayon 请求两个线程，GPU 计算在结果使用前同步。结果包含依赖版本、Python 版本、设备、CPU 信息、用例哈希和各阶段时间。一次观测只能说明该次运行，小幅差异不构成统计排名，也不能保证在其他设备上相同。

NVML 在测量轮内每 20 毫秒采样该 worker 的 GPU 驻留内存。峰值包含进程上下文与原生分配器缓存，不是分配增量。采样可能漏掉短暂峰值，因此该值是瞬时显存峰值的下界。主机峰值 RSS 覆盖整个进程生命周期，包括导入和预热，而非仅实测调用。两项均不计入其他无关进程的内存。

## 正确性与报告解读

- 比较从 `trade_start` 到倒数第三行的决策日 SMA。起始预热和末尾不可用的延迟输出不参与该比较。
- 精确比较入场标记和成交身份：候选、资产、日期、买卖方向。未配对成交与数量、价格、费用误差分开报告。
- 比较除以初始现金后的收盘净值、持仓和期末现金；有原生账本时独立重建核对。Foundry 重建成交标记为 `native_ledger=False`，不能把它当作独立订单账本证据。
- 各端输出统一按 365 日年化，使用总体标准差（`ddof=0`）、CAGR 和正值形式的最大回撤，零波动 Sharpe 定义为零。这些是共同报告公式，不替换各端账户执行。
- 零费归一化净值的最大绝对误差门槛为 `1e-8`，单独通过该项不能豁免决策或成交不一致。含费场景只报告差异，不宣称数值等价，同时展示未取整费用与按分取整费用的残差。

报告同时保留绝对误差和尺度误差，后者定义为 `abs(actual-reference) / max(1, abs(reference))`。参照实现为 Foundry CPU，不是外部真值模拟器。通过这一有限用例，不等于验收全部算子、永续资金费/强平、A 股限制、PIT 来源或研究准入规则。跨框架加速比也不重新验收 Foundry 独立的同卡 1.5 倍精度开销预算。

## 在授权云端环境复现

使用具备兼容 NVIDIA 驱动和原生构建工具的 Linux 或 WSL2。当前 Spectre 要求 Python 3.14，因此使用独立环境。依赖文件固定了环境准备时验证的直接依赖版本；运行元数据保存实际安装的完整发行包版本。它们不是完整的传递依赖锁文件。

```bash
python -m pip install uv
python -m uv python install 3.13.16 3.14.10
python -m uv venv --python 3.13.16 /content/framework-main
python -m uv pip install --python /content/framework-main/bin/python \
  -r validation/benchmarks/competition/requirements-main.txt
python -m uv venv --python 3.14.10 /content/framework-spectre
python -m uv pip install --python /content/framework-spectre/bin/python \
  -r validation/benchmarks/competition/requirements-spectre.txt
source /content/framework-main/bin/activate
bash build.sh
```

[主环境](../validation/benchmarks/competition/requirements-main.txt) 固定 vectorbt/vectorbt-rust 1.2.1 和 Freqtrade 2026.10。[Spectre 环境](../validation/benchmarks/competition/requirements-spectre.txt) 固定上游提交 `c059665ba33b3d98f523ee55fdecf5105af146f6`。已验证的 Linux Torch 运行时报告 `2.15.0+cu132`，发行包元数据报告 `2.15.0`。PyPI 上名为 `spectre` 的包与该项目无关，必须使用固定 Git URL，不绕过上游依赖要求。

先在全新输出目录运行小规模 smoke：

```bash
python -m foundry benchmark var/runs/framework-smoke --suite frameworks \
  --sizes 1 --assets 1 --spectre-python /content/framework-spectre/bin/python
```

smoke 验收后再运行既定网格：

```bash
python -m foundry benchmark var/runs/framework-comparison --suite frameworks \
  --sizes 1 16 64 --assets 8 --spectre-python /content/framework-spectre/bin/python
```

不要传 `market`、`backend` 等研究参数：该套件拥有公开的共同交易契约，并比较已配置的实现。输出目录必须尚不存在；worker 失败时保留日志和未完成工件，部分完成不能当作完整比较。

Python 入口为 `foundry.benchmark(output, suite="frameworks", sizes=(1,16,64), assets=8, spectre_python=...)`。结果包含计划、共同用例、各实现日志和数组、版本/计时元数据、`comparison.json` 与双语报告。每份结果应绑定实测源码和依赖记录，不把历史测量混入新运行。
