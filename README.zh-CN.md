# alpha-foundry —— 组合优先的公式化 alpha 研究

[![CI](https://github.com/Huang-Hg/alpha-foundry/actions/workflows/ci.yml/badge.svg)](https://github.com/Huang-Hg/alpha-foundry/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | **中文**

公式化 alpha 由人、语言模型或搜索策略写出,在 GPU 上批量评价,并且只按它给组合带来的变化来评判:每条候选都作为当前组合的"加项",
在一组权重上用真实的合成持仓并扣除交易成本回测。所有市场共用同一个执行内核;市场是插件,声明自己的数据字段,并把交易规则编译成数组。

## 架构

```
foundry/
  core/        Panel(字段即组件)与唯一一套收益统计
  ops/         算子表(registry.py)+ 后端:CPU C++ 参考实现、CUDA、可移植 numpy
  expr/        公式树、FastExpr 解析/校验、任意后端求值
  backtest/    ExecutionSpec + 执行内核(C++ 参考、CUDA 批量镜像)
  market/      crypto(Quantiacs 日线)、us_equity、a_share:字段、执行规则、可持有集合、日历
  engine.py    一个市场的研究窗口常驻 GPU:公式 → 持仓书 → 回测 → 统计
  portfolio/   组合库、参照组合、准入、假设族邻域评判
  search/      组合条件 GFlowNet(C++ 扩展)与挖矿循环
  service/     常驻评价服务(本机或经 SSH 的云端)及客户端
  monitor/     终端面板:类型化状态 schema、组件、布局;HTML 预览与终端同帧绘制
  __main__.py  python -m foundry <命令>
config/config.ini   唯一的可调参数源(每个 section 对应 foundry/config.py 里的一个 dataclass)
tests/              CPU 测试进 CI;数据与 GPU 测试在本机跑
```

依赖单向:`core ← ops ← expr ← backtest/market ← engine ← portfolio ← search/service ← CLI`;面板只读状态文件。

### 一个内核,多个市场

执行内核用昨日目标权重在今日开盘成交、按今日收盘盯市(Quantiacs `calc_relative_return` 递推)。市场之间的差别只在内核读取的数组:

| 规则 | 加密(Quantiacs) | 美股 | A 股 |
|---|---|---|---|
| 每股成本(买 / 卖) | ATR14 × 0.04 / 相同 | 费率 × 开盘价 / 相同 | (佣金 + 过户费) × 开盘价 / 再加印花税 |
| 方向冻结 | – | – | 涨停开盘不可买;跌停开盘不可卖;停牌双向冻结 |
| 融资成本(`carry`) | – | 空头借券费 | – |
| Sharpe | 算术,365 天 | 几何,252 | 几何,242 |

成本对称、无冻结无 carry 时,内核就是 Quantiacs 递推本身;CUDA 内核批量运行这种情形(至多 128 个资产),CPU 内核运行全部情形。

### Panel 组件

`Field` 声明名字、类目、公式能否读取,以及(派生字段)依赖哪些字段、如何用面板的算子后端(numpy 或 cupy)计算。
市场把自己的源字段与共享的派生字段(量价原语、趋势)组合成面板;新数据就是新的字段定义。

## 使用

```bash
pip install -r requirements.txt
bash build.sh                        # 原生扩展:ops/_ops、backtest/_bt、search/_gfn(Linux .so,WSL/msys2 .pyd)

python -m foundry fields             # 当前市场的公式词表
python -m foundry eval "gt(ma_dist_20, 0)" --file ideas.txt     # 对当前组合做准入
python -m foundry family hypotheses.txt                          # 按邻域评判每条公式
python -m foundry mine runs/gfn_01                               # 机器挖矿(组合目标)
python -m foundry monitor                                        # 面板
```

当前市场由 `config/config.ini` 的 `[market] name` 指定。市场数据(parquet 面板)不在本仓库中。

## 测试

```bash
python -m unittest tests.test_expr tests.test_native tests.test_policy tests.test_monitor   # CPU(CI)
python -m unittest tests.test_markets tests.test_portfolio tests.test_production tests.test_search   # 数据 + GPU
```

## 参考

- Malkin et al., *Trajectory balance: improved credit assignment in GFlowNets*(NeurIPS 2022)—— 策略的损失。
- Treynor & Black(1973)—— 单 alpha 目标的 appraisal ratio 前沿。
- Heidari & Nikolopoulos, **FACT**(arXiv:2604.26666)—— 子图识别与验证后组合,用于融合的逐元素算子与回测核内的组合融合。
- Quantiacs —— 内核复现的加密日线执行与计分口径。
