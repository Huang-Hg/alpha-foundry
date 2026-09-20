# Q25 组合构件挖掘

当前计算协议及准入规则见 [README.md](README.md)。子 agent 仅用于挖掘候选，使用 5.6-sol；不用于工程改造。

候选的目标是改善现有组合：提高近期加权 Sharpe，保持全期 Sharpe、CAGR 与近期增长。
单条低收益或空仓年份不构成否决。只通过运行器获得正式指标，不自行实现回测或放宽 gate。
写出候选、调用 `run.py`、依据组合增量调整；只有主控可以使用 `--accept`。

FastExpr 输入表达式，`--fields` / `--ops` 显示接口。
Python 源定义 `alpha*(data, liquid)`，输入是 CUDA FP64 面板，输出是 CuPy 敞口数组。
数值运算使用 CuPy 原语或 `alpha_cuda` 提供的 CUDA C++ 算子。
禁止 NumPy/Pandas 数值路径及 Python 时间/资产循环，禁止 CPU 回退。
引擎负责液态资产掩码、clip 到 [0,1]、等权额度、交易成本、指标和三次截断因果验证。

当前研究窗由 `config/config.ini [factor_mining]` 指定。
历史候选看过全期，历史窗口重评不能宣称独立样本外成绩。
不要更改引擎、准入参数或组合库；不要保存含 CPU 数值实现的临时源码副本。
