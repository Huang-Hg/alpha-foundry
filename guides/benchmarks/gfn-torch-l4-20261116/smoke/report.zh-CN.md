# GFN 挖掘端到端测速

NVIDIA L4, 23034 MiB, 580.82.07

GFN 网络前向、损失、反向和 SGD 更新在 PyTorch CUDA 上执行；CPU 负责语法和公式文本。评价和账户计算使用 CUDA。

| 阶段 | 累计秒数 |
| --- | ---: |
| sampling_seconds | 0.374587 |
| evaluation_seconds | 8.236063 |
| training_seconds | 6.388546 |
| commit_seconds | 1.055205 |
| reweight_seconds | 0.728717 |

Mining wall time: 17.999072 s; open through close: 33.477339 s.
Open/data/device setup: 15.403259 s; policy initialization: 0.055046 s.

训练与提交/重配重叠，阶段时间不能直接相加当作端到端耗时。首次运行包括遇到的 JIT；准备与预热条件由运行计划记录。

Epochs: 1; ordinary proposals: 32; including probes: 56; accepted: 1.
GPU sampled peak: 964.0 MiB. Policy changed: True.

这是同一历史研究窗口中的在线策略学习与准入，不是监督模型样本外 walk-forward 收益证明。此入口执行实际 mine 流程；独立 judge_family 稳健性审查不自动计入。

[Summary](summary.json) · [Plan](plan.json) · [Epoch reports](../raw.tar.gz)
