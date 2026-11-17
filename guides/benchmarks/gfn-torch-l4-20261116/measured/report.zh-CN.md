# GFN 挖掘端到端测速

NVIDIA L4, 23034 MiB, 580.82.07

GFN 网络前向、损失、反向和 SGD 更新在 PyTorch CUDA 上执行；CPU 负责语法和公式文本。评价和账户计算使用 CUDA。

| 阶段 | 累计秒数 |
| --- | ---: |
| sampling_seconds | 1.587836 |
| evaluation_seconds | 3.030296 |
| training_seconds | 2.205516 |
| commit_seconds | 0.232981 |
| reweight_seconds | 0.246478 |

Mining wall time: 7.888872 s; open through close: 8.921073 s.
Open/data/device setup: 0.912927 s; policy initialization: 0.009034 s.

训练与提交/重配重叠，阶段时间不能直接相加当作端到端耗时。首次运行包括遇到的 JIT；准备与预热条件由运行计划记录。

Epochs: 8; ordinary proposals: 4096; including probes: 4288; accepted: 8.
GPU sampled peak: 4340.0 MiB. Policy changed: True.

这是同一历史研究窗口中的在线策略学习与准入，不是监督模型样本外 walk-forward 收益证明。此入口执行实际 mine 流程；独立 judge_family 稳健性审查不自动计入。

[Summary](summary.json) · Plan（本地工件） · Epoch reports（本地工件）
