# GFN 挖掘端到端测速

NVIDIA L4, 23034 MiB, 580.82.07

现有 GFN 采样和训练在 C++ CPU 上执行，评价和账户计算使用 CUDA。未运行 CPU 回测对照。

| 阶段 | 累计秒数 |
| --- | ---: |
| sampling_seconds | 0.160146 |
| evaluation_seconds | 2.942735 |
| training_seconds | 1.107435 |
| commit_seconds | 0.150943 |
| reweight_seconds | 0.225563 |

Mining wall time: 5.032071 s; open through close: 6.186181 s.
Open/data/device setup: 1.025217 s; policy initialization: 0.004243 s.

训练与提交/重配重叠，阶段时间不能直接相加当作端到端耗时。首次运行包括遇到的 JIT；准备与预热条件由运行计划记录。

Epochs: 8; ordinary proposals: 4096; including probes: 4288; accepted: 8.
GPU sampled peak: 5010.0 MiB. Policy changed: True.

这是同一历史研究窗口中的在线策略学习与准入，不是监督模型样本外 walk-forward 收益证明。此入口执行实际 mine 流程；独立 judge_family 稳健性审查不自动计入。

[Summary](summary.json) · Plan（本地工件） · Epoch reports（本地工件）
