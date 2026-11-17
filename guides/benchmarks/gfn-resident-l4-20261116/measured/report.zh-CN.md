# GFN 挖掘端到端测速

NVIDIA L4, 23034 MiB, 580.82.07

GFN 状态、合法动作、抽样、前向和训练均驻留 CUDA；CPU 只处理探针批次结果和最终公式文本。

| 阶段 | 累计秒数 |
| --- | ---: |
| sampling_seconds | 1.066037 |
| evaluation_seconds | 3.201830 |
| training_seconds | 0.807742 |
| commit_seconds | 0.190410 |
| reweight_seconds | 0.251801 |

Mining wall time: 6.168272 s; open through close: 8.585818 s.
Open/data/device setup: 0.926112 s; policy initialization: 1.354557 s.

训练与提交/重配重叠，阶段时间不能直接相加当作端到端耗时。首次运行包括遇到的 JIT；准备与预热条件由运行计划记录。

Epochs: 8; ordinary proposals: 4096; including probes: 4288; accepted: 8.
GPU sampled peak: 5774.0 MiB. Policy changed: True.

这是同一历史研究窗口中的在线策略学习与准入，不是监督模型样本外 walk-forward 收益证明。此入口执行实际 mine 流程；独立 judge_family 稳健性审查不自动计入。

[Summary](summary.json) · Plan（本地工件） · Epoch reports（本地工件）
