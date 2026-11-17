# GFN 挖掘端到端测速

NVIDIA L4, 23034 MiB, 580.82.07

GFN 状态、合法动作、抽样、前向和训练均驻留 CUDA；CPU 只处理探针批次结果和最终公式文本。

| 阶段 | 累计秒数 |
| --- | ---: |
| sampling_seconds | 0.738971 |
| evaluation_seconds | 8.611890 |
| training_seconds | 0.253366 |
| commit_seconds | 0.842006 |
| reweight_seconds | 0.607151 |

Mining wall time: 13.828669 s; open through close: 34.831464 s.
Open/data/device setup: 15.301601 s; policy initialization: 5.681035 s.

训练与提交/重配重叠，阶段时间不能直接相加当作端到端耗时。首次运行包括遇到的 JIT；准备与预热条件由运行计划记录。

Epochs: 1; ordinary proposals: 32; including probes: 56; accepted: 1.
GPU sampled peak: 946.0 MiB. Policy changed: True.

这是同一历史研究窗口中的在线策略学习与准入，不是监督模型样本外 walk-forward 收益证明。此入口执行实际 mine 流程；独立 judge_family 稳健性审查不自动计入。

[Summary](summary.json) · Plan（本地工件） · Epoch reports（本地工件）
