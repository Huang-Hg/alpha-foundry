# L4 非公式 GPU 发现流程

同一公开接口，分别运行因果 MLP 与 GPU 张量因子程序。

| Case | Candidates | Accepted | Mine wall (s) | Open through close (s) |
| --- | ---: | ---: | ---: | ---: |
| mlp | 256 | 0 | 3.1989 | 5.6175 |
| factor | 256 | 0 | 0.8810 | 1.1842 |

报告包括输出注册、主机摘要/落盘、GPU 评价、驻留奖励、学习、有序提交和重配。MLP 预测拟合严格使用历史前缀；超参/因子选择使用同一研究窗口的反馈，因此这些收益不是独立样本外证据。

[Detailed stages and checks](summary.json) · [Run plan](plan.json)
