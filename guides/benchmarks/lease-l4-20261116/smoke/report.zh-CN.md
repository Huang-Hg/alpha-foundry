# L4 非公式 GPU 发现流程

同一公开接口，分别运行因果 MLP 与 GPU 张量因子程序。

| Case | Candidates | Accepted | Mine wall (s) | Open through close (s) |
| --- | ---: | ---: | ---: | ---: |
| mlp | 8 | 0 | 9.0730 | 11.4576 |
| factor | 8 | 0 | 0.4699 | 2.2741 |

报告包括驻留注册、GPU 摘要、GPU 评价、驻留奖励、学习、有序提交、重配及选择性落盘。注册仅传回摘要；检查点引用和入选候选的落盘计入完整用时。MLP 预测拟合严格使用历史前缀；超参/因子选择使用同一研究窗口的反馈，因此这些收益不是独立样本外证据。

[Detailed stages and checks](summary.json) · [Run plan](plan.json)
