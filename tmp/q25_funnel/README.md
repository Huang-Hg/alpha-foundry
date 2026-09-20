# Q25 CUDA 组合挖掘

协议为 `combo-cuda-v4`。Python 仅负责数据 I/O、AST 解析、CUDA 调度、服务和报告序列化。
因子与信号状态机保持 FP64，组合试探的每日持仓/资金递推用 FP32，统计保持 FP64。
默认 `precision=auto` 按误差校准、数值稳定性和 gate 距离自动升级到 GPU FP64。
生产路径没有 CPU 数值回退，也没有 GPU 计算后再跑 CPU 回测。

## 组合准入

每条 alpha 都是当前组合的候选构件，不建立独立单 alpha 准入池。
候选通过结构与因果检查后，试探 0.05..1.00 的 20 个权重以及解析建议权重。
每个试探点直接合成真实持仓并扣除交易成本；组合内部交易可对冲。
准入同时要求：

- 近期加权 Sharpe 至少提高 0.01（半衰期 1095 天）。
- 全期 Sharpe 不下降。
- 全期 CAGR 和近期加权对数增长不下降。

单条年度收益、空仓年份、敞口、相关性用于诊断，不否决互补构件。
满足约束的方案按近期加权 Sharpe、CAGR 排序选择。参数见 `config/config.ini [q25_combo]`。
0.01 是最小工程增量，不代表统计显著性；历史候选看过全期数据。

空组合从等权基准起步。`--accept` 每接受一条便重建组合参照；过期判定禁止入库。
Python 候选直接使用 CuPy/CUDA 原语，返回设备数组；禁止 NumPy/Pandas 数值实现及 Python 数值循环。
接受的 Python 源码按内容哈希保存，可从磁盘重建组合。

## 运行

```text
python tmp/q25_funnel/run.py --daemon serve
python tmp/q25_funnel/run.py --file expressions.txt
python tmp/q25_funnel/run.py --python alpha.py
python tmp/q25_funnel/run.py --file expressions.txt --accept
python tmp/q25_funnel/run.py --daemon dash
python scripts/remine_q25.py tmp/q25_funnel/candidates/r6/full_pool.json <新输出目录>
```

云端连接单源为 `mcp-ssh/hosts.json`；通过已有 MCP SSH 服务部署及启动。
本机端口状态文件 `.q25d.json` 指向远端；不要将主机凭据提交 Git。
Windows 独立 Shell 面板使用 `TERM=xterm-256color` 运行 `run.py --daemon dash`。
切换计算协议必须新建输出目录全量重挖，旧版本 checkpoint 不可混入新结果。
`--resume` 只恢复相同协议、候选顺序和组合参照的筛选阶段。

只读批次上限为 256 候选；合成持仓内核每批最多 16384 个试探方案，不物化巨型持仓张量。
加载后按空闲显存限制两项上限，4 GiB 本机显卡不会沿用 V100S 的大批次。
两项来自 `config/config.ini [q25_cuda]`。V100S 上 2048 个混合候选测试：
128/256/512/1024 分别约 496/558/552/467 条每秒，显存池峰值 4.0/6.9/10.5/24.0 GiB。
价格与面板常驻显存。首次运行包含 CUDA 编译与预热，吞吐比较应使用预热后的整批测量。
递推算法与算子实现在 `backtest/ops_cuda/kernels.cu` 和 `q25_native.cu`。

## 验证

```text
python scripts/smoke_q25_combo_gate.py
python scripts/verify_q25_combo_gpu.py --sources --benchmark
python scripts/verify_q25_combo_gpu.py --batch-sweep
python scripts/verify_q25_combo_gpu.py --precision --precision-count 256 --precision-batch 32
```

验证脚本用独立 C++ 核对照静态价格、每日净收益、融合持仓和 gate 选权，并检查生产路径不调用 CPU。
源码验证覆盖候选池全部 CUDA 算子及 Python 候选的三次截断因果检查。
输出写入 `tmp/q25_cuda/`。独立对照使用的 NumPy 仅存在于离线测试。

## 自动精度

`[q25_cuda] precision=auto` 为生产默认；`f64` 用于对照。`mixed` 只用于低层诊断，
不能绕过精算写组合库。精度选择不依赖显卡型号：

1. FP32 批量运行所有组合权重点，收益/Sharpe/CAGR 等统计用 FP64。
2. 每条候选用一个靠近 0.5 的权重点进行 GPU FP64 校准；同时比较权重 1 的结果
   与已有单候选 FP64 收益，以及组合参照的 FP32/FP64 收益。
3. 校准日收益误差超过 `mixed_return_atol=1e-6`、低波动、非有限值、破产、
   或超出非负非杠杆持仓域时，整条候选的权重网格升级 FP64。
4. 用 `max(1e-6,16×校准误差)` 扩展每日收益误差范围，推导 Sharpe/CAGR/近期增长的
   上界；任一权重点仍可能满足所有 gate 时，整条候选升级 FP64 后选权。
5. 精算发现误差突破扩大范围时，当前批次全部恢复 FP64，该组合参照后续也使用 FP64。
   所有通过者必须带 FP64 结果；近似结果只能用于远离边界的拒绝。

这里的误差范围是**经过安全倍数扩大的经验校准范围，不是全权重区间的数学误差证明**。
离线验证需比较全网格 FP64 的准入结果，记录误拒、误收和选权变化；不能只测数组均方误差。
因子中的比较、排名和状态转移会放大舍入误差，本次保持 FP64，不把回测核的精度结论套用到信号生成。
配置误差预算不会修改原 gate 的收益/Sharpe 门槛。

## FACT 思路的应用

参考 Heidari 与 Nikolopoulos 的 FACT (arXiv:2604.26666v2)，采用子图识别、验证后组合与端到端测量。
本任务是递推和滚动时序计算；本次混合精度使用普通 CUDA 浮点运算，
不使用论文针对 SM80/SM90 GEMM 的 Tensor Core 模板，适用于本机 Turing 和云端 Volta。

- `fusion.py` 识别连续逐元素子图，复用已验证算子的标量语义合成为一个 CUDA kernel。
- 滚动与状态机保留原有 CUDA C++ 内核；不改变累加顺序或阈值。
- 常数作为标量参数，编译缓存按结构复用；CuPy/NVRTC 缓存包含设备架构与类型。
- 合成持仓在回测内核读取时融合；日历指标复用对数收益，移除重复排序。
- `verify_q25_combo_gpu.py --all-composition` 对全池验证持仓、日收益、权重和准入一致性。

首次结构编译时间与预热后的吞吐分别报告；融合失败会直接报错，没有 CPU 回退。
