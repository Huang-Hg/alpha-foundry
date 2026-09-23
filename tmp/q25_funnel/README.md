# Q25 CUDA 组合挖掘

协议为 `combo-cuda-v6`。Python 仅负责数据 I/O、AST 解析、CUDA 调度、服务和报告序列化。
因子与信号状态机保持 FP64，组合试探的每日持仓/资金递推用 FP32，统计保持 FP64。
默认 `precision=auto` 按误差校准、数值稳定性和 gate 距离自动升级到 GPU FP64。
生产路径没有 CPU 数值回退，也没有 GPU 计算后再跑 CPU 回测。

## 组合准入

每条 alpha 都是当前组合的候选构件，不建立独立单 alpha 准入池。
候选通过结构与因果检查后，试探 0.05..1.00 的 20 个权重以及解析建议权重。
每个试探点直接合成真实持仓并扣除交易成本；组合内部交易可对冲。

**合成规则(v6,2026-09-21)**:`H = (U + a·A) / max(1, 当日总仓位)`,U 为成员书按权重直接相加的原始和。
候选是**加**到组合上的,只有相加后超过满仓的日子才整体等比缩回(即 Quantiacs 对权重的原生处理)。
成员权重是各自持仓的相加倍数 ∈ (0,1],**不是和为 1 的份额**;入库时已有成员原样不动,CEM 重配逐成员独立。
v5 及以前是凸混合 `(1−a)P + aA`:每给候选一份就从组合扣一份。实测生产组合平均仓位仅 0.379、满仓日 0.3%,
凸混合却按满仓计挤占,CAGR 约束于是等价于“候选自身收益≈整个组合”,gfn_0921 的 13 万候选无一可行;
Sharpe 增益前 64 名里 92% 只死在这一条上(`scripts/diag_q25_admission_operators.py`)。
等权 beta 与“组合自身放大”在新规则下仍过不了 Sharpe 约束。组合库已满时新成员在任何权重下都不可行(相加不构成替换)。
准入同时要求：

- 近期加权 Sharpe 至少提高 0.01（半衰期 1095 天）。
- 全期 Sharpe 不下降。
- 收益与回撤互相折算(Sharpe 为主维度,不参与折算;2026-09-22 用户定"非 Sharpe 维度可以降,只要其他维度够好"):
  全期 CAGR 与近期加权对数增长各自的相对变化 ≥ 最大回撤深度的相对变化,即 Δx ≥ |x_参照|·(|DD|/|DD_参照| − 1)。
  回撤不变 ⇒ 收益不降;回撤加深 10% ⇒ 收益至少多 10%;回撤变浅 10% ⇒ 收益可少 10%(`combo_gpu.return_requirements`)。

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
python -m monitoring --source tmp/q25_funnel/.q25d_status.json      # 本机服务的实时面板
python scripts/mirror_q25_status.py <host> /root/quant/tmp/q25_funnel/.q25d_status.json tmp/q25_machine/service_status.json
python -m monitoring --source tmp/q25_machine/service_status.json    # 云端服务:先起上面的镜像
python scripts/remine_q25.py tmp/q25_funnel/candidates/r6/full_pool.json <新输出目录>
```

云端连接单源为 `mcp-ssh/hosts.json`；通过已有 MCP SSH 服务部署及启动。
本机端口状态文件 `.q25d.json` 指向远端；不要将主机凭据提交 Git。
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

- 滚动与状态机保留原有 CUDA C++ 内核；不改变累加顺序或阈值。
- 合成持仓在回测内核读取时融合；日历指标复用对数收益，移除重复排序。
- **逐元素子图的运行时融合（`fusion.py`）已于 2026-09-21 证否删除**：机器搜索每轮的结构几乎都是新的，
  每个新结构要现场 NVRTC 编译一个 kernel（V100S 每轮约 48 个 × 0.11 s ≈ 5 s，占评价 6 s 的大头，GPU 利用率 0–1%）；
  而 kernel 编好之后，同批 125 个真实候选开/关融合耗时相同（T1000 0.89 s 对 0.89 s），持仓与日收益逐位相同。
  即它只有编译成本、没有运行收益。度量脚本 `scripts/diag_q25_search_throughput.py`。
