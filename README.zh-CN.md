# Foundry

面向时点一致数据的公式、模型信号与目标敞口研究和回测。支持美股、A 股、加密货币日线现货与 USDT 线性逐仓永续。A 股保留涨跌停、停牌、卖出税费和次日执行回归。Linux CPU/CUDA 执行明确的市场规格；股票和现货支持双 FP32，永续账户目前明确使用 FP64。

## 模块边界

采用模块化单体与端口适配器。AGENTS.md 规定边界；tests/test_architecture.py 检查依赖无环、禁止依赖及 import 副作用。

| 所有者 | 职责 |
| --- | --- |
| contracts、candidates、research_types | 研究区间、快照、后端能力、类型化来源/结果与存储端口 |
| core、expr、market、portfolio | 面板与指标语义、公式、交易规则和准入 |
| ops、backtest、engine | 基于明确数组和执行规格的 CPU/portable/CUDA 计算 |
| research | 共用评价、邻域探索和显式版本化提交 |
| search、CLI/RPC | 提案或传输，共用研究决策 |
| adapters | 数据读取、指纹、原子存储和序列化 |
| runtime | 不可变配置、后端准备与资源所有权 |
| monitor | 只读取发布状态，不持有行情、引擎或 GPU |

load_config() 显式加载配置。导入模块不读行情、不探测设备、不编译 kernel、不分配显存。open_runtime(settings, study) 拥有这些操作，并通过 close() 显式释放。领域依赖由参数传入，不依靠全局 service locator。

Study 固定研究区间；快照进一步绑定数据、执行规格、数值策略和实现身份。查询结果携带组合版本。评价不写组合，commit 是独立的 compare-and-swap 命令，一次原子发布替换结果。设备驻留接口与 CLI/RPC 序列化共用类型化用例。可导出 JSON Schema 是机器契约，python -m foundry --help 是命令清单。

旧请求、组合和 checkpoint 协议明确拒绝，不保留兼容层，不自动重录基线。Quantiacs 专属源码、提交 notebook、配置与测试已从当前工程移除；历史数据（包括 data/qnt_crypto）、来源 manifest 和私有生产工件继续保留。保留数据不代表其具备完整 PIT 来源证明。

## 候选接口

`research-v10` 使用 `Candidate(id, source)`。来源明确区分 `Formula(expression)`、`ArraySource(kind, digest, snapshot, producer)` 和组合基准 `EqualWeight()`。公式挖掘器与邻域探索构造 `Formula`；模型输出通过 `Research.register_batch(batch, context_id)` 注册，然后共用评价、权重试探、准入和提交规则。

通过显式存储实例构造 `Research(engine, library_store, array_store, null_shift=False)`。`evaluate(candidates, context_id)` 返回标量报告；`evaluate_device(sources, context_id, report=False)` 保留设备上的持仓与日收益。查询不发布工件、不修改组合；`commit(candidates, context_id)` 一次原子发布工作副本。注册是独立命令，保存按内容寻址的不可变工件并复制输入；返回后生产者可以复用原缓冲区。重载含数组成员的组合时，会重新验证工件和声明的时间截止点。

`foundry.candidates.ArrayBatch` 包含 `kind`、`ids`、`values`、`snapshot`、`producer`、`dates`、`assets`、`observed_through`、`fit_through`。`values` 是 NumPy/CuPy 的 `float32` 或 `float64 (B,T,S)`，必须覆盖研究快照完整且有序的日期/资产网格。股票与现货的 `signals` 与公式一样经过 `clip(0,1)/当日可持有资产数`，NaN 表示缺失。`target_weights` 是有限、非负的组合比例；无资格资产权重必须为零，单日总敞口不得超过 1。仅对 `[portfolio] weight_sum_atol` 内的微小超额归一化到 1，更大超额明确拒绝。目标权重继续服从市场执行规则和次日开盘时序。

两个截止点数组均为 `int64 (B,T)`，使用包含端点的面板行号，范围为 `-1` 到对应决策行；`-1` 表示未使用观测或拟合模型。引擎拒绝未来截止点、错误坐标和旧快照。时间信息来自生产者声明，不能证明外部训练或特征计算没有泄露未来；`producer` 应标识模型/程序版本。`research.describe` 发布完整坐标、快照及这些限制：batch 的 `snapshot` 使用 `context.snapshot.id`，查询/命令的版本令牌使用 `context.id`。

注册时向宿主传输一次数据，用于内容摘要和保存；最近输入的设备缓存最多保留 `candidate_batch` 个候选。缓存命中后评价继续使用设备数组，逐出的输入从已验证工件重载。CUDA 生产者须在提交前让输入就绪于 runtime 当前 stream；注册会等自有复制完成再返回。外部数组的噪声对照延迟只依赖生产者身份与输入类型，不依赖包含未来数值的全序列摘要。

RPC 使用 `foundry.adapters.arrays.write_bundle(path, batch)` 导出宿主数组，再发送 `research.register_batch` 的 `{context, path}`。路径位于服务所在机器，加载使用 `allow_pickle=False`。注册响应返回可直接用于 `research.evaluate` / `research.commit` 的类型化候选。公式请求形式为 `{"id":"alpha-1","source":{"kind":"formula","expression":"gt(ma_dist_20,0)"}}`。`python -m foundry schema OUTPUT` 导出请求和结果 schema；Python 结果与 JSON 序列化由同一组 dataclass 定义。

## 加密货币执行

`crypto_spot` 使用 UTC 日线、7×24 连续日历与 365 日年化。保留 Quantiacs 历史报价及资格，缺失的发布时间/修订版本通过数据契约公开。缺失报价时冻结已有持仓，不从未来末值填补过去。

`crypto_perpetual` 使用 USDT 线性合约和逐仓保证金。日线收盘信号在下一 UTC 日开盘调仓；每小时依次处理标记价跳空、旧仓资金费、计划调仓、标记价高低点触发强平、收盘估值。仅在每日调仓时重新分配逐仓抵押，空闲现金不自动补仓。`target_exposure` 接收有符号的名义本金/净值比，总绝对敞口上限为配置杠杆；公式与 `signals` 截到 [-1,1] 后除以合资格资产数，不隐式放大到最大杠杆。

`[crypto_perpetual]` 集中配置数据和执行参数；唯一 CLI 的导入命令下载并核验 Binance 成交 K 线、标记价 K 线和已结算资金费归档 SHA256。初始固定 BTC/ETH/SOL 集合使用 2025 数据。这是固定维持保证金/费用、小时不利价格触达的研究模型，未重建历史风险档位、VIP 费用、订单簿、ADL 或 tick 路径。保留真实资金费时间戳；执行映射到所属小时，归档缺少结算标记价时明确使用该小时开盘标记价。发现接口公开上述限制，不声称交易所精确仿真或完整供应商 PIT。

类型化模拟结果包含每日资金费、交易/强平费用、逐仓损失封顶调整、强平阶段位图、破产与终值。报告的现金流和次数按研究窗口汇总。损失封顶调整仅为本模型会计量，不冒充交易所保险赔付；行情不利变化后的实际最大杠杆可能高于入场上限。永续执行前明确规划为 FP64，股票/现货保留双 FP32 策略。

## PIT 与精度

共享指标中的 `turnover`、`exposure`、`max_single_weight` 和 `margin_bps` 分母描述每日目标持仓，不等同于永续实际成交本金。实际费用和账户杠杆见独立风险报告；日内强平时间标识所在小时，不能当作精确触碰时刻。

先截断源数据，再派生特征。收盘决策最早在次日开盘生效。横截面只包含当时可见且有资格的资产。报价历史与成分区间分离；迟到发布与修订不能覆盖旧决策行；forward 标签不进入特征面板。

A 股数据保留 365 只历史成分的完整报价，资格另按区间应用。信号使用原始 OHLCV，执行使用复权价格。manifest 明示缺少历史修订版本和原始成分公告时间戳。美股数据限制同样通过发现接口暴露。前缀与未来污染测试证明计算因果性，不代表完整供应商 PIT 认证，也不是样本外收益证据。

执行规格包含开盘可交易资格、成本、方向冻结与持仓费用，内核不凭股票代码猜交易规则。CPU/CUDA 保留相同完整资产范围与指标语义。

双 FP32 用两个分量计算递推、混合和统计，以缩放指数保留财富范围。转换、部分报告指标和最终准入复核仍使用 FP64。验收比较数值误差、交易、权重选择与阈值两侧决策，不承诺逐位 FP64。

历史驻留 kernel 测量：600 日 × 299 股 × 256 组合，CUDA event 毫秒。

| GPU | FP64 执行 + 统计 | 双 FP32 执行 + 统计 |
| --- | ---: | ---: |
| L4 | 43.356 | 10.904 |
| RTX PRO 6000 Blackwell Server Edition | 17.593 | 7.035 |
| A100 SXM4 40GB | 7.055 | 12.682 |

这些记录不含打包/分配、首次传输和 JIT，也不是本轮重构的端到端性能。此前统计实验保持日收益不变，含混合试探的最大统计绝对误差为 1.43e-13。双 FP32 不保证在所有设备上快于原生 FP64。

## 开发与验证

Python 负责研究、接口与编排，现有 C++/CUDA 内核负责数值计算，一个 Bash 脚本构建三个 Python 扩展。正式部署与回测环境为 Linux，包括 WSL2 和 Colab。WSL2 使用 Linux Python 生成原生 `.so`，已删除 Windows Python/MSYS2 交叉构建分支。

构建需要 Python 3.12 或 3.13、对应开发头文件，以及支持 C++17/OpenMP 的 GCC。在仓库根目录激活实际运行 Foundry 的 Linux 虚拟环境，安装依赖并构建：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash build.sh
```

`bash build.sh backtest` 只重建回测扩展；不指定目标时构建 `ops`、`backtest`、`search`。脚本从当前 `python3` 获取 Python/NumPy 头文件与 ABI 后缀，支持含空格的路径，不绑定旧机器的解释器或编译器路径。算子、回测和策略各自的数值编译选项保持不变。Linux CI 覆盖 Python 3.12/3.13 与 ASan/UBSan，不再维护独立 Windows 工具链。

GPU 后端单独安装匹配环境 CUDA 主版本的 CuPy；已验证的 Colab CUDA 13 环境使用 `python -m pip install cupy-cuda13x`。CPU 执行无需 CuPy 或 GPU。所有可调参数集中在 `config/config.ini`。架构与纯规则测试不依赖私有行情、组合或凭据。

CPU CI 覆盖架构、表达式、原生 ABI、搜索策略、发布时点和请求/存储协议。云端测试显式选择美股/A 股/加密现货/逐仓永续，使用隔离的临时组合库，不从私有生产库获取隐藏种子。test_production 单独校验明确提供的生产基线，不悄悄重录。

本轮任务的**全部测试仅在已授权 Colab PRO 6000 执行**；本地仅编辑和静态审阅。云端先运行 CPU smoke，再运行数据/GPU 套件：

```bash
PYTHONPATH=tests python -m unittest test_architecture test_expr test_native test_policy test_monitor test_pit test_api test_service test_runtime test_candidates test_crypto_spot test_perpetual test_perpetual_data
PYTHONPATH=tests python -m unittest test_cuda_backtest test_cuda_statistics test_cuda_markets test_cuda_pit test_cuda_candidates test_cuda_perpetual test_markets test_portfolio test_search
```

结构性决策记在 docs/architecture/。性能报告分别记录加载、传输/JIT、驻留批量与完整用例，注明设备、完整规模、精度和误差。
