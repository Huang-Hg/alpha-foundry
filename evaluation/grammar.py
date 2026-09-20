"""
句法槽型 CFG for formulaic alpha discovery
==========================================

rhs_kind:operand / const_add / const_mul / window / unary / binary / binary_const / ts / pair。
每个 production 自带:
    out_kind: SemKind             — 此节点的输出槽型(panel=RAW / 三类叶型)
    child_allowed: tuple[set[SemKind], ...] — 每个子槽接受的槽型集合
    cost: int                     — 长度成本(替代 token count)

槽型系统(纯句法,保证树可构造):
    SemKind ∈ {RAW, NUMERIC_ADD, NUMERIC_MULPOW, WINDOW_VAL};PANEL_KINDS = {RAW}。
    根 slot 期望 ∈ PANEL_KINDS(const 不能当根);binary_const 第二槽只接对应 NUMERIC 叶;
    ts/pair 末槽只接 window 叶 —— `ts_min(3, w)` / `tanh(0.5)` 等退化形式结构性禁止。

**语义 forbid 层(α-Sem 不变群剪枝)2026-07-22 A/B 证否删除**(scripts/diag_semkind_ab,
预注册判据 <5pp 删):SemON vs SemOFF 行为重复率 10% vs 13%(差 +2.9pp)、门通过率
96% vs 95%、edge 分布同 —— 时序书口径下 qnt_ts_state(ts_rank)对逐币单调包装本就不变、
池 Δ 门做行为级去重,语义层的边际剪枝量不值其复杂度。RANKED/SHIFTED/NEGATED 等
9 个语义 kind + _UNARY_FORBID/_TS_FORBID/根 ctx forbid 随删(恢复从 git 历史)。

剩余结构约束(槽型覆盖不到的):
    - 整树 ≥ 1 operand:evaluator G0(has_operand / raw-price-wrapper)结构静态拒。
    - 树深度 / op 数由搜索器侧约束(gp_baseline 走 DEAP staticLimit + cost 预算)。

cost 表见 PRODUCTION_COSTS;总 cost ≤ max_total_cost 由 ExpansionContext.remaining_cost 前瞻剪枝。

接口(gfn_codec 依赖,签名稳定):
    PRODUCTIONS    — 全部 production 元组(NUM_PRODUCTIONS 条)
    is_legal(p, ctx) — 给定父槽 ctx,该 production 是否合法
    child_context(parent, child_idx, sibling_kinds, outer_ctx) — 构造子 slot 的 ctx
    initial_context(max_total_cost) — 根 slot 的 ctx
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, FrozenSet, List, Optional, Tuple

from config.config import ini as _ini
from markets.profile import ACTIVE as _ACTIVE_PROFILE
from markets.vocabulary import OperandVocabulary


# ============================================================================
# 终结符:Operand —— 数据驱动词表(直接沿用 parquet 列名,无枚举)
# ============================================================================
# operand 身份 = parquet 列名字符串(进 AST Node.op / panels key / 序列化)。
# 无默认词表:挖矿入口经 set_vocabulary(from_columns(<panel 列>)) 注入后 PRODUCTIONS 才有
# operand 叶(手工 Node.make 构树不经产生式,不受影响)。词表/分类见 markets.vocabulary。


# 常数池 — 按算子类型分两组(避免 mul/pow 的 ±1 = identity 浪费搜索空间):
#   ADD_CONSTANTS    = {±0.5, ±1, ±2}      shift 用得着 ±1
#   MULPOW_CONSTANTS = {±0.5, ±2}          mul_const(*,±1) = identity/NEG;pow_const(*,±1) = identity/inv,
#                                          均与 UnaryOp 重复 → 删
ADD_CONSTANTS:    Tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
MULPOW_CONSTANTS: Tuple[float, ...] = (-2.0, -0.5, 0.5, 2.0)

# 窗口(ts/pair 的回看 bar 数,单位 = 市场 bar cadence)的**唯一来源 = markets.MarketProfile.windows**
# (与 bars_per_day 等市场常量同处,不在此重复字面量)。ACTIVE_WINDOWS = 当前生效集(默认 = active
# profile 的 windows),换市场经 set_windows(panel.profile.windows) 注入 → 重建 PRODUCTIONS
# (消除异市场窗集套错 panel 致长窗全 warmup 死)。


# ============================================================================
# 算子枚举
# ============================================================================

class UnaryOp(Enum):
    ABS      = 'abs'
    NEG      = 'neg'
    SIGN     = 'sign'
    LOG      = 'log'
    SQUARE   = 'square'
    SQRT     = 'sqrt'
    TANH     = 'tanh'
    INV      = 'inv'
    S_LOG_1P = 's_log_1p'


class BinaryOp(Enum):
    ADD = 'add'
    SUB = 'sub'
    MUL = 'mul'
    DIV = 'div'
    MAX = 'max'
    MIN = 'min'
    # GT/LT 2026-07-24 恢复(自由书口径):0/1 指示 = regime 条件门(如 gt(close,
    # ts_mean(close,240)) = 趋势择时),乘法复合即"信号×门"双层结构。旧删除理由
    # (cs_zscore+tanh 截面分类器过拟合)随截面栈 2026-07-22 退役而失效。
    GT  = 'gt'
    LT  = 'lt'


class TsOp(Enum):
    TS_MEAN    = 'ts_mean'
    TS_STD     = 'ts_std'
    TS_MAX     = 'ts_max'
    TS_MIN     = 'ts_min'
    TS_SUM     = 'ts_sum'
    TS_RANK    = 'ts_rank'
    TS_ARG_MAX = 'ts_arg_max'
    TS_ARG_MIN = 'ts_arg_min'
    TS_EMA     = 'ts_ema'
    TS_WMA     = 'ts_wma'
    TS_REF     = 'ts_ref'
    TS_DELTA   = 'ts_delta'
    TS_SKEW    = 'ts_skew'      # 滚动 3 阶中心矩
    TS_KURT    = 'ts_kurt'      # 滚动 4 阶中心矩
    TS_MAD     = 'ts_mad'       # 滚动 mean absolute deviation(robust 替代 ts_std)
    TS_SLOPE   = 'ts_slope'     # 滚动 OLS 斜率
    # 时序书口径新增融合算子(2026-07-22,文献对齐;max_len=5 下组合不可达 → 融合成单 production)
    TS_CHANPOS = 'ts_chanpos'   # Donchian 通道位置 (x−min)/(max−min) ∈[0,1](Zarattini 突破集成)
    TS_DDOWN   = 'ts_ddown'     # 距滚动高点回撤 x/max−1 ≤0(George-Hwang nearness-to-high)
    TS_ZS      = 'ts_zscore'    # 滚动 z=(x−μ)/σ,ddof=0(归一趋势)
    TS_SNR     = 'ts_snr'       # 趋势信噪比 μ/σ(rolling t-stat;AdaptiveTrend 趋势质量门)


class TernaryOp(Enum):
    """bool 组合子(2026-07-24,用户指令:原生事件驱动 / 更精细调控)。
    与 GT/LT 复合 = 条件切换书、事件闸、滞回持仓 —— rank-gate 与纯代数空间均不可表达。
    **注入层专用,不入搜索产生式**(2026-07-24 用户裁定):手工先验树(Node.make 直构)/
    部署层可用,GFN/GP 不可采样 —— 搜索自由度=噪声极值天花板(scaling 四轴证否同源)。"""
    IF_ELSE    = 'if_else'      # cond>0 ? a : b(条件分支;cond NaN→NaN,a/b NaN 自然传播)
    TRADE_WHEN = 'trade_when'   # exit>0→0;entry>0→alpha;否则持有前值(逐列时序状态机,因果)


class TsInjOp(Enum):
    """BRAIN time_series 族补齐,签名与 TsOp 相同(kind='ts')。**注入层专用,不入产生式**。
    实现:backtest/ops/ops_inject_ts.cpp。"""
    TS_BACKFILL = 'ts_backfill'   # 回看 d 行取最近有效值。**不套 min_periods** —— 它的用途
                                  # 就是填 NaN,套了等于自我否定(稀疏字段用它,稠密用 nan_handling)
    TS_MEDIAN   = 'ts_median'     # 滚动中位数(稳健中心,不被单日暴动拉走)
    TS_AV_DIFF  = 'ts_av_diff'    # x − mean(x,d),**均值忽略 NaN**(BRAIN 定义,与 ts_mean 相反)


class TsParamOp(Enum):
    """panel + window + **连续标量**(kind='ts_param')。BRAIN 里这些参数是有名可选参数
    (factor= / constant= / driver=),DSL 里作第三个位置参数。**注入层专用,不入产生式**。"""
    TS_DECAY_EXP_WINDOW = 'ts_decay_exp_window'  # (x,d,factor):权重 factor^i 归一化;
                                                 # factor=1 退化为等权均值,越小越重视近期
    TS_SCALE            = 'ts_scale'             # (x,d,k) = (x−min)/(max−min) + k;
                                                 # k=0 时逐字等于 ts_chanpos,k 用来平移出 clip 边界
    TS_QUANTILE         = 'ts_quantile'          # (x,d,driver):窗内序数位置经逆 CDF 映射。
                                                 # driver 0=gaussian 1=uniform 2=cauchy
                                                 # (BRAIN 用字符串,DSL 用数值码)


class PairParamOp(Enum):
    """panel + panel + window + **连续标量**(kind='pair_param')。**注入层专用,不入产生式**。"""
    TS_REGRESSION = 'ts_regression'   # (y,x,d,rettype):窗内 y ~ α+β·x 滚动 OLS。
                                      # rettype 0=残差 1=截距 2=斜率 3=拟合值 4=R²。
                                      # ⚠️ 0–3 按 BRAIN 文档化含义命名,但知识库未收录其
                                      # rettype 数值对照表 ⇒ **不保证与平台逐码一致**;
                                      # 4(R²)是本仓扩展。跨平台移植表达式须自行核对。
                                      # 用途:时序残差化 = 中性化的时序版(把 y 中能被 x
                                      # 解释的部分剔掉,留下特质分量)。


class LogicalUnaryOp(Enum):
    """BRAIN logical 族的一元部分(2026-08-13,用户指令:LLM 写 alpha 需要完整条件逻辑)。
    **注入层专用,不入搜索产生式** —— 同 TernaryOp 先例:手写/LLM 树可用,GFN/GP 不可采样。
    实现在 evaluation.ops_injection(numpy;C 内核只覆盖原搜索算子集)。"""
    NOT    = 'not'        # x>0 → 0,否则 → 1;NaN → NaN
    IS_NAN = 'is_nan'     # NaN → 1,否则 → 0(**唯一不传播 NaN 的算子**,它就是用来测 NaN 的)


class LogicalBinaryOp(Enum):
    """BRAIN logical 族的二元部分。greater/less 已由 BinaryOp.GT/LT 覆盖,此处补其余六个。
    统一 NaN 语义:任一侧 NaN → NaN(与既有 gt/lt 的 C 内核一致)。**注入层专用**。"""
    AND       = 'and'
    OR        = 'or'
    EQUAL     = 'equal'
    NOT_EQUAL = 'not_equal'
    GE        = 'greater_equal'
    LE        = 'less_equal'


class ParamOp(Enum):
    """panel ⊕ **连续**标量参数。槽型同 binary_const,但参数不受 ADD/MULPOW_CONSTANTS 约束 ——
    常数池是为 GP 枚举产生式而设的离散化,注入层不枚举,离散化只白扔表达力。
    **注入层专用,不入搜索产生式**。"""
    LEFT_TAIL  = 'left_tail'    # [BRAIN transformational] > maximum 的全部置 NaN
    RIGHT_TAIL = 'right_tail'   # [BRAIN transformational] < minimum 的全部置 NaN
    HUMP       = 'hump'         # [BRAIN time_series] 限制逐步变化幅度 ⇒ 压换手(turnover 是硬门)
                                # out[t] = x[t] if |x[t]−out[t−1]| > k else out[t−1]
                                # **列内时序递归**(同 trade_when)⇒ 批并行需 scan,默认走 CPU


class PairOp(Enum):
    TS_CORR = 'ts_corr'
    TS_COV  = 'ts_cov'


class ConstBinaryOp(Enum):
    """常数双目算子:第一槽 panel,第二槽 NUMERIC 常数。AlphaCFG-style typed slot。"""
    ADD_CONST = 'add_const'  # panel + k     (小偏移)
    MUL_CONST = 'mul_const'  # panel * k     (缩放)
    POW_CONST = 'pow_const'  # sign(panel)·|panel|^k(signed_power,WQ101 #1/#3/#4 等用)


# ============================================================================
# 槽型(纯句法;语义 kind 2026-07-22 A/B 证否删除)
# ============================================================================

class SemKind(IntEnum):
    RAW            = 0   # panel(operand 或任意算子输出)
    NUMERIC_ADD    = 8   # ADD_CONST 第二槽:{±0.5, ±1, ±2}
    NUMERIC_MULPOW = 9   # MUL_CONST/POW_CONST 第二槽:{±0.5, ±2}(去 ±1 防 identity / NEG 冗余)
    WINDOW_VAL     = 10  # 窗口整数(ts/pair 最后一槽)


ALL_KINDS:   FrozenSet[SemKind] = frozenset(SemKind)
# panel 槽型集合 = 排除三类叶型。
PANEL_KINDS: FrozenSet[SemKind] = ALL_KINDS - {SemKind.NUMERIC_ADD, SemKind.NUMERIC_MULPOW, SemKind.WINDOW_VAL}


# ============================================================================
# Production 定义
# ============================================================================

class NonTerminal(IntEnum):
    EXPR   = 0  # 主 NT,语义类型由 Production.out_kind / child_allowed 表达
    WINDOW = 1  # 窗口 leaf 子槽(ts/pair 最后一槽);只接 rhs_kind='window' 的 8 个 production


@dataclass(frozen=True)
class Production:
    """文法产生式 + α-Sem 类型标记。

    rhs_kind: 节点类别(用于 ast / encoder 派发)
    op:       具体算子(operand=parquet 列名 str / UnaryOp / BinaryOp / TsOp / PairOp)
    extra:    ts/pair 的窗口 w(int)
    children_nts: 子 NT 序列(占位用,数量决定 children 数)
    cost:     production 自身 cost
    out_kind: 此 production 的输出语义类型(α-Sem)
    child_allowed: 每个子槽允许的输入 SemKind 集合(典型地为 ALL_KINDS \\ {redundant})
    """
    rhs_kind: str
    op: object
    extra: object = None
    children_nts: Tuple[NonTerminal, ...] = ()
    cost: int = 1
    out_kind: SemKind = SemKind.RAW
    child_allowed: Tuple[FrozenSet[SemKind], ...] = ()


PRODUCTION_COSTS: Dict[str, int] = {
    'operand':      1,
    'const_add':    1,    # NUMERIC_ADD 标量,填 add_const 第二槽
    'const_mul':    1,    # NUMERIC_MULPOW 标量,填 mul/pow_const 第二槽
    'unary':        2,
    'binary':       3,
    'binary_const': 2,    # panel ⊕ const,比 binary(panel ⊕ panel)便宜:无需第二个 panel 子树
    'ternary':      4,    # 三 panel 子槽 bool 组合子(if_else / trade_when;仅注入树 total_cost 用,无 production)
    'ts':           3,
    'pair':         5,
    'ts_param':     3,    # panel+window+标量(注入层;标量子叶自带 const_add 的 1,合计 4)
    'pair_param':   5,    # panel+panel+window+标量(注入层;合计 6)
    # 'window' 不在表里 — window leaf 在 make_node 时被 strip 进父 ts/pair 的 Node.window
    # attr,不进 AST.total_cost 遍历;production cost inline 写 0(纯参数,免成本预算)。
}


def all_productions() -> Tuple[Production, ...]:
    """枚举所有合法 productions(纯句法槽型)。"""
    out: List[Production] = []

    # operand 叶(从 active 词表取,op = parquet 列名字符串)
    for _op in ACTIVE_VOCAB.operands:
        out.append(Production('operand', _op.name, None, (), PRODUCTION_COSTS['operand'],
                              out_kind=SemKind.RAW, child_allowed=()))

    # const 叶按 op 分组(per-op grid,#2):
    #   const_add → NUMERIC_ADD,只填 ADD_CONST 第二槽
    #   const_mul → NUMERIC_MULPOW,只填 MUL_CONST/POW_CONST 第二槽
    for c in ADD_CONSTANTS:
        out.append(Production('const_add', c, None, (), PRODUCTION_COSTS['const_add'],
                              out_kind=SemKind.NUMERIC_ADD, child_allowed=()))
    for c in MULPOW_CONSTANTS:
        out.append(Production('const_mul', c, None, (), PRODUCTION_COSTS['const_mul'],
                              out_kind=SemKind.NUMERIC_MULPOW, child_allowed=()))

    # window 叶 → out=WINDOW_VAL,**只能**填 ts/pair 最后一槽
    # 跟 const 同类设计但隔离类型,防止填错槽位。cost=0(纯参数,不占预算)。
    for w in ACTIVE_WINDOWS:
        out.append(Production('window', w, None, (), 0,
                              out_kind=SemKind.WINDOW_VAL, child_allowed=()))

    # unary(子槽只接 panel)
    for op in UnaryOp:
        out.append(Production('unary', op, None, (NonTerminal.EXPR,), PRODUCTION_COSTS['unary'],
                              out_kind=SemKind.RAW, child_allowed=(PANEL_KINDS,)))

    # binary(panel + panel,两槽都只接 panel,不接 NUMERIC)
    for op in BinaryOp:
        out.append(Production('binary', op, None, (NonTerminal.EXPR, NonTerminal.EXPR), PRODUCTION_COSTS['binary'],
                              out_kind=SemKind.RAW,
                              child_allowed=(PANEL_KINDS, PANEL_KINDS)))

    # binary_const(panel + 常数,第二槽按 op 接 NUMERIC_ADD 或 NUMERIC_MULPOW)
    # ADD_CONST:add_const(panel, k),k ∈ {±0.5, ±1, ±2}
    # MUL_CONST/POW_CONST:k ∈ {±0.5, ±2}(±1 跟 NEG/inv 重复 → 已删)
    only_add: FrozenSet[SemKind] = frozenset({SemKind.NUMERIC_ADD})
    only_mul: FrozenSet[SemKind] = frozenset({SemKind.NUMERIC_MULPOW})
    for op in ConstBinaryOp:
        slot_allowed = only_add if op == ConstBinaryOp.ADD_CONST else only_mul
        out.append(Production('binary_const', op, None, (NonTerminal.EXPR, NonTerminal.EXPR),
                              PRODUCTION_COSTS['binary_const'],
                              out_kind=SemKind.RAW,
                              child_allowed=(PANEL_KINDS, slot_allowed)))

    # ternary 无 production:注入层专用算子,不入搜索预算(见 TernaryOp docstring)

    # ts(panel + window leaf):每个 op 1 production,window 由独立 len(ACTIVE_WINDOWS) 个 productions 填
    only_window: FrozenSet[SemKind] = frozenset({SemKind.WINDOW_VAL})
    for op in TsOp:
        out.append(Production('ts', op, None,
                              (NonTerminal.EXPR, NonTerminal.WINDOW),
                              PRODUCTION_COSTS['ts'],
                              out_kind=SemKind.RAW,
                              child_allowed=(PANEL_KINDS, only_window)))

    # pair(panel + panel + window leaf)
    for op in PairOp:
        out.append(Production('pair', op, None,
                              (NonTerminal.EXPR, NonTerminal.EXPR, NonTerminal.WINDOW),
                              PRODUCTION_COSTS['pair'],
                              out_kind=SemKind.RAW,
                              child_allowed=(PANEL_KINDS, PANEL_KINDS, only_window)))

    return tuple(out)


# active operand 词表 + 窗口集 —— 词表无默认(空占位,挖矿入口 set_vocabulary 注入;旧 5m 默认
# 词表随 1h crypto 栈 2026-07-25 删除);窗口集默认 = active profile.windows(单一来源)。
# set_vocabulary/set_windows() 换市场重建 PRODUCTIONS。
ACTIVE_VOCAB:    OperandVocabulary = OperandVocabulary(operands=(), name='unset')
ACTIVE_WINDOWS:  Tuple[int, ...]   = tuple(_ACTIVE_PROFILE.windows)


def set_vocabulary(vocab: OperandVocabulary) -> None:
    """切换 active operand 词表(数据驱动),重建模块级 PRODUCTIONS / NUM_PRODUCTIONS。
    换市场时由数据层用 from_columns(<parquet 列>) 建词表后调用一次。下游 ast/expression/
    adapter 经 panels[name] 字符串键自动对齐,无需改签名。"""
    global ACTIVE_VOCAB, PRODUCTIONS, NUM_PRODUCTIONS
    ACTIVE_VOCAB = vocab
    PRODUCTIONS = all_productions()
    NUM_PRODUCTIONS = len(PRODUCTIONS)


def set_windows(windows: Tuple[int, ...]) -> None:
    """切换 ts/pair 窗口集(单位 = 市场 bar cadence),重建 PRODUCTIONS。
    换市场时由挖矿入口注入 `panel.profile.windows` 后调用一次。window 叶随之刷新,gp_baseline
    `_build_pset` 运行时读 ACTIVE_WINDOWS → rebuild_pset() 同步(消除 import 顺序陷阱)。"""
    global ACTIVE_WINDOWS, PRODUCTIONS, NUM_PRODUCTIONS
    ACTIVE_WINDOWS = tuple(windows)
    PRODUCTIONS = all_productions()
    NUM_PRODUCTIONS = len(PRODUCTIONS)


PRODUCTIONS: Tuple[Production, ...] = all_productions()
NUM_PRODUCTIONS: int = len(PRODUCTIONS)


# ============================================================================
# 扩展上下文 + 合法性
# ============================================================================

@dataclass
class ExpansionContext:
    """要被扩展的 slot 的上下文。
    expected_out_kinds: 此 slot 允许的输出 SemKind 集合(从父 production 的 child_allowed 传下)。
    parent_kind / parent_op:仅留作 caller 反查诊断,不进 is_legal。
    remaining_cost:cost 前瞻剪枝。"""
    expected_out_kinds: FrozenSet[SemKind] = ALL_KINDS
    parent_kind:        Optional[str] = None
    parent_op:          Optional[object] = None
    remaining_cost:     int = 999_999_999


def is_legal(prod: Production, ctx: ExpansionContext) -> bool:
    """α-Sem 合法性:返回 True 表示该 production 可在 ctx 下扩展。"""
    # 1) cost 前瞻
    min_total = prod.cost + (1 if prod.children_nts else 0)
    if min_total > ctx.remaining_cost:
        return False

    # 2) α-Sem 类型:此 prod 的 out_kind 必须落在 slot 允许集合内
    if prod.out_kind not in ctx.expected_out_kinds:
        return False

    # ts/pair same-op-same-window 嵌套不剪:冗余如 ts_mean(ts_mean(x,24),24) 保留在搜索空间里
    # (罕见,且 inner ts_mean 仍是合法子特征)。
    return True


def child_context(parent: Production, child_idx: int, sibling_kinds: Tuple[str, ...],
                  outer_ctx: ExpansionContext) -> ExpansionContext:
    """构造扩展某个子节点时的 context。"""
    return ExpansionContext(
        expected_out_kinds=parent.child_allowed[child_idx] if child_idx < len(parent.child_allowed) else ALL_KINDS,
        parent_kind=parent.rhs_kind,
        parent_op=parent.op,
        remaining_cost=outer_ctx.remaining_cost - parent.cost,
    )


def initial_context(max_total_cost: int) -> ExpansionContext:
    """根 slot:整树根必须输出 panel(const/window 叶不能当根)。"""
    return ExpansionContext(
        expected_out_kinds=PANEL_KINDS,
        parent_kind=None, parent_op=None,
        remaining_cost=max_total_cost,
    )


# ============================================================================
# DSL 搜索空间预算([dsl] 唯一来源;GFN(gfn_codec)与 GP(gp_baseline)统一经此读,
# 2026-07-03 收单链 —— 此前 25/3 硬编码,改 ini 只动 GP 不动 GFN 的静默分叉已修)
# ============================================================================

DEFAULT_MAX_COST: int = int(_ini('dsl', 'max_cost', 25))   # 单棵 alpha 最大累计 cost
DEFAULT_MIN_COST: int = int(_ini('dsl', 'min_cost', 3))    # 整树最小 cost(防 alpha = 单 operand,无意义)
DEFAULT_MAX_LEN:  int = int(_ini('dsl', 'max_len', 5))     # 推导 production 数上限(含 window leaf;5=实测极限)
