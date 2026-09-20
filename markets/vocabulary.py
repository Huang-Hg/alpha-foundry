"""OperandVocabulary —— 数据驱动的 DSL 终结符词表(直接沿用 parquet 列名,无枚举)。

operand 身份 = **parquet 列名字符串本身**(直接进 AST `Node.op` / `panels` key / 序列化),
不再走手写 `OperandToken` 枚举。词表从 panel 列名自动建(`from_columns`),自动分类
价(PRICE)/量(VOLUME)/特征(FEATURE);缺基础量价列(high/low/close/volume,不含衍生)
→ 直接报错(信任前置条件,出错大声崩)。

grammar 无默认词表(空占位);任何市场 = 挖矿入口
`grammar.set_vocabulary(from_columns(<其 parquet 列>))` → 重建 PRODUCTIONS。

分类口径(name-based,保守:歧义归 FEATURE):
  - PRICE  : 原始价格水平(open/high/low/close/adj_close/vwap/mark/index_price…)。
  - VOLUME : 原始成交/流动性水平(volume/quote_volume/turnover/amount/number_of_trades…)。
  - FEATURE: 其余一切(funding/oi/lsr/premium/rv/momentum/换手率/基本面…)。
PRICE ∪ VOLUME = "raw-level" 叶 → `is_raw_price_wrapper`(G0 gate)据此拒「纯 raw-level 无 ts」
退化树(截面 ≈ symbol identity,承载不了时序信息)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import FrozenSet, Iterable, Mapping, Optional, Tuple


class OperandKind(IntEnum):
    """operand 的语义类别 —— 自动分类(classify_kind)产出 PRICE/VOLUME/FEATURE,
    用户可经 from_columns(overrides=...) 覆盖(含声明 NORMALIZED 补语义精度)。"""
    PRICE = 0       # 原始价格水平 → raw-level(G0 拒纯 wrapper);grammar 初始 SemKind=RAW
    VOLUME = 1      # 原始成交/流动性水平 → raw-level;SemKind=RAW
    FEATURE = 2     # 一般特征(funding/oi/动量/波动…)→ 非 raw-level;SemKind=RAW
    NORMALIZED = 3  # 已**截面归一**的特征(cs-zscore/percentile 类)→ 非 raw-level;SemKind=NORMALIZED
                    #   仅用户显式声明(自动分类从不产出)→ 让现成 cs-forbid 剪冗余 cs_*


# ---- 自动分类规则(保守;只对明确命中的归 PRICE/VOLUME,其余 FEATURE)----
_PRICE_EXACT = frozenset({
    'open', 'high', 'low', 'close', 'adj_close', 'adjclose', 'adjusted_close',
    'vwap', 'twap', 'mark', 'mark_price', 'index_price', 'prev_close', 'preclose', 'price',
})
_VOLUME_EXACT = frozenset({
    'volume', 'quote_volume', 'quotevolume', 'taker_buy_quote_volume', 'taker_buy_base_volume',
    'number_of_trades', 'num_trades', 'trades', 'amount', 'turnover', 'qty', 'quantity',
    'dollar_volume', 'notional', 'vol',
})
# 后缀规则(end-with);避免 'volatility'/'rv_*'/'vol_zscore_288' 误判为量。
_VOLUME_SUFFIX = ('_volume', '_turnover', '_amount', '_trades', '_qty', '_notional')

# 基础量价列(**不含任何衍生**);parquet 缺其一 → from_columns 报错。
_REQUIRED_BASE: Tuple[str, ...] = ('high', 'low', 'close', 'volume')


def classify_kind(name: str) -> OperandKind:
    """按列名自动分类 价/量/特征。"""
    n = name.strip().lower()
    if n in _PRICE_EXACT:
        return OperandKind.PRICE
    if n in _VOLUME_EXACT or n.endswith(_VOLUME_SUFFIX):
        return OperandKind.VOLUME
    return OperandKind.FEATURE


# 列名尾部窗长(派生列约定:ret_288 / ma_dist_48 / rv_12 / vol_zscore_288 …)。
_TRAILING_WINDOW = re.compile(r'(\d+)$')
# 构造余量(shift/lag/ema 暖机)上界 —— 加在尾部窗长上得保守 warmup;远 << panel 深度故无害。
_WARMUP_MARGIN = 3


def warmup_depth(name: str) -> int:
    """operand 列自身的 warmup 深度(bar 数),**数据驱动**:取列名尾部窗长(ret_288→288、
    ma_dist_48→48、rv_12→12、oi_chg_48→48)+ 构造余量;无尾部窗长的列(raw OHLCV、价量
    基础、桶内 bar-local body_pct/log_ret_oc、快照 premium/funding)= 1。
    任何市场的派生列按名自动得 warmup,**无需手工枚举**。required_depth() 用:树最后一行非
    NaN 所需的最小 panel 深度从叶往上累加。"""
    m = _TRAILING_WINDOW.search(name)
    return int(m.group(1)) + _WARMUP_MARGIN if m else 1


@dataclass(frozen=True)
class Operand:
    name: str            # parquet 列名 = operand 身份(进 AST Node.op / panels key / 序列化)
    kind: OperandKind


@dataclass(frozen=True)
class OperandVocabulary:
    """一组 operand 的不可变词表 + 索引。operand 身份 = name(str)。"""
    operands: Tuple[Operand, ...]
    name: str = 'vocab'

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(o.name for o in self.operands)

    @property
    def price_names(self) -> FrozenSet[str]:
        return frozenset(o.name for o in self.operands if o.kind is OperandKind.PRICE)

    @property
    def volume_names(self) -> FrozenSet[str]:
        return frozenset(o.name for o in self.operands if o.kind is OperandKind.VOLUME)

    @property
    def raw_level_names(self) -> FrozenSet[str]:
        """PRICE ∪ VOLUME —— 截面 ≈ symbol identity 的 raw-level 叶(is_raw_price_wrapper 用)。"""
        return frozenset(o.name for o in self.operands
                         if o.kind in (OperandKind.PRICE, OperandKind.VOLUME))

    def kind_of(self, name: str) -> OperandKind:
        return self._by_name[name].kind

    def __post_init__(self):
        object.__setattr__(self, '_by_name', {o.name: o for o in self.operands})


def from_columns(columns: Iterable[str],
                 exclude: Iterable[str] = (),
                 name: str = 'vocab',
                 require_base: bool = True,
                 overrides: Optional[Mapping[str, OperandKind]] = None) -> OperandVocabulary:
    """从 parquet 列名自动建词表(自动分类 价/量/特征,**保留列序**)。
    exclude:非 operand 列(date/symbol/标签/复权因子);
    require_base:缺基础量价列(high/low/close/volume,非衍生)即报错;
    overrides:{列名: OperandKind} —— **使用者自定义** operand 语义判定,覆盖该列自动分类。
      唯一 operand-耦合的语义钩子由此定制:声明 raw-level 归属(PRICE/VOLUME)、或把已截面归一的
      特征声明 NORMALIZED(顺带补语义精度,让 cs-forbid 剪冗余)。"""
    skip = set(exclude)
    ov = dict(overrides or {})
    names = [c for c in columns if c not in skip]
    if require_base:
        missing = [b for b in _REQUIRED_BASE if b not in names]
        if missing:
            raise ValueError(
                f'operand 词表缺基础量价列 {missing}(需全含 {list(_REQUIRED_BASE)});'
                f'当前列={names}。基础量价数据(非衍生)必须存在,否则无法建 DSL operand 词表。')
    ops = tuple(Operand(name=c, kind=ov.get(c, classify_kind(c))) for c in names)
    return OperandVocabulary(operands=ops, name=name)
