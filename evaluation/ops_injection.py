"""注入层算子的分发表注册 —— BRAIN logical / transformational 族(2026-08-13 用户指令)。

**实现全在 C**(`backtest/ops/ops_xs.cpp` + `_ops_module.cpp`),与既有算子同一套内核、
同一套 NaN 语义、同一条构建链路(`backtest/build.sh`)。本文件只做**枚举 → C 函数**的
挂表,不含任何算子实现 —— 保证「一个算子一份实现」。

这些算子**注入层专用,不入 GP/GFN 搜索产生式**(枚举见 `grammar.LogicalUnaryOp /
LogicalBinaryOp / ParamOp`,与 `TernaryOp` 同待遇):手写 / LLM 写的树可用,搜索器不可采样。
理由同 TernaryOp —— 搜索自由度 = 噪声极值天花板。

复用既有节点 kind('unary' / 'binary' / 'binary_const')⇒ `expression._eval_block` 无需
新分支;而 `grammar.all_productions()` 只遍历 UnaryOp/BinaryOp/ConstBinaryOp/TsOp/PairOp,
⇒ 新枚举不产生 production,搜索空间逐字不变。

`ParamOp` 的标量参数是**连续**的:`ADD_CONSTANTS`/`MULPOW_CONSTANTS` 常数池是为 GP 枚举
产生式而设的离散化,注入层不枚举,离散化只白扔表达力。
"""
from __future__ import annotations

from backtest import ops as _c


def register(unary: dict, binary: dict, binary_const: dict,
             ts: dict, ts_param: dict, pair_param: dict) -> None:
    """把注入层算子挂进 `expression` 的分发表。由 expression 在模块尾调用(避免循环导入)。

    前四张表复用既有节点 kind;`ts_param` / `pair_param` 是本轮新增的两个 kind
    (panel+window+标量 / panel+panel+window+标量),`_eval_block` 已加对应分支。"""
    from evaluation.grammar import (LogicalBinaryOp, LogicalUnaryOp, PairParamOp,
                                    ParamOp, TsInjOp, TsParamOp)

    unary.update({
        LogicalUnaryOp.NOT:    _c.not_,      # x>0 → 0,否则 1;NaN → NaN
        LogicalUnaryOp.IS_NAN: _c.is_nan,    # NaN → 1,否则 0(**唯一不传播 NaN 的**)
    })
    binary.update({
        LogicalBinaryOp.AND:       _c.and_,
        LogicalBinaryOp.OR:        _c.or_,
        LogicalBinaryOp.EQUAL:     _c.equal,
        LogicalBinaryOp.NOT_EQUAL: _c.not_equal,
        LogicalBinaryOp.GE:        _c.greater_equal,
        LogicalBinaryOp.LE:        _c.less_equal,
    })
    binary_const.update({
        ParamOp.LEFT_TAIL:  _c.left_tail,    # >k 全置 NaN
        ParamOp.RIGHT_TAIL: _c.right_tail,   # <k 全置 NaN
        ParamOp.HUMP:       _c.hump,         # 压换手;逐列时序状态机(同 trade_when)
    })
    ts.update({
        TsInjOp.TS_BACKFILL: _c.ts_backfill,   # 回看 d 行取最近有效值(不套 min_periods)
        TsInjOp.TS_MEDIAN:   _c.ts_median,     # 滚动中位数
        TsInjOp.TS_AV_DIFF:  _c.ts_av_diff,    # x − mean(x,d),均值忽略 NaN
    })
    ts_param.update({
        TsParamOp.TS_DECAY_EXP_WINDOW: _c.ts_decay_exp_window,
        TsParamOp.TS_SCALE:            _c.ts_scale,
        TsParamOp.TS_QUANTILE:         _c.ts_quantile,
    })
    pair_param.update({
        PairParamOp.TS_REGRESSION: _c.ts_regression,
    })
