/* backtest/engine/qnt_bt.hpp —— Quantiacs qnt 组合回测 1:1 复刻(独立于 run_portfolio 引擎)。
 *
 * 语义 = qnt 0.0.507 stats.calc_relative_return(per_asset=False)+ cryptodaily 口径统计
 * (arithmetic 年化 Sharpe / std ddof=0 / ppy=365);规格冻结件 scripts/qnt_bt_ref.py
 * (vs qnt 真实数据逐位 ~1e-15 对账)。递推依赖 IEEE NaN 传染(0×NaN=NaN→归零),
 * 构建必须保持 -fno-finite-math-only(backtest/build.sh 已带)。
 *
 * 权重无关静态量(ATR14×fract 滑点 / ffill 价 / prev_close / unlocked)由 Python 侧
 * rl.backtest_reward.prepare_qnt_static 一次算好传入;内核只跑权重相关热路径,
 * 每候选 O(T·S) —— 供公式挖矿 reward 循环毫秒级评估。
 */
#pragma once
#include <cstdint>

namespace bt {

struct QntStats {
    double sharpe;        // arithmetic 年化:mean·ppy / (std(ddof=0)·√ppy),窗 [i0:]
    double mean_return;   // 简单年化:mean·ppy
    double volatility;    // std(ddof=0)·√ppy
    double max_drawdown;  // 窗 [i0:](equity 基 1 重锚)水下最小值(≤0)
    double total_return;  // Π(1+rr[i0:]) − 1
};

/* w / open_raw / prev_close / open_ff0 / close_ff0 / slip:(T,S) C-连续 f64;
 * unlocked:(T,S) uint8(非 0 = 可交易);i0:统计窗起点(竞赛口径 = 2016-01-01 的行号)。
 * rr_out / eq_out:(T,) 调用方分配 —— 全网格 RR 与 equity=cumprod(1+rr);统计只看 [i0:]。
 * 注意:与 qnt 全等的前提是权重在 [i0:] 之前为 0(竞赛 checker 流);否则 [<i0] 段
 * 持仓会改变递推状态,qnt 侧 maxDD/equity 锚点语义随权重网格起点漂移,不在本内核复刻范围。 */
QntStats qnt_bt(const double* w,
                const double* open_raw, const double* prev_close,
                const double* open_ff0, const double* close_ff0,
                const double* slip, const uint8_t* unlocked,
                int64_t T, int64_t S, int64_t i0, double ppy,
                double* rr_out, double* eq_out);

}  // namespace bt
