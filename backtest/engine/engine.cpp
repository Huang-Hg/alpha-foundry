/* backtest/engine/engine.cpp —— 市场无关组合回测引擎核心。
 *
 * bar 循环:市场专属块(成本 / carry)为 MarketPolicy 回调:
 *   (a) MTM open gap → cash 底线 → (b) rebalance(policy.trade_cost)→ (c) MTM 余 →
 *   (d) policy.carry。
 * (旧 perp 专属块 isolated·cross 强平 / trailing-stop / stop_scan 随 1h crypto 栈
 *  2026-07-25 删除。)
 */
#include "engine.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bt {

namespace {

/* 持久 scratch(thread_local std::vector:防并发 BT race + RAII;resize 保 capacity → grow-only 复用)。 */
thread_local std::vector<double>  g_weight_TS, g_pos, g_target;

/* step 1: alpha → weight 行内变换(市场无关:cs_zscore → clip(±3) → demean → L1 cap)。 */
inline void row_signal_to_weight(const double* a, double* w, int64_t S, double cap) {
    double sum = 0.0, sumsq = 0.0;
    int64_t n = 0;
    for (int64_t s = 0; s < S; s++) {
        double v = a[s];
        if (std::isfinite(v)) { sum += v; sumsq += v * v; n++; }
    }
    if (n < 2) { for (int64_t s = 0; s < S; s++) w[s] = 0.0; return; }
    double m   = sum / (double)n;
    double var = (sumsq - (double)n * m * m) / (double)(n - 1);
    double sd  = (var > 0.0) ? std::sqrt(var) : 0.0;
    if (sd < 1e-12) { for (int64_t s = 0; s < S; s++) w[s] = 0.0; return; }
    double inv_sd = 1.0 / sd;

    double clip_sum = 0.0;
    for (int64_t s = 0; s < S; s++) {
        double v = a[s];
        double z = std::isfinite(v) ? (v - m) * inv_sd : 0.0;
        if (z >  3.0) z =  3.0;
        if (z < -3.0) z = -3.0;
        w[s] = z;
        clip_sum += z;
    }
    const double clip_mean = clip_sum / (double)n;
    double l1 = 0.0;
    for (int64_t s = 0; s < S; s++) {
        double v = a[s];
        if (!std::isfinite(v)) { w[s] = 0.0; continue; }
        w[s] -= clip_mean;
        l1   += std::fabs(w[s]);
    }
    double inv_l1 = (l1 > cap && cap > 0.0) ? (cap / l1) : 1.0;
    for (int64_t s = 0; s < S; s++) w[s] *= inv_l1;
}

} // anonymous namespace


void run_portfolio(
    const double* alpha, const double* close_TS, const double* open_TS,
    const int8_t* trade_block,
    int64_t T, int64_t S,
    double initial_cash, int64_t skip_warmup,
    double leverage_cap, int raw_weights,
    const MarketPolicy& policy,
    double* equity, double* turnover, double* cost_drag, int64_t* liq_at_out)
{
    g_weight_TS.resize((size_t)T * (size_t)S);
    g_pos.resize(S); g_target.resize(S);
    double*  weight_TS = g_weight_TS.data();
    double*  pos       = g_pos.data();
    double*  target    = g_target.data();

    /* === Step 1: alpha → weight,跨 t 并行 === */
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double* w_row = &weight_TS[t * S];
        if (t < skip_warmup) {
            for (int64_t s = 0; s < S; s++) w_row[s] = 0.0;
        } else if (raw_weights) {
            const double* a_row = &alpha[t * S];
            for (int64_t s = 0; s < S; s++) {
                double v = a_row[s];
                w_row[s] = std::isfinite(v) ? v : 0.0;
            }
        } else {
            row_signal_to_weight(&alpha[t * S], w_row, S, leverage_cap);
        }
    }

    /* === Step 2: 组合 bt,串行 T,inner SIMD over S === */
    for (int64_t s = 0; s < S; s++) { pos[s] = 0.0; target[s] = 0.0; }

    double cash = initial_cash;
    *liq_at_out = -1;
    equity[0] = cash; turnover[0] = 0.0; cost_drag[0] = 0.0;

    for (int64_t t = 1; t < T; t++) {
        const double* p_open  = &open_TS[t * S];
        const double* p_close = &close_TS[t * S];
        const double* p_prev  = &close_TS[(t - 1) * S];
        const double* w_lag   = &weight_TS[(t - 1) * S];

        /* (a) MTM 跨 open gap */
        for (int64_t s = 0; s < S; s++) cash += pos[s] * (p_open[s] - p_prev[s]);

        if (cash <= 0.0) {                                  /* cash 底线(市场无关硬停)*/
            cash = 0.0;
            for (int64_t s = 0; s < S; s++) pos[s] = 0.0;
            *liq_at_out = t;
            equity[t] = 0.0; turnover[t] = 0.0; cost_drag[t] = 0.0;
            for (int64_t j = t + 1; j < T; j++) { equity[j] = 0.0; turnover[j] = 0.0; cost_drag[j] = 0.0; }
            return;
        }

        /* (b) rebal + 执行:单 bar 直达 target(成本经 policy)*/
        const double equity_pre_trade = cash;
        double notional_total = 0.0;
        double cost_total     = 0.0;

        for (int64_t s = 0; s < S; s++) {
            double op = p_open[s];
            target[s] = (op > 0.0) ? (w_lag[s] * equity_pre_trade / op) : 0.0;
        }
        const int8_t* tb_row = trade_block ? &trade_block[t * S] : nullptr;
        for (int64_t s = 0; s < S; s++) {
            double op = p_open[s];
            if (!(op > 0.0)) continue;
            double tgt = target[s];
            if (tb_row) {                                   /* 涨跌停/停牌:受限方向冻结仓位(target←pos)*/
                const int8_t b = tb_row[s];
                const double d = tgt - pos[s];
                if (((b & 1) && d > 0.0) || ((b & 2) && d < 0.0)) tgt = pos[s];
            }
            const double delta = tgt - pos[s];
            const double notional = std::fabs(delta) * op;
            if (notional <= 0.0) { pos[s] = tgt; continue; }
            const double cost = policy.trade_cost(notional, delta < 0.0);   /* delta<0=减仓(卖)*/
            cash    -= cost;
            pos[s]   = tgt;
            notional_total += notional;
            cost_total     += cost;
        }
        turnover[t]  = (equity_pre_trade > 0.0) ? notional_total / equity_pre_trade : 0.0;
        cost_drag[t] = cost_total;

        /* (c) MTM 余下 */
        for (int64_t s = 0; s < S; s++) cash += pos[s] * (p_close[s] - p_open[s]);

        /* (d) carry(借券费 / 分红)*/
        policy.carry(pos, p_close, S, cash);

        equity[t] = cash;
    }
}

} // namespace bt
