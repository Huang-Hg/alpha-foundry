/* foundry/backtest/src/kernel.cpp —— 日频执行内核,语义见 kernel.hpp。
 *
 * 可选输入按"是否提供"展开成模板实参:未提供的推广在编译期整段消失,基底实例的循环与 qnt 复刻逐字相同。
 * 这是逐位一致的前提 —— -ffast-math 允许向量化器重排归约的结合顺序,顺序随循环形状与编译器版本而变;
 * 故推广部分只加独立的循环或逐元素的选择,不改动基底归约循环的形状。逐位比对只在同一工具链内成立
 * (跨 GCC 版本、或与 numpy pairwise-sum 相比,差异在 1e-16 级)。 */
#include "kernel.hpp"

#include <cmath>
#include <vector>

namespace bt {

static constexpr double QNT_EPS = 1e-7;   // qnt.stats.EPS = 10**-7

template <bool SELL, bool BLOCK, bool CARRY>
static void run(const double* w,
                const double* open_raw, const double* prev_close,
                const double* open_ff0, const double* close_ff0,
                const double* cost_buy, const double* cost_sell,
                const uint8_t* live, const uint8_t* block, const double* carry,
                int64_t T, int64_t S, double* rr_out) {
    std::vector<double> wt(S), shares(S, 0.0);
    std::vector<uint8_t> tradable(BLOCK ? S : 0);

    double eq_after = 0.0;
    double prev_tonight = 1.0;             // Ep[0] = 1
    for (int64_t day = 0; day < T; day++) {
        const double*  o_raw = open_raw   + day * S;
        const double*  p_cl  = prev_close + day * S;
        const double*  o     = open_ff0   + day * S;
        const double*  c     = close_ff0  + day * S;
        const double*  sl    = cost_buy   + day * S;
        const uint8_t* ul    = live       + day * S;

        /* --- 当日生效权重:shift-1 → normalize → open 调整 → normalize --- */
        if (day == 0) {
            for (int64_t a = 0; a < S; a++) wt[a] = 0.0;
        } else {
            const double* wrow = w + (day - 1) * S;
            double s1 = 0.0;
            for (int64_t a = 0; a < S; a++) {
                double x = wrow[a];
                if (!std::isfinite(x)) x = 0.0;
                wt[a] = x;
                s1 += std::fabs(x);
            }
            if (s1 < 1.0) s1 = 1.0;
            double s2 = 0.0;
            for (int64_t a = 0; a < S; a++) {
                double v = (wt[a] / s1) * o_raw[a] / p_cl[a];   // NaN 价 → NaN → 0(含 0×NaN)
                if (!std::isfinite(v)) v = 0.0;
                wt[a] = v;
                s2 += std::fabs(v);
            }
            if (s2 < 1.0) s2 = 1.0;
            for (int64_t a = 0; a < S; a++) wt[a] /= s2;
        }

        /* --- equity 递推(calc_relative_return_np 逐行对照)--- */
        double eq_before;
        if (day == 0) {
            eq_before = 1.0;
        } else {
            const double* o_prev = open_ff0 + (day - 1) * S;
            double mtm = 0.0;
            for (int64_t a = 0; a < S; a++) mtm += (o[a] - o_prev[a]) * shares[a];
            eq_before = eq_after + mtm;
        }

        /* 冻结判定在分配之前:被冻结的资产与 live==0 一样退出当日分配(市值锁定、权重不参与),
         * 只是份额不清零。首轮按"全部可交易"时的目标市值判方向。 */
        const uint8_t* tr = ul;
        if constexpr (BLOCK) {
            const uint8_t* bk = block + day * S;
            double w_all = 0.0;
            for (int64_t a = 0; a < S; a++) w_all += std::fabs(wt[a]);
            const double w_den = w_all > 1.0 ? w_all : 1.0;
            for (int64_t a = 0; a < S; a++) {
                const double target = eq_before * wt[a] / w_den;
                const double cur = o[a] * shares[a];
                const bool frozen = ((bk[a] & 1) && target > cur) || ((bk[a] & 2) && target < cur);
                tradable[a] = ul[a] && !frozen;
            }
            tr = tradable.data();
        }

        double w_sum, w_unlocked, locked, w_operable, eq_operable;
        for (;;) {
            w_sum = 0.0; w_unlocked = 0.0; locked = 0.0;
            for (int64_t a = 0; a < S; a++) {
                const double aw = std::fabs(wt[a]);
                w_sum += aw;
                if (tr[a]) w_unlocked += aw;
                else       locked += o[a] * std::fabs(shares[a]);
            }
            const double w_free_cash = (w_sum > 1.0 ? w_sum : 1.0) - w_sum;
            w_operable  = w_unlocked + w_free_cash;
            eq_operable = eq_before - locked;
            if constexpr (!BLOCK) {
                break;
            } else {
                /* 冻结改变了可操作权益与权重之比,仍可交易的受限资产的实际分配可能越过当前持仓、
                 * 朝被禁方向成交。这些资产同样冻结后重算,直到无人朝被禁方向成交
                 * (冻结集单调增长,至多 S 轮;无受限资产时一轮即出)。 */
                if (w_operable < QNT_EPS) break;
                const uint8_t* bk = block + day * S;
                bool refrozen = false;
                for (int64_t a = 0; a < S; a++) {
                    if (!tradable[a] || !bk[a]) continue;
                    const double ns = eq_operable * wt[a] / (w_operable * o[a]);
                    if (((bk[a] & 1) && ns > shares[a]) || ((bk[a] & 2) && ns < shares[a])) {
                        tradable[a] = 0;
                        refrozen = true;
                    }
                }
                if (!refrozen) break;
            }
        }

        if (w_operable < QNT_EPS) {
            eq_after = eq_before;
        } else {
            double step_slip = 0.0;
            for (int64_t a = 0; a < S; a++) {
                if (!tr[a]) continue;
                const double ns = eq_operable * wt[a] / (w_operable * o[a]);
                const double dn = (day > 0) ? ns - shares[a] : ns;
                if constexpr (SELL) step_slip += (dn < 0.0 ? cost_sell[day * S + a] : sl[a]) * std::fabs(dn);
                else                step_slip += sl[a] * std::fabs(dn);
                shares[a] = ns;
            }
            eq_after = eq_before - step_slip;
        }

        double mtm2 = 0.0;
        for (int64_t a = 0; a < S; a++) mtm2 += (c[a] - o[a]) * shares[a];
        if constexpr (CARRY) {
            /* 记入逐日递推的状态量 eq_after(与成交成本同处),当晚权益随之扣减且持续;
             * 只扣 eq_tonight 的话,次日 eq_before 从 eq_after 起算,这笔成本会在次日 rr 里原样退回。 */
            const double* cr = carry + day * S;
            double financing = 0.0;
            for (int64_t a = 0; a < S; a++)
                if (shares[a] < 0.0) financing += cr[a] * c[a] * -shares[a];
            eq_after -= financing;
        }
        const double eq_tonight = eq_after + mtm2;

        double rr = eq_tonight / prev_tonight - 1.0;
        if (!std::isfinite(rr)) rr = 0.0;
        rr_out[day] = rr;
        prev_tonight = eq_tonight;

        for (int64_t a = 0; a < S; a++)
            if (!ul[a]) shares[a] = 0.0;                       // 不可交易资产份额清零(tonight 之后)
    }
}

void simulate(const double* w,
              const double* open_raw, const double* prev_close,
              const double* open_ff0, const double* close_ff0,
              const double* cost_buy, const double* cost_sell,
              const uint8_t* live, const uint8_t* block, const double* carry,
              int64_t T, int64_t S, double* rr_out) {
    using Run = decltype(&run<false, false, false>);
    static constexpr Run variants[8] = {
        run<false, false, false>, run<true, false, false>, run<false, true, false>, run<true, true, false>,
        run<false, false, true>,  run<true, false, true>,  run<false, true, true>,  run<true, true, true>,
    };
    const int k = (cost_sell != nullptr) | (block != nullptr) << 1 | (carry != nullptr) << 2;
    variants[k](w, open_raw, prev_close, open_ff0, close_ff0, cost_buy, cost_sell, live, block, carry,
                T, S, rr_out);
}

}  // namespace bt
