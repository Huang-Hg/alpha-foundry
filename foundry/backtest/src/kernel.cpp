/* foundry/backtest/src/kernel.cpp — daily execution kernel; see kernel.hpp for semantics.
 *
 * Optional inputs expand into template bools by presence: unused generalisations vanish at compile time, leaving the
 * base instantiation's loops identical to the qnt reproduction verbatim. That is required for bitwise agreement —
 * -ffast-math lets the vectoriser reassociate reductions depending on loop shape and compiler version, so the
 * generalisations add separate loops or elementwise selects without altering the base reduction loops. Bitwise parity
 * holds within the same toolchain (across GCC versions or vs numpy pairwise-sum, differences are ~1e-16). */
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

        /* --- Effective weights for today: shift-1 -> normalize -> open adjustment -> normalize --- */
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
                double v = (wt[a] / s1) * o_raw[a] / p_cl[a];   // NaN price -> NaN -> 0 (including 0 * NaN)
                if (!std::isfinite(v)) v = 0.0;
                wt[a] = v;
                s2 += std::fabs(v);
            }
            if (s2 < 1.0) s2 = 1.0;
            for (int64_t a = 0; a < S; a++) wt[a] /= s2;
        }

        /* --- Equity recurrence (line-by-line match with calc_relative_return_np) --- */
        double eq_before;
        if (day == 0) {
            eq_before = 1.0;
        } else {
            const double* o_prev = open_ff0 + (day - 1) * S;
            double mtm = 0.0;
            for (int64_t a = 0; a < S; a++) mtm += (o[a] - o_prev[a]) * shares[a];
            eq_before = eq_after + mtm;
        }

        /* Freeze check precedes allocation: frozen assets leave today's allocation just like live == 0 (notional
         * locked, weight excluded), except their shares are not zeroed after close. Pass 1 tests direction against
         * target notional assuming all assets are tradable. */
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
                /* Freezing alters the operable equity-to-weight ratio, so a restricted asset still marked tradable
                 * could now cross its current position into the forbidden direction. Freeze those too and recompute
                 * until none does (the frozen set grows monotonically, at most S rounds; 1 round when unrestricted). */
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
            /* Deducted from the recurrent state eq_after (alongside trading costs) so tonight's equity drops
             * permanently; deducting from eq_tonight alone would be refunded in tomorrow's rr because tomorrow's
             * eq_before starts from eq_after. */
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
            if (!ul[a]) shares[a] = 0.0;                       // zero untradable shares (after tonight's mark)
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
