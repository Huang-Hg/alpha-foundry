/* backtest/engine/qnt_bt.cpp —— qnt calc_relative_return_np 递推 + cryptodaily 统计。
 * 逐步对照 scripts/qnt_bt_ref.py(= qnt 0.0.507 已对账的规格冻结件);求和顺序保持
 * 逐 asset 顺序累加(numpy pairwise-sum 的树形差异在 1e-16 级,由 golden 门收口)。 */
#include "qnt_bt.hpp"

#include <cmath>
#include <vector>

namespace bt {

static constexpr double QNT_EPS = 1e-7;   // qnt.stats.EPS = 10**-7

QntStats qnt_bt(const double* w,
                const double* open_raw, const double* prev_close,
                const double* open_ff0, const double* close_ff0,
                const double* slip, const uint8_t* unlocked,
                int64_t T, int64_t S, int64_t i0, double ppy,
                double* rr_out, double* eq_out) {
    std::vector<double> wt(S), shares(S, 0.0);

    double eq_after = 0.0;
    double prev_tonight = 1.0;             // Ep[0] = 1
    for (int64_t day = 0; day < T; day++) {
        const double*  o_raw = open_raw   + day * S;
        const double*  p_cl  = prev_close + day * S;
        const double*  o     = open_ff0   + day * S;
        const double*  c     = close_ff0  + day * S;
        const double*  sl    = slip       + day * S;
        const uint8_t* ul    = unlocked   + day * S;

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

        double w_sum = 0.0, w_unlocked = 0.0, locked = 0.0;
        for (int64_t a = 0; a < S; a++) {
            const double aw = std::fabs(wt[a]);
            w_sum += aw;
            if (ul[a]) w_unlocked += aw;
            else       locked += o[a] * std::fabs(shares[a]);
        }
        const double w_free_cash = (w_sum > 1.0 ? w_sum : 1.0) - w_sum;
        const double w_operable  = w_unlocked + w_free_cash;
        const double eq_operable = eq_before - locked;

        if (w_operable < QNT_EPS) {
            eq_after = eq_before;
        } else {
            double step_slip = 0.0;
            for (int64_t a = 0; a < S; a++) {
                if (!ul[a]) continue;
                const double ns = eq_operable * wt[a] / (w_operable * o[a]);
                const double dn = (day > 0) ? ns - shares[a] : ns;
                step_slip += sl[a] * std::fabs(dn);
                shares[a] = ns;
            }
            eq_after = eq_before - step_slip;
        }

        double mtm2 = 0.0;
        for (int64_t a = 0; a < S; a++) mtm2 += (c[a] - o[a]) * shares[a];
        const double eq_tonight = eq_after + mtm2;

        double rr = eq_tonight / prev_tonight - 1.0;
        if (!std::isfinite(rr)) rr = 0.0;
        rr_out[day] = rr;
        prev_tonight = eq_tonight;

        for (int64_t a = 0; a < S; a++)
            if (!ul[a]) shares[a] = 0.0;                       // 无价资产份额清零(tonight 之后)
    }

    /* equity = cumprod(1+rr)(全网格,qnt calc_equity 口径) */
    double cum = 1.0;
    for (int64_t t = 0; t < T; t++) { cum *= 1.0 + rr_out[t]; eq_out[t] = cum; }

    /* --- 统计窗 [i0:]:arithmetic 年化 Sharpe(ddof=0)/ maxDD / 总收益 --- */
    const int64_t n = T - i0;
    double sum = 0.0;
    for (int64_t t = i0; t < T; t++) sum += rr_out[t];
    const double mean = sum / (double)n;
    double ss = 0.0;
    for (int64_t t = i0; t < T; t++) { const double d = rr_out[t] - mean; ss += d * d; }
    const double sd = std::sqrt(ss / (double)n);

    double cum2 = 1.0, peak = 1.0, dd = 0.0;
    for (int64_t t = i0; t < T; t++) {
        cum2 *= 1.0 + rr_out[t];
        if (cum2 > peak) peak = cum2;
        const double u = cum2 / peak - 1.0;
        if (u < dd) dd = u;
    }

    QntStats st;
    st.mean_return  = mean * ppy;
    st.volatility   = sd * std::sqrt(ppy);
    st.sharpe       = st.mean_return / st.volatility;         // sd=0 → ±inf/nan,与 qnt 一致
    st.max_drawdown = dd;
    st.total_return = cum2 - 1.0;
    return st;
}

}  // namespace bt
