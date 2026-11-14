/* One scalar recurrence compiled for both the native CPU and CUDA executors.
 * Inputs have passed the data boundary. Stages: gap, funding, daily rebalance,
 * adverse-mark touch, close. No cash-market normalization or financing rules. */
#pragma once

#ifdef __CUDACC__
#define PERP_HD __host__ __device__
#else
#define PERP_HD
#endif

namespace bt {

struct PerpetualInput {
    const long long* times;
    const long long* day;
    const unsigned char* rebalance;
    const double *trade, *open, *high, *low, *close;
    const double *funding, *funding_mark, *maintenance, *fee, *liquidation_fee;
    long long events, days, assets;
    double leverage;
};

struct PerpetualOutput {
    double *returns, *funding, *trading_fees, *liquidation_fees, *isolated_loss_cap;
    long long* liquidations;
    unsigned char* liquidation_events;
    long long* first_liquidation;
    unsigned char* bankrupt;
    double *ending_equity, *max_gross;
};

PERP_HD inline double perp_abs(double x) { return x < 0.0 ? -x : x; }
PERP_HD inline double perp_max(double x, double y) { return x > y ? x : y; }
PERP_HD inline double perp_min(double x, double y) { return x < y ? x : y; }

PERP_HD inline double perp_equity(double cash, const double* bucket, long long assets) {
    for (long long a = 0; a < assets; ++a) cash += bucket[a];
    return cash;
}

PERP_HD inline void perp_exposure(const double* q, const double* bucket, const double* mark,
                                 double cash, long long assets, double* maximum) {
    double gross = 0.0;
    for (long long a = 0; a < assets; ++a) gross += perp_abs(q[a]) * mark[a];
    const double equity = perp_equity(cash, bucket, assets);
    const double ratio = gross == 0.0 ? 0.0 : gross / (equity > 0.0 ? equity : 0.0);
    *maximum = perp_max(*maximum, ratio);
}

PERP_HD inline void perp_liquidate(const PerpetualInput& in, PerpetualOutput& out,
                                 long long event, long long asset, unsigned char stage,
                                 double price, double equity, double* q, double* bucket,
                                 double& cash) {
    const long long k = event * in.assets + asset, day = in.day[event];
    const double nominal_fee = in.liquidation_fee[k] * perp_abs(q[asset]) * price;
    const double available = perp_max(equity, 0.0);
    const double paid_fee = perp_min(available, nominal_fee);
    cash += perp_max(equity - nominal_fee, 0.0);
    out.liquidation_fees[day] += paid_fee;
    out.isolated_loss_cap[day] += perp_max(-equity, 0.0);
    out.liquidations[day] += 1;
    out.liquidation_events[k] |= stage;
    if (*out.first_liquidation < 0) *out.first_liquidation = in.times[event];
    q[asset] = 0.0;
    bucket[asset] = 0.0;
}

PERP_HD inline void run_perpetual(const PerpetualInput& in, const double* weights,
                                 PerpetualOutput out, double* q, double* bucket, double* lastmark) {
    const long long S = in.assets;
    double cash = 1.0, previous_nav = 1.0;
    bool bankrupt = false;
    *out.first_liquidation = -1;
    *out.max_gross = 0.0;
    for (long long a = 0; a < S; ++a) {
        q[a] = 0.0; bucket[a] = 0.0; lastmark[a] = in.open[a];
    }
    for (long long event = 0; event < in.events; ++event) {
        const long long offset = event * S, day = in.day[event];
        const double* mark = in.open + offset;
        if (!bankrupt) {
            for (long long a = 0; a < S; ++a)
                bucket[a] += q[a] * (mark[a] - lastmark[a]);
            perp_exposure(q, bucket, mark, cash, S, out.max_gross);
            for (long long a = 0; a < S; ++a) {
                const long long k = offset + a;
                if (q[a] != 0.0 && bucket[a] <= in.maintenance[k] * perp_abs(q[a]) * mark[a])
                    perp_liquidate(in, out, event, a, 1, mark[a], bucket[a], q, bucket, cash);
            }
            for (long long a = 0; a < S; ++a) {
                const long long k = offset + a;
                const double flow = -q[a] * in.funding_mark[k] * in.funding[k];
                bucket[a] += flow;
                out.funding[day] += flow;
                if (q[a] != 0.0 && bucket[a] <= in.maintenance[k] * perp_abs(q[a]) * mark[a])
                    perp_liquidate(in, out, event, a, 2, mark[a], bucket[a], q, bucket, cash);
            }
            double equity = perp_equity(cash, bucket, S);
            if (equity <= 0.0) bankrupt = true;
            if (!bankrupt && in.rebalance[event] && day > 0) {
                const double* w = weights + (day - 1) * S;
                double A = equity, C = 0.0, fees_zero = 0.0;
                for (long long a = 0; a < S; ++a) {
                    const long long k = offset + a;
                    A += q[a] * (in.trade[k] - mark[a]);
                    C += w[a] * (1.0 - in.trade[k] / mark[a]);
                    fees_zero += in.fee[k] * perp_abs(q[a]) * in.trade[k];
                }
                if (A - fees_zero <= 0.0) {
                    out.trading_fees[day] += perp_min(perp_max(A, 0.0), fees_zero);
                    bankrupt = true;
                } else {
                    double lo = 0.0, hi = A / (1.0 - C);
                    for (int iteration = 0; iteration < 64; ++iteration) {
                        const double mid = (lo + hi) * 0.5;
                        double fees = 0.0;
                        for (long long a = 0; a < S; ++a) {
                            const long long k = offset + a;
                            const double next = w[a] * mid / mark[a];
                            fees += in.fee[k] * perp_abs(next - q[a]) * in.trade[k];
                        }
                        const double f = mid * (1.0 - C) - A + fees;
                        if (f > 0.0) hi = mid; else lo = mid;
                    }
                    const double nav = (lo + hi) * 0.5;
                    double collateral = 0.0, fees = 0.0;
                    for (long long a = 0; a < S; ++a) {
                        const long long k = offset + a;
                        const double next = w[a] * nav / mark[a];
                        fees += in.fee[k] * perp_abs(next - q[a]) * in.trade[k];
                        q[a] = next;
                        bucket[a] = perp_abs(w[a]) / in.leverage * nav;
                        collateral += bucket[a];
                    }
                    cash = nav - collateral;
                    out.trading_fees[day] += fees;
                    perp_exposure(q, bucket, mark, cash, S, out.max_gross);
                    for (long long a = 0; a < S; ++a) {
                        const long long k = offset + a;
                        if (q[a] != 0.0 && bucket[a] <= in.maintenance[k] * perp_abs(q[a]) * mark[a])
                            perp_liquidate(in, out, event, a, 4, mark[a], bucket[a], q, bucket, cash);
                    }
                }
            }
            if (!bankrupt) {
                for (long long a = 0; a < S; ++a) {
                    const long long k = offset + a;
                    const double bad = q[a] > 0.0 ? in.low[k] : in.high[k];
                    const double adverse_equity = bucket[a] + q[a] * (bad - mark[a]);
                    if (q[a] != 0.0 && adverse_equity <= in.maintenance[k] * perp_abs(q[a]) * bad) {
                        const double price = (q[a] * mark[a] - bucket[a])
                                             / (q[a] - in.maintenance[k] * perp_abs(q[a]));
                        const double at_touch = bucket[a] + q[a] * (price - mark[a]);
                        perp_liquidate(in, out, event, a, 8, price, at_touch, q, bucket, cash);
                    }
                    bucket[a] += q[a] * (in.close[k] - mark[a]);
                    lastmark[a] = in.close[k];
                }
                perp_exposure(q, bucket, in.close + offset, cash, S, out.max_gross);
                if (perp_equity(cash, bucket, S) <= 0.0) bankrupt = true;
            }
            if (bankrupt) {
                cash = 0.0;
                for (long long a = 0; a < S; ++a) { q[a] = 0.0; bucket[a] = 0.0; }
            }
        }
        if (event + 1 == in.events || in.day[event + 1] != day) {
            const double nav = bankrupt ? 0.0 : perp_equity(cash, bucket, S);
            out.returns[day] = previous_nav > 0.0 ? nav / previous_nav - 1.0 : 0.0;
            previous_nav = nav;
        }
    }
    *out.bankrupt = bankrupt;
    *out.ending_equity = previous_nav;
}

#ifndef __CUDACC__
void simulate_perpetual(const PerpetualInput& in, const double* weights, long long books,
                        PerpetualOutput out);
#endif

}  // namespace bt
