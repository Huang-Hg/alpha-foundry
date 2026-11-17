/* Preserve the isolated-account recurrence's explicit operation order. */
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
#include "perpetual.hpp"
#include <vector>

namespace bt {

void simulate_perpetual(const PerpetualInput& in, const double* weights, long long books,
                        PerpetualOutput out) {
    std::vector<double> q(in.assets), bucket(in.assets), lastmark(in.assets);
    for (long long b = 0; b < books; ++b) {
        PerpetualOutput row {
            out.returns + b * in.measurements, out.funding + b * in.measurements,
            out.trading_fees + b * in.measurements, out.liquidation_fees + b * in.measurements,
            out.isolated_loss_cap + b * in.measurements,
            out.liquidations + b * in.measurements,
            out.liquidation_events + b * in.events * in.assets,
            out.first_liquidation + b, out.bankrupt + b,
            out.ending_equity + b, out.max_gross + b
        };
        run_perpetual(in, weights + b * in.decisions * in.assets, row,
                      q.data(), bucket.data(), lastmark.data());
    }
}

}  // namespace bt
