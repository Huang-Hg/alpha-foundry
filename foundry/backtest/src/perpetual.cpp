/* Preserve explicit operation order without altering the established cash ABI. */
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
#include "perpetual.hpp"
#include <vector>

namespace bt {

void simulate_perpetual(const PerpetualInput& in, const double* weights, long long books,
                        PerpetualOutput out) {
    std::vector<double> q(in.assets), bucket(in.assets), lastmark(in.assets);
    for (long long b = 0; b < books; ++b) {
        PerpetualOutput row {
            out.returns + b * in.days, out.funding + b * in.days,
            out.trading_fees + b * in.days, out.liquidation_fees + b * in.days,
            out.isolated_loss_cap + b * in.days,
            out.liquidations + b * in.days,
            out.liquidation_events + b * in.events * in.assets,
            out.first_liquidation + b, out.bankrupt + b,
            out.ending_equity + b, out.max_gross + b
        };
        run_perpetual(in, weights + b * in.days * in.assets, row,
                      q.data(), bucket.data(), lastmark.data());
    }
}

}  // namespace bt
