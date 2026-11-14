/* foundry/backtest/src/kernel.hpp — market-agnostic daily execution kernel (CPU reference).
 *
 * Daily execution semantics:
 * yesterday's weights -> rebalance to target notional at today's open (cost per traded share) -> mark to market at
 * close. Three generalisations let the same recurrence cover multiple markets:
 *   cost_sell  separate per-share cost for sells (share reductions, e.g. sell-only stamp duty); nullptr = symmetric
 *              with cost_buy.
 *   block      direction freeze bits: bit 0 forbids increasing shares today, bit 1 forbids decreasing shares today
 *              (limit-up/down; suspension = 3). Direction is first tested against the all-tradable target notional
 *              eq * wt / max(sum|wt|, 1) vs current notional; if hit, the asset does not trade today, its notional is
 *              deducted from operable equity and its weight leaves the allocation, while its shares carry overnight
 *              (unlike live == 0, which zeroes shares after tonight's mark). Freezing shifts the allocation ratio for
 *              remaining assets: if another restricted asset would then trade in its forbidden direction, it is frozen
 *              too and recomputed until closure — no restricted asset ever trades in a forbidden direction.
 *   carry      daily short financing rate: sum_{shares < 0} carry * close * |shares| is deducted from daily equity and
 *              recorded in the recurrent state so it persists across days.
 * When all three are nullptr, the symmetric execution operation order is preserved; summary statistics
 * (Sharpe, drawdown, etc.) are computed uniformly from rr by the caller.
 *
 * Relies on IEEE NaN propagation (NaN price -> weight zeroed); build must retain -fno-finite-math-only.
 */
#pragma once
#include <cstdint>

namespace bt {

/* All panels are (T,S) C-contiguous, row = trading day:
 *   w           target weights; row t executes at day t+1 open (non-finite treated as 0)
 *   open_raw    raw open price (NaN = missing); prev_close = prior close; their ratio adjusts weights to the open
 *   open_ff0    open price forward-filled with remaining NaNs set to 0, close_ff0 likewise — pricing basis for shares,
 *               mark-to-market, and costs
 *   cost_buy    per-share cost (in price units); cost_sell likewise for share reductions only, may be nullptr
 *   live        non-zero = tradable today; 0 = untradable, notional locked, shares zeroed after tonight's close
 *   block       direction freeze bits, may be nullptr; carry = daily short financing rate, may be nullptr
 * rr_out: (T,) daily returns, caller-allocated. */
void simulate(const double* w,
              const double* open_raw, const double* prev_close,
              const double* open_ff0, const double* close_ff0,
              const double* cost_buy, const double* cost_sell,
              const uint8_t* live, const uint8_t* block, const double* carry,
              int64_t T, int64_t S, double* rr_out);

}  // namespace bt
