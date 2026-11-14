#pragma once
namespace bt {
struct PerpetualOutput {
    double *returns, *funding, *trading_fees, *liquidation_fees, *isolated_loss_cap;
    long long* liquidations;
    unsigned char* liquidation_events;
    long long* first_liquidation;
    unsigned char* bankrupt;
    double *ending_equity, *max_gross;
};
}
