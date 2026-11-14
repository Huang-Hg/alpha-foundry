// PerpetualSpec fixes ASSETS for this runtime; no truncation or global state.
extern "C" __global__ void perpetual(
    const double* weights, const long long* times, const long long* day,
    const unsigned char* rebalance, const double* trade, const double* open,
    const double* high, const double* low, const double* close,
    const double* funding, const double* funding_mark, const double* maintenance,
    const double* fee, const double* liquidation_fee,
    long long events, long long days, long long books, double leverage,
    double* returns, double* funding_out, double* trading_fees, double* liquidation_fees, double* isolated_loss_cap,
    long long* liquidations, unsigned char* liquidation_events,
    long long* first_liquidation, unsigned char* bankrupt, double* ending_equity, double* max_gross,
    const unsigned char* uncertain,const double* stats,double metric_atol) {
    const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= books) return;
    // Explicit numerical refinement: ill-conditioned Sharpe or a risk boundary
    // uses the original scalar FP64 operation order. No host synchronization.
    if (!uncertain[b] && !(fabs(stats[b*5+1])>metric_atol && stats[b*5+2]<=metric_atol)) return;
    bt::PerpetualInput in {times, day, rebalance, trade, open, high, low, close,
                          funding, funding_mark, maintenance, fee, liquidation_fee,
                          events, days, ASSETS, leverage};
    bt::PerpetualOutput out {
        returns + b * days, funding_out + b * days, trading_fees + b * days,
        liquidation_fees + b * days, isolated_loss_cap + b * days, liquidations + b * days,
        liquidation_events + b * events * ASSETS, first_liquidation + b,
        bankrupt + b, ending_equity + b, max_gross + b
    };
    double q[ASSETS], bucket[ASSETS], lastmark[ASSETS];
    for(long long t=0;t<days;t++) {
        out.returns[t]=out.funding[t]=out.trading_fees[t]=out.liquidation_fees[t]=out.isolated_loss_cap[t]=0.;
        out.liquidations[t]=0;
    }
    for(long long j=0;j<events*ASSETS;j++) out.liquidation_events[j]=0;
    bt::run_perpetual(in, weights + b * days * ASSETS, out, q, bucket, lastmark);
}

// The CPU contract uses population variance and a peak including initial NAV=1.
// Wealth is scanned in time order, including ruin; nonfinite input stays visible.
extern "C" __global__ void perpetual_statistics(const double* returns, long long days,
    long long books, long long start, double ppy, double* statistics) {
    const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= books) return;
    const double* r = returns + b * days;
    const double n = (double)(days - start);
    double mean = 0.0;
    for (long long t = start; t < days; ++t) mean += r[t];
    mean /= n;
    double variance = 0.0, wealth = 1.0, peak = 1.0, drawdown = 0.0;
    for (long long t = start; t < days; ++t) {
        const double delta = r[t] - mean;
        variance += delta * delta;
        wealth *= 1.0 + r[t];
        peak = isnan(wealth) || isnan(peak) ? nan("") : (wealth > peak ? wealth : peak);
        const double depth = wealth / peak - 1.0;
        drawdown = isnan(depth) || isnan(drawdown) ? nan("") : (depth < drawdown ? depth : drawdown);
    }
    const double sd = sqrt(variance / n), annual_root = sqrt(ppy);
    double* out = statistics + b * 5;
    out[0] = sd > 0.0 ? mean / sd * annual_root : 0.0;
    out[1] = mean * ppy;
    out[2] = sd * annual_root;
    out[3] = drawdown;
    out[4] = wealth - 1.0;
}

extern "C" __global__ void perpetual_fingerprints(const double* weights,
    unsigned long long* out, long long count) {
    __shared__ unsigned long long hashes[256];
    unsigned long long h = 0;
    for (long long j = threadIdx.x; j < count; j += blockDim.x) {
        const double v = weights[(long long)blockIdx.x * count + j];
        unsigned long long x = v == 0.0 ? 0ULL : __double_as_longlong(v);
        x ^= (unsigned long long)(j + 1) * 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        h ^= x ^ (x >> 31);
    }
    hashes[threadIdx.x] = h;
    __syncthreads();
    for (int offset = 128; offset; offset >>= 1) {
        if (threadIdx.x < offset) hashes[threadIdx.x] ^= hashes[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = hashes[0];
}
