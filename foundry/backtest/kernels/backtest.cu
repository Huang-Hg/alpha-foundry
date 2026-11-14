/* foundry/backtest/kernels/backtest.cu — batched execution kernel (CUDA mirror of foundry/backtest/src/kernel.cpp).
 *
 * Covers the symmetric-cost case (one per-share cost for buys and sells, no direction blocks, no carry).
 * general.cu covers larger cross-sections and asymmetric trading rules.
 * One warp simulates one book: block = 128 threads = 4 books, grid = ceil(B/4). The cross-section (S <= 128) is
 * held in registers, ASSETS_PER_LANE assets per lane; the ~7 reductions a day are warp shuffles, with no shared memory
 * and no __syncthreads. The daily chain is latency-bound (the batch time does not depend on B), hence the
 * prefetching below.
 * W (B,T,S); prices/cost (T,S); live uint8 (T,S). Outputs rr (B,T) f64 and stats (B,5) f64 =
 * [sharpe, mean_return, volatility, max_drawdown, total_return] over rows [i0:], arithmetic annualisation.
 * Recurrence scalars are replicated in every lane (same FP order -> deterministic); accumulators and shares are
 * paired by default; the mixed instantiation uses single components. Relies on IEEE NaN propagation (0*NaN = NaN
 * -> zeroed), like the CPU kernel.
 */
#define ASSETS_PER_LANE 4                        /* assets per lane => S <= 32*4 = 128 */
#define PRICE_EPS 1e-7

/* Four independent sums on one shuffle chain: same latency as one. */
template<typename A>
__device__ __forceinline__ void warp_sum4_bcast(A& a, A& b, A& c, A& d) {
    for (int off = warpSize >> 1; off > 0; off >>= 1) {
        a += __shfl_down_sync(0xffffffff, a, off);
        b += __shfl_down_sync(0xffffffff, b, off);
        c += __shfl_down_sync(0xffffffff, c, off);
        d += __shfl_down_sync(0xffffffff, d, off);
    }
    a = __shfl_sync(0xffffffff, a, 0); b = __shfl_sync(0xffffffff, b, 0);
    c = __shfl_sync(0xffffffff, c, 0); d = __shfl_sync(0xffffffff, d, 0);
}

/* Blended book = (ref + alpha*W) / max(1, exposure): the candidate is ADDED to the portfolio and both shrink only
 * on days the sum is above full exposure. Books are nonnegative, so the day's exposure is
 * eref[day] + alpha*eown[day] from two precomputed row sums - no reduction here. */
__device__ __forceinline__ double blend_add(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double blend_mul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ FloatPair blend_add(FloatPair a, FloatPair b) { return a + b; }
__device__ __forceinline__ FloatPair blend_mul(FloatPair a, FloatPair b) { return a * b; }

template<bool MIX, typename C>
__device__ __forceinline__ C blend_den(const C* eref, const C* eown, int day, C alpha) {
    if (!MIX) return C(1);
    const C e = blend_add(eref[day], blend_mul(alpha, eown[day]));
    return e > C(1) ? e : C(1);
}

template<typename T, bool MIX, typename C>
__device__ __forceinline__ T blend_weight(const T* W, const T* ref, long long i, C alpha, C den) {
    if (!MIX) return W[i];
    // Keep the materialised expression's separate multiply/add/divide rounding.
    return (T)(blend_add((C)ref[i], blend_mul(alpha, (C)W[i])) / den);
}

// Explicit FP32 blending keeps double arithmetic out of the low-precision daily path.
template<>
__device__ __forceinline__ float blend_weight<float,true,double>(const float* W, const float* ref,
                                                          long long i, double alpha, double den) {
    float a = (float)alpha;
    return __fadd_rn(ref[i],__fmul_rn(a,W[i])) / (float)den;
}

template<typename T, bool MIX = false, typename A = double, typename C = double>
__device__ void simulate_core(const T* __restrict__ W,
                              const T* __restrict__ open_raw, const T* __restrict__ prev_close,
                              const T* __restrict__ open_ff0, const T* __restrict__ close_ff0,
                              const T* __restrict__ cost, const unsigned char* __restrict__ live,
                              int Tn, int S, int B, long long i0, double ppy,
                              double* __restrict__ rr_out, double* __restrict__ stats_out,
                              const T* ref = nullptr, const int* owners = nullptr,
                              const C* weights = nullptr,
                              const C* eref = nullptr, const C* eown_all = nullptr) {
    const int lane = threadIdx.x & 31;
    const long long b = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (b >= B) return;                                    /* warp-uniform: the whole warp leaves */
    const T* Wb = W + (MIX ? owners[b] : b) * (long long)Tn * S;
    const C alpha = MIX ? weights[b] : C(0);
    const C* eown = MIX ? eown_all + owners[b] * (long long)Tn : nullptr;
    double* rr = rr_out + b * (long long)Tn;
    A wt[ASSETS_PER_LANE], shares[ASSETS_PER_LANE], off_prev[ASSETS_PER_LANE];
    /* The daily chain is latency-bound: (1) the next W row is prefetched while today's reductions run (day+1 trades
     * W[day]); (2) today's static rows are loaded into registers at once; (3) yesterday's open is carried in a
     * register; (4) four independent sums share one shuffle chain. The per-sum order is unchanged => bitwise the
     * same results as reducing term by term. */
    T wcur[ASSETS_PER_LANE];
    #pragma unroll
    for (int j = 0; j < ASSETS_PER_LANE; j++) {
        const int a = lane + 32 * j;
        wt[j] = A(0); shares[j] = A(0); off_prev[j] = A(0);
        wcur[j] = (a < S) ? blend_weight<T,MIX>(Wb,ref,a,alpha,blend_den<MIX>(eref,eown,0,alpha)) : (T)0;
    }

    A eq_after = A(0), prev_tonight = A(1);
    for (int day = 0; day < Tn; day++) {
        const long long r0 = (long long)day * S;
        const C den = blend_den<MIX>(eref, eown, day, alpha);
        A oraw[ASSETS_PER_LANE], pcl[ASSETS_PER_LANE], off[ASSETS_PER_LANE], cff[ASSETS_PER_LANE], sl[ASSETS_PER_LANE];
        bool ul[ASSETS_PER_LANE];
        T wnext[ASSETS_PER_LANE];
        #pragma unroll
        for (int j = 0; j < ASSETS_PER_LANE; j++) {
            const int a = lane + 32 * j;
            const bool in = a < S;
            oraw[j] = in ? (A)open_raw[r0 + a] : A(0);
            pcl[j]  = in ? (A)prev_close[r0 + a] : A(1);
            off[j]  = in ? (A)open_ff0[r0 + a] : A(0);
            cff[j]  = in ? (A)close_ff0[r0 + a] : A(0);
            sl[j]   = in ? (A)cost[r0 + a] : A(0);
            ul[j]   = in ? (live[r0 + a] != 0) : false;
            wnext[j] = (in && day + 1 < Tn) ? blend_weight<T,MIX>(Wb,ref,r0+a,alpha,den) : (T)0;
        }

        /* --- effective weights: shift-1 -> normalise -> open adjustment -> normalise --- */
        if (day > 0) {
            A part = A(0);
            #pragma unroll
            for (int j = 0; j < ASSETS_PER_LANE; j++) {
                A x = (A)wcur[j];
                if (!isfinite(x)) x = A(0);
                wt[j] = x; part += fabs(x);
            }
            A s1 = warp_sum_bcast(part);
            if (s1 < A(1)) s1 = A(1);
            part = A(0);
            #pragma unroll
            for (int j = 0; j < ASSETS_PER_LANE; j++) {
                A v = (wt[j] / s1) * oraw[j] / pcl[j];
                if (!isfinite(v)) v = A(0);                     /* NaN price -> 0 (0*NaN included) */
                wt[j] = v; part += fabs(v);
            }
            A s2 = warp_sum_bcast(part);
            if (s2 < A(1)) s2 = A(1);
            #pragma unroll
            for (int j = 0; j < ASSETS_PER_LANE; j++) wt[j] /= s2;
        }

        /* --- equity recurrence --- */
        A eq_part = A(0), p_sum = A(0), p_unl = A(0), p_lock = A(0);
        #pragma unroll
        for (int j = 0; j < ASSETS_PER_LANE; j++) {
            const int a = lane + 32 * j;
            if (a < S) {
                if (day > 0) eq_part += (off[j] - off_prev[j]) * shares[j];
                const A aw = fabs(wt[j]);
                p_sum += aw;
                if (ul[j]) p_unl += aw;
                else p_lock += off[j] * fabs(shares[j]);
            }
        }
        warp_sum4_bcast(eq_part, p_sum, p_unl, p_lock);
        const A eq_before = (day > 0) ? eq_after + eq_part : A(1);
        const A w_free = (p_sum > A(1) ? p_sum : A(1)) - p_sum;
        const A w_op   = p_unl + w_free;
        const A eq_oper = eq_before - p_lock;

        if (w_op < A(PRICE_EPS)) {                              /* w_op is warp-uniform => uniform branch */
            eq_after = eq_before;
        } else {
            A part = A(0);
            #pragma unroll
            for (int j = 0; j < ASSETS_PER_LANE; j++) {
                const int a = lane + 32 * j;
                if (a < S && ul[j]) {
                    const A ns = eq_oper * wt[j] / (w_op * off[j]);
                    const A dn = (day > 0) ? ns - shares[j] : ns;
                    part += sl[j] * fabs(dn);
                    shares[j] = ns;
                }
            }
            eq_after = eq_before - warp_sum_bcast(part);
        }
        A part2 = A(0);
        #pragma unroll
        for (int j = 0; j < ASSETS_PER_LANE; j++) {
            const int a = lane + 32 * j;
            if (a < S) part2 += (cff[j] - off[j]) * shares[j];
        }
        const A eq_tonight = eq_after + warp_sum_bcast(part2);

        A v = eq_tonight / prev_tonight - A(1);
        if (!isfinite(v)) v = A(0);
        // An unrepresentable recurrence remains invalid; it cannot masquerade
        // as a valid zero return and is never rerun in another precision.
        if (LimitedRange<A>::value &&
            (!isfinite(eq_tonight) || !isfinite(eq_after) ||
             !isfinite(prev_tonight) || fabs(prev_tonight) < A(1e-30))) v = (A)__int_as_float(0x7fffffff);
        if (lane == 0) rr[day] = (double)v;
        prev_tonight = eq_tonight;
        #pragma unroll
        for (int j = 0; j < ASSETS_PER_LANE; j++) {
            if (!ul[j]) shares[j] = A(0);                     /* no price: the position is closed after tonight */
            off_prev[j] = off[j];
            wcur[j] = wnext[j];
        }
        rebase(eq_after, prev_tonight, shares);
    }
    __syncwarp();                                          /* lane 0 wrote rr; the whole warp reads it below */

}

extern "C" __global__ void k_simulate_mixed(const float* W, const float* o_raw, const float* pcl,
                                            const float* off, const float* cff, const float* cost,
                                            const unsigned char* live, int T, int S, int B, long long i0,
                                            double ppy, double* rr, double* st) {
    simulate_core<float,false,float>(W, o_raw, pcl, off, cff, cost, live, T, S, B, i0, ppy, rr, st);
}
extern "C" __global__ void k_blend_mixed(const float* W, const float* ref,
                                         const int* owners, const double* weights,
                                         const double* eref, const double* eown,
                                         const float* o_raw, const float* pcl,
                                         const float* off, const float* cff, const float* cost,
                                         const unsigned char* live, int T, int S, int B, long long i0,
                                         double ppy, double* rr, double* st) {
    simulate_core<float,true,float>(W, o_raw, pcl, off, cff, cost, live, T, S, B, i0, ppy, rr, st,
                                    ref, owners, weights, eref, eown);
}

// Paired holdings and blended-book arithmetic retain low input bits.
extern "C" __global__ void k_simulate_paired(const FloatPair* W, const FloatPair* o_raw, const FloatPair* pcl,
                                           const FloatPair* off, const FloatPair* cff, const FloatPair* cost,
                                           const unsigned char* live, int T, int S, int B, long long i0,
                                           double ppy, double* rr, double* st) {
    simulate_core<FloatPair,false,FloatPair,FloatPair>(W, o_raw, pcl, off, cff, cost, live, T, S, B, i0, ppy, rr, st);
}
extern "C" __global__ void k_blend_paired(const FloatPair* W, const FloatPair* ref,
                                        const int* owners, const FloatPair* weights,
                                        const FloatPair* eref, const FloatPair* eown,
                                        const FloatPair* o_raw, const FloatPair* pcl,
                                        const FloatPair* off, const FloatPair* cff, const FloatPair* cost,
                                        const unsigned char* live, int T, int S, int B, long long i0,
                                        double ppy, double* rr, double* st) {
    simulate_core<FloatPair,true,FloatPair,FloatPair>(W, o_raw, pcl, off, cff, cost, live, T, S, B, i0, ppy, rr, st,
                                                   ref, owners, weights, eref, eown);
}
