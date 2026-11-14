/* Paired statistics use FP32 arithmetic. Wide arrays remain at the return-input
 * and five-scalar output boundaries. Integer exponents preserve cumulative wealth
 * outside the FP32 exponent range; no float log/exp approximations are used.
 */
template<typename A>
__device__ __forceinline__ A block_sum(A v) {
    __shared__ A sums[4];
    v = warp_reduce_sum(v);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) sums[0] = ((sums[0] + sums[1]) + sums[2]) + sums[3];
    __syncthreads();
    A out = sums[0];
    __syncthreads();
    return out;
}
template<int N> struct StatGroup;
template<> struct StatGroup<32> {
    __device__ static FloatPair sum(FloatPair v) { return warp_sum_bcast(v); }
    __device__ static bool any(bool v) { return __any_sync(0xffffffff,v); }
};
template<> struct StatGroup<128> {
    __device__ static FloatPair sum(FloatPair v) { return block_sum(v); }
    __device__ static bool any(bool v) { return __syncthreads_or(v); }
};

struct ScaledPair {
    FloatPair value;
    int exponent;
    __device__ void multiply(FloatPair x) {
        value = value*x;
        if (isfinite(value) && value.hi != 0.0f) {
            int e;
            frexpf(value.hi, &e);
            value = FloatPair(scalbnf(value.hi,-e),scalbnf(value.lo,-e));
            exponent += e;
        }
    }
    __device__ bool exceeds_positive(const ScaledPair& peak) const {
        return value.hi > 0.0f && (exponent > peak.exponent ||
               (exponent == peak.exponent && value > peak.value));
    }
    __device__ FloatPair ratio(const ScaledPair& peak) const {
        FloatPair q=value/peak.value;
        int shift=exponent-peak.exponent;
        return FloatPair(scalbnf(q.hi,shift),scalbnf(q.lo,shift));
    }
    __device__ double output() const { return ldexp((double)value, exponent); }
};

__device__ ScaledPair scaled_product(ScaledPair a, ScaledPair b) {
    a.multiply(b.value); a.exponent+=b.exponent; return a;
}
__device__ ScaledPair scaled_down(ScaledPair a,int offset) {
    return {__shfl_down_sync(0xffffffff,a.value,offset),
            __shfl_down_sync(0xffffffff,a.exponent,offset)};
}

// Ordered segment summary. The cross-segment drawdown is the right segment's
// minimum wealth, rescaled by the left terminal wealth, divided by the left peak.
// Positive gross returns make this merge associative without logarithms.
struct WealthSummary {
    ScaledPair total, peak, trough;
    FloatPair dd;
    __device__ static WealthSummary identity() {
        ScaledPair one={FloatPair(.5f,0.0f),1};
        return {one,one,one,FloatPair(0.0f,0.0f)};
    }
    __device__ void append(FloatPair gross) {
        total.multiply(gross);
        if(total.exceeds_positive(peak)) peak=total;
        if(trough.exceeds_positive(total)) trough=total;
        FloatPair d=total.ratio(peak)-FloatPair(1.0f,0.0f);
        if(d<dd) dd=d;
    }
    __device__ void merge(WealthSummary right) {
        ScaledPair low=scaled_product(total,right.trough);
        FloatPair cross=low.ratio(peak)-FloatPair(1.0f,0.0f);
        if(right.dd<dd) dd=right.dd;
        if(cross<dd) dd=cross;
        ScaledPair high=scaled_product(total,right.peak);
        if(high.exceeds_positive(peak)) peak=high;
        if(trough.exceeds_positive(low)) trough=low;
        total=scaled_product(total,right.total);
    }
};
__device__ WealthSummary wealth_warp(WealthSummary v) {
    const int lane=threadIdx.x&31;
    for(int offset=1;offset<32;offset*=2) {
        WealthSummary right={scaled_down(v.total,offset),scaled_down(v.peak,offset),
                             scaled_down(v.trough,offset),__shfl_down_sync(0xffffffff,v.dd,offset)};
        if((lane%(offset*2))==0) v.merge(right);
    }
    return v;
}
template<int N> struct WealthGroup;
template<> struct WealthGroup<32> {
    __device__ static WealthSummary reduce(WealthSummary v) { return wealth_warp(v); }
};
template<> struct WealthGroup<128> {
    __device__ static WealthSummary reduce(WealthSummary v) {
        __shared__ WealthSummary segments[4];
        v=wealth_warp(v);
        if((threadIdx.x&31)==0) segments[threadIdx.x>>5]=v;
        __syncthreads();
        if(threadIdx.x<32) {
            v=threadIdx.x<4?segments[threadIdx.x]:WealthSummary::identity();
            v=wealth_warp(v);
        }
        return v;
    }
};

template<int N>
__device__ void paired_statistics(const double* rr, int T, int start, double ppy, double* st) {
    const int lane = threadIdx.x % N;
    FloatPair part(0.0f,0.0f), n((double)(T-start)), origin(rr[start]);
    bool nonpositive=false;
    for (int t=start+lane;t<T;t+=N) {
        FloatPair r(rr[t]); part+=r-origin;
        nonpositive|=!(r>FloatPair(-1.0f,0.0f));
    }
    const bool serial=StatGroup<N>::any(nonpositive);
    FloatPair shift=StatGroup<N>::sum(part)/n, mean=origin+shift;
    part=FloatPair(0.0f,0.0f);
    for (int t=start+lane;t<T;t+=N) {
        FloatPair d=(FloatPair(rr[t])-origin)-shift;
        part+=d*d;
    }
    FloatPair sd=sqrt(StatGroup<N>::sum(part)/n);
    WealthSummary wealth=WealthSummary::identity();
    if(!serial) {
        const int size=(T-start+N-1)/N;
        const int first=start+lane*size, last=min(first+size,T);
        for(int t=first;t<last;++t) wealth.append(FloatPair(1.0f,0.0f)+FloatPair(rr[t]));
        wealth=WealthGroup<N>::reduce(wealth);
    } else if(lane==0) {
        // Ruined/nonfinite books retain the sequential signed-wealth semantics.
        for (int t=start;t<T;++t) {
            wealth.append(FloatPair(1.0f,0.0f)+FloatPair(rr[t]));
        }
    }
    if (lane==0) {
        FloatPair annual(ppy), mr=mean*annual, vol=sd*sqrt(annual);
        st[0]=vol>FloatPair(0.f,0.f) ? (double)(mr/vol) : 0.;
        st[1]=(double)mr; st[2]=(double)vol; st[3]=(double)wealth.dd;
        st[4]=wealth.total.output()-1.0;
    }
}

extern "C" __global__ void k_statistics_paired(const double* r,int T,int start,double ppy,double* st) {
    paired_statistics<128>(r+(long long)blockIdx.x*T,T,start,ppy,st+5*blockIdx.x);
}
