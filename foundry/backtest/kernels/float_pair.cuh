/* Unevaluated sum of two FP32 values (~48 significant bits, FP32 exponent range).
 * Explicit round-to-nearest operations keep the error-free transforms from being
 * contracted by NVRTC. FMA recovers the product residual, not a TF32/FP16 product.
 * No FP64 arithmetic in the operators; double conversion is an I/O boundary only.
 */
struct FloatPair {
    float hi, lo;
    __device__ FloatPair() = default;
    __device__ FloatPair(float h, float l) : hi(h), lo(l) {}
    __device__ explicit FloatPair(double v) : hi((float)v), lo((float)(v - (double)hi)) {}
    __device__ explicit operator double() const { return (double)hi + (double)lo; }
};
template<typename T> struct LimitedRange { static constexpr bool value = false; };
template<> struct LimitedRange<float> { static constexpr bool value = true; };
template<> struct LimitedRange<FloatPair> { static constexpr bool value = true; };
template<typename T> struct PairStats { static constexpr bool enabled = false; };
template<> struct PairStats<FloatPair> { static constexpr bool enabled = true; };
__device__ __forceinline__ FloatPair pair_sum(float a, float b) {
    float s = __fadd_rn(a, b), v = __fsub_rn(s, a);
    return FloatPair(s, __fadd_rn(__fsub_rn(a, __fsub_rn(s, v)), __fsub_rn(b, v)));
}
__device__ __forceinline__ FloatPair operator+(FloatPair a, FloatPair b) {
    FloatPair s = pair_sum(a.hi, b.hi);
    return pair_sum(s.hi, __fadd_rn(s.lo, __fadd_rn(a.lo, b.lo)));
}
__device__ __forceinline__ FloatPair operator-(FloatPair a, FloatPair b) {
    return a + FloatPair(-b.hi, -b.lo);
}
__device__ __forceinline__ FloatPair operator*(FloatPair a, FloatPair b) {
    float p = __fmul_rn(a.hi, b.hi);
    float e = __fmaf_rn(a.hi, b.hi, -p);
    e = __fmaf_rn(a.hi, b.lo, e);
    e = __fmaf_rn(a.lo, b.hi, e);
    return pair_sum(p, e);
}
__device__ __forceinline__ FloatPair operator/(FloatPair a, FloatPair b) {
    float q = __fdiv_rn(a.hi, b.hi);
    FloatPair r = a - b * FloatPair(q, 0.0f);
    return pair_sum(q, __fdiv_rn(__fadd_rn(r.hi, r.lo), b.hi));
}
__device__ __forceinline__ FloatPair& operator+=(FloatPair& a, FloatPair b) { return a = a + b; }
__device__ __forceinline__ FloatPair& operator/=(FloatPair& a, FloatPair b) { return a = a / b; }
__device__ __forceinline__ bool operator<(FloatPair a, FloatPair b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
__device__ __forceinline__ bool operator>(FloatPair a, FloatPair b) { return b < a; }
__device__ __forceinline__ bool isfinite(FloatPair a) { return isfinite(a.hi) && isfinite(a.lo); }
__device__ __forceinline__ FloatPair fabs(FloatPair a) {
    return a.hi < 0.0f ? FloatPair(-a.hi, -a.lo) : a;
}
__device__ __forceinline__ FloatPair __shfl_down_sync(unsigned m, FloatPair a, int d) {
    return FloatPair(__shfl_down_sync(m, a.hi, d), __shfl_down_sync(m, a.lo, d));
}
__device__ __forceinline__ FloatPair __shfl_sync(unsigned m, FloatPair a, int s) {
    return FloatPair(__shfl_sync(m, a.hi, s), __shfl_sync(m, a.lo, s));
}

template<typename A, int N>
__device__ __forceinline__ void rebase(A& equity, A& previous, A (&shares)[N]) {}

template<int N>
__device__ __forceinline__ void rebase(FloatPair& equity, FloatPair& previous, FloatPair (&shares)[N]) {
    // Returns are homogeneous in wealth and shares. Exact power-of-two rescaling
    // prevents years of compounding from exhausting the FP32 exponent range.
    float magnitude = fabsf(previous.hi);
    if (isfinite(previous) && magnitude > 0.0f &&
        (magnitude > 4294967296.0f || magnitude < 2.3283064365386963e-10f)) {
        int exponent;
        frexpf(magnitude, &exponent);
        equity = FloatPair(scalbnf(equity.hi, -exponent), scalbnf(equity.lo, -exponent));
        previous = FloatPair(scalbnf(previous.hi, -exponent), scalbnf(previous.lo, -exponent));
        #pragma unroll
        for (int j = 0; j < N; j++)
            shares[j] = FloatPair(scalbnf(shares[j].hi, -exponent), scalbnf(shares[j].lo, -exponent));
    }
}
