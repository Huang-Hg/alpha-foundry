/* Unevaluated sum of two FP32 values (~48 significant bits, FP32 exponent range).
 * Explicit round-to-nearest operations keep the error-free transforms from being
 * contracted by NVRTC. FMA recovers the product residual, not a TF32/FP16 product.
 * Wide arrays are an interchange format; arithmetic uses the two components.
 */
struct FloatPair {
    float hi, lo;
    __device__ FloatPair() = default;
    __device__ FloatPair(float h, float l) : hi(h), lo(l) {}
    __device__ FloatPair(float v) : hi(v), lo(0.f) {}
    __device__ FloatPair(int v) : hi((float)v), lo((float)((long long)v-(long long)hi)) {}
    __device__ FloatPair(double v) : hi((float)v), lo(isfinite(v)?(float)(v - (double)hi):0.f) {}
    __device__ explicit operator long long() const { return (long long)hi; }
    __device__ explicit operator double() const { return (double)hi + (double)lo; }
};
template<typename T> struct LimitedRange { static constexpr bool value = false; };
template<> struct LimitedRange<float> { static constexpr bool value = true; };
template<> struct LimitedRange<FloatPair> { static constexpr bool value = true; };
__device__ __forceinline__ FloatPair pair_sum(float a, float b) {
    float s = __fadd_rn(a, b);
    if (!isfinite(s)) return FloatPair(s,0.f);
    float v = __fsub_rn(s, a);
    return FloatPair(s, __fadd_rn(__fsub_rn(a, __fsub_rn(s, v)), __fsub_rn(b, v)));
}
// Quick-Two-Sum renormalizes a leading term and its rounding correction.
// The pair operations below establish this ordering; arbitrary inputs use pair_sum.
__device__ __forceinline__ FloatPair pair_normalize(float leading, float correction) {
    float s = __fadd_rn(leading, correction);
    if (!isfinite(s)) return FloatPair(s, 0.f);
    return FloatPair(s, __fsub_rn(correction, __fsub_rn(s, leading)));
}
__device__ __forceinline__ FloatPair operator+(FloatPair a, FloatPair b) {
    FloatPair s = pair_sum(a.hi, b.hi);
    return pair_normalize(s.hi, __fadd_rn(s.lo, __fadd_rn(a.lo, b.lo)));
}
__device__ __forceinline__ FloatPair operator-(FloatPair a, FloatPair b) {
    return a + FloatPair(-b.hi, -b.lo);
}
__device__ __forceinline__ FloatPair operator*(FloatPair a, FloatPair b) {
    float p = __fmul_rn(a.hi, b.hi);
    if (!isfinite(p)) return FloatPair(p,0.f);
    float e = __fmaf_rn(a.hi, b.hi, -p);
    e = __fmaf_rn(a.hi, b.lo, e);
    e = __fmaf_rn(a.lo, b.hi, e);
    return pair_normalize(p, e);
}
__device__ __forceinline__ FloatPair operator/(FloatPair a, FloatPair b) {
    if (b.hi == 1.f && b.lo == 0.f) return a;
    float q = __fdiv_rn(a.hi, b.hi);
    if (!isfinite(q)) return FloatPair(q,0.f);
    // FMA forms the high-product residual directly. Only its correction needs
    // the low components; constructing and subtracting a full pair product is
    // redundant. Both divisions retain round-to-nearest, including subnormals.
    float residual = __fmaf_rn(-q, b.hi, a.hi);
    residual = __fsub_rn(residual, __fmul_rn(q, b.lo));
    residual = __fadd_rn(residual, a.lo);
    return pair_normalize(q, __fdiv_rn(residual, b.hi));
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
__device__ __forceinline__ FloatPair __shfl_down_sync(unsigned m, FloatPair a, unsigned d) {
    return FloatPair(__shfl_down_sync(m, a.hi, d), __shfl_down_sync(m, a.lo, d));
}
__device__ __forceinline__ FloatPair __shfl_sync(unsigned m, FloatPair a, int s) {
    return FloatPair(__shfl_sync(m, a.hi, s), __shfl_sync(m, a.lo, s));
}

__device__ __forceinline__ FloatPair operator-(FloatPair a) { return FloatPair(-a.hi,-a.lo); }
__device__ __forceinline__ FloatPair& operator-=(FloatPair& a,FloatPair b) {return a=a-b;}
__device__ __forceinline__ FloatPair& operator*=(FloatPair& a,FloatPair b) {return a=a*b;}
__device__ __forceinline__ bool operator==(FloatPair a,FloatPair b) {return a.hi==b.hi && a.lo==b.lo;}
__device__ __forceinline__ bool operator!=(FloatPair a,FloatPair b) {return !(a==b);}
__device__ __forceinline__ bool operator<=(FloatPair a,FloatPair b) {return a<b || a==b;}
__device__ __forceinline__ bool operator>=(FloatPair a,FloatPair b) {return a>b || a==b;}
__device__ __forceinline__ bool isnan(FloatPair a) {return isnan(a.hi) || isnan(a.lo);}
__device__ __forceinline__ FloatPair fmin(FloatPair a,FloatPair b) {return isnan(a)||b<a?b:a;}
__device__ __forceinline__ FloatPair fmax(FloatPair a,FloatPair b) {return isnan(a)||b>a?b:a;}
__device__ __forceinline__ FloatPair sqrt(FloatPair x) {
    float q=sqrtf(x.hi);
    if(q==0.f || !isfinite(q)) return FloatPair(q,0.f);
    FloatPair p(q,0.f);
    return p+(x-p*p)/FloatPair(__fadd_rn(q,q),0.f);
}
__device__ __forceinline__ FloatPair pair_scale(FloatPair a,int e) {
    return FloatPair(scalbnf(a.hi,e),scalbnf(a.lo,e));
}
__device__ __forceinline__ FloatPair pair_ln2() {return FloatPair(0.693147182464599609375f,-1.904654323148236e-9f);}
__device__ __forceinline__ FloatPair log(FloatPair a) {
    if(!(a>FloatPair(0.f,0.f)) || !isfinite(a)) return FloatPair(logf(a.hi),0.f);
    int e;frexpf(a.hi,&e);
    FloatPair m=pair_scale(a,-e);
    if(m<FloatPair(0.7071067811865476)) {m=pair_scale(m,1);--e;}
    FloatPair z=(m-FloatPair(1.f,0.f))/(m+FloatPair(1.f,0.f)),z2=z*z;
    FloatPair term=z,sum=z;
    for(int k=3;k<=31;k+=2) {term*=z2;sum+=term/FloatPair(k);}
    return FloatPair(2.f,0.f)*sum+FloatPair(e)*pair_ln2();
}
__device__ __forceinline__ FloatPair log1p(FloatPair a) {return log(FloatPair(1.f,0.f)+a);}
__device__ __forceinline__ FloatPair pair_exp_mantissa(FloatPair a,int& exponent) {
    exponent=(int)rintf(a.hi/0.693147182464599609375f);
    FloatPair r=a-FloatPair(exponent)*pair_ln2(),term(1.f,0.f),sum=term;
    for(int k=1;k<=16;++k) {term=term*r/FloatPair(k);sum+=term;}
    return sum;
}
__device__ __forceinline__ double pair_expm1_output(FloatPair a) {
    if(!isfinite(a) || fabsf(a.hi)>710.f) return (double)expm1f(a.hi);
    int e;FloatPair m=pair_exp_mantissa(a,e);
    if(e==0) return double(m-FloatPair(1.f,0.f));
    return ldexp(double(m),e)-1.;
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
