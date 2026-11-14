/* foundry/ops/kernels/ops.cu — CUDA formula operators (NVRTC via cupy), the device backend of foundry/ops.
 *
 * Semantics follow the CPU reference (foundry/ops/src); accumulation order differs, so agreement is to ~1e-12
 * relative in f64, not bitwise.
 *   - Row-major (T,S): x[t*S+s]; panels are tall (T >> S), one thread per output (t,s).
 *   - Rolling windows use a shared-memory tile: a block of (SS assets) x (TT days) loads (TT+w-1) x SS with halo.
 *   - out[t,s] = (t >= w-1 and no NaN in [t-w+1, t]) ? stat : NaN (the CPU range trimming is only an optimisation).
 *   - Centering anchor K: kmode 0 = oldest value of the window (sum/mean/mad); kmode 1 = first valid value of
 *     the column (skew/kurt, matching the CPU degenerate-window threshold).
 */

/* Typed view over dynamic shared memory: avoids conflicting extern __shared__ declarations across template T. */
template<typename T> __device__ __forceinline__ T* shmem() {
    extern __shared__ __align__(8) unsigned char smem[];
    return reinterpret_cast<T*>(smem);
}


/* ============================================================================
 * Rolling moment kernel: op 0=sum 1=mean 3=skew 4=kurt 5=mad
 * ========================================================================== */
template<typename T>
__device__ void moment_core(const T* __restrict__ x, T* __restrict__ out,
                            int Tt, int S, int w, int op, const T* __restrict__ Kcol, int kmode) {
    T* sh = shmem<T>();
    int SS = blockDim.x, TT = blockDim.y;
    int strip_s0 = blockIdx.x * SS, tile_t0 = blockIdx.y * TT;
    int rows = TT + w - 1, base_t = tile_t0 - (w - 1);
    int tid = threadIdx.y * SS + threadIdx.x, nth = SS * TT;
    for (int idx = tid; idx < rows * SS; idx += nth) {
        int i = idx / SS, sx = idx - i * SS;
        int gt = base_t + i, gs = strip_s0 + sx;
        sh[i * SS + sx] = (gt < 0 || gt >= Tt || gs >= S) ? (T)nan("")
                                                          : x[(long long)gt * S + gs];
    }
    __syncthreads();

    int sx = threadIdx.x, ty = threadIdx.y;
    int s = strip_s0 + sx, t = tile_t0 + ty;
    if (s >= S || t >= Tt) return;
    long long o = (long long)t * S + s;

    T K = (kmode == 1) ? Kcol[s] : sh[ty * SS + sx];
    T s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    int bad = 0;
    for (int k = 0; k < w; k++) {
        T v = sh[(ty + k) * SS + sx];
        if (isnan(v)) { bad = 1; break; }
        T d = v - K;
        s1 += d;
        if (op >= 2 && op != 5) { T d2 = d * d; s2 += d2;
            if (op >= 3) { s3 += d2 * d;
                if (op >= 4) s4 += d2 * d2; } }
    }
    if (bad) { out[o] = (T)nan(""); return; }

    T wd = (T)w, res;
    if (op == 0)      res = s1 + wd * K;                          /* sum  */
    else if (op == 1) res = (s1 + wd * K) / wd;                   /* mean */
    else if (op == 5) {                                           /* mad = mean(|x-mean|) */
        T mean = (s1 + wd * K) / wd, mad = 0;
        for (int k = 0; k < w; k++) mad += fabs(sh[(ty + k) * SS + sx] - mean);
        res = mad / wd;
    } else {                                                      /* skew/kurt: two-pass central moments per window */
        /* Pass 1 window mean -> pass 2 direct central moments sum(v - mean)^p (eliminates e2 - mu^2 cancellation).
         * Same algorithm and oldest->newest order as CPU fm_ts_skew/kurt -> bitwise match (replaces incremental power
         * sums, which diverged from GPU on real f64 panels on ~40% of kurt cells; two-pass keeps near-constant windows
         * identical too). */
        T mean = 0;
        for (int k = 0; k < w; k++) mean += sh[(ty + k) * SS + sx];
        mean /= wd;
        T m2 = 0, m3 = 0, m4 = 0;
        for (int k = 0; k < w; k++) {
            T d = sh[(ty + k) * SS + sx] - mean, d2 = d * d;
            m2 += d2; m3 += d2 * d; m4 += d2 * d2;
        }
        m2 /= wd; m3 /= wd; m4 /= wd;
        if (m2 <= (T)0) { out[o] = (T)nan(""); return; }            /* constant window -> NaN */
        res = (op == 3) ? m3 / pow(m2, (T)1.5) : m4 / (m2 * m2) - (T)3;
    }
    out[o] = res;
}


/* ============================================================================
 * Extended rolling kernel: op 0=max 1=min 2=wma 4=rank 5=arg_max 6=arg_min
 *   arg_*: emit (w-1)-argk = bars since extremum in [0, w-1]; ties -> most recent (>= / <=, matching C deque).
 * ========================================================================== */
template<typename T>
__device__ void ext_core(const T* __restrict__ x, T* __restrict__ out,
                         int Tt, int S, int w, int op) {
    T* sh = shmem<T>();
    int SS = blockDim.x, TT = blockDim.y;
    int strip_s0 = blockIdx.x * SS, tile_t0 = blockIdx.y * TT;
    int rows = TT + w - 1, base_t = tile_t0 - (w - 1);
    int tid = threadIdx.y * SS + threadIdx.x, nth = SS * TT;
    for (int idx = tid; idx < rows * SS; idx += nth) {
        int i = idx / SS, sx = idx - i * SS, gt = base_t + i, gs = strip_s0 + sx;
        sh[i * SS + sx] = (gt < 0 || gt >= Tt || gs >= S) ? (T)nan("") : x[(long long)gt * S + gs];
    }
    __syncthreads();
    int sx = threadIdx.x, ty = threadIdx.y, s = strip_s0 + sx, t = tile_t0 + ty;
    if (s >= S || t >= Tt) return;
    long long o = (long long)t * S + s;
    int bad = 0;
    if (op == 0 || op == 1) {                                    /* max / min */
        T m = sh[ty * SS + sx];                                  /* seed from oldest bar in window (NVRTC lacks INFINITY) */
        if (isnan(m)) { out[o] = (T)nan(""); return; }
        for (int k = 1; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; } m = (op == 0) ? fmax(m, v) : fmin(m, v); }
        out[o] = bad ? (T)nan("") : m;
    } else if (op == 5 || op == 6) {                             /* arg_max / arg_min */
        T m = sh[ty * SS + sx]; int argk = 0;
        if (isnan(m)) { out[o] = (T)nan(""); return; }
        for (int k = 1; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; }
            if ((op == 5 && v >= m) || (op == 6 && v <= m)) { m = v; argk = k; } }
        out[o] = bad ? (T)nan("") : (T)(w - 1 - argk);
    } else if (op == 2) {                                        /* wma */
        T wsum = 0, denom = (T)w * ((T)w + (T)1) / (T)2;
        for (int k = 0; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; } wsum += (T)(k + 1) * v; }
        out[o] = bad ? (T)nan("") : wsum / denom;
    } else {                                                     /* rank */
        T cur = sh[(ty + w - 1) * SS + sx];
        if (isnan(cur)) { out[o] = (T)nan(""); return; }
        int cl = 0, ce = 0;
        for (int k = 0; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; } if (v < cur) cl++; else if (v == cur) ce++; }
        out[o] = bad ? (T)nan("") : ((T)cl + (T)0.5 * (T)(ce - 1)) / (T)(w - 1);
    }
}


/* ts_ema: pandas ewm(span, adjust=False, min_periods=span). Sequential recurrence -> one thread per column along T. */
template<typename T>
__device__ void ema_core(const T* __restrict__ x, T* __restrict__ out, int Tt, int S, int span) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    T alpha = (T)2 / (T)(span + 1);
    T ema = 0; int has_seed = 0; long long cnt = 0;
    for (int t = 0; t < Tt; t++) {
        long long o = (long long)t * S + s;
        T v = x[o];
        if (isnan(v)) { out[o] = (T)nan(""); continue; }
        ema = has_seed ? (alpha * v + ((T)1 - alpha) * ema) : v;
        has_seed = 1; cnt++;
        out[o] = (cnt >= span) ? ema : (T)nan("");
    }
}


/* ============================================================================
 * Cross-sectional warp-shuffle kernel: op 0=zscore 1=demean 2=scale (L1)
 * ========================================================================== */
template<typename T> __device__ __forceinline__ T warp_reduce_sum(T v) {
    for (int off = warpSize >> 1; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

template<typename T>
__device__ void cs_core(const T* __restrict__ x, T* __restrict__ out, int S, int op) {
    int t = blockIdx.x;
    const T* row = x + (long long)t * S;
    T* orow = out + (long long)t * S;
    int tid = threadIdx.x, B = blockDim.x;
    int lane = tid & 31, wid = tid >> 5, nwarp = B >> 5;

    T lsum = 0, lsq = 0; int ln = 0;
    for (int s = tid; s < S; s += B) {
        T v = row[s];
        if (!isnan(v)) { lsum += v; ln++;
            if (op == 0) lsq += v * v; else if (op == 2) lsq += fabs(v); }
    }
    lsum = warp_reduce_sum<T>(lsum);
    T lsqr = warp_reduce_sum<T>(lsq);
    T lnr  = warp_reduce_sum<T>((T)ln);
    __shared__ T wsum[32], wsq[32], wn[32];
    if (lane == 0) { wsum[wid] = lsum; wsq[wid] = lsqr; wn[wid] = lnr; }
    __syncthreads();
    __shared__ T mean_sh, p_sh; __shared__ int ok_sh;
    if (wid == 0) {
        T sm = (lane < nwarp) ? wsum[lane] : (T)0;
        T sq = (lane < nwarp) ? wsq[lane]  : (T)0;
        T nn = (lane < nwarp) ? wn[lane]   : (T)0;
        sm = warp_reduce_sum<T>(sm); sq = warp_reduce_sum<T>(sq); nn = warp_reduce_sum<T>(nn);
        if (lane == 0) {
            long long n = (long long)nn; ok_sh = 0;
            if (op == 1) { if (n >= 1) { mean_sh = sm / (T)n; ok_sh = 1; } }
            else if (op == 2) { if (sq > (T)1e-12) { p_sh = sq; ok_sh = 1; } }
            else { if (n >= 2) { T mean = sm / (T)n;
                       T var = (sq - mean * sm) / (T)(n - 1);
                       if (var > 0) { mean_sh = mean; p_sh = sqrt(var); ok_sh = 1; } } }
        }
    }
    __syncthreads();
    if (!ok_sh) { for (int s = tid; s < S; s += B) orow[s] = (T)nan(""); return; }
    T mean = mean_sh, p = p_sh;
    for (int s = tid; s < S; s += B) {
        T v = row[s];
        if (isnan(v)) { orow[s] = (T)nan(""); continue; }
        if (op == 0)      orow[s] = (v - mean) / p;
        else if (op == 1) orow[s] = v - mean;
        else              orow[s] = v / p;
    }
}


/* cs_rank: per-row cross-sectional average rank in [0, 1]; full row NaN if n < 2. One block per row in shared, O(n^2) count. */
template<typename T>
__device__ void cs_rank_core(const T* __restrict__ x, T* __restrict__ out, int S) {
    T* r = shmem<T>();
    int t = blockIdx.x; const T* row = x + (long long)t * S; T* orow = out + (long long)t * S;
    int tid = threadIdx.x, B = blockDim.x;
    for (int s = tid; s < S; s += B) r[s] = row[s];
    __shared__ int nval; if (tid == 0) nval = 0;
    __syncthreads();
    int ln = 0; for (int s = tid; s < S; s += B) if (!isnan(r[s])) ln++;
    atomicAdd(&nval, ln); __syncthreads();
    int n = nval;
    if (n < 2) { for (int s = tid; s < S; s += B) orow[s] = (T)nan(""); return; }
    for (int s = tid; s < S; s += B) {
        T v = r[s];
        if (isnan(v)) { orow[s] = (T)nan(""); continue; }
        int cl = 0, ce = 0;
        for (int j = 0; j < S; j++) { T u = r[j]; if (isnan(u)) continue;
            if (u < v) cl++; else if (u == v) ce++; }
        orow[s] = ((T)cl + (T)0.5 * (T)(ce - 1)) / (T)(n - 1);
    }
}


/* Formula values are always double: comparisons, ranks and state machines amplify rounding (the mixed-precision
 * path is confined to the backtest kernel). */
extern "C" __global__ void k_moment(const double* x, double* o, int T, int S, int w, int op, const double* K, int km) { moment_core<double>(x, o, T, S, w, op, K, km); }
extern "C" __global__ void k_ext(const double* x, double* o, int T, int S, int w, int op) { ext_core<double>(x, o, T, S, w, op); }
extern "C" __global__ void k_ema(const double* x, double* o, int T, int S, int span) { ema_core<double>(x, o, T, S, span); }
extern "C" __global__ void k_cs(const double* x, double* o, int S, int op) { cs_core<double>(x, o, S, op); }
extern "C" __global__ void k_cs_rank(const double* x, double* o, int S) { cs_rank_core<double>(x, o, S); }
