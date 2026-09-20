/* factor_mining/ops_cuda/kernels.cu — 主程序级 CUDA alpha 算子(templated C++,NVRTC 经 cupy 编译)
 *
 * 与 CPU 侧 factor_mining/ops/*.c 一一对应、语义对齐(真 C = ground truth,容差 ~1e-6;
 * GPU/CPU FP 累加序不同 → 非 bitwise)。**分开写程序**:CPU=C 扩展,CUDA=此 .cu。
 *
 * 设计(源自 FACT + V100/T1000 实证,见 reference_openbayes_v100_cuda):
 *   - 行优先 (T,S):x[t*S+s];面板瘦高(T≫S)→ 一线程一输出 (t,s) 喂满 SM。
 *   - 滑窗 shared-mem tile:一 block=(SS symbol)×(TT 时间)协作载入 (TT+w-1)×SS tile+halo。
 *   - rolling 规则化简:out[t,s] = (t>=w-1 且窗口[t-w+1,t]无 NaN) ? stat : NaN
 *     (列 range-trim 在 C 里只是性能优化;跨上市边界窗口必含 NaN → 语义等价此简单规则)。
 *   - 中心化锚 K:kmode0=窗口最旧值(sum/mean/std/mad,数值干净);kmode1=列首有效值
 *     Kcol[s](skew/kurt,复刻 C 大偏移抵消 → 对齐 m2<=1e-12 退化门)。
 *
 * **templated C++**:template<typename T> __device__ 核心 + extern "C" f64/f32 实例化
 *   (cupy RawModule 按平名 get_function('k_*_f64'/'k_*_f32'));替掉旧 .replace('double','float')。
 *   消费卡(FP64 1:32)走 f32 实例(2-3.6× 快),数据中心卡走 f64。
 *   scan(flat-in-w)是 f64-only 性能优化、且 fp32 全局 cumsum 已实证灾难 → 不在此文件,
 *   主程序求值器用 tile 核(全 dtype 安全)。
 */

/* 动态 shared 的 typed 视图:模板 T 下避开多个 extern __shared__ 同名异型冲突 */
template<typename T> __device__ __forceinline__ T* shmem() {
    extern __shared__ __align__(8) unsigned char smem[];
    return reinterpret_cast<T*>(smem);
}


/* ============================================================================
 * 滑窗矩核:op 0=sum 1=mean 2=std 3=skew 4=kurt 5=mad
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
    } else if (op == 2) {                                         /* std  */
        T Sx = s1 + wd * K, Sxx = s2 + (T)2 * K * s1 + wd * K * K;
        T mean = Sx / wd, var = (Sxx - mean * Sx) / (wd - (T)1);
        if (var < 0) var = 0;
        res = sqrt(var);
    } else {                                                      /* skew/kurt:窗内两遍中心矩 */
        /* pass1 窗均值 → pass2 直接中心矩 Σ(v-mean)^p(消 e2−mu² 抵消)。与 C fm_ts_skew/kurt
         * 同算法同顺序(窗最旧→最新)→ 逐位一致(替旧增量幂和:真盘 f64 与 GPU 分叉 kurt~40%,
         * m2<=1e-8·e2 floor 证伪;两遍稳定后近常数窗也产同一(大)值 → 不再分叉)。 */
        T mean = 0;
        for (int k = 0; k < w; k++) mean += sh[(ty + k) * SS + sx];
        mean /= wd;
        T m2 = 0, m3 = 0, m4 = 0;
        for (int k = 0; k < w; k++) {
            T d = sh[(ty + k) * SS + sx] - mean, d2 = d * d;
            m2 += d2; m3 += d2 * d; m4 += d2 * d2;
        }
        m2 /= wd; m3 /= wd; m4 /= wd;
        if (m2 <= (T)0) { out[o] = (T)nan(""); return; }            /* 常数窗 → NaN */
        res = (op == 3) ? m3 / pow(m2, (T)1.5) : m4 / (m2 * m2) - (T)3;
    }
    out[o] = res;
}


/* ============================================================================
 * 滑窗扩展核:op 0=max 1=min 2=wma 3=slope 4=rank 5=arg_max 6=arg_min
 *   arg_*:emit (w-1)-argk = 距今多少 bar(∈[0,w-1]),ties→最近(>= / <=,贴 C deque)。
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
        T m = sh[ty * SS + sx];                                  /* 窗口最旧播种(避 NVRTC 无 INFINITY) */
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
    } else if (op == 3) {                                        /* slope */
        T Sx = 0, Skx = 0;
        for (int k = 0; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; } Sx += v; Skx += (T)k * v; }
        if (bad) { out[o] = (T)nan(""); return; }
        T mean_t = (T)(w - 1) / (T)2, var_t = (T)((long long)w * w - 1) / (T)12;
        out[o] = (Skx - mean_t * Sx) / (T)w / var_t;
    } else {                                                     /* rank */
        T cur = sh[(ty + w - 1) * SS + sx];
        if (isnan(cur)) { out[o] = (T)nan(""); return; }
        int cl = 0, ce = 0;
        for (int k = 0; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; } if (v < cur) cl++; else if (v == cur) ce++; }
        out[o] = bad ? (T)nan("") : ((T)cl + (T)0.5 * (T)(ce - 1)) / (T)(w - 1);
    }
}


/* ts_corr / ts_cov:成对 tile;op 0=corr 1=cov;任一侧 NaN→NaN。shared=2*rows*SS*sizeof(T)。 */
template<typename T>
__device__ void pair_core(const T* __restrict__ a, const T* __restrict__ b,
                         T* __restrict__ out, int Tt, int S, int w, int op) {
    T* sh = shmem<T>();
    int SS = blockDim.x, TT = blockDim.y, rows = TT + w - 1;
    T* sha = sh; T* shb = sh + rows * SS;
    int strip_s0 = blockIdx.x * SS, tile_t0 = blockIdx.y * TT, base_t = tile_t0 - (w - 1);
    int tid = threadIdx.y * SS + threadIdx.x, nth = SS * TT;
    for (int idx = tid; idx < rows * SS; idx += nth) {
        int i = idx / SS, sx = idx - i * SS, gt = base_t + i, gs = strip_s0 + sx;
        int oob = (gt < 0 || gt >= Tt || gs >= S);
        sha[i * SS + sx] = oob ? (T)nan("") : a[(long long)gt * S + gs];
        shb[i * SS + sx] = oob ? (T)nan("") : b[(long long)gt * S + gs];
    }
    __syncthreads();
    int sx = threadIdx.x, ty = threadIdx.y, s = strip_s0 + sx, t = tile_t0 + ty;
    if (s >= S || t >= Tt) return;
    long long o = (long long)t * S + s;
    /* K-centering(窗口最旧值,平移不变 → cov/var 不变):压住大幅值(如 close~1e2 vs vol~1e6)
     * 的乘积 Σab 灾难性抵消,这是 f32 下 corr/cov 数值稳定的关键(同 moment 核 std 思路)。*/
    T Ka = sha[ty * SS + sx], Kb = shb[ty * SS + sx];
    T Sa = 0, Sb = 0, Sab = 0, Saa = 0, Sbb = 0; int bad = 0;
    for (int k = 0; k < w; k++) {
        T va = sha[(ty + k) * SS + sx], vb = shb[(ty + k) * SS + sx];
        if (isnan(va) || isnan(vb)) { bad = 1; break; }
        T da = va - Ka, db = vb - Kb;
        Sa += da; Sb += db; Sab += da * db; Saa += da * da; Sbb += db * db;
    }
    if (bad) { out[o] = (T)nan(""); return; }
    T wd = (T)w, ma = Sa / wd, mb = Sb / wd, cov = (Sab - ma * Sb) / (wd - (T)1);
    if (op == 1) { out[o] = cov; return; }
    T va_ = (Saa - ma * Sa) / (wd - (T)1), vb_ = (Sbb - mb * Sb) / (wd - (T)1);
    out[o] = (va_ > 0 && vb_ > 0) ? cov / sqrt(va_ * vb_) : (T)nan("");
}


/* ts_ema:pandas ewm(span,adjust=False,min_periods=span)。序列递推 → 一线程一列(沿 T 串行)。 */
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
 * cross-sectional warp-shuffle 核:op 0=zscore 1=demean 2=scale(L1)
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


/* cs_rank:逐 t 截面平均-rank ∈[0,1];n<2 整行 NaN。一 block 一 t,行入 shared,O(n²) count。 */
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


/* ============================================================================
 * metrics 融合核:逐 t 一 block,warp-shuffle 单遍归约 → (T,) f64
 *   (替 metrics.py 的多遍整盘 .sum(1):per_t_ic ~10 遍 / per_t_pnl ~7 遍 → 1 核单遍)。
 *   累加器恒 double(对齐 metrics.py _F64,消费卡 f32 panel 仍保全盘 reduction 精度)。
 *   块 256 线程(8 warp);S 列 grid-stride 累入线程私有,warp_reduce + shared 跨 warp 归约。
 * ========================================================================== */

/* per_t_ic:逐 t Pearson IC。n<3 或 dx<=0 或 dy<=0 → NaN(贴 fm_per_t_ic / metrics.per_t_ic)。 */
template<typename T>
__device__ void per_t_ic_core(const T* __restrict__ x, const T* __restrict__ y,
                              double* __restrict__ out, int S) {
    int t = blockIdx.x;
    const T* xr = x + (long long)t * S;
    const T* yr = y + (long long)t * S;
    int tid = threadIdx.x, B = blockDim.x, lane = tid & 31, wid = tid >> 5, nwarp = B >> 5;
    double n = 0, sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
    for (int s = tid; s < S; s += B) {
        double vx = (double)xr[s], vy = (double)yr[s];
        if (isfinite(vx) && isfinite(vy)) { n += 1; sx += vx; sy += vy; sxy += vx * vy; sxx += vx * vx; syy += vy * vy; }
    }
    n = warp_reduce_sum<double>(n); sx = warp_reduce_sum<double>(sx); sy = warp_reduce_sum<double>(sy);
    sxy = warp_reduce_sum<double>(sxy); sxx = warp_reduce_sum<double>(sxx); syy = warp_reduce_sum<double>(syy);
    __shared__ double wn[32], wsx[32], wsy[32], wsxy[32], wsxx[32], wsyy[32];
    if (lane == 0) { wn[wid] = n; wsx[wid] = sx; wsy[wid] = sy; wsxy[wid] = sxy; wsxx[wid] = sxx; wsyy[wid] = syy; }
    __syncthreads();
    if (wid == 0) {
        n = (lane < nwarp) ? wn[lane] : 0.0; sx = (lane < nwarp) ? wsx[lane] : 0.0; sy = (lane < nwarp) ? wsy[lane] : 0.0;
        sxy = (lane < nwarp) ? wsxy[lane] : 0.0; sxx = (lane < nwarp) ? wsxx[lane] : 0.0; syy = (lane < nwarp) ? wsyy[lane] : 0.0;
        n = warp_reduce_sum<double>(n); sx = warp_reduce_sum<double>(sx); sy = warp_reduce_sum<double>(sy);
        sxy = warp_reduce_sum<double>(sxy); sxx = warp_reduce_sum<double>(sxx); syy = warp_reduce_sum<double>(syy);
        if (lane == 0) {
            double r;
            if (n < 3.0) r = nan("");
            else {
                double mx = sx / n, my = sy / n;
                double num = sxy - mx * sy, dx = sxx - mx * sx, dy = syy - my * sy;
                r = (dx > 0.0 && dy > 0.0) ? num / sqrt(dx * dy) : nan("");
            }
            out[t] = r;
        }
    }
}

/* per_t_pnl:v=values−cs_mean(finite),w=v/Σ|v|,out=Σ w·y(v&y finite);n_v==0 或 Σ|v|<1e-12 → 0。
 * 两轮归约(先 cs_mean 全块广播,再 Σ|v| 与 Σ v·y),贴 fm_per_t_pnl / metrics.per_t_pnl。 */
template<typename T>
__device__ void per_t_pnl_core(const T* __restrict__ vals, const T* __restrict__ y,
                               double* __restrict__ out, int S) {
    int t = blockIdx.x;
    const T* vr = vals + (long long)t * S;
    const T* yr = y + (long long)t * S;
    int tid = threadIdx.x, B = blockDim.x, lane = tid & 31, wid = tid >> 5, nwarp = B >> 5;
    double sv = 0, nv = 0;
    for (int s = tid; s < S; s += B) { double v = (double)vr[s]; if (isfinite(v)) { sv += v; nv += 1; } }
    sv = warp_reduce_sum<double>(sv); nv = warp_reduce_sum<double>(nv);
    __shared__ double wsv[32], wnv[32], mean_sh; __shared__ int ok_sh;
    if (lane == 0) { wsv[wid] = sv; wnv[wid] = nv; }
    __syncthreads();
    if (wid == 0) {
        double s2 = (lane < nwarp) ? wsv[lane] : 0.0, n2 = (lane < nwarp) ? wnv[lane] : 0.0;
        s2 = warp_reduce_sum<double>(s2); n2 = warp_reduce_sum<double>(n2);
        if (lane == 0) { if (n2 < 1.0) { ok_sh = 0; } else { ok_sh = 1; mean_sh = s2 / n2; } }
    }
    __syncthreads();
    if (!ok_sh) { if (tid == 0) out[t] = 0.0; return; }
    double mean = mean_sh, sab = 0, pnl = 0;
    for (int s = tid; s < S; s += B) {
        double v = (double)vr[s];
        if (isfinite(v)) {
            double dv = v - mean; sab += fabs(dv);
            double yv = (double)yr[s];
            if (isfinite(yv)) pnl += dv * yv;
        }
    }
    sab = warp_reduce_sum<double>(sab); pnl = warp_reduce_sum<double>(pnl);
    __shared__ double wsab[32], wpnl[32];
    if (lane == 0) { wsab[wid] = sab; wpnl[wid] = pnl; }
    __syncthreads();
    if (wid == 0) {
        double a = (lane < nwarp) ? wsab[lane] : 0.0, p = (lane < nwarp) ? wpnl[lane] : 0.0;
        a = warp_reduce_sum<double>(a); p = warp_reduce_sum<double>(p);
        if (lane == 0) out[t] = (a < 1e-12) ? 0.0 : (p / a);
    }
}


/* ============================================================================
 * qnt_bt:crypto 组合回测批量核 = Quantiacs qnt 1:1(engine/qnt_bt.cpp 的 CUDA 镜像)
 *   **一 warp 一候选**(2026-09-19;grid=ceil(B/4), block=128 = 4 warp = 4 候选):截面 S≤128
 *   由 32 lane 各持 ≤4 个资产的 wt/shares **寄存器**,逐日 ~7 次归约全走 warp shuffle
 *   (零 shared、零 __syncthreads)。前身 block-per-candidate(128 线程 + 7 次 block 归约/日)
 *   在 A800 上是纯延迟束缚:B=32/128/512 整批耗时恒 ~36 ms;thread-per-candidate 版(零同步、
 *   W 转 (T,S,B))同一形状实测再慢 10×。二者 2026-09-19 删除,本版为唯一实现。
 *   W (B,T,S) DTYPE 常驻;静态面板 (T,S)(rl.backtest_reward.prepare_qnt_static 产物上卡);
 *   rr (B,T) f64;stats (B,5) f64 = [sharpe, mean_return, volatility, max_drawdown,
 *   total_return](窗 [i0:],arithmetic 年化)。递推标量状态 warp 内冗余复制(同 FP 序 →
 *   确定性一致);累加器/份额恒 double。依赖 IEEE NaN 传染(0×NaN=NaN→归零),与 CPU 内核同语义。
 * ========================================================================== */
#define QNT_LANE_A 4                        /* 每 lane 资产数上限 ⇒ S ≤ 32×4 = 128 */

template<typename A>
__device__ __forceinline__ A warp_sum_bcast(A v) {
    v = warp_reduce_sum<A>(v);
    return __shfl_sync(0xffffffff, v, 0);
}

/* 4 路并行 warp 归约 + 广播:同一条 shuffle 链上并行摊 4 个独立累加量,延迟与单路相同。 */
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

template<typename T, bool MIX>
__device__ __forceinline__ T qnt_weight(const T* W, const T* ref, long long i, double alpha) {
    if (!MIX) return W[i];
    // Keep the materialized f64 expression's separate multiply/add rounding.
    return (T)__dadd_rn(__dmul_rn(1.0-alpha, (double)ref[i]),
                       __dmul_rn(alpha, (double)W[i]));
}

// Explicit FP32 mixing avoids double arithmetic in the low-precision daily path.
template<>
__device__ __forceinline__ float qnt_weight<float,true>(const float* W, const float* ref,
                                                       long long i, double alpha) {
    float a = (float)alpha;
    return __fadd_rn(__fmul_rn(1.0f-a,ref[i]),__fmul_rn(a,W[i]));
}

template<typename T, bool MIX = false, typename A = double>
__device__ void qnt_bt_core(const T* __restrict__ W,
                            const T* __restrict__ open_raw, const T* __restrict__ prev_close,
                            const T* __restrict__ open_ff0, const T* __restrict__ close_ff0,
                            const T* __restrict__ slip, const unsigned char* __restrict__ unlocked,
                            int Tn, int S, int B, long long i0, double ppy,
                            double* __restrict__ rr_out, double* __restrict__ stats_out,
                            const T* ref = nullptr, const int* owners = nullptr,
                            const double* weights = nullptr) {
    const int lane = threadIdx.x & 31;
    const long long b = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (b >= B) return;                                    /* b 为 warp 一致量 → 整 warp 退出 */
    const T* Wb = W + (MIX ? owners[b] : b) * (long long)Tn * S;
    const double alpha = MIX ? weights[b] : 0.0;
    double* rr = rr_out + b * (long long)Tn;
    A wt[QNT_LANE_A], shares[QNT_LANE_A], off_prev[QNT_LANE_A];
    /* 每日串行链是延迟束缚(整批耗时不随 B 变):① W 行双缓冲预取(day+1 生效的是 W[day],
     * 在 day 迭代内就发出加载,与本日归约链重叠);② 本日静态面板一次性预取进寄存器;
     * ③ 昨日 open_ff0 寄存器携带,不再回读;④ 4 个独立累加量并到一条 shuffle 链。
     * 数学与逐项归约完全相同(各累加量的求和序不变)⇒ 结果逐位不变。 */
    T wcur[QNT_LANE_A];
    #pragma unroll
    for (int j = 0; j < QNT_LANE_A; j++) {
        const int a = lane + 32 * j;
        wt[j] = A(0); shares[j] = A(0); off_prev[j] = A(0);
        wcur[j] = (a < S) ? qnt_weight<T,MIX>(Wb,ref,a,alpha) : (T)0;
    }

    A eq_after = A(0), prev_tonight = A(1);
    for (int day = 0; day < Tn; day++) {
        const long long r0 = (long long)day * S;
        A oraw[QNT_LANE_A], pcl[QNT_LANE_A], off[QNT_LANE_A], cff[QNT_LANE_A], sl[QNT_LANE_A];
        bool ul[QNT_LANE_A];
        T wnext[QNT_LANE_A];
        #pragma unroll
        for (int j = 0; j < QNT_LANE_A; j++) {
            const int a = lane + 32 * j;
            const bool in = a < S;
            oraw[j] = in ? (A)open_raw[r0 + a] : A(0);
            pcl[j]  = in ? (A)prev_close[r0 + a] : A(1);
            off[j]  = in ? (A)open_ff0[r0 + a] : A(0);
            cff[j]  = in ? (A)close_ff0[r0 + a] : A(0);
            sl[j]   = in ? (A)slip[r0 + a] : A(0);
            ul[j]   = in ? (unlocked[r0 + a] != 0) : false;
            wnext[j] = (in && day + 1 < Tn) ? qnt_weight<T,MIX>(Wb,ref,r0+a,alpha) : (T)0;
        }

        /* --- 当日生效权重:shift-1 → normalize → open 调整 → normalize --- */
        if (day > 0) {
            A part = A(0);
            #pragma unroll
            for (int j = 0; j < QNT_LANE_A; j++) {
                A x = (A)wcur[j];
                if (!isfinite(x)) x = A(0);
                wt[j] = x; part += fabs(x);
            }
            A s1 = warp_sum_bcast(part);
            if (s1 < A(1)) s1 = A(1);
            part = A(0);
            #pragma unroll
            for (int j = 0; j < QNT_LANE_A; j++) {
                A v = (wt[j] / s1) * oraw[j] / pcl[j];
                if (!isfinite(v)) v = A(0);                     /* NaN 价 → 0(含 0×NaN) */
                wt[j] = v; part += fabs(v);
            }
            A s2 = warp_sum_bcast(part);
            if (s2 < A(1)) s2 = A(1);
            #pragma unroll
            for (int j = 0; j < QNT_LANE_A; j++) wt[j] /= s2;
        }

        /* --- equity 递推(calc_relative_return_np 对照)--- */
        A eq_part = A(0), p_sum = A(0), p_unl = A(0), p_lock = A(0);
        #pragma unroll
        for (int j = 0; j < QNT_LANE_A; j++) {
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

        if (w_op < A(1e-7)) {                                 /* qnt EPS;w_op 全 warp 一致 → 分支均匀 */
            eq_after = eq_before;
        } else {
            A part = A(0);
            #pragma unroll
            for (int j = 0; j < QNT_LANE_A; j++) {
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
        for (int j = 0; j < QNT_LANE_A; j++) {
            const int a = lane + 32 * j;
            if (a < S) part2 += (cff[j] - off[j]) * shares[j];
        }
        const A eq_tonight = eq_after + warp_sum_bcast(part2);

        A v = eq_tonight / prev_tonight - A(1);
        if (!isfinite(v)) v = A(0);
        // A low-precision overflow/underflow must trigger precision escalation,
        // never masquerade as the legacy zero-return invalid-value handling.
        if (sizeof(A) == sizeof(float) &&
            (!isfinite(eq_tonight) || !isfinite(eq_after) ||
             !isfinite(prev_tonight) || fabs(prev_tonight) < A(1e-30))) v = (A)__int_as_float(0x7fffffff);
        if (lane == 0) rr[day] = v;
        prev_tonight = eq_tonight;
        #pragma unroll
        for (int j = 0; j < QNT_LANE_A; j++) {
            if (!ul[j]) shares[j] = A(0);                     /* 无价资产份额清零(tonight 之后) */
            off_prev[j] = off[j];
            wcur[j] = wnext[j];
        }
    }
    __syncwarp();                                          /* lane 0 的 rr 全局写 → 统计段全 warp 读 */

    /* --- 统计窗 [i0:]:arithmetic 年化(ddof=0)/ maxDD / 总收益 --- */
    const double n = (double)(Tn - (int)i0);
    double part = 0.0;
    for (int t = (int)i0 + lane; t < Tn; t += 32) part += rr[t];
    const double mean = warp_sum_bcast(part) / n;
    part = 0.0;
    for (int t = (int)i0 + lane; t < Tn; t += 32) { const double d = rr[t] - mean; part += d * d; }
    const double sd = sqrt(warp_sum_bcast(part) / n);
    if (lane == 0) {
        double cum = 1.0, peak = 1.0, dd = 0.0;
        for (int t = (int)i0; t < Tn; t++) {
            cum *= 1.0 + rr[t];
            if (cum > peak) peak = cum;
            const double u = cum / peak - 1.0;
            if (u < dd) dd = u;
        }
        const double mr = mean * ppy, vol = sd * sqrt(ppy);
        double* st = stats_out + b * 5;
        st[0] = mr / vol; st[1] = mr; st[2] = vol; st[3] = dd; st[4] = cum - 1.0;
    }
}

/* ============================================================================
 * extern "C" f64/f32 实例化(cupy 按平名 get_function)
 * ========================================================================== */
#define INST_TS(name, core) \
    extern "C" __global__ void name##_f64(const double* x, double* o, int T, int S, int w, int op, const double* K, int km) { core<double>(x, o, T, S, w, op, K, km); } \
    extern "C" __global__ void name##_f32(const float*  x, float*  o, int T, int S, int w, int op, const float*  K, int km) { core<float >(x, o, T, S, w, op, K, km); }
INST_TS(k_moment, moment_core)

#define INST_EXT(name, core) \
    extern "C" __global__ void name##_f64(const double* x, double* o, int T, int S, int w, int op) { core<double>(x, o, T, S, w, op); } \
    extern "C" __global__ void name##_f32(const float*  x, float*  o, int T, int S, int w, int op) { core<float >(x, o, T, S, w, op); }
INST_EXT(k_ext, ext_core)

#define INST_PAIR(name, core) \
    extern "C" __global__ void name##_f64(const double* a, const double* b, double* o, int T, int S, int w, int op) { core<double>(a, b, o, T, S, w, op); } \
    extern "C" __global__ void name##_f32(const float*  a, const float*  b, float*  o, int T, int S, int w, int op) { core<float >(a, b, o, T, S, w, op); }
INST_PAIR(k_pair, pair_core)

extern "C" __global__ void k_ema_f64(const double* x, double* o, int T, int S, int span) { ema_core<double>(x, o, T, S, span); }
extern "C" __global__ void k_ema_f32(const float*  x, float*  o, int T, int S, int span) { ema_core<float >(x, o, T, S, span); }

extern "C" __global__ void k_cs_f64(const double* x, double* o, int S, int op) { cs_core<double>(x, o, S, op); }
extern "C" __global__ void k_cs_f32(const float*  x, float*  o, int S, int op) { cs_core<float >(x, o, S, op); }

extern "C" __global__ void k_cs_rank_f64(const double* x, double* o, int S) { cs_rank_core<double>(x, o, S); }
extern "C" __global__ void k_cs_rank_f32(const float*  x, float*  o, int S) { cs_rank_core<float >(x, o, S); }

extern "C" __global__ void k_per_t_ic_f64(const double* x, const double* y, double* o, int S) { per_t_ic_core<double>(x, y, o, S); }
extern "C" __global__ void k_per_t_ic_f32(const float*  x, const float*  y, double* o, int S) { per_t_ic_core<float >(x, y, o, S); }
extern "C" __global__ void k_per_t_pnl_f64(const double* v, const double* y, double* o, int S) { per_t_pnl_core<double>(v, y, o, S); }
extern "C" __global__ void k_per_t_pnl_f32(const float*  v, const float*  y, double* o, int S) { per_t_pnl_core<float >(v, y, o, S); }

extern "C" __global__ void k_qnt_bt_f64(const double* W, const double* o_raw, const double* pcl,
                                        const double* off, const double* cff, const double* sl,
                                        const unsigned char* ul, int T, int S, int B, long long i0,
                                        double ppy, double* rr, double* st) {
    qnt_bt_core<double>(W, o_raw, pcl, off, cff, sl, ul, T, S, B, i0, ppy, rr, st);
}
extern "C" __global__ void k_qnt_combo_f64(const double* W, const double* ref,
                                        const int* owners, const double* weights,
                                        const double* o_raw, const double* pcl,
                                        const double* off, const double* cff, const double* sl,
                                        const unsigned char* ul, int T, int S, int B, long long i0,
                                        double ppy, double* rr, double* st) {
    qnt_bt_core<double,true>(W, o_raw, pcl, off, cff, sl, ul, T, S, B, i0, ppy, rr, st,
                            ref, owners, weights);
}
extern "C" __global__ void k_qnt_bt_f32(const float* W, const float* o_raw, const float* pcl,
                                        const float* off, const float* cff, const float* sl,
                                        const unsigned char* ul, int T, int S, int B, long long i0,
                                        double ppy, double* rr, double* st) {
    qnt_bt_core<float >(W, o_raw, pcl, off, cff, sl, ul, T, S, B, i0, ppy, rr, st);
}

// Mixed precision: FP32 holdings/prices/daily state, FP64 return
// storage and all final statistics. Factor construction remains independent.
extern "C" __global__ void k_qnt_bt_mixed(const float* W, const float* o_raw, const float* pcl,
                                        const float* off, const float* cff, const float* sl,
                                        const unsigned char* ul, int T, int S, int B, long long i0,
                                        double ppy, double* rr, double* st) {
    qnt_bt_core<float,false,float>(W,o_raw,pcl,off,cff,sl,ul,T,S,B,i0,ppy,rr,st);
}
extern "C" __global__ void k_qnt_combo_mixed(const float* W, const float* ref,
                                        const int* owners, const double* weights,
                                        const float* o_raw, const float* pcl,
                                        const float* off, const float* cff, const float* sl,
                                        const unsigned char* ul, int T, int S, int B, long long i0,
                                        double ppy, double* rr, double* st) {
    qnt_bt_core<float,true,float>(W,o_raw,pcl,off,cff,sl,ul,T,S,B,i0,ppy,rr,st,ref,owners,weights);
}
