// One recurrence definition serves global and asynchronously staged histories.
// Preserve chronological rounding, missing values and the native reset clock.
template<bool slope>
__device__ void rolling_history(const double* x, int stride, double* out, int* counts,
        int T, int S, int s, int w, int begin, int end) {
    double sum=0., weighted=0.; int count=0, lo=0;
    if(slope) {
        while(lo<end && isnan(x[lo*stride]))++lo;
        for(int t=begin;t<min(end,lo+w-1);++t)out[t*S+s]=NAN;
        if(lo+w>end)return;
        for(int k=0;k<w;++k) {
            double v=x[(lo+k)*stride];
            if(isnan(v))++count;else{sum+=v;weighted+=double(k)*v;}
        }
        double mean=(w-1)/2.;
        if(lo+w-1>=begin)out[(lo+w-1)*S+s]=count?NAN:weighted-mean*sum;
        for(int t=lo+w;t<end;++t) {
            if(((t-lo)&65535)==0) {
                sum=weighted=0.;count=0;
                for(int k=0;k<w;++k) {
                    double v=x[(t-w+1+k)*stride];
                    if(isnan(v))++count;else{sum+=v;weighted+=double(k)*v;}
                }
            } else {
                double next=x[t*stride],old=x[(t-w)*stride];
                bool nn=isnan(next),no=isnan(old);
                double nv=nn?0.:next,ov=no?0.:old;
                weighted=weighted-sum+ov+double(w-1)*nv;sum=sum+nv-ov;
                count+=int(nn)-int(no);
            }
            if(t>=begin)out[t*S+s]=count?NAN:weighted-mean*sum;
        }
    } else {
        for(int t=0;t<end;++t) {
            double v=x[t*stride];
            if(!isnan(v)){sum+=v;++count;}
            if(t>=w){double old=x[(t-w)*stride];if(!isnan(old)){sum-=old;--count;}}
            if(t>=begin) {
                out[t*S+s]=count>0&&!isnan(v)?sum:NAN;
                counts[t*S+s]=count;
            }
        }
    }
}

// Four interleaved assets fill a warp's recurrence lanes and align TMA copies.
extern "C" __global__ void k_history_pack(const double* x,double* packed,int T,int S,int pitch) {
    __shared__ double panel[32][33];
    int s=blockIdx.x*32+threadIdx.x,t=blockIdx.y*32+threadIdx.y;
    for(int j=0;j<32;j+=8)panel[threadIdx.y+j][threadIdx.x]=
        s<S && t+j<T?x[(t+j)*S+s]:NAN;
    __syncthreads();
    int group=blockIdx.x*8+threadIdx.y,asset=threadIdx.x%4;
    t=blockIdx.y*32+threadIdx.x/4;
    for(int j=0;j<32;j+=8)if(group*4<S && t+j<pitch)
        packed[(group*pitch+t+j)*4+asset]=panel[threadIdx.x/4+j][threadIdx.y*4+asset];
}

template<bool slope>
__device__ void staged_history(const double* x,double* out,int* counts,int T,int S,int w,int pitch,int tile) {
    extern __shared__ __align__(16) unsigned char storage[];
    double* history=reinterpret_cast<double*>(storage);
    int s=blockIdx.x*4+threadIdx.x%4;
#if __CUDA_ARCH__ >= 900
    __shared__ unsigned long long barrier;
    unsigned bar=static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
    if(threadIdx.x==0) {
        unsigned dst=static_cast<unsigned>(__cvta_generic_to_shared(history));
        unsigned bytes=pitch*4*sizeof(double);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
            :: "r"(dst), "l"(x+blockIdx.x*pitch*4), "r"(bytes), "r"(bar) : "memory");
    }
    __syncthreads();
    asm volatile("{ .reg .pred ready; wait_copy: mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], 0; @!ready bra wait_copy; }"
        :: "r"(bar) : "memory");
#else
    for(int t=threadIdx.x;t<pitch*4;t+=blockDim.x)history[t]=x[blockIdx.x*pitch*4+t];
    __syncthreads();
#endif
    int begin=(threadIdx.x/4)*tile,end=min(T,begin+tile);
    if(begin<T && s<S)rolling_history<slope>(history+threadIdx.x%4,4,out,counts,T,S,s,w,begin,end);
}

extern "C" __global__ void k_slope_staged(const double* x,double* out,int* counts,int T,int S,int w,int pitch,int tile) {
    staged_history<true>(x,out,counts,T,S,w,pitch,tile);
}
extern "C" __global__ void k_av_diff_staged(const double* x,double* out,int* counts,int T,int S,int w,int pitch,int tile) {
    staged_history<false>(x,out,counts,T,S,w,pitch,tile);
}
extern "C" __global__ void k_slope(const double* x,double* out,int* counts,int T,int S,int w,int tile) {
    int s=blockIdx.x*blockDim.x+threadIdx.x,begin=blockIdx.y*tile;
    if(s<S)rolling_history<true>(x+s,S,out,counts,T,S,s,w,begin,min(T,begin+tile));
}
extern "C" __global__ void k_av_diff(const double* x,double* out,int* counts,int T,int S,int w,int tile) {
    int s=blockIdx.x*blockDim.x+threadIdx.x,begin=blockIdx.y*tile;
    if(s<S)rolling_history<false>(x+s,S,out,counts,T,S,s,w,begin,min(T,begin+tile));
}

// Normalize all output cells together, outside the divergent prefix loops.
extern "C" __global__ void k_history_finish(const double* x,double* out,const int* counts,int n,int w,int slope) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    if(slope)out[i]=out[i]/double(w)/((double(w)*w-1.)/12.);
    else out[i]=x[i]-out[i]/double(counts[i]);
}
