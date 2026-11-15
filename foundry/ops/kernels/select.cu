// Each asset lane gathers one window. Integer quickselect preserves exact order
// without materialising/sorting a (days, assets, window) device tensor.
__device__ __forceinline__ double rank_value(unsigned long long key) {
    unsigned long long bits=(key>>63)?key^(1ULL<<63):~key;
    return __longlong_as_double((long long)bits);
}
template<int N> __device__ void median_core(const double* x,double* out,int T,int S,int w) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=T*S)return;
    if(index/S<w-1){out[index]=nan("");return;}
    unsigned long long values[N];
    for(int j=0;j<w;++j) {
        unsigned long long key=rank_key(x[index-(w-1-j)*S]);
        if(key==~0ULL){out[index]=nan("");return;}
        values[j]=key;
    }
    int target=w/2,left=0,right=w-1;
    while(left<right) {
        unsigned long long pivot=values[(left+right)/2];
        int i=left,j=right;
        while(i<=j) {
            while(values[i]<pivot)++i;
            while(values[j]>pivot)--j;
            if(i<=j){unsigned long long swap=values[i];values[i++]=values[j];values[j--]=swap;}
        }
        if(target<=j)right=j;
        else if(target>=i)left=i;
        else break;
    }
    double hi=rank_value(values[target]);
    if(w&1){out[index]=hi;return;}
    unsigned long long lower=values[0];
    for(int j=1;j<target;++j)lower=lower>values[j]?lower:values[j];
    out[index]=(rank_value(lower)+hi)*0.5;
}
extern "C" __global__ void k_median32(const double* x,double* o,int T,int S,int w){median_core<32>(x,o,T,S,w);}
extern "C" __global__ void k_median128(const double* x,double* o,int T,int S,int w){median_core<128>(x,o,T,S,w);}
extern "C" __global__ void k_median512(const double* x,double* o,int T,int S,int w){median_core<512>(x,o,T,S,w);}
extern "C" __global__ void k_median2048(const double* x,double* o,int T,int S,int w){median_core<2048>(x,o,T,S,w);}
