// Immutable per-day prefixes, shared by all candidates. The first settlement
// belongs to the old book and is excluded; no output reads a future event.
extern "C" __global__ void funding_prefix(const double* mark,const double* rate,
    const long long* offsets,double* output,int assets) {
    int day=blockIdx.x, asset=blockIdx.y*32+threadIdx.x;
    if(asset>=assets) return;
    double sum=0.;
    for(long long h=offsets[day];h<offsets[day+1];h++) {
        if(h>offsets[day]) sum+=mark[h*assets+asset]*rate[h*assets+asset];
        output[h*assets+asset]=sum;
    }
}
