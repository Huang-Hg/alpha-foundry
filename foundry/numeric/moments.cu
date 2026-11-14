// One block per return row: centered moments, weighted moments and logarithmic
// growth all use pairs. This is the same metric contract as core.metrics.
__device__ FloatPair sum_block(FloatPair x) {
    __shared__ FloatPair parts[4];
    for(int n=16;n;n/=2) x+=__shfl_down_sync(0xffffffffu,x,n);
    if((threadIdx.x&31)==0) parts[threadIdx.x/32]=x;
    __syncthreads();
    if(threadIdx.x==0) parts[0]=((parts[0]+parts[1])+parts[2])+parts[3];
    __syncthreads();
    FloatPair out=parts[0];__syncthreads();return out;
}
extern "C" __global__ void return_moments(const double* returns,const double* weights,
    int days,double ppy,double* output) {
    const double* row=returns+(long long)blockIdx.x*days;
    FloatPair sum(0.f,0.f),weighted=sum,logs=sum,wlogs=sum,origin(row[0]);
    for(int t=threadIdx.x;t<days;t+=blockDim.x) {
        FloatPair r(row[t]),p(weights[t]),lg=log1p(r);
        sum+=r-origin;weighted+=(r-origin)*p;logs+=lg;wlogs+=lg*p;
    }
    FloatPair shift=sum_block(sum)/FloatPair(days),mean=origin+shift;
    FloatPair wshift=sum_block(weighted),wm=origin+wshift,lg=sum_block(logs)/FloatPair(days),wg=sum_block(wlogs);
    FloatPair variance(0.f,0.f),wvariance(0.f,0.f);
    for(int t=threadIdx.x;t<days;t+=blockDim.x) {
        FloatPair r(row[t]),d=(r-origin)-shift,wd=(r-origin)-wshift;
        variance+=d*d;wvariance+=wd*wd*FloatPair(weights[t]);
    }
    FloatPair sd=sqrt(sum_block(variance)/FloatPair(days)),wsd=sqrt(sum_block(wvariance));
    if(threadIdx.x==0) {
        FloatPair annual(ppy),scale=sqrt(annual);double* out=output+4*blockIdx.x;
        out[0]=sd>FloatPair(0.f,0.f)?double(mean/sd*scale):0.;
        out[1]=pair_expm1_output(lg*annual);
        out[2]=wsd>FloatPair(0.f,0.f)?double(wm/wsd*scale):0.;
        out[3]=double(wg*annual);
    }
}
