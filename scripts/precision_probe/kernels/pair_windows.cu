// Experimental paired arithmetic for bounded rolling reductions.
extern "C" __global__ void pair_window(const double* x,double* out,int T,int S,int w,int mode,double factor) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=T*S)return;
    if(i/S<w-1){out[i]=nan("");return;}
    int first=i-(w-1)*S;
    FloatPair anchor(x[first]),sum(0),den(0),weight(1),f(factor);
    for(int k=0;k<w;++k) {
        FloatPair v(x[first+k*S]);
        if(mode==3) sum+=FloatPair(k+1)*v;
        else if(mode==4) {sum+=weight*FloatPair(x[i-k*S]);den+=weight;weight*=f;}
        else sum+=v-anchor;
    }
    FloatPair n(w),mean=mode==2?anchor+sum/n:(sum+n*anchor)/n,result;
    if(mode==0) result=sum+n*anchor;
    else if(mode==1)result=mean;
    else if(mode==2) {
        FloatPair mad(0);
        for(int k=0;k<w;++k)mad+=fabs(FloatPair(x[first+k*S])-mean);
        result=mad/n;
    } else if(mode==3)result=sum/(n*FloatPair(w+1)/FloatPair(2));
    else result=sum/den;
    out[i]=double(result);
}
