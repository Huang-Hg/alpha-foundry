// foundry/ops/kernels/paths.cu — path-dependent operators: out[t] depends on out[t-1], so one thread walks one
// column. Compiled with --fmad=false to keep the CPU reference's rounding.
#define NAN __longlong_as_double((long long)0x7ff8000000000000ULL)

// mode 0 trade_when(entry=x, alpha=y, exit=z): exit>0 -> 0; entry>0 -> alpha; else hold.
// mode 1 hump(x, k): take x only when it moves more than k from the held value.
// mode 2 ts_backfill(x, window): the last valid value, if it is less than `window` rows old.
extern "C" __global__ void k_recursive(const double* x, const double* y, const double* z,
        double* out, int T, int S, int mode, int window, double k) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s>=S)return;
    double prev=mode==1?NAN:0.; int last=-1;
    for(int t=0;t<T;t++) {
        int i=t*S+s; double v=x[i];
        if(mode==0) prev=z[i]>0?0.:(v>0?y[i]:prev);
        else if(mode==1) { if(!isnan(v)&&(isnan(prev)||fabs(v-prev)>k))prev=v; }
        else { if(!isnan(v)){prev=v;last=t;} }
        out[i]=(mode==2&&(last<0||t-last>=window))?NAN:prev;
    }
}
// ts_decay_exp_window(x, w, factor): weights factor^j over the last w rows, normalised.
extern "C" __global__ void k_decay(const double* x,double* out,int T,int S,int w,double factor) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=T*S)return;
    if(i/S<w-1){out[i]=NAN;return;}
    double weight=1,den=0,acc=0;
    for(int j=0;j<w;j++){acc+=weight*x[i-j*S];den+=weight;weight*=factor;}
    out[i]=acc/den;
}
