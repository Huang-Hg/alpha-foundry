// foundry/ops/kernels/paths.cu — path-dependent operators: out[t] depends on out[t-1], so one thread walks one
// column. Compiled with --fmad=false to keep the CPU reference's rounding.
#define NAN __longlong_as_double((long long)0x7ff8000000000000ULL)

extern "C" __global__ void k_std(const double* x,double* out,int T,int S,int w) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s>=S)return;
    int lo=0;while(lo<T && isnan(x[lo*S+s]))lo++;
    double sum=0.,squares=0.;int missing=0;
    for(int t=0;t<T;t++) {
        int i=t*S+s;out[i]=NAN;
        if(t<lo)continue;
        double v=x[i];
        if(isnan(v))missing++;else{sum+=v;squares+=v*v;}
        if(t-lo>=w) {
            double old=x[i-w*S];
            if(isnan(old))missing--;else{sum-=old;squares-=old*old;}
        }
        if(t-lo>=w-1 && missing==0) {
            double mean=sum/double(w),variance=(squares-mean*sum)/double(w-1);
            if(variance<0.)variance=0.;
            out[i]=sqrt(variance);
        }
    }
}

// Keep the native reference's chronological recurrence and resynchronization.
// A window-local recomputation is not numerically interchangeable after an
// extreme observation leaves the rolling window.
extern "C" __global__ void k_slope(const double* x,double* out,int T,int S,int w) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s>=S)return;
    for(int t=0;t<T;t++)out[t*S+s]=NAN;
    int lo=0;while(lo<T && isnan(x[lo*S+s]))lo++;
    if(lo+w>T)return;
    double sx=0.,skx=0.,mean=(w-1)/2.,var=(double(w)*w-1.)/12.;int missing=0;
    for(int k=0;k<w;k++) {
        double v=x[(lo+k)*S+s];
        if(isnan(v))missing++;else{sx+=v;skx+=double(k)*v;}
    }
    out[(lo+w-1)*S+s]=missing?NAN:(skx-mean*sx)/double(w)/var;
    for(int t=lo+w;t<T;t++) {
        if(((t-lo)&65535)==0) {
            sx=skx=0.;missing=0;
            for(int k=0;k<w;k++) {
                double v=x[(t-w+1+k)*S+s];
                if(isnan(v))missing++;else{sx+=v;skx+=double(k)*v;}
            }
        } else {
            double next=x[t*S+s],old=x[(t-w)*S+s];
            bool nn=isnan(next),no=isnan(old);
            double nv=nn?0.:next,ov=no?0.:old;
            skx=skx-sx+ov+double(w-1)*nv;sx=sx+nv-ov;
            missing+=int(nn)-int(no);
        }
        out[t*S+s]=missing?NAN:(skx-mean*sx)/double(w)/var;
    }
}

extern "C" __global__ void k_pair(const double* a,const double* b,double* out,int T,int S,int w,int mode) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s>=S)return;
    int la=0,lb=0;
    while(la<T && isnan(a[la*S+s]))la++;
    while(lb<T && isnan(b[lb*S+s]))lb++;
    int lo=la>lb?la:lb,missing=0,constant_a=0,constant_b=0;
    double sa=0.,sb=0.,sab=0.,saa=0.,sbb=0.;
    for(int t=0;t<T;t++) {
        int i=t*S+s;out[i]=NAN;
        if(t<lo)continue;
        double va=a[i],vb=b[i];
        constant_a=t>lo && va==a[i-S]?constant_a+1:1;
        constant_b=t>lo && vb==b[i-S]?constant_b+1:1;
        if(isnan(va)||isnan(vb))missing++;
        else{sa+=va;sb+=vb;sab+=va*vb;saa+=va*va;sbb+=vb*vb;}
        if(t-lo>=w) {
            double oa=a[i-w*S],ob=b[i-w*S];
            if(isnan(oa)||isnan(ob))missing--;
            else{sa-=oa;sb-=ob;sab-=oa*ob;saa-=oa*oa;sbb-=ob*ob;}
        }
        if(t-lo>=w-1 && missing==0) {
            double ma=sa/double(w),mb=sb/double(w);
            double cov=(sab-ma*sb)/double(w-1);
            double va=(saa-ma*sa)/double(w-1),vb=(sbb-mb*sb)/double(w-1);
            bool constant=constant_a>=w || constant_b>=w;
            out[i]=mode==1?(constant?0.:cov):((!constant && va>0. && vb>0.)?cov/sqrt(va*vb):NAN);
        }
    }
}

// Native zscore/snr use direct two-pass population variance, including their
// strict sd > 1e-12 rule. Reconstructing variance from raw moments changes
// constant-window validity through cancellation.
extern "C" __global__ void k_standardize(const double* x,double* out,int T,int S,int w,int mode) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=T*S)return;
    if(i/S<w-1){out[i]=NAN;return;}
    int first=i-(w-1)*S;double sum=0.;
    for(int k=0;k<w;k++)sum+=x[first+k*S];
    double mean=sum/double(w),ss=0.;
    for(int k=0;k<w;k++){double d=x[first+k*S]-mean;ss+=d*d;}
    double sd=sqrt(ss/double(w));
    out[i]=sd>1e-12?(mode==0?(x[i]-mean)/sd:mean/sd):NAN;
}

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
