// Q25 path-dependent primitives. Python only binds arrays and launch dimensions.
#define NAN __longlong_as_double((long long)0x7ff8000000000000ULL)

// Rolling means with separate Kahan compensation for additions/removals.
// Keep threshold decisions stable at window boundaries (same recurrence as pandas).
extern "C" __global__ void q25_rolling_mean(const double* x, double* out, int T, int S, int w) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s>=S)return;
    double sum=0, ca=0, cr=0, previous=NAN;
    int n=0, negative=0, same=0;
    for(int t=0;t<T;++t){
        if(t>=w){
            double v=x[(t-w)*S+s];
            if(!isnan(v)){
                --n; negative-=signbit(v)!=0;
                double y=-v-cr, z=sum+y; cr=(z-sum)-y; sum=z;
            }
        }
        double v=x[t*S+s];
        if(!isnan(v)){
            ++n; negative+=signbit(v)!=0;
            double y=v-ca, z=sum+y; ca=(z-sum)-y; sum=z;
            same=(v==previous)?same+1:1; previous=v;
        }
        double r=NAN;
        if(n>=w){
            r=sum/n;
            if(same>=n)r=previous;
            else if(negative==0 && r<0)r=0;
            else if(negative==n && r>0)r=0;
        }
        out[t*S+s]=r;
    }
}
extern "C" __global__ void q25_recursive(const double* x,const double* y,const double* z,
        double* out,int T,int S,int mode,int window,double k) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s>=S)return;
    double prev=mode==1?NAN:0.; int last=-1;
    for(int t=0;t<T;t++) {
        int i=t*S+s; double v=x[i];
        if(mode==0) prev=z[i]>0?0.:(v>0?y[i]:prev);
        else if(mode==1) { if(!isnan(v)&&(isnan(prev)||fabs(v-prev)>k))prev=v; }
        else if(mode==3) prev=t==0?v:fmax(prev,v);
        else { if(!isnan(v)){prev=v;last=t;} }
        out[i]=(mode==2&&(last<0||t-last>=window))?NAN:prev;
    }
}
extern "C" __global__ void q25_decay(const double* x,double* out,int T,int S,int w,double factor) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=T*S)return;
    if(i/S<w-1){out[i]=NAN;return;}
    double weight=1,den=0,acc=0;
    for(int j=0;j<w;j++){acc+=weight*x[i-j*S];den+=weight;weight*=factor;}
    out[i]=acc/den;
}
extern "C" __global__ void q25_hysteresis(const unsigned char* enter,const unsigned char* exit,
        unsigned char* out,int T,int S) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s>=S)return;
    bool state=false;
    for(int t=0;t<T;t++){int i=t*S+s; if(exit[i])state=false;else if(enter[i])state=true;out[i]=state;}
}
extern "C" __global__ void q25_hold(const double* x,double* out,int T,double enter,double exit,int hold) {
    if(blockIdx.x||threadIdx.x)return;
    bool state=false;int held=0;
    for(int t=0;t<T;t++) {
        if(isfinite(x[t])) {
            if(!state){if(x[t]>enter){state=true;held=0;}}
            else {held++;if(x[t]<exit&&held>=hold)state=false;}
        }
        out[t]=state;
    }
}
extern "C" __global__ void q25_state(const double* c,const unsigned char* ent,const unsigned char* exit,
        const double* stop,const unsigned char* allow,const unsigned char* ent_loss,
        const unsigned char* force,double* out,double* dd,double* gain,
        int T,int S,int flags,int cool,double tight_k,double tight_min,
        double stop_days,double stop_gain,double initial_stop,double initial_gain,double breakeven) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s>=S)return;
    bool state=false,lost=false;double peak=1,pin=1;int age=0,since=1000000;
    for(int t=0;t<T;t++) {
        int i=t*S+s;double price=c[i];bool fin=isfinite(price);
        if(state&&fin&&price>peak)peak=price;
        double g=peak/pin-1,dist=stop[i];
        if(flags&1)dist=fmax(tight_min,dist-tight_k*g);
        if((flags&2)&&g<initial_gain)dist=initial_stop;
        if((flags&4)&&g>=breakeven)dist=fmin(dist,1-pin/peak);
        bool ex=state&&(exit[i]||(fin&&price<peak*(1-dist)));
        if(flags&8)ex=ex||(state&&age>=stop_days&&g<stop_gain);
        if(flags&16)ex=ex||(state&&force[i]);
        if(ex)lost=price<pin;
        state=state&&!ex;
        since=ex?0:since+1;
        bool ev=(flags&32)&&lost?ent_loss[i]:ent[i];
        bool en=!state&&!ex&&ev&&fin&&since>cool&&allow[i];
        state=state||en;age=en?0:age+1;
        if(en){peak=price;pin=price;}
        out[i]=state;dd[i]=state&&fin?1-price/peak:0.;gain[i]=state?peak/pin-1:0.;
    }
}
// Rolling percentile rank used by the legacy alpha interface (average ties).
extern "C" __global__ void q25_rank_pct(const double* x,double* out,int T,int S,int w) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=T*S)return;
    if(i/S<w-1){out[i]=NAN;return;}
    double cur=x[i];int lower=0,equal=0;
    for(int j=0;j<w;j++){double v=x[i-j*S];if(isnan(v)){out[i]=NAN;return;}lower+=v<cur;equal+=v==cur;}
    out[i]=(lower+(equal+1)*0.5)/w;
}
