// One warp owns a book; lanes evaluate all hourly events within each daily
// collateral interval. Funding is a parallel prefix, liquidation a first-hit vote.
// Daily rebalance remains ordered. CPU's scalar recurrence is the reference.
__device__ FloatPair operator-(FloatPair a) { return FloatPair(-a.hi,-a.lo); }
__device__ bool operator==(FloatPair a,FloatPair b) { return a.hi==b.hi && a.lo==b.lo; }
__device__ bool operator!=(FloatPair a,FloatPair b) { return !(a==b); }
__device__ bool operator<=(FloatPair a,FloatPair b) { return !(a>b); }
template<class R> __device__ R pabs(R x) { return x<R(0.) ? -x:x; }
template<class R> __device__ R pmax(R x,R y) { return x>y?x:y; }
template<class R> __device__ R pmin(R x,R y) { return x<y?x:y; }
template<class R> __device__ R ratio(R gross,R equity) {
    if(gross==R(0.)) return R(0.);
    if(equity<=R(0.)) return R(__int_as_float(0x7f800000));
    return gross/equity;
}
// FloatPair's infinity is represented without an inf-inf low component.
template<> __device__ FloatPair ratio(FloatPair gross,FloatPair equity) {
    if(gross==FloatPair(0.)) return FloatPair(0.);
    if(equity<=FloatPair(0.)) return FloatPair(__int_as_float(0x7f800000),0.f);
    return gross/equity;
}
template<class R> __device__ R warpmax(R v) {
    for(int step=16;step;step/=2) v=pmax(v,__shfl_down_sync(0xffffffffu,v,step));
    return __shfl_sync(0xffffffffu,v,0);
}
template<class R> struct PerpArrays {
    const R *trade,*open,*high,*low,*close,*funding,*funding_mark,*maintenance,*fee,*liqfee,*funding_prefix;
};
template<class R> __device__ R liquidate_day(bt::PerpetualOutput out,const long long* times,
    int day,int event,int asset,int stage,R price,R balance,R quantity,R rate) {
    const R nominal=rate*pabs(quantity)*price;
    out.liquidation_fees[day]+=double(pmin(pmax(balance,R(0.)),nominal));
    out.isolated_loss_cap[day]+=double(pmax(-balance,R(0.)));
    out.liquidations[day]++;
    out.liquidation_events[(long long)event*ASSETS+asset]|=stage;
    if(*out.first_liquidation<0 || times[event]<*out.first_liquidation) *out.first_liquidation=times[event];
    return pmax(balance-nominal,R(0.));
}
template<class R> __device__ void warp_account(const R* weights,PerpArrays<R> in,
    const long long* times,const long long* offsets,const unsigned char* rebalance,
    int days,R leverage,bt::PerpetualOutput out,R* q,R* bucket,R* last,R* state,unsigned char* uncertain,R tolerance) {
    const int lane=threadIdx.x;
    // state: cash, prior NAV, maximum sampled leverage.
    if(lane==0) {
        state[0]=state[1]=R(1.);state[2]=R(0.);
        *out.first_liquidation=-1;*out.bankrupt=0;
        for(int a=0;a<ASSETS;a++){q[a]=bucket[a]=R(0.);last[a]=in.open[a];}
    }
    __syncwarp();
    for(int day=0;day<days;day++) {
        const int begin=offsets[day],end=offsets[day+1],hours=end-begin;
        if(lane==0) {
            out.returns[day]=out.funding[day]=out.trading_fees[day]=0.;
            out.liquidation_fees[day]=out.isolated_loss_cap[day]=0.;out.liquidations[day]=0;
        }
        for(int j=lane;j<hours*ASSETS;j+=32) out.liquidation_events[(long long)begin*ASSETS+j]=0;
        __syncwarp();
        if(*out.bankrupt) continue;
        const int first=begin*ASSETS;
        if(lane==0) {
            R equity=state[0],gross=R(0.);
            for(int a=0;a<ASSETS;a++) {
                bucket[a]+=q[a]*(in.open[first+a]-last[a]);
                equity+=bucket[a];gross+=pabs(q[a])*in.open[first+a];
            }
            state[2]=pmax(state[2],ratio(gross,equity));
            for(int a=0;a<ASSETS;a++) {
                int k=first+a;
                R boundary=in.maintenance[k]*pabs(q[a])*in.open[k];
                if(q[a]!=R(0.) && pabs(bucket[a]-boundary)<=tolerance*pmax(pabs(bucket[a])+pabs(q[a])*in.open[k],R(1.))) *uncertain=1;
                if(q[a]!=R(0.) && bucket[a]<=in.maintenance[k]*pabs(q[a])*in.open[k]) {
                    state[0]+=liquidate_day(out,times,day,begin,a,1,in.open[k],bucket[a],q[a],in.liqfee[k]);
                    q[a]=bucket[a]=R(0.);
                }
                R flow=-q[a]*in.funding_mark[k]*in.funding[k];
                bucket[a]+=flow;out.funding[day]+=double(flow);
                if(q[a]!=R(0.) && pabs(bucket[a]-boundary)<=tolerance*pmax(pabs(bucket[a])+pabs(q[a])*in.open[k],R(1.))) *uncertain=1;
                if(q[a]!=R(0.) && bucket[a]<=in.maintenance[k]*pabs(q[a])*in.open[k]) {
                    state[0]+=liquidate_day(out,times,day,begin,a,2,in.open[k],bucket[a],q[a],in.liqfee[k]);
                    q[a]=bucket[a]=R(0.);
                }
            }
            equity=state[0];for(int a=0;a<ASSETS;a++) equity+=bucket[a];
            if(equity<=R(0.)) *out.bankrupt=1;
            if(!*out.bankrupt && rebalance[begin] && day>0) {
                const R* w=weights+(day-1)*ASSETS;
                R A=equity,C=R(0.),zero_fee=R(0.),alpha[ASSETS];
                for(int a=0;a<ASSETS;a++) {
                    int k=first+a;
                    alpha[a]=w[a]/in.open[k];
                    A+=q[a]*(in.trade[k]-in.open[k]);
                    C+=w[a]*(R(1.)-in.trade[k]/in.open[k]);
                    zero_fee+=in.fee[k]*pabs(q[a])*in.trade[k];
                }
                if(pabs(A-zero_fee)<=tolerance*pmax(pabs(A)+zero_fee,R(1.))) *uncertain=1;
                if(A-zero_fee<=R(0.)) {
                    out.trading_fees[day]+=double(pmin(pmax(A,R(0.)),zero_fee));*out.bankrupt=1;
                } else {
                    // Monotone convex piecewise-linear self-financing equation.
                    // Newton from the fee-free upper bound crosses at most S knots.
                    R nav=A/(R(1.)-C);
                    for(int pass=0;pass<ASSETS+2;pass++) {
                        R intercept=A,slope=R(1.)-C;
                        for(int a=0;a<ASSETS;a++) {
                            int k=first+a;
                            R sign=alpha[a]*nav-q[a]<R(0.) ? R(-1.):R(1.);
                            R cost=in.fee[k]*in.trade[k]*sign;
                            intercept+=cost*q[a];slope+=cost*alpha[a];
                        }
                        R next=intercept/slope;
                        if(next==nav) break;
                        nav=next;
                    }
                    R collateral=R(0.),fees=R(0.);
                    for(int a=0;a<ASSETS;a++) {
                        int k=first+a;R next=alpha[a]*nav;
                        fees+=in.fee[k]*pabs(next-q[a])*in.trade[k];
                        q[a]=next;bucket[a]=pabs(w[a])/leverage*nav;collateral+=bucket[a];
                    }
                    state[0]=nav-collateral;out.trading_fees[day]+=double(fees);
                    gross=R(0.);equity=state[0];
                    for(int a=0;a<ASSETS;a++){gross+=pabs(q[a])*in.open[first+a];equity+=bucket[a];}
                    state[2]=pmax(state[2],ratio(gross,equity));
                    for(int a=0;a<ASSETS;a++) {
                        int k=first+a;
                        if(q[a]!=R(0.) && pabs(bucket[a]-in.maintenance[k]*pabs(q[a])*in.open[k])<=tolerance*pmax(pabs(bucket[a])+pabs(q[a])*in.open[k],R(1.))) *uncertain=1;
                        if(q[a]!=R(0.) && bucket[a]<=in.maintenance[k]*pabs(q[a])*in.open[k]) {
                            state[0]+=liquidate_day(out,times,day,begin,a,4,in.open[k],bucket[a],q[a],in.liqfee[k]);
                            q[a]=bucket[a]=R(0.);
                        }
                    }
                }
            }
        }
        __syncwarp();
        if(!*out.bankrupt) {
            int hour=lane<hours?lane:hours-1,k=(begin+hour)*ASSETS;
            R open_nav=state[0],close_nav=state[0],open_gross=R(0.),close_gross=R(0.);
            R residual_total=R(0.);
            for(int a=0;a<ASSETS;a++) {
                R quantity=q[a],collateral=bucket[a];
                R flow=lane<hours && lane>0 ? -quantity*in.funding_mark[k+a]*in.funding[k+a]:R(0.);
                R paid=-quantity*in.funding_prefix[k+a];
                R prior=hour>0 ? -quantity*in.funding_prefix[k-ASSETS+a]:R(0.);
                R balance=collateral+quantity*(in.open[k+a]-in.open[first+a])+prior;
                R after=balance+flow;
                R bad=quantity>R(0.)?in.low[k+a]:in.high[k+a];
                R adverse=after+quantity*(bad-in.open[k+a]);
                R scale=tolerance*pmax(pabs(collateral)+pabs(quantity)*in.open[k+a],R(1.));
                bool near=quantity!=R(0.) && lane<hours &&
                    (pabs(balance-in.maintenance[k+a]*pabs(quantity)*in.open[k+a])<=scale ||
                     pabs(after-in.maintenance[k+a]*pabs(quantity)*in.open[k+a])<=scale ||
                     pabs(adverse-in.maintenance[k+a]*pabs(quantity)*bad)<=scale);
                if(__any_sync(0xffffffffu,near) && lane==0) *uncertain=1;
                int stage=0;
                if(lane<hours && quantity!=R(0.)) {
                    if(lane>0 && balance<=in.maintenance[k+a]*pabs(quantity)*in.open[k+a]) stage=1;
                    else if(lane>0 && after<=in.maintenance[k+a]*pabs(quantity)*in.open[k+a]) stage=2;
                    else if(adverse<=in.maintenance[k+a]*pabs(quantity)*bad) stage=8;
                }
                unsigned hits=__ballot_sync(0xffffffffu,stage!=0);
                int hit=hits?__ffs(hits)-1:hours;
                int picked=hit<hours?hit:hours-1;
                int phase=__shfl_sync(0xffffffffu,stage,picked);
                R price=in.open[k+a],at_touch=balance;
                if(stage==2) at_touch=after;
                if(stage==8) {
                    price=(quantity*in.open[k+a]-after)/(quantity-in.maintenance[k+a]*pabs(quantity));
                    at_touch=after+quantity*(price-in.open[k+a]);
                }
                R liquidation_price=__shfl_sync(0xffffffffu,price,picked);
                R liquidation_balance=__shfl_sync(0xffffffffu,at_touch,picked);
                R rate=__shfl_sync(0xffffffffu,in.liqfee[k+a],picked);
                R residual=hit<hours ? pmax(liquidation_balance-rate*pabs(quantity)*liquidation_price,R(0.)):R(0.);
                R total_funding=__shfl_sync(0xffffffffu,paid,picked);
                R final_flow=__shfl_sync(0xffffffffu,flow,picked);
                if(hit<hours && phase==1) total_funding=total_funding-final_flow;
                open_nav+=lane<=hit ? balance:residual;
                open_gross+=lane<=hit ? pabs(quantity)*in.open[k+a]:R(0.);
                R close_balance=after+quantity*(in.close[k+a]-in.open[k+a]);
                close_nav+=lane<hit ? close_balance:residual;
                close_gross+=lane<hit ? pabs(quantity)*in.close[k+a]:R(0.);
                R final_bucket=__shfl_sync(0xffffffffu,close_balance,hours-1);
                if(lane==0) {
                    out.funding[day]+=double(total_funding);
                    if(hit<hours) {
                        liquidate_day(out,times,day,begin+hit,a,phase,liquidation_price,liquidation_balance,quantity,rate);
                        q[a]=bucket[a]=R(0.);
                    } else bucket[a]=final_bucket;
                    last[a]=in.close[(end-1)*ASSETS+a];
                }
                residual_total+=residual;
                __syncwarp();
            }
            R maximum=lane<hours?pmax(ratio(open_gross,open_nav),ratio(close_gross,close_nav)):R(0.);
            maximum=warpmax(maximum);
            if(lane==0) {
                state[0]+=residual_total;state[2]=pmax(state[2],maximum);
                R equity=state[0];for(int a=0;a<ASSETS;a++) equity+=bucket[a];
                if(equity<=R(0.)) *out.bankrupt=1;
            }
        }
        __syncwarp();
        if(lane==0) {
            R nav=state[0];for(int a=0;a<ASSETS;a++) nav+=bucket[a];
            if(!isfinite(nav) || (LimitedRange<R>::value && nav!=R(0.) &&
                (pabs(nav)<R(2.3283064365386963e-10) || pabs(nav)>R(4294967296.)))) *uncertain=1;
            if(pabs(nav)<=tolerance*pmax(pabs(state[1]),R(1.))) *uncertain=1;
            if(*out.bankrupt) {nav=state[0]=R(0.);for(int a=0;a<ASSETS;a++) q[a]=bucket[a]=R(0.);}
            out.returns[day]=double(nav/state[1]-R(1.));state[1]=nav;
        }
        __syncwarp();
    }
    if(lane==0){*out.ending_equity=double(state[1]);*out.max_gross=double(state[2]);}
}
#define WARP_ENTRY(NAME,R) \
extern "C" __global__ void NAME(const R* weights,const long long* times,const long long* offsets, \
 const unsigned char* rebalance,const R* trade,const R* open,const R* high,const R* low,const R* close, \
 const R* funding,const R* funding_mark,const R* maintenance,const R* fee,const R* liqfee,const R* funding_prefix, \
 long long events,long long days,double leverage,double* returns,double* funding_out,double* fees, \
 double* liquidation_fees,double* loss_cap,long long* counts,unsigned char* flags,long long* first, \
 unsigned char* bankrupt,double* ending,double* maximum,unsigned char* uncertain,double tolerance) { \
 long long b=blockIdx.x; \
 bt::PerpetualOutput out{returns+b*days,funding_out+b*days,fees+b*days,liquidation_fees+b*days,loss_cap+b*days, \
 counts+b*days,flags+b*events*ASSETS,first+b,bankrupt+b,ending+b,maximum+b}; \
 PerpArrays<R> in{trade,open,high,low,close,funding,funding_mark,maintenance,fee,liqfee,funding_prefix}; \
 __shared__ R q[ASSETS],bucket[ASSETS],last[ASSETS],state[3]; \
 warp_account(weights+b*days*ASSETS,in,times,offsets,rebalance,(int)days,R(leverage),out,q,bucket,last,state,uncertain+b,R(tolerance)); }
WARP_ENTRY(perpetual_warp_f64,double)
WARP_ENTRY(perpetual_warp_mixed,float)
WARP_ENTRY(perpetual_warp_paired,FloatPair)
