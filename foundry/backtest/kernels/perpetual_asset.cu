// One warp owns a book; lanes own assets. Hourly funding, risk checks and daily
// rebalance remain chronological. CPU's scalar recurrence is the reference.
template<class R> __device__ R pabs(R x) { return x<R(0.) ? -x:x; }
template<class R> __device__ R pmax(R x,R y) { return x>y?x:y; }
template<class R> __device__ R pmin(R x,R y) { return x<y?x:y; }
template<class R> __device__ R ratio(R gross,R equity) {
    if(gross==R(0.)) return R(0.);
    if(equity<=R(0.)) return R(__int_as_float(0x7f800000));
    return gross/equity;
}
template<class R> struct PerpArrays {
    const R *trade,*open,*high,*low,*close,*funding,*funding_mark,*maintenance,*fee,*liqfee;
};
// Lanes own assets; hourly settlement stays chronological.
// Rebalance reductions run across the warp instead of serializing all assets
// through lane zero. The same code is instantiated at each arithmetic width.
#define SLOTS ((ASSETS+31)/32)
template<class R> __device__ R asset_liquidate(R price,R quantity,R balance,R rate,
    int asset,int event,int stage,unsigned char* flags,long long& first,
    long long& count,R& fee,R& loss) {
    R nominal=rate*pabs(quantity)*price;
    fee+=pmin(pmax(balance,R(0.)),nominal);loss+=pmax(-balance,R(0.));
    flags[(long long)event*ASSETS+asset]|=stage;count++;
    if(first<0)first=event;
    return pmax(balance-nominal,R(0.));
}
template<class R> __device__ void asset_account(const R* weights,PerpArrays<R> in,
    const long long* times,const long long* offsets,const unsigned char* rebalance,
    int days,R leverage,bt::PerpetualOutput out) {
    const int lane=threadIdx.x;
    R q[SLOTS],bucket[SLOTS],last[SLOTS];
    R cash(1.),previous(1.),maximum(0.);bool bankrupt=false;
    long long first=-1;
    for(int j=0;j<SLOTS;j++){int a=lane+32*j;q[j]=bucket[j]=R(0.);last[j]=a<ASSETS?in.open[a]:R(0.);}
    for(int day=0;day<days;day++) {
        R funding(0.),fees(0.),liqfees(0.),loss(0.);long long count=0;
        for(int event=offsets[day];event<offsets[day+1];event++) {
            for(int j=0;j<SLOTS;j++) {int a=lane+32*j;if(a<ASSETS)out.liquidation_events[(long long)event*ASSETS+a]=0;}
            if(bankrupt)continue;
            R equity(0.),gross(0.);
            for(int j=0;j<SLOTS;j++) {
                int a=lane+32*j;if(a>=ASSETS)continue;int k=event*ASSETS+a;
                bucket[j]+=q[j]*(in.open[k]-last[j]);
                equity+=bucket[j];gross+=pabs(q[j])*in.open[k];
            }
            equity=cash+warp_sum_bcast(equity);gross=warp_sum_bcast(gross);
            maximum=pmax(maximum,ratio(gross,equity));
            R residual(0.);
            for(int j=0;j<SLOTS;j++) {
                int a=lane+32*j;if(a>=ASSETS)continue;int k=event*ASSETS+a;
                if(q[j]!=R(0.)&&bucket[j]<=in.maintenance[k]*pabs(q[j])*in.open[k]) {
                    residual+=asset_liquidate(in.open[k],q[j],bucket[j],in.liqfee[k],a,event,1,
                        out.liquidation_events,first,count,liqfees,loss);q[j]=bucket[j]=R(0.);
                }
                R flow=-q[j]*in.funding_mark[k]*in.funding[k];bucket[j]+=flow;funding+=flow;
                if(q[j]!=R(0.)&&bucket[j]<=in.maintenance[k]*pabs(q[j])*in.open[k]) {
                    residual+=asset_liquidate(in.open[k],q[j],bucket[j],in.liqfee[k],a,event,2,
                        out.liquidation_events,first,count,liqfees,loss);q[j]=bucket[j]=R(0.);
                }
            }
            cash+=warp_sum_bcast(residual);equity=R(0.);
            for(int j=0;j<SLOTS;j++)equity+=bucket[j];equity=cash+warp_sum_bcast(equity);
            if(equity<=R(0.))bankrupt=true;
            if(!bankrupt&&rebalance[event]&&day>0) {
                R A(0.),C(0.),zero_fee(0.),alpha[SLOTS],w[SLOTS];
                for(int j=0;j<SLOTS;j++) {
                    int a=lane+32*j;w[j]=alpha[j]=R(0.);if(a>=ASSETS)continue;int k=event*ASSETS+a;
                    w[j]=weights[(day-1)*ASSETS+a];alpha[j]=w[j]/in.open[k];
                    A+=q[j]*(in.trade[k]-in.open[k]);C+=w[j]*(R(1.)-in.trade[k]/in.open[k]);
                    zero_fee+=in.fee[k]*pabs(q[j])*in.trade[k];
                }
                A=equity+warp_sum_bcast(A);C=warp_sum_bcast(C);zero_fee=warp_sum_bcast(zero_fee);
                if(A-zero_fee<=R(0.)) {if(lane==0)fees+=pmin(pmax(A,R(0.)),zero_fee);bankrupt=true;}
                else {
                    R nav=A/(R(1.)-C);
                    for(int pass=0;pass<ASSETS+2;pass++) {
                        R intercept(0.),slope(0.);
                        for(int j=0;j<SLOTS;j++) {
                            int a=lane+32*j;if(a>=ASSETS)continue;int k=event*ASSETS+a;
                            R sign=alpha[j]*nav-q[j]<R(0.)?R(-1.):R(1.);
                            R cost=in.fee[k]*in.trade[k]*sign;intercept+=cost*q[j];slope+=cost*alpha[j];
                        }
                        intercept=A+warp_sum_bcast(intercept);slope=R(1.)-C+warp_sum_bcast(slope);
                        R next=intercept/slope;if(next==nav)break;nav=next;
                    }
                    R collateral(0.);
                    for(int j=0;j<SLOTS;j++) {
                        int a=lane+32*j;if(a>=ASSETS)continue;int k=event*ASSETS+a;R next=alpha[j]*nav;
                        fees+=in.fee[k]*pabs(next-q[j])*in.trade[k];q[j]=next;
                        bucket[j]=pabs(w[j])/leverage*nav;collateral+=bucket[j];
                    }
                    cash=nav-warp_sum_bcast(collateral);equity=R(0.);gross=R(0.);
                    for(int j=0;j<SLOTS;j++) {
                        int a=lane+32*j;if(a>=ASSETS)continue;
                        equity+=bucket[j];gross+=pabs(q[j])*in.open[event*ASSETS+a];
                    }
                    maximum=pmax(maximum,ratio(warp_sum_bcast(gross),cash+warp_sum_bcast(equity)));
                    residual=R(0.);
                    for(int j=0;j<SLOTS;j++) {
                        int a=lane+32*j;if(a>=ASSETS)continue;int k=event*ASSETS+a;
                        if(q[j]!=R(0.)&&bucket[j]<=in.maintenance[k]*pabs(q[j])*in.open[k]) {
                            residual+=asset_liquidate(in.open[k],q[j],bucket[j],in.liqfee[k],a,event,4,
                                out.liquidation_events,first,count,liqfees,loss);q[j]=bucket[j]=R(0.);
                        }
                    }
                    cash+=warp_sum_bcast(residual);
                }
            }
            if(!bankrupt) {
                residual=R(0.);equity=R(0.);gross=R(0.);
                for(int j=0;j<SLOTS;j++) {
                    int a=lane+32*j;if(a>=ASSETS)continue;int k=event*ASSETS+a;
                    R bad=q[j]>R(0.)?in.low[k]:in.high[k];
                    R adverse=bucket[j]+q[j]*(bad-in.open[k]);
                    if(q[j]!=R(0.)&&adverse<=in.maintenance[k]*pabs(q[j])*bad) {
                        R price=(q[j]*in.open[k]-bucket[j])/(q[j]-in.maintenance[k]*pabs(q[j]));
                        R balance=bucket[j]+q[j]*(price-in.open[k]);
                        residual+=asset_liquidate(price,q[j],balance,in.liqfee[k],a,event,8,
                            out.liquidation_events,first,count,liqfees,loss);q[j]=bucket[j]=R(0.);
                    }
                    bucket[j]+=q[j]*(in.close[k]-in.open[k]);last[j]=in.close[k];
                    equity+=bucket[j];gross+=pabs(q[j])*in.close[k];
                }
                cash+=warp_sum_bcast(residual);equity=cash+warp_sum_bcast(equity);
                maximum=pmax(maximum,ratio(warp_sum_bcast(gross),equity));
                if(equity<=R(0.))bankrupt=true;
            }
            if(bankrupt){cash=R(0.);for(int j=0;j<SLOTS;j++)q[j]=bucket[j]=R(0.);}
        }
        R equity(0.);for(int j=0;j<SLOTS;j++)equity+=bucket[j];equity=cash+warp_sum_bcast(equity);
        funding=warp_sum_bcast(funding);fees=warp_sum_bcast(fees);liqfees=warp_sum_bcast(liqfees);loss=warp_sum_bcast(loss);
        count=warp_sum_bcast(count);
        if(lane==0) {
            out.returns[day]=bankrupt&&previous==R(0.)?0.:double(equity/previous-R(1.));
            out.funding[day]=double(funding);out.trading_fees[day]=double(fees);
            out.liquidation_fees[day]=double(liqfees);out.isolated_loss_cap[day]=double(loss);out.liquidations[day]=count;
        }
        previous=equity;
    }
    long long earliest=first<0?0x7fffffffffffffffLL:first;
    for(int n=16;n;n/=2){long long other=__shfl_down_sync(0xffffffffu,earliest,n);earliest=earliest<other?earliest:other;}
    if(lane==0) {
        *out.first_liquidation=earliest==0x7fffffffffffffffLL?-1:times[earliest];
        *out.bankrupt=bankrupt;*out.ending_equity=double(previous);*out.max_gross=double(maximum);
    }
}
#define ASSET_ENTRY(NAME,R) \
extern "C" __global__ void NAME(const R* weights,const long long* times,const long long* offsets, \
 const unsigned char* rebalance,const R* trade,const R* open,const R* high,const R* low,const R* close, \
 const R* funding,const R* funding_mark,const R* maintenance,const R* fee,const R* liqfee, \
 long long events,long long days,double leverage,double* returns,double* funding_out,double* fees, \
 double* liquidation_fees,double* loss_cap,long long* counts,unsigned char* flags,long long* first, \
 unsigned char* bankrupt,double* ending,double* maximum) { \
 long long b=blockIdx.x; \
 bt::PerpetualOutput out{returns+b*days,funding_out+b*days,fees+b*days,liquidation_fees+b*days,loss_cap+b*days, \
 counts+b*days,flags+b*events*ASSETS,first+b,bankrupt+b,ending+b,maximum+b}; \
 PerpArrays<R> in{trade,open,high,low,close,funding,funding_mark,maintenance,fee,liqfee}; \
 asset_account(weights+b*days*ASSETS,in,times,offsets,rebalance,(int)days,R(leverage),out); }
ASSET_ENTRY(perpetual_asset_native,double)
