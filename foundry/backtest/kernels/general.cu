/* Full execution rules, one 128-thread block per book. GEN_SLOTS is compiled
 * from asset count, not a market limit. Each thread keeps ceil(S/128) positions.
 * Blocks never communicate: all books/trials run independently on the GPU.
 */
template<typename T, typename A, typename C, bool MIX>
__device__ void general_core(const T* W, const T* ref, const int* owners, const C* weights,
                            const C* eref, const C* eown_all,
                            const T* o_raw, const T* pcl, const T* off, const T* cff,
                            const T* buy, const T* sell, const T* carry,
                            const unsigned char* live, const unsigned char* block, int flags,
                            int Tn, int S, long long i0, double ppy, double* rr_out, double* stats) {
    const int tid = threadIdx.x, b = blockIdx.x;
    const T* Wb = W + (MIX ? owners[b] : b) * (long long)Tn * S;
    const C alpha = MIX ? weights[b] : C(0);
    const C* eown = MIX ? eown_all + owners[b] * (long long)Tn : nullptr;
    double* rr = rr_out + (long long)b * Tn;
    A shares[GEN_SLOTS], last_open[GEN_SLOTS];
    #pragma unroll
    for (int j=0; j<GEN_SLOTS; ++j) { shares[j]=A(0); last_open[j]=A(0); }
    A eq_after=A(0), previous=A(1);
    for (int day=0; day<Tn; ++day) {
        const long long row=(long long)day*S;
        const C den=blend_den<MIX>(eref,eown,day>0?day-1:0,alpha);
        A w[GEN_SLOTS], o[GEN_SLOTS], c[GEN_SLOTS];
        bool ul[GEN_SLOTS], tr[GEN_SLOTS];
        unsigned char bk[GEN_SLOTS];
        A s1=A(0), mtm=A(0);
        #pragma unroll
        for (int j=0; j<GEN_SLOTS; ++j) {
            const int a=tid+128*j;
            w[j]=(day>0 && a<S) ? A(blend_weight<T,MIX>(Wb,ref,row-S+a,alpha,den)) : A(0);
            if (!isfinite(w[j])) w[j]=A(0);
            s1+=fabs(w[j]);
            o[j]=a<S ? A(off[row+a]) : A(0);
            c[j]=a<S ? A(cff[row+a]) : A(0);
            ul[j]=a<S && live[row+a]; tr[j]=ul[j];
            bk[j]=(a<S && (flags&2)) ? block[row+a] : 0;
            if (day>0 && a<S) mtm+=(o[j]-last_open[j])*shares[j];
        }
        s1=block_sum(s1); if(s1<A(1)) s1=A(1);
        const A eq_before=day>0 ? eq_after+block_sum(mtm) : A(1);
        A s2=A(0);
        #pragma unroll
        for(int j=0;j<GEN_SLOTS;++j) {
            const int a=tid+128*j;
            if(day>0 && a<S) {
                A v=(w[j]/s1)*A(o_raw[row+a])/A(pcl[row+a]);
                w[j]=isfinite(v)?v:A(0);
            }
            s2+=fabs(w[j]);
        }
        s2=block_sum(s2); if(s2<A(1)) s2=A(1);
        A total=A(0);
        #pragma unroll
        for(int j=0;j<GEN_SLOTS;++j) { w[j]/=s2; total+=fabs(w[j]); }
        total=block_sum(total);
        const A wd=total>A(1)?total:A(1), cash=wd-total;
        if(flags&2) {
            #pragma unroll
            for(int j=0;j<GEN_SLOTS;++j) {
                const A target=eq_before*w[j]/wd, current=o[j]*shares[j];
                tr[j]=ul[j] && !(((bk[j]&1)&&target>current)||((bk[j]&2)&&target<current));
            }
        }
        A operable, equity;
        for(;;) {
            A unlocked=A(0), locked=A(0);
            #pragma unroll
            for(int j=0;j<GEN_SLOTS;++j) {
                if(tr[j]) unlocked+=fabs(w[j]); else locked+=o[j]*fabs(shares[j]);
            }
            operable=block_sum(unlocked)+cash;
            equity=eq_before-block_sum(locked);
            if(!(flags&2) || operable<A(PRICE_EPS)) break;
            bool changed=false;
            #pragma unroll
            for(int j=0;j<GEN_SLOTS;++j) {
                if(tr[j] && bk[j]) {
                    const A ns=equity*w[j]/(operable*o[j]);
                    if(((bk[j]&1)&&ns>shares[j])||((bk[j]&2)&&ns<shares[j])) {
                        tr[j]=false; changed=true;
                    }
                }
            }
            // Frozen positions grow monotonically; includes the CPU's cascading freeze pass.
            if(!__syncthreads_or(changed)) break;
        }
        A fee=A(0);
        if(operable<A(PRICE_EPS)) eq_after=eq_before;
        else {
            #pragma unroll
            for(int j=0;j<GEN_SLOTS;++j) {
                const int a=tid+128*j;
                if(tr[j]) {
                    const A ns=equity*w[j]/(operable*o[j]);
                    const A dn=day>0?ns-shares[j]:ns;
                    const A cost=((flags&1)&&dn<A(0))?A(sell[row+a]):A(buy[row+a]);
                    fee+=cost*fabs(dn); shares[j]=ns;
                }
            }
            eq_after=eq_before-block_sum(fee);
        }
        A intraday=A(0), financing=A(0);
        #pragma unroll
        for(int j=0;j<GEN_SLOTS;++j) {
            const int a=tid+128*j;
            intraday+=(c[j]-o[j])*shares[j];
            if((flags&4)&&a<S&&shares[j]<A(0)) financing+=A(carry[row+a])*c[j]*(A(0)-shares[j]);
        }
        if(flags&4) eq_after=eq_after-block_sum(financing);
        const A tonight=eq_after+block_sum(intraday);
        A v=tonight/previous-A(1);
        if(!isfinite(v)) v=A(0);
        if(LimitedRange<A>::value && (!isfinite(tonight)||!isfinite(eq_after)||
                                     !isfinite(previous)||fabs(previous)<A(1e-30))) v=A(__int_as_float(0x7fffffff));
        if(tid==0) rr[day]=(double)v;
        previous=tonight;
        #pragma unroll
        for(int j=0;j<GEN_SLOTS;++j) { if(!ul[j]) shares[j]=A(0); last_open[j]=o[j]; }
        rebase(eq_after,previous,shares);
    }
    __syncthreads();

}

#define GENERAL(NAME,T,A,C,MIX) \
extern "C" __global__ void NAME(const T* W,const T* ref,const int* owners,const C* weights, \
 const C* eref,const C* eown,const T* o,const T* p,const T* off,const T* c,const T* buy, \
 const T* sell,const T* carry,const unsigned char* live,const unsigned char* block,int flags, \
 int Tn,int S,long long start,double ppy,double* rr,double* st) { \
 general_core<T,A,C,MIX>(W,ref,owners,weights,eref,eown,o,p,off,c,buy,sell,carry,live,block,flags,Tn,S,start,ppy,rr,st); }
GENERAL(k_general_mixed,float,float,double,false)
GENERAL(k_general_paired,FloatPair,FloatPair,FloatPair,false)
GENERAL(k_general_blend_mixed,float,float,double,true)
GENERAL(k_general_blend_paired,FloatPair,FloatPair,FloatPair,true)
