// One thread owns a chronological account; candidates are independent.
template<class R> __device__ void order_step_device(
    const double* panels,const unsigned char* rules,double* assets,double* cash,long long* meta,
    long long* orders,double* values,const long long* commands,const double* command_values,
    long long* audit,double* audit_values,long long* counts,double* owner_quantity,long long* highwater,
    const double* targets,const unsigned char* updates,const long long* units,const double* coefficients,
    double* pending,unsigned char* dirty,double* activity,unsigned int* digest,double* locked,double* scratch,
    long long B,long long S,long long Q,long long K,long long F,long long time,long long M,
    int margin_mode,int path,int measured,double leverage,double gross,double participation) {
    namespace o=foundry_orders;
    const long long b=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=B)return;
    o::Input in{panels,rules,S,Q,K,F,time,M,margin_mode,path,measured,leverage,gross,participation};
    o::State state{assets+b*S*o::ASSET_VALUES,cash+b*o::CASH_VALUES,meta+b*o::META_VALUES,
        orders+b*Q*o::ORDER_INTS,values+b*Q*o::ORDER_VALUES,commands+b*K*o::COMMAND_INTS,
        command_values+b*K*o::COMMAND_VALUES,audit+b*F*o::AUDIT_INTS,audit_values+b*F*o::AUDIT_VALUES,counts+b,
        owner_quantity+b*M*S,highwater+b*M,targets+b*M*S,updates+b*M*S,units,coefficients+b*M,
        pending+b*M*S,dirty+b*M*S,activity+b*o::ACTIVITY_VALUES,digest+b*foundry_order_digest::WORDS,locked+b*M*S};
    // Scratch stores R directly, avoiding a per-thread asset-count-sized stack.
    R* capacity=reinterpret_cast<R*>(scratch)+b*S;
    o::Account<R>{in,state}.run(capacity);
}
#define ORDER_ARGS const double* p,const unsigned char* r,double* a,double* c,long long* m,long long* o,double* v,const long long* ci,const double* cv,long long* ai,double* av,long long* n,double* oq,long long* hi,const double* ta,const unsigned char* up,const long long* un,const double* co,double* pe,unsigned char* di,double* ac,unsigned int* dg,double* lo,double* scratch,long long B,long long S,long long Q,long long K,long long F,long long time,long long M,int margin_mode,int path,int measured,double leverage,double gross,double participation
#define ORDER_PASS p,r,a,c,m,o,v,ci,cv,ai,av,n,oq,hi,ta,up,un,co,pe,di,ac,dg,lo,scratch,B,S,Q,K,F,time,M,margin_mode,path,measured,leverage,gross,participation
extern "C" __global__ void orders_paired(ORDER_ARGS) {order_step_device<FloatPair>(ORDER_PASS);}
extern "C" __global__ void orders_mixed(ORDER_ARGS) {order_step_device<float>(ORDER_PASS);}
#undef ORDER_ARGS
#undef ORDER_PASS

extern "C" __global__ void orders_digest_finish(unsigned int* state,long long B) {
    const long long b=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(b<B)foundry_order_digest::finish(state+b*foundry_order_digest::WORDS);
}

extern "C" __global__ void orders_sample_independent(const double* quantities,const double* equities,
    const double* marks,const double* unallocated,double* state,double* output,long long B,long long M,long long S) {
    const long long b=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(b<B)foundry_orders::sample_independent<FloatPair>(quantities+b*M*S,equities+b*M,marks,
        FloatPair(unallocated[b]),M,S,state+b*7,output+b*6);
}

extern "C" __global__ void orders_independent_turnover(const double* values,const double* unallocated,
    double* state,long long B,long long M) {
    const long long b=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(b<B)foundry_orders::independent_turnover<FloatPair>(values+b*M*2,FloatPair(unallocated[b]),M,state+b*7);
}
