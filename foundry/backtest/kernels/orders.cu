// One thread owns a chronological account; candidates are independent.
template<class R> __device__ void order_step_device(
    const double* panels,const unsigned char* rules,double* assets,double* cash,long long* meta,
    long long* orders,double* values,const long long* commands,const double* command_values,
    long long* audit,double* audit_values,long long* counts,double* scratch,
    long long B,long long S,long long Q,long long K,long long F,long long time,
    int isolated,int path,int measured,double leverage,double gross,double participation) {
    namespace o=foundry_orders;
    const long long b=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=B)return;
    o::Input in{panels,rules,S,Q,K,F,time,isolated,path,measured,leverage,gross,participation};
    o::State state{assets+b*S*o::ASSET_VALUES,cash+b*o::CASH_VALUES,meta+b*o::META_VALUES,
        orders+b*Q*o::ORDER_INTS,values+b*Q*o::ORDER_VALUES,commands+b*K*o::COMMAND_INTS,
        command_values+b*K*o::COMMAND_VALUES,audit+b*F*o::AUDIT_INTS,audit_values+b*F*o::AUDIT_VALUES,counts+b};
    // Scratch stores R directly, avoiding a per-thread asset-count-sized stack.
    R* capacity=reinterpret_cast<R*>(scratch)+b*S;
    o::Account<R>{in,state}.run(capacity);
}
#define ORDER_ARGS const double* p,const unsigned char* r,double* a,double* c,long long* m,long long* o,double* v,const long long* ci,const double* cv,long long* ai,double* av,long long* n,double* scratch,long long B,long long S,long long Q,long long K,long long F,long long time,int isolated,int path,int measured,double leverage,double gross,double participation
#define ORDER_PASS p,r,a,c,m,o,v,ci,cv,ai,av,n,scratch,B,S,Q,K,F,time,isolated,path,measured,leverage,gross,participation
extern "C" __global__ void orders_paired(ORDER_ARGS) {order_step_device<FloatPair>(ORDER_PASS);}
extern "C" __global__ void orders_mixed(ORDER_ARGS) {order_step_device<float>(ORDER_PASS);}
#undef ORDER_ARGS
#undef ORDER_PASS
