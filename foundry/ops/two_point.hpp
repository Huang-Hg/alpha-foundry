/* Closed forms for two observations. Raw moments lose the small differences
 * when an upstream logarithm differs by one ulp, manufacturing a signal from
 * a zero residual or moving a Pearson correlation away from exactly +/-1. */
#ifdef __CUDACC__
#define FM_POINT __host__ __device__ inline
#else
#include <cmath>
using std::isfinite;
using std::nan;
#define FM_POINT inline
#endif
template<class T> FM_POINT T two_point_moment(T a,T b,int order) {
    return !isfinite(a)||!isfinite(b)||a==b ? T(nan("")) : T(order==3?0:-2);
}
template<class T> FM_POINT T two_point_pair(T a,T b,T c,T d,int mode) {
    if(!isfinite(a)||!isfinite(b)||!isfinite(c)||!isfinite(d))return T(nan(""));
    if(mode==1)return (b-a)*(d-c)/T(2);
    return a==b||c==d?T(nan("")):T((b>a)==(d>c)?1:-1);
}
template<class T> FM_POINT T two_point_regression(T y0,T y1,T x0,T x1,int mode) {
    if(!isfinite(y0)||!isfinite(y1)||!isfinite(x0)||!isfinite(x1)||x0==x1)return T(nan(""));
    T beta=(y1-y0)/(x1-x0);
    return mode==1?y0-beta*x0:mode==2?beta:mode==3?y1:
           mode==4?(y0==y1?T(nan("")):T(1)):T(0);
}
#undef FM_POINT
