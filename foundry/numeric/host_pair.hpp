/* Host counterpart of the device FloatPair arithmetic used by order accounts. */
#pragma once
#include <cmath>

struct HostPair {
    float hi, lo;
    HostPair() = default;
    HostPair(float h, float l) : hi(h), lo(l) {}
    HostPair(double v) : hi(static_cast<float>(v)), lo(std::isfinite(v) ? static_cast<float>(v-double(hi)) : 0.f) {}
    explicit operator double() const { return double(hi)+double(lo); }
};
inline HostPair host_pair_normalize(float h,float l) {
    const float s=h+l;
    return HostPair(s,std::isfinite(s)?l-(s-h):0.f);
}
inline HostPair operator+(HostPair a,HostPair b) {
    const float s=a.hi+b.hi;
    if(!std::isfinite(s)) return HostPair(s,0.f);
    const float v=s-a.hi;
    return host_pair_normalize(s,((a.hi-(s-v))+(b.hi-v))+(a.lo+b.lo));
}
inline HostPair operator-(HostPair a) {return HostPair(-a.hi,-a.lo);}
inline HostPair operator-(HostPair a,HostPair b) {return a+(-b);}
inline HostPair operator*(HostPair a,HostPair b) {
    const float p=a.hi*b.hi;
    if(!std::isfinite(p)) return HostPair(p,0.f);
    float e=std::fma(a.hi,b.hi,-p);
    e=std::fma(a.hi,b.lo,e);
    e=std::fma(a.lo,b.hi,e);
    return host_pair_normalize(p,e);
}
inline HostPair operator/(HostPair a,HostPair b) {
    if(b.hi==1.f&&b.lo==0.f)return a;
    const float q=a.hi/b.hi;
    if(!std::isfinite(q))return HostPair(q,0.f);
    float residual=std::fma(-q,b.hi,a.hi);
    residual-=q*b.lo;
    residual+=a.lo;
    return host_pair_normalize(q,residual/b.hi);
}
inline bool operator<(HostPair a,HostPair b) {return a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo);}
inline bool operator>(HostPair a,HostPair b) {return b<a;}
inline bool operator==(HostPair a,HostPair b) {return a.hi==b.hi&&a.lo==b.lo;}
inline bool operator!=(HostPair a,HostPair b) {return !(a==b);}
inline bool operator<=(HostPair a,HostPair b) {return a<b||a==b;}
inline bool operator>=(HostPair a,HostPair b) {return a>b||a==b;}
inline HostPair& operator+=(HostPair& a,HostPair b) {return a=a+b;}
inline HostPair& operator-=(HostPair& a,HostPair b) {return a=a-b;}
inline HostPair& operator*=(HostPair& a,HostPair b) {return a=a*b;}
inline HostPair& operator/=(HostPair& a,HostPair b) {return a=a/b;}
