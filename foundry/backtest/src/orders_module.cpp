/* NumPy boundary for bounded streaming order accounts. */
#include "npy_glue.h"
#include "../../numeric/host_pair.hpp"
#include "orders.hpp"
#include <vector>
#include <string>

namespace o=foundry_orders;

template<class R> static void step_accounts(const o::Input& in,PyArrayObject** a,long long books) {
    #pragma omp parallel for schedule(static)
    for(long long b=0;b<books;++b) {
        std::vector<R> capacity(static_cast<size_t>(in.assets));
        o::State state {
            (double*)PyArray_DATA(a[2])+b*in.assets*o::ASSET_VALUES,
            (double*)PyArray_DATA(a[3])+b*o::CASH_VALUES,
            (long long*)PyArray_DATA(a[4])+b*o::META_VALUES,
            (long long*)PyArray_DATA(a[5])+b*in.capacity*o::ORDER_INTS,
            (double*)PyArray_DATA(a[6])+b*in.capacity*o::ORDER_VALUES,
            (const long long*)PyArray_DATA(a[7])+b*in.commands*o::COMMAND_INTS,
            (const double*)PyArray_DATA(a[8])+b*in.commands*o::COMMAND_VALUES,
            (long long*)PyArray_DATA(a[9])+b*in.audit_capacity*o::AUDIT_INTS,
            (double*)PyArray_DATA(a[10])+b*in.audit_capacity*o::AUDIT_VALUES,
            (long long*)PyArray_DATA(a[11])+b,
            (double*)PyArray_DATA(a[12])+b*in.owners*in.assets,
            (long long*)PyArray_DATA(a[13])+b*in.owners,
            (const double*)PyArray_DATA(a[14])+b*in.owners*in.assets,
            (const unsigned char*)PyArray_DATA(a[15])+b*in.owners*in.assets,
            (const long long*)PyArray_DATA(a[16]),
            (const double*)PyArray_DATA(a[17])+b*in.owners,
            (double*)PyArray_DATA(a[18])+b*in.owners*in.assets,
            (unsigned char*)PyArray_DATA(a[19])+b*in.owners*in.assets,
            (double*)PyArray_DATA(a[20])+b*o::ACTIVITY_VALUES,
            (unsigned int*)PyArray_DATA(a[21])+b*foundry_order_digest::WORDS
        };
        o::Account<R>{in,state}.run(capacity.data());
    }
}

static PyObject* py_step(PyObject*,PyObject* args) {
    PyObject* inputs;
    long long time;
    int isolated,path,measured;
    double leverage,gross,participation;
    const char* precision;
    if(!PyArg_ParseTuple(args,"OLiiiddds",&inputs,&time,&isolated,&path,&measured,&leverage,&gross,&participation,&precision))return nullptr;
    if(!PyTuple_Check(inputs)||PyTuple_GET_SIZE(inputs)!=22) {
        PyErr_SetString(PyExc_ValueError,"Order step requires twenty-two packed arrays");return nullptr;
    }
    PyArrayObject* a[22]={};
    const int types[22]={NPY_DOUBLE,NPY_UINT8,NPY_DOUBLE,NPY_DOUBLE,NPY_INT64,NPY_INT64,
        NPY_DOUBLE,NPY_INT64,NPY_DOUBLE,NPY_INT64,NPY_DOUBLE,NPY_INT64,NPY_DOUBLE,NPY_INT64,
        NPY_DOUBLE,NPY_UINT8,NPY_INT64,NPY_DOUBLE,NPY_DOUBLE,NPY_UINT8,NPY_DOUBLE,NPY_UINT32};
    const int dimensions[22]={2,2,3,2,2,3,3,3,3,3,3,1,3,2,3,3,1,2,3,3,2,2};
    for(int i=0;i<22;++i) {
        PyObject* value=PyTuple_GET_ITEM(inputs,i);
        if(!PyArray_Check(value)) {PyErr_SetString(PyExc_TypeError,"Order step arrays must be NumPy arrays");return nullptr;}
        a[i]=(PyArrayObject*)value;
        if(PyArray_TYPE(a[i])!=types[i]||PyArray_NDIM(a[i])!=dimensions[i]||!PyArray_IS_C_CONTIGUOUS(a[i])
                ||!PyArray_ISALIGNED(a[i])||(i>=2&&i!=7&&i!=8&&i!=14&&i!=15&&i!=16&&i!=17&&!PyArray_ISWRITEABLE(a[i]))) {
            PyErr_SetString(PyExc_ValueError,"Order step array dtype, dimensions, contiguity or ownership is invalid");return nullptr;
        }
    }
    const long long S=PyArray_DIM(a[0],1),B=PyArray_DIM(a[2],0),Q=PyArray_DIM(a[5],1),K=PyArray_DIM(a[7],1),F=PyArray_DIM(a[9],1);
    const long long M=PyArray_DIM(a[12],1);
    bool bad=M<1||S<1||B<1||Q<1||K<0||F<K+M*S+4*Q+(M+1)*(Q+S)||PyArray_DIM(a[0],0)!=o::PANELS||PyArray_DIM(a[1],0)!=o::RULES||PyArray_DIM(a[1],1)!=S;
    const long long width[12]={0,0,o::ASSET_VALUES,o::CASH_VALUES,o::META_VALUES,o::ORDER_INTS,o::ORDER_VALUES,
                              o::COMMAND_INTS,o::COMMAND_VALUES,o::AUDIT_INTS,o::AUDIT_VALUES,0};
    for(int i=2;i<12&&!bad;++i) {
        bad=PyArray_DIM(a[i],0)!=B;
        if(i==2)bad=bad||PyArray_DIM(a[i],1)!=S;
        if(i==5||i==6)bad=bad||PyArray_DIM(a[i],1)!=Q;
        if(i==7||i==8)bad=bad||PyArray_DIM(a[i],1)!=K;
        if(i==9||i==10)bad=bad||PyArray_DIM(a[i],1)!=F;
        if(i<11)bad=bad||PyArray_DIM(a[i],dimensions[i]-1)!=width[i];
    }
    for(int i=12;i<22&&!bad;++i) {
        if(i==16) {bad=PyArray_DIM(a[i],0)!=M;continue;}
        bad=PyArray_DIM(a[i],0)!=B;
        if(i<=19)bad=bad||PyArray_DIM(a[i],1)!=M;
        if(i==12||i==14||i==15||i==18||i==19)bad=bad||PyArray_DIM(a[i],2)!=S;
        if(i==20)bad=bad||PyArray_DIM(a[i],1)!=o::ACTIVITY_VALUES;
        if(i==21)bad=bad||PyArray_DIM(a[i],1)!=foundry_order_digest::WORDS;
    }
    const bool paired=std::string(precision)=="paired",mixed=std::string(precision)=="mixed";
    bad=bad||(!paired&&!mixed)||(isolated!=0&&isolated!=1)||(path!=0&&path!=1)||leverage<1||gross<=0||gross>leverage||participation<=0||participation>1;
    if(bad) {PyErr_SetString(PyExc_ValueError,"Order step packed shapes or execution parameters are invalid");return nullptr;}
    // The adapter validates public command values. This check protects direct
    // extension callers from indexing outside the native state buffers.
    const long long* commands=(const long long*)PyArray_DATA(a[7]);
    for(long long k=0;k<B*K;++k) {
        const long long* c=commands+k*o::COMMAND_INTS;
        if(c[o::ACTION]<0||c[o::ACTION]>3||(c[o::ACTION]&&
                (c[o::C_ASSET]<0||c[o::C_ASSET]>=S||c[o::C_ID]<=0||c[o::C_OWNER]<0||c[o::C_OWNER]>=M
                 ||(c[o::C_SIDE]!=1&&c[o::C_SIDE]!=-1)||c[o::C_TYPE]<1||c[o::C_TYPE]>3))) {
            PyErr_SetString(PyExc_ValueError,"Order command indices lie outside the native account contract");return nullptr;
        }
    }
    const o::Input in {(const double*)PyArray_DATA(a[0]),(const unsigned char*)PyArray_DATA(a[1]),
                       S,Q,K,F,time,M,isolated,path,measured,leverage,gross,participation};
    Py_BEGIN_ALLOW_THREADS
    if(paired)step_accounts<HostPair>(in,a,B);else step_accounts<float>(in,a,B);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject* py_digest(PyObject*,PyObject* args) {
    PyObject* value;if(!PyArg_ParseTuple(args,"O",&value))return nullptr;
    if(!PyArray_Check(value)) {PyErr_SetString(PyExc_TypeError,"Digest state must be a NumPy array");return nullptr;}
    auto* state=(PyArrayObject*)value;
    if(PyArray_TYPE(state)!=NPY_UINT32||PyArray_NDIM(state)!=2||PyArray_DIM(state,1)!=foundry_order_digest::WORDS
            ||!PyArray_IS_C_CONTIGUOUS(state)||!PyArray_ISALIGNED(state)||!PyArray_ISWRITEABLE(state)) {
        PyErr_SetString(PyExc_ValueError,"Digest state requires owned contiguous uint32 (B,26)");return nullptr;
    }
    auto* data=(unsigned int*)PyArray_DATA(state);const long long B=PyArray_DIM(state,0);
    Py_BEGIN_ALLOW_THREADS
    for(long long b=0;b<B;++b)foundry_order_digest::finish(data+b*foundry_order_digest::WORDS);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}
static HostPair pair_sqrt(HostPair value) {
    const float root=std::sqrt(value.hi);
    if(root==0.f||!std::isfinite(root))return HostPair(root,0.f);
    return HostPair(double(root))+(value-HostPair(double(root))*HostPair(double(root)))/HostPair(double(2.f*root));
}
struct ScaledWealth {
    HostPair value=HostPair(.5,0.);int exponent=1;
    void multiply(HostPair amount) {
        value*=amount;
        if(std::isfinite(value.hi)&&value.hi!=0.f) {
            int shift;std::frexp(value.hi,&shift);
            value=HostPair(std::scalbn(value.hi,-shift),std::scalbn(value.lo,-shift));exponent+=shift;
        }
    }
    bool exceeds(const ScaledWealth& other) const {
        return value.hi>0.f&&(exponent>other.exponent||(exponent==other.exponent&&value>other.value));
    }
    HostPair ratio(const ScaledWealth& other) const {
        const HostPair r=value/other.value;const int shift=exponent-other.exponent;
        return HostPair(std::scalbn(r.hi,shift),std::scalbn(r.lo,shift));
    }
};
static PyObject* py_statistics(PyObject*,PyObject* args) {
    PyObject* value;double ppy;if(!PyArg_ParseTuple(args,"Od",&value,&ppy))return nullptr;
    if(!PyArray_Check(value)) {PyErr_SetString(PyExc_TypeError,"Returns must be NumPy arrays");return nullptr;}
    auto* r=(PyArrayObject*)value;
    if(PyArray_TYPE(r)!=NPY_DOUBLE||PyArray_NDIM(r)!=2||!PyArray_IS_C_CONTIGUOUS(r)||!PyArray_ISALIGNED(r)
            ||PyArray_DIM(r,1)<1||!std::isfinite(ppy)||ppy<=0.) {
        PyErr_SetString(PyExc_ValueError,"Statistics require contiguous double (B,V) returns and positive ppy");return nullptr;
    }
    const long long B=PyArray_DIM(r,0),V=PyArray_DIM(r,1);npy_intp shape[2]={B,5};
    auto* result=(PyArrayObject*)PyArray_EMPTY(2,shape,NPY_DOUBLE,0);if(!result)return nullptr;
    const double* data=(const double*)PyArray_DATA(r);double* output=(double*)PyArray_DATA(result);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for(long long b=0;b<B;++b) {
        const double* row=data+b*V;HostPair shift(0.),origin(row[0]),n{double(V)};
        for(long long v=0;v<V;++v)shift+=HostPair(row[v])-origin;
        shift/=n;const HostPair mean=origin+shift;HostPair variance(0.);
        ScaledWealth wealth,peak;HostPair dd(0.);
        for(long long v=0;v<V;++v) {
            const HostPair centered=(HostPair(row[v])-origin)-shift;variance+=centered*centered;
            wealth.multiply(HostPair(1.)+HostPair(row[v]));if(wealth.exceeds(peak))peak=wealth;
            const HostPair drawdown=wealth.ratio(peak)-HostPair(1.);if(drawdown<dd)dd=drawdown;
        }
        const HostPair annual(ppy),vol=pair_sqrt(variance/n)*pair_sqrt(annual),ret=mean*annual;
        double* out=output+b*5;out[0]=vol>HostPair(0.)?double(ret/vol):0.;out[1]=double(ret);
        out[2]=double(vol);out[3]=double(dd);out[4]=std::ldexp(double(wealth.value),wealth.exponent)-1.;
    }
    Py_END_ALLOW_THREADS
    return (PyObject*)result;
}
static PyObject* py_sample_independent(PyObject*,PyObject* args) {
    PyObject *qo,*eo,*mo,*uo,*so;
    if(!PyArg_ParseTuple(args,"OOOOO",&qo,&eo,&mo,&uo,&so))return nullptr;
    PyObject* inputs[5]={qo,eo,mo,uo,so};PyArrayObject* arrays[5];const int dimensions[5]={3,2,1,1,2};
    for(int i=0;i<5;++i) {
        if(!PyArray_Check(inputs[i])) {PyErr_SetString(PyExc_TypeError,"Independent summary requires NumPy arrays");return nullptr;}
        arrays[i]=(PyArrayObject*)inputs[i];
        if(PyArray_TYPE(arrays[i])!=NPY_DOUBLE||PyArray_NDIM(arrays[i])!=dimensions[i]
                ||!PyArray_IS_C_CONTIGUOUS(arrays[i])||!PyArray_ISALIGNED(arrays[i])
                ||(i==4&&!PyArray_ISWRITEABLE(arrays[i]))) {
            PyErr_SetString(PyExc_ValueError,"Independent summary arrays violate the contiguous double contract");return nullptr;
        }
    }
    const long long B=PyArray_DIM(arrays[0],0),M=PyArray_DIM(arrays[0],1),S=PyArray_DIM(arrays[0],2);
    if(B<1||M<1||S<1||PyArray_DIM(arrays[1],0)!=B||PyArray_DIM(arrays[1],1)!=M
            ||PyArray_DIM(arrays[2],0)!=S||PyArray_DIM(arrays[3],0)!=B
            ||PyArray_DIM(arrays[4],0)!=B||PyArray_DIM(arrays[4],1)!=7) {
        PyErr_SetString(PyExc_ValueError,"Independent summary axes must align with (B,M,S)");return nullptr;
    }
    npy_intp shape[2]={B,6};auto* out=(PyArrayObject*)PyArray_EMPTY(2,shape,NPY_DOUBLE,0);if(!out)return nullptr;
    const double *q=(const double*)PyArray_DATA(arrays[0]),*e=(const double*)PyArray_DATA(arrays[1]),
        *m=(const double*)PyArray_DATA(arrays[2]),*u=(const double*)PyArray_DATA(arrays[3]);
    double* state=(double*)PyArray_DATA(arrays[4]);double* output=(double*)PyArray_DATA(out);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for(long long b=0;b<B;++b)o::sample_independent<HostPair>(q+b*M*S,e+b*M,m,HostPair(u[b]),M,S,state+b*7,output+b*6);
    Py_END_ALLOW_THREADS
    return (PyObject*)out;
}
static PyObject* py_independent_turnover(PyObject*,PyObject* args) {
    PyObject *vo,*uo,*so;if(!PyArg_ParseTuple(args,"OOO",&vo,&uo,&so))return nullptr;
    PyObject* inputs[3]={vo,uo,so};PyArrayObject* arrays[3];const int dims[3]={3,1,2};
    for(int i=0;i<3;++i) {
        if(!PyArray_Check(inputs[i])) {PyErr_SetString(PyExc_TypeError,"Turnover requires NumPy arrays");return nullptr;}
        arrays[i]=(PyArrayObject*)inputs[i];
        if(PyArray_TYPE(arrays[i])!=NPY_DOUBLE||PyArray_NDIM(arrays[i])!=dims[i]
                ||!PyArray_IS_C_CONTIGUOUS(arrays[i])||!PyArray_ISALIGNED(arrays[i])
                ||(i==2&&!PyArray_ISWRITEABLE(arrays[i]))) {
            PyErr_SetString(PyExc_ValueError,"Turnover arrays violate contiguous double contracts");return nullptr;
        }
    }
    const long long B=PyArray_DIM(arrays[0],0),M=PyArray_DIM(arrays[0],1);
    if(B<1||M<1||PyArray_DIM(arrays[0],2)!=2||PyArray_DIM(arrays[1],0)!=B
            ||PyArray_DIM(arrays[2],0)!=B||PyArray_DIM(arrays[2],1)!=7) {
        PyErr_SetString(PyExc_ValueError,"Turnover axes must align with (B,M,2) and (B,7) state");return nullptr;
    }
    const double *v=(const double*)PyArray_DATA(arrays[0]),*u=(const double*)PyArray_DATA(arrays[1]);
    double* state=(double*)PyArray_DATA(arrays[2]);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for(long long b=0;b<B;++b)o::independent_turnover<HostPair>(v+b*M*2,HostPair(u[b]),M,state+b*7);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}
static PyMethodDef methods[]={{"independent_turnover",py_independent_turnover,METH_VARARGS,"Accumulate actual aggregate bar-open-normalized turnover."},{"sample_independent",py_sample_independent,METH_VARARGS,"Accumulate actual independent account exposure at one valuation."},{"statistics",py_statistics,METH_VARARGS,"Paired-FP32 measured account statistics."},{"digest",py_digest,METH_VARARGS,"Finalize a private SHA-256 state copy."},{"step",py_step,METH_VARARGS,"Advance bounded CPU order accounts by one declared bar."},{nullptr,nullptr,0,nullptr}};
BT_MODULE(_orders,methods,"Bounded FP32 and paired-FP32 order account execution.")
