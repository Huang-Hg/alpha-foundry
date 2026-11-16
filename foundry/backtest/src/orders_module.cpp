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
            (long long*)PyArray_DATA(a[11])+b
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
    if(!PyTuple_Check(inputs)||PyTuple_GET_SIZE(inputs)!=12) {
        PyErr_SetString(PyExc_ValueError,"Order step requires twelve packed arrays");return nullptr;
    }
    PyArrayObject* a[12]={};
    const int types[12]={NPY_DOUBLE,NPY_UINT8,NPY_DOUBLE,NPY_DOUBLE,NPY_INT64,NPY_INT64,
                         NPY_DOUBLE,NPY_INT64,NPY_DOUBLE,NPY_INT64,NPY_DOUBLE,NPY_INT64};
    const int dimensions[12]={2,2,3,2,2,3,3,3,3,3,3,1};
    for(int i=0;i<12;++i) {
        PyObject* value=PyTuple_GET_ITEM(inputs,i);
        if(!PyArray_Check(value)) {PyErr_SetString(PyExc_TypeError,"Order step arrays must be NumPy arrays");return nullptr;}
        a[i]=(PyArrayObject*)value;
        if(PyArray_TYPE(a[i])!=types[i]||PyArray_NDIM(a[i])!=dimensions[i]||!PyArray_IS_C_CONTIGUOUS(a[i])
                ||!PyArray_ISALIGNED(a[i])||(i>=2&&i!=7&&i!=8&&!PyArray_ISWRITEABLE(a[i]))) {
            PyErr_SetString(PyExc_ValueError,"Order step array dtype, dimensions, contiguity or ownership is invalid");return nullptr;
        }
    }
    const long long S=PyArray_DIM(a[0],1),B=PyArray_DIM(a[2],0),Q=PyArray_DIM(a[5],1),K=PyArray_DIM(a[7],1),F=PyArray_DIM(a[9],1);
    bool bad=S<1||B<1||Q<1||K<0||F<K+4*Q+S||PyArray_DIM(a[0],0)!=o::PANELS||PyArray_DIM(a[1],0)!=o::RULES||PyArray_DIM(a[1],1)!=S;
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
    const bool paired=std::string(precision)=="paired",mixed=std::string(precision)=="mixed";
    bad=bad||(!paired&&!mixed)||(isolated!=0&&isolated!=1)||(path!=0&&path!=1)||leverage<1||gross<=0||gross>leverage||participation<=0||participation>1;
    if(bad) {PyErr_SetString(PyExc_ValueError,"Order step packed shapes or execution parameters are invalid");return nullptr;}
    // The adapter validates public command values. This check protects direct
    // extension callers from indexing outside the native state buffers.
    const long long* commands=(const long long*)PyArray_DATA(a[7]);
    for(long long k=0;k<B*K;++k) {
        const long long* c=commands+k*o::COMMAND_INTS;
        if(c[o::ACTION]<0||c[o::ACTION]>3||(c[o::ACTION]&&
                (c[o::C_ASSET]<0||c[o::C_ASSET]>=S||c[o::C_ID]<=0||c[o::C_OWNER]<0
                 ||(c[o::C_SIDE]!=1&&c[o::C_SIDE]!=-1)||c[o::C_TYPE]<1||c[o::C_TYPE]>3))) {
            PyErr_SetString(PyExc_ValueError,"Order command indices lie outside the native account contract");return nullptr;
        }
    }
    const o::Input in {(const double*)PyArray_DATA(a[0]),(const unsigned char*)PyArray_DATA(a[1]),
                       S,Q,K,F,time,isolated,path,measured,leverage,gross,participation};
    Py_BEGIN_ALLOW_THREADS
    if(paired)step_accounts<HostPair>(in,a,B);else step_accounts<float>(in,a,B);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyMethodDef methods[]={{"step",py_step,METH_VARARGS,"Advance bounded CPU order accounts by one declared bar."},{nullptr,nullptr,0,nullptr}};
BT_MODULE(_orders,methods,"Bounded FP32 and paired-FP32 order account execution.")
