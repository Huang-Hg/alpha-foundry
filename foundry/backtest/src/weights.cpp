// Native additive member-weight sampling and elite refitting.
#include "npy_glue.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool yes,const char* why) { if(!yes) throw std::runtime_error(why); }
void cem_sample(const double*mean,const double*sd,npy_intp n,npy_intp count,uint64_t seed,double*out){
    require(n>0&&n<=64&&count>0,"CEM shape");std::mt19937_64 rng(seed);std::normal_distribution<double> normal;
    // Member weights are additive scales in (0,1), independent of each other: no simplex.
    for(npy_intp b=0;b<count;b++)for(npy_intp j=0;j<n;j++){require(std::isfinite(mean[j])&&std::isfinite(sd[j])&&sd[j]>=0,"CEM parameters");out[b*n+j]=1/(1+std::exp(-(mean[j]+sd[j]*normal(rng))));}
}
void cem_update(const double*weights,const double*scores,npy_intp n,npy_intp count,npy_intp elite,double*mean,double*sd){
    require(n>0&&count>=elite&&elite>0,"CEM update shape");std::vector<npy_intp> ids(count);std::iota(ids.begin(),ids.end(),npy_intp{0});
    std::stable_sort(ids.begin(),ids.end(),[&](npy_intp a,npy_intp b){return scores[a]>scores[b];});
    std::vector<double> x(static_cast<std::size_t>(elite)*n);for(npy_intp k=0;k<elite;k++)for(npy_intp j=0;j<n;j++){double w=std::clamp(weights[ids[k]*n+j],1e-12,1-1e-12);x[k*n+j]=std::log(w/(1-w));}
    for(npy_intp j=0;j<n;j++){double mu=0,var=0;for(npy_intp k=0;k<elite;k++)mu+=x[k*n+j]/elite;for(npy_intp k=0;k<elite;k++)var+=(x[k*n+j]-mu)*(x[k*n+j]-mu)/elite;mean[j]=.3*mean[j]+.7*mu;sd[j]=std::max(.1,.3*sd[j]+.7*std::sqrt(var));}
}

// Weight kernel errors surface as RuntimeError carrying the original message.
template<class Body> bool guarded(Body&&body) {
    try { body(); return true; }
    catch(const std::exception&e) { PyErr_SetString(PyExc_RuntimeError,e.what()); return false; }
}

int seed_arg(PyObject*obj,void*out) {
    PyObject*index=PyNumber_Index(obj);
    if(!index) return 0;
    unsigned long long v=PyLong_AsUnsignedLongLong(index); Py_DECREF(index);
    if(PyErr_Occurred()) return 0;
    *static_cast<uint64_t*>(out)=v; return 1;
}

const double* cdata(PyArrayObject*a) { return static_cast<const double*>(PyArray_DATA(a)); }
double* wdata(PyObject*a) { return static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(a))); }

// mean/sd are updated in place, so they must already be the exact buffer (no conversion copy).
bool inout(PyObject*obj,npy_intp n) {
    auto*a=reinterpret_cast<PyArrayObject*>(obj);
    return PyArray_Check(obj)&&PyArray_TYPE(a)==NPY_DOUBLE&&PyArray_NDIM(a)==1&&PyArray_DIM(a,0)==n
        &&PyArray_IS_C_CONTIGUOUS(a)&&PyArray_ISALIGNED(a)&&PyArray_ISWRITEABLE(a);
}

PyObject* py_cem_sample(PyObject*,PyObject*args,PyObject*kw) {
    static const char*names[]={"mean","sd","count","seed",nullptr};
    PyObject*mean_obj,*sd_obj; int count; uint64_t seed;
    if(!PyArg_ParseTupleAndKeywords(args,kw,"OOiO&:cem_sample",const_cast<char**>(names),&mean_obj,&sd_obj,&count,seed_arg,&seed))
        return nullptr;
    PyArrayObject*mean=bt_as_1d_f64(mean_obj);
    if(!mean) return nullptr;
    PyArrayObject*sd=bt_as_1d_f64(sd_obj);
    if(!sd) { Py_DECREF(mean); return nullptr; }
    npy_intp n=PyArray_DIM(mean,0); PyArrayObject*out=nullptr;
    if(PyArray_DIM(sd,0)!=n) PyErr_SetString(PyExc_ValueError,"mean and sd lengths differ");
    else {
        npy_intp dims[2]={count,n};
        out=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(2,dims,NPY_DOUBLE,0));
        if(out&&!guarded([&]{ cem_sample(cdata(mean),cdata(sd),n,count,seed,static_cast<double*>(PyArray_DATA(out))); })) Py_CLEAR(out);
    }
    Py_DECREF(mean); Py_DECREF(sd);
    return reinterpret_cast<PyObject*>(out);
}

PyObject* py_cem_update(PyObject*,PyObject*args,PyObject*kw) {
    static const char*names[]={"weights","scores","elite","mean","sd",nullptr};
    PyObject*weights_obj,*scores_obj,*mean,*sd; int elite;
    if(!PyArg_ParseTupleAndKeywords(args,kw,"OOiOO:cem_update",const_cast<char**>(names),&weights_obj,&scores_obj,&elite,&mean,&sd))
        return nullptr;
    PyArrayObject*weights=bt_as_2d(weights_obj);
    if(!weights) return nullptr;
    PyArrayObject*scores=bt_as_1d_f64(scores_obj);
    if(!scores) { Py_DECREF(weights); return nullptr; }
    npy_intp count=PyArray_DIM(weights,0),n=PyArray_DIM(weights,1); bool ok=false;
    if(PyArray_DIM(scores,0)!=count||!inout(mean,n)||!inout(sd,n))
        PyErr_SetString(PyExc_ValueError,"cem_update needs one score per weight row and writable float64 mean/sd of the row length");
    else ok=guarded([&]{ cem_update(cdata(weights),cdata(scores),n,count,elite,wdata(mean),wdata(sd)); });
    Py_DECREF(weights); Py_DECREF(scores);
    if(!ok) return nullptr;
    Py_RETURN_NONE;
}

template<class Fn> PyCFunction method(Fn fn) { return reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(fn)); }

PyMethodDef module_methods[]={
    {"cem_sample",method(py_cem_sample),METH_VARARGS|METH_KEYWORDS,
     "cem_sample(mean, sd, count, seed) -> (count, n) float64 member weights in (0,1)"},
    {"cem_update",method(py_cem_update),METH_VARARGS|METH_KEYWORDS,
     "cem_update(weights, scores, elite, mean, sd): refits mean/sd in place on the elite rows"},
    {nullptr,nullptr,0,nullptr}};
}

BT_MODULE(_weights, module_methods, "Native additive member-weight sampling and elite refitting.")
