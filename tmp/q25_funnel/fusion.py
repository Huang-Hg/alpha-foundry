"""FACT-style pointwise subgraph composition over the existing CUDA operators.

Python constructs code/launch arguments only. Constants are scalar parameters:
candidate parameter sweeps reuse the same architecture-specific NVRTC kernel.
Rolling/stateful nodes are boundaries, preserving their validated FP64 recurrence.
"""
import hashlib
from collections import OrderedDict
import cupy as cp

# Scalar semantics mirror ops_cuda and gpu.py, including NaN/domain rules.
_BODY={
    'abs':'fabs(a)', 'neg':'-a', 'square':'a*a', 'tanh':'tanh(a)',
    'sign':'isnan(a)?a:(a>0?1.0:(a<0?-1.0:0.0))',
    'log':'(isnan(a)||a<=0)?NAN:log(a)',
    'sqrt':'(isnan(a)||a<0)?NAN:sqrt(a)',
    'inv':'(isnan(a)||a==0)?NAN:1.0/a',
    's_log_1p':'isnan(a)?a:(a>=0?log1p(a):-log1p(-a))',
    'add':'a+b', 'sub':'a-b', 'mul':'a*b', 'div':'b==0?NAN:a/b',
    'max':'(isnan(a)||isnan(b))?NAN:fmax(a,b)',
    'min':'(isnan(a)||isnan(b))?NAN:fmin(a,b)',
    'add_const':'a+b', 'mul_const':'a*b',
    'pow_const':'fabs(a)==0?0.0:(a<0?-pow(fabs(a),b):pow(fabs(a),b))',
    'not':'isnan(a)?NAN:(a>0?0.0:1.0)',
    'is_nan':'isnan(a)?1.0:0.0',
    'left_tail':'a>b?NAN:a', 'right_tail':'a<b?NAN:a',
    'if_else':'isnan(a)?NAN:(a>0?b:c)',
}
for _name,_cond in {'gt':'a>b','lt':'a<b','and':'a>0&&b>0','or':'a>0||b>0',
                   'equal':'a==b','not_equal':'a!=b','greater_equal':'a>=b','less_equal':'a<=b'}.items():
    _BODY[_name]=f'(isnan(a)||isnan(b))?NAN:(({_cond})?1.0:0.0)'

_PRE='#define NAN __longlong_as_double((long long)0x7ff8000000000000ULL)'
_PLANS=OrderedDict()
_KERNELS={}
ENABLED=True


def supported(node):
    return node.kind in ('unary','binary','binary_const','ternary') and node.op.value in _BODY


def _plan(root):
    """Discover a maximal pointwise island and realize a reusable scalar kernel."""
    if root.hash in _PLANS:
        _PLANS.move_to_end(root.hash)
        return _PLANS[root.hash]
    import re
    inputs=[];args=[];lines=[];seen={};count=0
    def visit(n):
        nonlocal count
        if n.hash in seen:return seen[n.hash]
        if n.kind in ('const_add','const_mul'):
            var=f'x{len(inputs)}';inputs.append(('constant',float(n.op)));args.append(f'float64 {var}')
        elif not supported(n):
            var=f'x{len(inputs)}';inputs.append(('panel',n));args.append(f'float64 {var}')
        else:
            children=[visit(c) for c in n.children]
            mapping=dict(zip(('a','b','c'),children))
            body=re.sub(r'\b[a-c]\b',lambda m:mapping[m[0]],_BODY[n.op.value])
            var=f'v{len(lines)}';lines.append(f'double {var}=({body});');count+=1
        seen[n.hash]=var
        return var
    result=visit(root)
    body='\n'.join(lines+[f'o={result};'])
    signature=(','.join(args),body)
    # A single primitive already has one launch; composing only saves work for >=2.
    if count<2:plan=None
    else:
        if signature not in _KERNELS:
            name='q25_fused_'+hashlib.sha256(repr(signature).encode()).hexdigest()[:20]
            _KERNELS[signature]=cp.ElementwiseKernel(signature[0],'float64 o',body,name,
                preamble=_PRE,options=('--fmad=false',))
        plan=(_KERNELS[signature],inputs)
    _PLANS[root.hash]=plan
    while len(_PLANS)>8192:_PLANS.popitem(last=False)
    return plan


def evaluate(node,panels,cache,evaluate_boundary):
    if not ENABLED or not supported(node):return None
    plan=_plan(node)
    if plan is None:return None
    kernel,inputs=plan
    values=[value if kind=='constant' else evaluate_boundary(value,panels,cache) for kind,value in inputs]
    return kernel(*values)
