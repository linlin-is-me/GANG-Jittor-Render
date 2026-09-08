"""Inference-only three-component reductions, preserving input precision."""
import math
import numpy as np
import jittor as jt


def dot3(a, b):
    if not jt.flags.no_grad:
        raise RuntimeError('vector3_cuda requires jt.no_grad(); backward is not implemented')
    if not jt.flags.use_cuda:
        raise RuntimeError('vector3_cuda requires CUDA')
    if str(a.dtype) not in ('float32', 'float64') or a.dtype != b.dtype:
        raise TypeError('dot3 requires matching FP32 or FP64 inputs')
    if not a.ndim or not b.ndim or a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError('dot3 requires last dimension 3')
    leading = np.broadcast_shapes(tuple(a.shape[:-1]), tuple(b.shape[:-1]))
    shape = tuple(leading) + (1,)
    count = math.prod(leading)
    if count == 0:
        return jt.zeros(shape, dtype=a.dtype)
    def index_code(var):
        dims = (1,)*(len(leading)-var.ndim+1) + tuple(var.shape[:-1])
        strides = [math.prod(dims[i+1:]) for i in range(len(dims))]
        terms = [f'((i/{math.prod(leading[j+1:])}LL)%{d}LL)*{strides[j]}LL'
                 for j,d in enumerate(dims) if d != 1]
        return '(' + ('+'.join(terms) if terms else '0') + ')*3'
    is_double = str(a.dtype) == 'float64'
    ctype, mul, add = ('double', '__dmul_rn', '__dadd_rn') if is_double else ('float', '__fmul_rn', '__fadd_rn')
    header = f'''
__global__ void sg_dot3(const {ctype}* a, const {ctype}* b, {ctype}* out, long long count) {{
    long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    long long ai={index_code(a)}, bi={index_code(b)};
    {ctype} p0={mul}(a[ai],b[bi]), p1={mul}(a[ai+1],b[bi+1]), p2={mul}(a[ai+2],b[bi+2]);
    out[i]={add}({add}(p0,p1),p2);
}}
'''
    return jt.code(shape, a.dtype, [a,b], cuda_header=header,
                   cuda_src='sg_dot3<<<(out0->num+255)/256,256>>>(in0_p,in1_p,out0_p,out0->num);')


def squared_norm3(a):
    return dot3(a, a)


def norm3(a, eps=None):
    squared = squared_norm3(a)
    if eps is not None:
        squared = squared.maximum(eps)
    return jt.sqrt(squared)
