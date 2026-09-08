"""Small CUDA correctness suite, executed with the inference Jittor environment."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import jittor as jt

spec = importlib.util.spec_from_file_location('sg_vector3', Path(__file__).resolve().parents[1]/'scene/NVDIFFREC/sg_vector3.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
jt.flags.use_cuda = 1
rows = []
rng = np.random.default_rng(42)
for dtype in (np.float32, np.float64):
    cases = [(rng.normal(size=a).astype(dtype), rng.normal(size=b).astype(dtype))
             for a,b in [((257,3),(257,3)), ((17,1,3),(1,16,3)), ((3,),(13,16,3)), ((0,16,3),(1,16,3))]]
    cases += [(np.array([[0,0,0],[-1,2,-3],[1e-18,-1e-18,2e-18],[1e8,1,-1e8]],dtype=dtype),
               np.ones((4,3), dtype=dtype))]
    with jt.no_grad():
        for i,(a,b) in enumerate(cases):
            x,y = jt.array(a,dtype=str(np.dtype(dtype))),jt.array(b,dtype=str(np.dtype(dtype)))
            for name, actual, native, aa,bb in [
                ('dot',m.dot3(x,y),jt.sum(x*y,-1,keepdims=True),a,b),
                ('square',m.squared_norm3(x),jt.sum(x*x,-1,keepdims=True),a,a),
                ('norm',m.norm3(x),jt.sqrt(jt.sum(x*x,-1,keepdims=True)),a,a)]:
                out, old = actual.numpy(),native.numpy()
                ref = np.sum(aa.astype(np.longdouble)*bb.astype(np.longdouble),-1,keepdims=True)
                if name == 'norm': ref = np.sqrt(ref)
                scale = np.sum(np.abs(aa.astype(np.longdouble)*bb.astype(np.longdouble)),-1,keepdims=True)
                if name == 'norm': scale = np.sqrt(scale)
                assert out.dtype == dtype and out.shape == ref.shape, (dtype,i,name,out.dtype,out.shape,ref.shape)
                assert np.isfinite(out).all()
                error = np.abs(out.astype(np.longdouble)-ref)
                bound = 16*np.finfo(dtype).eps*scale + np.finfo(dtype).tiny
                # Keep Jittor's existing sqrt implementation, including its
                # native accuracy. Dot/square still require dtype precision.
                if name == 'norm':
                    bound += np.abs(old.astype(np.longdouble)-ref)
                assert np.all(error <= bound), (str(dtype),i,name,float(error.max(initial=0)))
                rows.append(dict(dtype=str(np.dtype(dtype)),case=i,op=name,max_error=float(error.max(initial=0)),native_max_error=float(np.abs(out-old).max(initial=0))))
        # Jittor logical layouts must remain valid without external replication.
        x = jt.array(rng.normal(size=(3,257)).astype(dtype),dtype=str(np.dtype(dtype))).transpose(1,0)
        assert np.allclose(m.squared_norm3(x).numpy(), np.sum(x.numpy()**2,-1,keepdims=True),rtol=1e-6,atol=1e-12)
        z=jt.zeros((3,3),dtype=str(np.dtype(dtype)))
        assert np.array_equal(m.norm3(z,eps=1e-30).numpy(),jt.norm(z,dim=-1,keepdim=True).numpy())
    try:
        m.dot3(jt.ones((1,3)),jt.ones((1,3)))
    except RuntimeError as e:
        assert 'no_grad' in str(e)
    else: raise AssertionError('gradient-enabled use accepted')
print(json.dumps(dict(passed=True, checks=len(rows), rows=rows)),flush=True)
