"""N4 (§21): deterministic emitter-sample positions shared by the shadow-context
builder and the BRDF branch. Pure NumPy — no jittor, no side effects.

The old area-light branch recomputed its golden-angle disk from a hardcoded
`_scene=[0,1.45,1.0]` inside `light.py`, so the context builder and the BRDF
could drift apart. This module is the single source of truth; `_relight_views.py`
builds the context list from `make_emitter_samples` and `light.py` consumes the
same positions (guide §21 N4: "context 构建器和 BRDF 分支必须使用同一组显式
sample positions；不可在两处各算一遍圆盘采样").
"""
import hashlib
import numpy as np

_GOLDEN_ANGLE = 2.39996323


def make_emitter_samples(center, radius, count, target=None, geometry='disk'):
    """Deterministic disk (or sphere) samples around `center`.

    disk:   facing `target` (an explicit scene point — default +Y up when no
            target is given, so a lamp with no target is a ceiling-facing disk).
    sphere: golden-angle Fibonacci lattice — isotropic bulb.

    Returns (samples [S,3] float32, meta dict with hash / geometry / radius /
    count / target) so the params JSON records the exact positions.
    """
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    radius = float(radius)
    count = int(count)
    if count < 1:
        return np.zeros((0, 3), dtype=np.float32), {}
    _up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    i = np.arange(count, dtype=np.float64)
    if geometry == 'sphere':
        ga = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - (2.0 * i + 1.0) / count
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        th = i * ga
        off = np.stack([r * np.cos(th), r * np.sin(th), z], axis=-1).astype(np.float32) * radius
    else:  # disk
        fwd = (np.asarray(target, dtype=np.float32) - center) if target is not None else _up
        fn = float(np.linalg.norm(fwd))
        fwd = fwd / fn if fn > 1e-9 else _up
        t1 = np.cross(fwd, _up)
        if float(np.linalg.norm(t1)) < 1e-6:
            t1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t1 = t1 / float(np.linalg.norm(t1))
        t2 = np.cross(fwd, t1)
        r = radius * np.sqrt((i + 0.5) / count)
        th = i * _GOLDEN_ANGLE
        off = (t1[None] * (r[:, None] * np.cos(th)[:, None]) +
               t2[None] * (r[:, None] * np.sin(th)[:, None])).astype(np.float32)
    samples = center[None] + off
    meta = {'geometry': geometry, 'radius': radius, 'count': count,
            'target': (np.asarray(target, dtype=np.float32).tolist()
                       if target is not None else None),
            'hash': hashlib.md5(samples.tobytes()).hexdigest()[:12]}
    return samples, meta
