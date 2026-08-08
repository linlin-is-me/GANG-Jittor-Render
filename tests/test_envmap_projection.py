"""Projection validation for relight_envmap.cubemap_to_latlong.

The projection is a visualization-only tool (envmap thumbnails); this test does
NOT change the production sampling convention. Two suites:

  A. six solid-color faces  -> every face appears in the projection and each
     face-center direction (lat-long centre / poles / equator) maps to the
     correct face color — catches default-face aliasing bugs.
  B. workshop HDR round-trip -> production `util.latlong_to_cubemap` then back
     through our projection: bounded MAE, no missing-face band, no central
     spike.

Usage (WSL, cwd=GANG-Jittor-Render-master):
    python3 -u tests/test_envmap_projection.py     # exit 0 = PROJECTION_ALL_PASS
"""
import os, sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'tools'))
sys.path.insert(0, os.path.join(BASE, 'submodules'))

from relight_envmap import cubemap_to_latlong

_OK, _FAIL = [], []


def _expect(cond, label):
    if cond:
        _OK.append(label)
    else:
        _FAIL.append(label)
        print(f'  [FAIL] {label}', flush=True)


# ---------------------------------------------------------------------------
# A. six solid-color faces + directional probes
# ---------------------------------------------------------------------------
_FACE_COLORS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                         [1, 1, 0], [0, 1, 1], [1, 0, 1]], np.float32)  # +X -X +Y -Y +Z -Z


def _dir_to_face(d):
    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
    adx, ady, adz = abs(dx), abs(dy), abs(dz)
    if adx >= ady and adx >= adz:
        return 0 if dx > 0 else 1
    if ady >= adz:
        return 2 if dy > 0 else 3
    return 4 if dz > 0 else 5


def _probes():
    """(name, theta, phi) at the six face centres. Direction convention:
    d = (sin t sin p, cos t, -sin t cos p);  gy=t/pi, gx=p/pi."""
    return [('+X', np.pi / 2, np.pi / 2), ('-X', np.pi / 2, -np.pi / 2),
            ('+Y', 0.0, 0.0), ('-Y', np.pi, 0.0),
            ('+Z', np.pi / 2, np.pi), ('-Z', np.pi / 2, 0.0)]


def test_six_faces():
    R = 64
    cm = np.zeros((6, R, R, 3), np.float32)
    for s in range(6):
        cm[s, ...] = _FACE_COLORS[s]
    ll = cubemap_to_latlong(cm, (128, 256))
    _expect(ll.shape == (128, 256, 3) and np.isfinite(ll).all(),
            'six-face latlong shape/finite')
    present = {s for s in range(6) if (np.abs(ll - _FACE_COLORS[s]).max(axis=-1) < 0.01).any()}
    _expect(present == set(range(6)), f'all 6 faces present ({sorted(present)})')
    H, W = ll.shape[:2]
    for name, theta, phi in _probes():
        gy, gx = theta / np.pi, phi / np.pi
        yi = int(round(np.clip(gy, 0, 1) * (H - 1)))
        xi = int(round(np.clip((gx + 1) / 2, 0, 1) * (W - 1)))
        d = np.array([np.sin(theta) * np.sin(phi), np.cos(theta),
                      -np.sin(theta) * np.cos(phi)])
        face = _dir_to_face(d)
        got = ll[yi, xi]
        _expect(np.abs(got - _FACE_COLORS[face]).max() < 0.01,
                f'probe {name} -> face {face} ({got})')
    # the assert inside cubemap_to_latlong already guarantees full assignment
    print('  A: six-face + probes PASS')


# ---------------------------------------------------------------------------
# B. workshop HDR round-trip through production latlong_to_cubemap
# ---------------------------------------------------------------------------
def test_workshop_roundtrip():
    import jittor as jt
    jt.flags.use_cuda = 1
    from scene.NVDIFFREC import util
    import cv2

    hdr = os.path.join(BASE, 'scene', 'NVDIFFREC', 'irrmaps', 'aerodynamics_workshop_2k.hdr')
    with open(hdr, 'rb') as h:
        b = np.frombuffer(h.read(), np.uint8)
    rgb = cv2.cvtColor(cv2.imdecode(b, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    orig_ll = np.clip(rgb / 255.0, 0.0, 1.0) * 255.0                    # [1024,2048,3]
    cm = util.latlong_to_cubemap(jt.array(orig_ll), [256, 256]).numpy()  # [6,256,256,3]
    back = cubemap_to_latlong(cm, (512, 1024))                          # [512,1024,3]
    _expect(back.shape == (512, 1024, 3) and np.isfinite(back).all(),
            'workshop back-projection shape/finite')
    orig_res = cv2.resize(orig_ll, (1024, 512), interpolation=cv2.INTER_LINEAR)
    mae = float(np.abs(back - orig_res).mean())
    rng = float(np.percentile(orig_ll, 99.9) - np.percentile(orig_ll, 0.1))
    _expect(mae / max(rng, 1e-6) < 0.10,
            f'workshop round-trip MAE/range={mae / max(rng, 1e-6):.4f} < 0.10')
    # content-aware structural guards (a dropped face / spike must NOT match the
    # original; the workshop HDR legitimately has large dark regions + bright
    # lamps, so dark rows / a bright neighbour are only suspicious if the
    # back-projection differs from the original there).
    rb = back.mean(axis=(1, 2))
    ro = orig_res.mean(axis=(1, 2))
    dim_b = rb < 0.02 * rb.max()
    dim_o = ro < 0.02 * ro.max()
    frac_mismatch = float(np.logical_xor(dim_b, dim_o).mean())
    _expect(frac_mismatch < 0.05,
            f'no missing-face band (row dark-mismatch={frac_mismatch:.4f})')
    _expect(float(np.abs(back[256, 512] - orig_res[256, 512]).mean() / rng) < 0.10,
            'no central spike (centre matches original)')
    print('  B: workshop round-trip PASS')


def main():
    print('=== envmap projection tests ===')
    test_six_faces()
    test_workshop_roundtrip()
    if _FAIL:
        print('FAILED:', _FAIL)
        sys.exit(1)
    print('PROJECTION_ALL_PASS')


if __name__ == '__main__':
    main()
