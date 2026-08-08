"""Pure-NumPy receiver-to-query-list layout contract.

Deterministically maps each receiver to the cubemap (face, pixel, q) used by the
production nearest-transmittance query. Face/UV/pixel semantics mirror
`scene/NVDIFFREC/texture/jittor_texture.py::_texture_cube()` exactly:
  * mutually-exclusive dominant-axis face selection with X>Y>Z tie priority
    (jittor_texture.py L101-112) — NOT the six-independent masks of
    `_sample_single`/`util._cubemap_sample_jt`;
  * per-face UV projection (L116-127) with the +1e-10 denominator;
  * uv_01 = u*0.5+0.5, then pixel = floor(uv_01*res) clamped to [0, res-1]
    (align_corners=False nearest; the half-texel rule follows grid_sample).

This module computes NO transmittance — it only produces the deterministic
sorted query layout the CUDA kernel (E1) consumes. Pure NumPy, no Jittor import.

Layout outputs:
  q_original[G]           float32  max(||p-L||-bias, znear)
  face_original[G]        uint8    mutually-exclusive face index 0..5
  pixel_original[G]       int32    nearest pixel 0..res*res-1
  order[G]                int64    argsort permutation: sorted slot i holds original order[i]
  inverse_order[G]        int64    inverse permutation: original j sits at sorted slot
                                   inverse_order[j]; X_orig == X_sorted[inverse_order]
  face_offsets[7]         int64    per-face cumulative receiver count (face_offsets[6]==G)
  pixel_offsets[6,R*R+1]  int64    per-face per-pixel cumulative (last entry == that
                                   face's query count; monotone non-decreasing)
  q_sorted[G]             float32  q in (face, pixel, q, receiver_id) order
  receiver_id_sorted[G]   int64    original receiver index at each sorted slot
  layout_hash             str      md5 identity of the sorted layout (§28.7 schema v3)

Edge cases (all legal): G=0 (empty arrays, offsets all zero), empty faces, repeated
q, receiver coincident with the light (maps to +X face centre pixel with q=znear).
"""
import hashlib
import numpy as np


def _dominant_face(d):
    """Mutually-exclusive dominant axis, X>Y>Z tie priority (jittor_texture L101-112)."""
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    adx, ady, adz = np.abs(dx), np.abs(dy), np.abs(dz)
    x_dom = (adx >= ady) & (adx >= adz)
    y_dom = (~x_dom) & (ady >= adz)
    z_dom = ~x_dom & ~y_dom
    faces = np.zeros(d.shape[0], dtype=np.int64)
    faces[x_dom & (dx >= 0)] = 0   # +X
    faces[x_dom & (dx < 0)] = 1    # -X
    faces[y_dom & (dy >= 0)] = 2   # +Y
    faces[y_dom & (dy < 0)] = 3    # -Y
    faces[z_dom & (dz >= 0)] = 4   # +Z
    faces[z_dom & (dz < 0)] = 5    # -Z
    return faces


def _face_uv(d, faces):
    """Per-face UV projection (jittor_texture L116-127, +1e-10 denominator)."""
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    u = np.zeros_like(dx)
    v = np.zeros_like(dx)
    eps = 1e-10
    m = faces == 0                                       # +X
    u[m] = -dz[m] / (dx[m] + eps); v[m] = -dy[m] / (dx[m] + eps)
    m = faces == 1                                       # -X
    u[m] = dz[m] / (-dx[m] + eps); v[m] = -dy[m] / (-dx[m] + eps)
    m = faces == 2                                       # +Y
    u[m] = dx[m] / (dy[m] + eps); v[m] = dz[m] / (dy[m] + eps)
    m = faces == 3                                       # -Y
    u[m] = dx[m] / (-dy[m] + eps); v[m] = -dz[m] / (-dy[m] + eps)
    m = faces == 4                                       # +Z
    u[m] = dx[m] / (dz[m] + eps); v[m] = -dy[m] / (dz[m] + eps)
    m = faces == 5                                       # -Z
    u[m] = -dx[m] / (-dz[m] + eps); v[m] = -dy[m] / (-dz[m] + eps)
    return u, v


def _nearest_pixel(uv_01, res):
    """align_corners=False nearest: floor(uv_01*res) clamped to [0, res-1].

    The half-texel tie rule follows jittor_texture.grid_sample. Keep this helper
    synchronized with the production texture sampler.
    """
    p = np.floor(uv_01 * res)
    return np.clip(p, 0, res - 1).astype(np.int64)


def build_query_layout(receiver_points, light_position, shadow_res,
                       bias=0.02, znear=0.01):
    """Receiver -> deterministic sorted (face, pixel, q) query layout.

    Parameters mirror the production query (`apply_query_distance`):
      receiver_points [G,3]  float  receiver world positions
      light_position [3]     float  point-light origin
      shadow_res             int    cubemap face resolution
      bias / znear           float  q = max(dist - bias, znear)
    """
    pts = np.asarray(receiver_points, np.float64).reshape(-1, 3)
    G = int(pts.shape[0])
    res = int(shadow_res)
    lp = np.asarray(light_position, np.float64).reshape(3)
    rel = pts - lp[None]                                   # [G,3]
    norm = np.linalg.norm(rel, axis=-1)                    # [G]
    q_original = np.maximum(norm - bias, znear).astype(np.float32)
    d = rel / np.maximum(norm[:, None], 1e-12)             # unit; zero -> (0,0,0) -> +X
    faces = _dominant_face(d)
    u, v = _face_uv(d, faces)
    uv_01 = np.stack([u * 0.5 + 0.5, v * 0.5 + 0.5], axis=-1)   # [G,2] (u=x, v=y)
    pix2 = _nearest_pixel(uv_01, res)                    # [G,2] -> [pu, pv]
    # flattened pixel index (v*res + u) — the per-pixel key the CUDA kernel uses.
    pixel_original = (pix2[:, 1] * res + pix2[:, 0]).astype(np.int32)
    receiver_id = np.arange(G, dtype=np.int64)

    # stable sort by (face, pixel, q, receiver_id) — lexsort's last key is primary.
    order = np.lexsort((receiver_id, q_original.astype(np.float64),
                        pixel_original, faces))
    faces_s = faces[order]
    pixels_s = pixel_original[order]
    q_sorted = q_original[order]
    receiver_id_sorted = receiver_id[order]
    inverse_order = np.empty(G, dtype=np.int64)
    inverse_order[order] = receiver_id

    face_offsets = np.zeros(7, dtype=np.int64)
    for f in range(6):
        face_offsets[f + 1] = face_offsets[f] + int(np.sum(faces_s == f))
    pixel_offsets = np.zeros((6, res * res + 1), dtype=np.int64)
    for f in range(6):
        lo, hi = int(face_offsets[f]), int(face_offsets[f + 1])
        sub_pix = pixels_s[lo:hi]
        cnt = np.bincount(sub_pix, minlength=res * res).astype(np.int64)
        offs = np.zeros(res * res + 1, dtype=np.int64)
        offs[1:] = np.cumsum(cnt)
        pixel_offsets[f] = offs

    id_src = (face_offsets.tobytes() + pixel_offsets.tobytes()
              + q_sorted.astype(np.float32).tobytes())
    layout_hash = hashlib.md5(id_src).hexdigest()[:12]

    return {
        'q_original': q_original,
        'face_original': faces.astype(np.uint8),
        'pixel_original': pixel_original.astype(np.int32),
        'order': order,
        'inverse_order': inverse_order,
        'face_offsets': face_offsets,
        'pixel_offsets': pixel_offsets,
        'q_sorted': q_sorted,
        'receiver_id_sorted': receiver_id_sorted,
        'layout_hash': layout_hash,
        'shadow_res': res, 'bias': float(bias), 'znear': float(znear),
        'G': G,
    }


def restore_sorted(X_sorted, inverse_order):
    """Restore a per-receiver array from sorted order back to original receiver
    order: X_orig[j] == X_sorted[inverse_order[j]] (exact, no copy loss)."""
    return X_sorted[np.asarray(inverse_order)]
