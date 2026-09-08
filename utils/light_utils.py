# light_gaussian imports — handle both old (C++ ext) and new (Jittor jt.code) paths
_C = None  # compiled C++ extension (lite_rasterize_gaussians) — not in Jittor path
GaussianRasterizationSettings = None
GaussianRasterizer = None
_RASTERIZER_IMPORT_ERRORS = ()

import jittor as jt
import jittor.nn as F
from utils.graphics_utils import getProjectionMatrix
from utils.shadow_bounds import (apply_query_distance, _strict_fill,
                                 calibrate_shadow_bounds, derive_nested,
                                 marker_meta, compute_interval_error,
                                 weighted_q_quantiles_in_interval, adaptive_refine)
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from copy import deepcopy

try:
    from light_gaussian import _C as _C_ext, GaussianRasterizationSettings, GaussianRasterizer
    _C = _C_ext
except (ImportError, ModuleNotFoundError, ValueError, RuntimeError, OSError) as primary_import_error:
    # Jittor path: _C not available; GaussianRasterizationSettings/GaussianRasterizer
    # are inside light_gaussian.light_gaussian sub-package
    try:
        from light_gaussian.light_gaussian import GaussianRasterizationSettings, GaussianRasterizer
    except (ImportError, ModuleNotFoundError, ValueError, RuntimeError, OSError) as fallback_import_error:
        _RASTERIZER_IMPORT_ERRORS = (primary_import_error, fallback_import_error)


def _require_point_shadow_rasterizer():
    """Fail at the optional point-shadow boundary with the original cause."""
    if GaussianRasterizer is not None and GaussianRasterizationSettings is not None:
        return
    details = "; ".join(
        f"{type(error).__name__}: {error}" for error in _RASTERIZER_IMPORT_ERRORS
    ) or "no rasterizer import exception was recorded"
    error = RuntimeError(
        "point-shadow rendering requires the Jittor light_gaussian rasterizer; "
        f"dependency loading failed ({details})")
    if _RASTERIZER_IMPORT_ERRORS:
        raise error from _RASTERIZER_IMPORT_ERRORS[-1]
    raise error

def get_canonical_rays(H: int, W: int, tan_fovx: float, tan_fovy: float) -> jt.Var:
    cen_x = W / 2
    cen_y = H / 2
    focal_x = W / (2.0 * tan_fovx)
    focal_y = H / (2.0 * tan_fovy)

    x, y = jt.meshgrid(
        jt.arange(W),
        jt.arange(H),
        indexing="xy",
    )
    x = x.flatten()  # [H * W]
    y = y.flatten()  # [H * W]
    camera_dirs = F.pad(
        jt.stack(
            [
                (x - cen_x + 0.5) / focal_x,
                (y - cen_y + 0.5) / focal_y,
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [H * W, 3]
    # NOTE: it is not normalized
    return camera_dirs


def getWorld2ViewTorch(R: jt.Var, t: jt.Var) -> jt.Var:
    Rt = jt.zeros((4, 4))
    Rt[:3, :3] = R[:3, :3].transpose(0, 1)   # Jittor: no `.T`
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


# inverse the mapping from https://github.com/NVlabs/nvdiffrec/blob/dad3249af8ede96c7dd72c30328272117fabb710/render/light.py#L22
def get_envmap_dirs(res = [256, 512]) -> jt.Var:
    gy, gx = jt.meshgrid(
        jt.linspace(0.0, 1.0 - 1.0 / res[0], res[0]),
        jt.linspace(-1.0, 1.0 - 1.0 / res[1], res[1]),
        indexing="ij",
    )

    sintheta, costheta = jt.sin(gy * np.pi), jt.cos(gy * np.pi)
    sinphi, cosphi = jt.sin(gx * np.pi), jt.cos(gx * np.pi)

    reflvec = jt.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]

    return reflvec

def get_depth_cubemap(get_xyz,get_opacity,get_scaling,get_rotation,get_features, position, res = 512
):
    canonical_rays = get_canonical_rays(H=res, W=res, tan_fovx=1.0, tan_fovy=1.0)  # [HW, 3]
    norm = jt.norm(canonical_rays, p=2, dim=-1).reshape(res, res, 1)  # [H, W]

    bg_color = jt.zeros([3, res, res])
    rotations: List[jt.Var] = [
        jt.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([-1.0, 0.0, 0.0]), jt.array([0.0, -1.0, 0.0]))  [eye, center, up]
        jt.array(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([1.0, 0.0, 0.0]), jt.array([0.0, -1.0, 0.0]))  [eye, center, up]
        jt.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([0.0, -1.0, 0.0]), jt.array([0.0, 0.0, -1.0]))  [eye, center, up]
        jt.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([0.0, 1.0, 0.0]), jt.array([0.0, 0.0, 1.0]))  [eye, center, up]
        jt.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([0.0, 0.0, -1.0]), jt.array([0.0, 1.0, 0.0]))  [eye, center, up]
        jt.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([0.0, 0.0, 1.0]), jt.array([0.0, -1.0, 0.0]))  [eye, center, up]
    ]
    zfar = 100.0
    znear = 0.01
    projection_matrix = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=np.pi * 0.5, fovY=np.pi * 0.5)
        .transpose(0, 1)
        
    )

    depth_cubemap = []
    opacity_cubemap = []
    for r_idx, rotation in enumerate(rotations):
        c2w = rotation
        c2w[:3, 3] = position
        w2c = jt.linalg.inv(c2w)
        T = w2c[:3, 3]
        R = w2c[:3, :3].T
        world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        input_args = (
            bg_color,
            # bg_colors[r_idx],
            get_xyz,
            jt.Var([]),
            get_opacity,
            get_scaling,
            get_rotation,
            jt.Var([]),
            get_features,
            camera_center,  # campos,
            world_view_transform,  # viewmatrix,
            full_proj_transform,  # projmatrix,
            1.0,  # scale_modifier
            1.0,  # tanfovx,
            1.0,  # tanfovy,
            res,  # image_height,
            res,  # image_width,
            1,
            False,  # prefiltered,
            True,  # argmax_depth, 
        )
        if _C is None:
            return None, None
        (num_rendered, rendered_image, opacity_map, radii, depth_map) = _C.lite_rasterize_gaussians(*input_args)

        # depth_cubemap.append(depth_map.permute(1, 2, 0) * norm)
        depth_cubemap.append(depth_map.permute(1, 2, 0))
        opacity_cubemap.append(opacity_map.permute(1, 2, 0))

    return jt.stack(depth_cubemap), jt.stack(opacity_cubemap)


# P1-b shadow cubemap (POINT_LIGHT_RENDER_QUALITY_FIX_PLAN.md §3.4): render a
# 6-face depth(+alpha) cubemap from the point-light position using the SAME
# Gaussian arrays as the main render (no MLP re-run). Face order and uv axes
# match jittor_texture._texture_cube / scene.NVDIFFREC.util.cube_to_dir:
#   0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
# Each row is a c2w rotation; columns = [right, up, forward]. This rasterizer
# treats +Z_cam as forward (in_frustum keeps p_view.z>0.2), so column 2 is the
# camera forward = the face's dominant axis in _texture_cube. right/up columns
# reproduce cube_to_dir's horizontal/vertical axes exactly (NDC_x==u, NDC_y==v),
# so no per-face flip is needed. Stage B (BLOCKER doc §4.1): corrected the X/Z
# face swap in the previous matrix set.
_CUBE_C2W = [
    [[0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],    # +X  fwd=(1,0,0)
    [[0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],    # -X  fwd=(-1,0,0)
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],      # +Y  fwd=(0,1,0)
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],    # -Y  fwd=(0,-1,0)
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],     # +Z  fwd=(0,0,1)
    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],   # -Z  fwd=(0,0,-1)
]


def _lite_depth_face(means3D, colors, opacity, scales, rotations, viewmatrix,
                     projmatrix, campos, W, H, argmax_depth):
    """P3b: one shadow-cubemap face via the precompiled .so's
    CudaRasterizer::Rasterizer::lite_forward (max-contribution depth when
    argmax_depth, else D/O expected depth). Keeps the main two-phase rasterizer
    path untouched. Returns (depth [1,H,W], opacity [1,H,W]) in VIEW-space z."""
    from light_gaussian import rasterize_points_jt as _rpj
    P = int(means3D.shape[0])
    bg = jt.zeros([3, 1, 1])
    out_c = jt.zeros([3, W, H], dtype='float32')
    out_o = jt.zeros([1, W, H], dtype='float32')
    out_d = jt.zeros([1, W, H], dtype='float32')
    nr = jt.zeros([1], dtype='int32')
    with jt.flag_scope(compile_options=_rpj.proj_options):
        nr, out_c, out_o, out_d = jt.code(
            outputs=[nr, out_c, out_o, out_d],
            inputs=[means3D, colors, opacity, scales, rotations,
                    viewmatrix, projmatrix, campos, bg],
            data={'P': P, 'D': 0, 'M': 0, 'W': W, 'H': H, 'sc': 1.0,
                  'tfx': 1.0, 'tfy': 1.0, 'argmax': 1 if argmax_depth else 0},
            cuda_header=_rpj.cuda_header,
            cuda_src=r'''
@alias(nr, out0) @alias(out_color, out1) @alias(out_opacity, out2) @alias(out_depth, out3)
@alias(means3D, in0) @alias(colors, in1) @alias(opacity, in2) @alias(scales, in3)
@alias(rotations, in4) @alias(viewmatrix, in5) @alias(projmatrix, in6) @alias(campos, in7)
@alias(background, in8)

const int P = data["P"];
char* geom = nullptr; char* bin = nullptr; char* img = nullptr;
int num_rendered = 0;
if (P != 0) {
    num_rendered = CudaRasterizer::Rasterizer::lite_forward(
        [&](size_t s) -> char* { if (!geom) cudaMalloc((void**)&geom, s); return geom; },
        [&](size_t s) -> char* { if (!bin)  cudaMalloc((void**)&bin,  s); return bin; },
        [&](size_t s) -> char* { if (!img)  cudaMalloc((void**)&img,  s); return img; },
        P, data["D"], data["M"],
        (const float*)background_p, data["W"], data["H"],
        means3D_p, nullptr, colors_p, opacity_p, scales_p, data["sc"], rotations_p,
        nullptr, viewmatrix_p, projmatrix_p, campos_p, data["tfx"], data["tfy"],
        false, (bool)data["argmax"],
        out_color_p, out_opacity_p, out_depth_p, nullptr);
}
cudaMemcpy(nr_p, &num_rendered, sizeof(int), cudaMemcpyHostToDevice);
if (geom) cudaFree(geom);
if (bin)  cudaFree(bin);
if (img)  cudaFree(img);
''')
    for o in (nr, out_c, out_o, out_d):
        o.compile_options = _rpj.proj_options
    return out_d, out_o


def _lite_transmit_face(means3D, colors, opacity, scales, rotations, viewmatrix,
                        projmatrix, campos, W, H, num_boundaries, boundaries):
    """P4.1: one shadow-cubemap face via lite_forward with PREFIX-TRANSMITTANCE
    boundaries. liteRenderCUDA accumulates T*=1-alpha front-to-back and records T
    at each ray-distance boundary (record-before semantics).

    Contract (P4.1): `boundaries` [num_boundaries] are RAY-DISTANCE sampling
    boundaries; out_transmit[b] = transmittance strictly BEFORE boundaries[b].
    Returns (transmit [num_boundaries, W, H], final_t [1, W, H]) where final_t
    is the end-of-ray transmittance (covers receivers past the last boundary).
    """
    from light_gaussian import rasterize_points_jt as _rpj
    P = int(means3D.shape[0])
    bg = jt.zeros([3, 1, 1])
    out_c = jt.zeros([3, W, H], dtype='float32')
    out_o = jt.zeros([1, W, H], dtype='float32')
    out_d = jt.zeros([1, W, H], dtype='float32')
    out_t = jt.zeros([num_boundaries, W, H], dtype='float32')
    out_f = jt.zeros([1, W, H], dtype='float32')
    nr = jt.zeros([1], dtype='int32')
    bb = jt.array(np.asarray(boundaries, dtype=np.float32).reshape(-1))  # [N]
    with jt.flag_scope(compile_options=_rpj.proj_options):
        nr, out_c, out_o, out_d, out_t, out_f = jt.code(
            outputs=[nr, out_c, out_o, out_d, out_t, out_f],
            inputs=[means3D, colors, opacity, scales, rotations,
                    viewmatrix, projmatrix, campos, bg, bb],
            data={'P': P, 'D': 0, 'M': 0, 'W': W, 'H': H, 'sc': 1.0,
                  'tfx': 1.0, 'tfy': 1.0, 'argmax': 0, 'NB': num_boundaries},
            cuda_header=_rpj.cuda_header,
            cuda_src=r'''
@alias(nr, out0) @alias(out_color, out1) @alias(out_opacity, out2) @alias(out_depth, out3)
@alias(out_transmit, out4) @alias(out_final, out5)
@alias(means3D, in0) @alias(colors, in1) @alias(opacity, in2) @alias(scales, in3)
@alias(rotations, in4) @alias(viewmatrix, in5) @alias(projmatrix, in6) @alias(campos, in7)
@alias(background, in8) @alias(boundaries, in9)

const int P = data["P"];
char* geom = nullptr; char* bin = nullptr; char* img = nullptr;
int num_rendered = 0;
if (P != 0) {
    num_rendered = CudaRasterizer::Rasterizer::lite_forward(
        [&](size_t s) -> char* { if (!geom) cudaMalloc((void**)&geom, s); return geom; },
        [&](size_t s) -> char* { if (!bin)  cudaMalloc((void**)&bin,  s); return bin; },
        [&](size_t s) -> char* { if (!img)  cudaMalloc((void**)&img,  s); return img; },
        P, data["D"], data["M"],
        (const float*)background_p, data["W"], data["H"],
        means3D_p, nullptr, colors_p, opacity_p, scales_p, data["sc"], rotations_p,
        nullptr, viewmatrix_p, projmatrix_p, campos_p, data["tfx"], data["tfy"],
        false, (bool)data["argmax"],
        out_color_p, out_opacity_p, out_depth_p, nullptr,
        data["NB"], (const float*)boundaries_p, out_transmit_p, out_final_p);
}
cudaMemcpy(nr_p, &num_rendered, sizeof(int), cudaMemcpyHostToDevice);
if (geom) cudaFree(geom);
if (bin)  cudaFree(bin);
if (img)  cudaFree(img);
''')
    for o in (nr, out_c, out_o, out_d, out_t, out_f):
        o.compile_options = _rpj.proj_options
    return out_t, out_f


def _lite_transmit_query_face(means3D, colors, opacity, scales, rotations, viewmatrix,
                              projmatrix, campos, pixel_offsets, q_sorted, res):
    """§28.6 E1: one exact-receiver cubemap face via `receiver_forward`.

    `pixel_offsets` [res*res+1] and `q_sorted` [G_face] are the §28.4 layout for
    THIS face; returns the face's SORTED per-receiver transmittance T [G_face].
    The CUDA `receiverQueryCUDA` mirrors liteRenderCUDA's alpha / alpha<1/255 /
    min 0.99 / test_T<1e-4 early-stop and record-before (g_ray < q) semantics
    exactly. `colors` is a dummy (receiver query uses no colour; it skips the SH
    path in preprocess). 64-bit offsets stay 64-bit.
    """
    from light_gaussian import rasterize_points_jt as _rpj
    P = int(means3D.shape[0])
    G_face = int(q_sorted.shape[0])
    # §29.2.1: an EMPTY caster (P==0) leaves every receiver on this face
    # unobstructed -> T must be 1 (the frozen §28.2 result). The CUDA kernel is
    # never launched (P==0 guard in the host wrapper), so do NOT rely on it to
    # overwrite the initialisation — return ones explicitly instead of zeros.
    if P == 0:
        return jt.ones([G_face], dtype='float32')
    out_t = jt.zeros([G_face], dtype='float32')
    nr = jt.zeros([1], dtype='int32')
    # NOTE: Jittor 1.3.11 downcasts int64 numpy -> int32 Var on jt.array; the kernel
    # reads pixel_offsets as int64 (64-bit offset contract §28.4), so force int64
    # explicitly with jt.cast — otherwise the kernel reads two int32s per int64 and
    # goes OOB (E1 root cause, confirmed by dtype probe).
    po = jt.cast(jt.array(np.asarray(pixel_offsets, dtype=np.int64).reshape(-1)), 'int64')
    qs = jt.array(np.asarray(q_sorted, dtype=np.float32).reshape(-1))
    with jt.flag_scope(compile_options=_rpj.proj_options):
        nr, out_t = jt.code(
            outputs=[nr, out_t],
            inputs=[means3D, colors, opacity, scales, rotations,
                    viewmatrix, projmatrix, campos, po, qs],
            data={'P': P, 'D': 0, 'M': 0, 'W': res, 'H': res, 'sc': 1.0,
                  'tfx': 1.0, 'tfy': 1.0},
            cuda_header=_rpj.cuda_header,
            cuda_src=r'''
@alias(nr, out0) @alias(out_transmit, out1)
@alias(means3D, in0) @alias(colors, in1) @alias(opacity, in2) @alias(scales, in3)
@alias(rotations, in4) @alias(viewmatrix, in5) @alias(projmatrix, in6) @alias(campos, in7)
@alias(pixel_offsets, in8) @alias(q_sorted, in9)

const int P = data["P"];
char* geom = nullptr; char* bin = nullptr; char* img = nullptr;
int num_rendered = 0;
if (P != 0) {
    num_rendered = CudaRasterizer::Rasterizer::receiver_forward(
        [&](size_t s) -> char* { if (!geom) cudaMalloc((void**)&geom, s); return geom; },
        [&](size_t s) -> char* { if (!bin)  cudaMalloc((void**)&bin,  s); return bin; },
        [&](size_t s) -> char* { if (!img)  cudaMalloc((void**)&img,  s); return img; },
        P, data["D"], data["M"],
        data["W"], data["H"],
        means3D_p, colors_p, opacity_p, scales_p, data["sc"], rotations_p,
        nullptr, viewmatrix_p, projmatrix_p, campos_p, data["tfx"], data["tfy"],
        false,
        (const int64_t*)pixel_offsets_p, (const float*)q_sorted_p, out_transmit_p);
}
cudaMemcpy(nr_p, &num_rendered, sizeof(int), cudaMemcpyHostToDevice);
if (geom) cudaFree(geom);
if (bin)  cudaFree(bin);
if (img)  cudaFree(img);
''')
    for o in (nr, out_t):
        o.compile_options = _rpj.proj_options
    return out_t


def render_receiver_transmittance(caster_xyz, caster_opacity, caster_scaling,
                                  caster_rotation, receiver_xyz, light_position,
                                  shadow_res, bias=0.02, znear=0.01, return_meta=False):
    """§28.6 E1: per-receiver exact transmittance [G,1] for the point light.

    CPU layout (§28.4, allowed) -> per-face GPU query (one face at a time; the
    face's geometry/binning/image temp buffers are freed before the next face) ->
    concat sorted T -> inverse_order restore via GPU indexing. The T never leaves
    the GPU. Mirrors liteRenderCUDA alpha / early-stop exactly (record-before).

    §29.3 E2-A `return_meta` levels:
      False (default) -> T [G,1]
      True / 'compact' -> (T, compact_meta, None)
      'full'           -> (T, compact_meta, full_diag)

    §30.2.1/§30.5.1 E2-C: the compact_meta ALWAYS carries the query identity
    (layout_hash, receiver_points_sha256, receiver_order_sha256, shadow_res, bias,
    znear), the per-face query counts and FIVE phase timings with a real
    materialization point: after T is fully restored to [G,1] the GPU is synced
    ONCE (never inside a cubemap face), so t_gpu_query_materialized_s reflects the
    actual CUDA kernel time, not merely the issue time. The receiver_xyz.numpy()
    device->host sync is charged to t_receiver_to_cpu_s, never to the GPU query.
    t_cpu_layout_s / t_gpu_query_s remain as E2-A backward-compat aliases. full_diag
    additionally carries points/dist/q/face/pixel in the ORIGINAL receiver order
    (dist saved float32) and is populated only under explicit return_meta='full'.
    """
    from utils.receiver_query import build_query_layout
    import math as _math, time as _time, hashlib as _hashlib
    _t0 = _time.perf_counter()
    pts_np = np.asarray(receiver_xyz.numpy(), dtype=np.float32).reshape(-1, 3)
    _t1 = _time.perf_counter()                    # t_receiver_to_cpu_s = t1-t0
    lay = build_query_layout(pts_np, light_position, shadow_res, bias, znear)
    G = int(lay['G'])
    res = int(shadow_res)
    _t2 = _time.perf_counter()                    # t_layout_build_s = t2-t1

    def _sha256_bytes(arr):
        h = _hashlib.sha256()
        h.update(np.ascontiguousarray(arr).tobytes(order='C'))
        return h.hexdigest()

    _recv_pts_sha = _sha256_bytes(pts_np)          # float32, C-order (query identity)
    _recv_order_sha = _sha256_bytes(lay['order'])  # int64 permutation

    def _meta(_t_issue, _t_mat, _t_total, _T_var=None):
        counts = [int(lay['face_offsets'][f + 1] - lay['face_offsets'][f])
                  for f in range(6)]
        compact = {
            'layout_hash': lay['layout_hash'],
            'G': G,
            'face_query_counts': counts,
            'receiver_points_sha256': _recv_pts_sha,
            'receiver_order_sha256': _recv_order_sha,
            'shadow_res': int(res),
            'bias': float(bias),
            'znear': float(znear),
            # §30.2.1 five phase timings (materialized after the ONE sync).
            't_receiver_to_cpu_s': float(_t1 - _t0),
            't_layout_build_s': float(_t2 - _t1),
            't_gpu_query_issue_s': float(_t_issue),
            't_gpu_query_materialized_s': float(_t_mat),
            't_query_total_s': float(_t_total),
            # E2-A backward-compat aliases.
            't_cpu_layout_s': float(_t2 - _t0),
            't_gpu_query_s': float(_t_issue),
        }
        full = None
        if return_meta == 'full':
            lp = np.asarray(light_position, np.float64).reshape(3)
            full = {
                'points': pts_np,
                'dist': np.linalg.norm(pts_np.astype(np.float64) - lp[None],
                                       axis=-1).astype(np.float32),
                'q': lay['q_original'],
                'face': lay['face_original'],
                'pixel': lay['pixel_original'],
            }
            # §30.6.1 E2-D: the restored T is part of the schema-v3 diagnostic.
            if _T_var is not None:
                full['T_exact'] = _T_var
        return compact, full

    if G == 0:
        T = jt.zeros([0, 1], dtype='float32')
        if not return_meta:
            return T
        compact, full = _meta(0.0, 0.0, _t2 - _t0)
        return T, compact, full

    w2c = _cube_w2c(light_position)
    proj = getProjectionMatrix(znear=znear, zfar=100.0, fovX=_math.pi * 0.5,
                               fovY=_math.pi * 0.5).transpose(0, 1)
    dummy_col = jt.zeros([caster_xyz.shape[0], 3])
    campos = jt.array(np.asarray(light_position, dtype=np.float32))
    parts = []
    for f in range(6):
        Gf = int(lay['face_offsets'][f + 1] - lay['face_offsets'][f])
        if Gf == 0:
            continue
        wvt = jt.array(np.asarray(w2c[f], dtype=np.float32)).transpose(0, 1)
        fpt = (wvt.unsqueeze(0) @ proj.unsqueeze(0)).squeeze(0)
        tf = _lite_transmit_query_face(
            caster_xyz, dummy_col, caster_opacity, caster_scaling, caster_rotation,
            wvt, fpt, campos, lay['pixel_offsets'][f],
            lay['q_sorted'][int(lay['face_offsets'][f]):int(lay['face_offsets'][f + 1])],
            res)
        parts.append(tf)
    _t3 = _time.perf_counter()                    # t_gpu_query_issue_s = t3-t2
    if not parts:
        T = jt.ones([G, 1], dtype='float32')
    else:
        T_sorted = jt.concat(parts, dim=0) if len(parts) > 1 else parts[0]
        inv = jt.array(lay['inverse_order'], dtype='int64')
        T = T_sorted[inv]
        T = T.reshape(-1, 1)
    # §30.2.1: ONE materialization point after T is fully restored (never inside a
    # cubemap face — that would change the real peak and split the six faces into
    # more sync segments).
    jt.sync_all(True)
    _t4 = _time.perf_counter()                    # t_gpu_query_materialized_s = t4-t3
    if not return_meta:
        return T
    compact, full = _meta(_t3 - _t2, _t4 - _t3, _t4 - _t0, T)
    return T, compact, full


def receiver_identity_meta(receiver_xyz, light_position, shadow_res, bias=0.02, znear=0.01):
    """§30.7.3 / §〇.31 E3: receiver-query IDENTITY for the boundary method A/B control.

    Computes the same per-view receiver identity the exact route produces
    (layout_hash + receiver_points_sha256 + receiver_order_sha256 + G) WITHOUT the
    CUDA query. The boundary control calls this so method_identity_check(boundary,
    exact) can validate receiver / caster / manifest identity (§30.6.3). Mirrors
    render_receiver_transmittance's CPU-layout identity exactly — same
    build_query_layout on the same float32 receiver points with the same
    light / shadow_res / bias / znear — so the hashes match the exact render of the
    same view."""
    from utils.receiver_query import build_query_layout
    import hashlib as _hashlib
    pts_np = np.asarray(receiver_xyz.numpy(), dtype=np.float32).reshape(-1, 3)
    lay = build_query_layout(
        pts_np, np.asarray(light_position, np.float32).reshape(3),
        int(shadow_res), float(bias), float(znear))

    def _sha256_bytes(arr):
        h = _hashlib.sha256()
        h.update(np.ascontiguousarray(arr).tobytes(order='C'))
        return h.hexdigest()

    return {
        'layout_hash': lay['layout_hash'],
        'G': int(lay['G']),
        'receiver_points_sha256': _sha256_bytes(pts_np),
        'receiver_order_sha256': _sha256_bytes(lay['order']),
        'shadow_res': int(shadow_res),
        'bias': float(bias),
        'znear': float(znear),
    }


# N3-A (§21): `_strict_fill`, `calibrate_shadow_bounds`, `derive_nested` and
# `apply_query_distance` live in the pure-NumPy module `utils/shadow_bounds.py`
# (no jittor import) so the bounds contract is testable on both Windows/Python
# and WSL/Jittor. They are imported at the top of this module and re-exported
# here for backward compatibility with `_relight_views.py` / `_scan_light_los.py`.

def _cube_w2c(light_position):
    """CPU analytic rigid-body inverse for the 6 cube faces (Stage A)."""
    light_pos_np = np.asarray(light_position, dtype=np.float32).reshape(-1)
    out = []
    for rot_m in _CUBE_C2W:
        R_c2w = np.asarray(rot_m, dtype=np.float32)[:3, :3].copy()
        w2c_np = np.eye(4, dtype=np.float32)
        w2c_np[:3, :3] = R_c2w.T
        w2c_np[:3, 3] = -(R_c2w.T @ light_pos_np)
        out.append(w2c_np)
    return out


def render_transmit_cubemap(xyz, opacity, scaling, rot, light_position, res=512,
                            num_boundaries=4, boundaries=None, zfar=100.0,
                            znear=0.01):
    """P4.1: render a 6-face PREFIX-TRANSMITTANCE cubemap from `light_position`.

    Contract (P4.1): `boundaries` [num_boundaries] are RAY-DISTANCE sampling
    boundaries; transmit[b] = transmittance strictly BEFORE boundaries[b].
    If None, log-spaced over the Gaussian distance percentiles [p5, p99] and
    the LAST boundary is extended to cover the max receiver (Gaussian) distance.

    Returns (transmit_cube [num_boundaries,6,res,res],
    final_cube [6,res,res]  <- a single 6-face map, same shape as
    transmit_cube[b] so dr.texture's cube sampler can index it as
    final_cube[None,...,None] == [1,6,res,res,1],
    boundaries [num_boundaries]).
    """
    _require_point_shadow_rasterizer()
    if boundaries is None:
        xyz_np = np.asarray(xyz.numpy(), dtype=np.float32)
        lp = np.asarray(light_position, dtype=np.float32).reshape(1, 3)
        d = np.linalg.norm(xyz_np - lp, axis=-1)
        lo = max(float(np.percentile(d, 5)), 0.3)
        hi_grid = max(float(np.percentile(d, 99)), lo * 4.0)
        boundaries = np.geomspace(lo, hi_grid, num_boundaries)
        # P4.1: the last boundary must cover the actual max receiver distance
        # (receivers are the same Gaussian centres). Otherwise far receivers
        # would fall back to final_T even when occluders sit between the last
        # boundary and themselves.
        max_d = float(np.max(d))
        if boundaries[-1] < max_d * 1.02:
            boundaries[-1] = max_d * 1.02
        print(f"[transmit] boundaries={num_boundaries} bounds="
              f"{np.round(boundaries, 3).tolist()}", flush=True)
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=np.pi * 0.5,
                               fovY=np.pi * 0.5).transpose(0, 1)
    dummy_col = jt.zeros([xyz.shape[0], 3])
    campos = jt.array(light_position)
    faces, finals = [], []
    for w2c_np in _cube_w2c(light_position):
        wvt = jt.array(w2c_np).transpose(0, 1)
        fpt = (wvt.unsqueeze(0) @ proj.unsqueeze(0)).squeeze(0)
        t, tf = _lite_transmit_face(xyz, dummy_col, opacity, scaling, rot,
                                    wvt, fpt, campos, res, res, num_boundaries,
                                    boundaries)
        faces.append(t)                                    # [N, res, res]
        finals.append(tf)                                  # [1, res, res]
    return (jt.stack(faces, dim=1),                          # [N,6,res,res]
            jt.stack([tf.squeeze(0) for tf in finals], dim=0),  # [6,res,res]
            boundaries)                                    # [N]


def render_shadow_cubemap(xyz, opacity, scaling, rot, light_position, res=512,
                          zfar=100.0, znear=0.01, argmax_depth=False):
    """Render a 6-face depth(+alpha) cubemap from `light_position`.

    Reuses the already-materialised per-Gaussian arrays from the main render
    (no neural-gaussian MLP re-run). Returns RAY distances (not camera-axis
    depth) so the point-light shadow test can compare directly against
    `dist = ||light_pos - point||`.

    P3b: `argmax_depth=True` uses the precompiled .so's lite_forward to output
    the MAX-CONTRIBUTION Gaussian depth (instead of the D/O expected depth);
    the main two-phase rasterizer path is untouched for argmax_depth=False.

    Args:
        xyz     [G,3]  Gaussian centres (world)
        opacity [G,1]  opacity
        scaling [G,3]  scales
        rot     [G,4]  rotations
        light_position [3] world-space lamp position
        argmax_depth bool  use max-contribution depth (P3b) vs D/O (default)
    Returns:
        depth_cube [6,res,res] float32 ray distance (zfar where empty)
        alpha_cube [6,res,res] float32
    """
    _require_point_shadow_rasterizer()
    # Canonical rays for a 90-degree FOV (focal = res/2, z=1), Jittor meshgrid
    # does not support indexing="xy", so build via linspace + broadcast.
    cen_x = res / 2
    cen_y = res / 2
    focal = res / 2.0
    xs = (jt.linspace(0, res - 1, res) - cen_x + 0.5) / focal          # [W]
    ys = (jt.linspace(0, res - 1, res) - cen_y + 0.5) / focal          # [H]
    xg = xs.unsqueeze(0).broadcast([res, res])                         # [H,W]
    yg = ys.unsqueeze(1).broadcast([res, res])                         # [H,W]
    canonical = jt.stack([xg, yg, jt.ones((res, res))], dim=-1).reshape(-1, 3)  # [HW,3]
    ray_len = jt.norm(canonical, p=2, dim=-1).reshape(res, res, 1)            # 1/cos
    bg = jt.zeros([3, res, res])
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=np.pi * 0.5, fovY=np.pi * 0.5).transpose(0, 1)
    # Must be non-empty, else the CUDA forward reads shs[0] and segfaults.
    dummy_col = jt.zeros([xyz.shape[0], 3])

    # Stage A (POINT_LIGHT_BLOCKER_ROOT_CAUSE_AND_NEXT_STEPS.md): build the six
    # camera w2c matrices on CPU with an analytic rigid-body inverse. jt.linalg.inv
    # is implemented via jt.numpy_code, which imports cupy under use_cuda=1 —
    # that was the cupy blocker. _CUBE_C2W rotations are signed permutation
    # matrices, so  w2c = [R_c2w.T | -R_c2w.T @ t; 0 1]  is exact in float32.
    light_pos_np = np.asarray(light_position, dtype=np.float32).reshape(-1)
    w2c_np_list = []
    for rot_m in _CUBE_C2W:
        R_c2w = np.asarray(rot_m, dtype=np.float32)[:3, :3].copy()
        w2c_np = np.eye(4, dtype=np.float32)
        w2c_np[:3, :3] = R_c2w.T
        w2c_np[:3, 3] = -(R_c2w.T @ light_pos_np)
        w2c_np_list.append(w2c_np)

    depth_faces, alpha_faces = [], []
    campos = jt.array(light_position)
    for w2c_np in w2c_np_list:
        # == getWorld2ViewTorch(w2c_rot.T, w2c_t).transpose(0,1) == w2c.T
        wvt = jt.array(w2c_np).transpose(0, 1)
        fpt = (wvt.unsqueeze(0) @ proj.unsqueeze(0)).squeeze(0)   # Jittor: no .bmm
        if argmax_depth:
            # P3b: precompiled .so lite_forward -> max-contribution depth.
            d, opac = _lite_depth_face(xyz, dummy_col, opacity, scaling, rot,
                                       wvt, fpt, campos, res, res, True)
            d_eff = d.squeeze(0)                          # view-z (already D/O or argmax)
            o = opac.squeeze(0)
        else:
            rs = GaussianRasterizationSettings(
                image_height=res, image_width=res, tanfovx=1.0, tanfovy=1.0,
                bg=bg, scale_modifier=1.0, viewmatrix=wvt, projmatrix=fpt,
                sh_degree=1, campos=campos, prefiltered=False, debug=False)
            rz = GaussianRasterizer(rs)
            _nc, _col, depth, opac, _nm, _dn, alpha, _rd, _ex = rz._execute_inference(
                means3D=xyz, means2D=None, shs=None, colors_precomp=dummy_col,
                opacities=opacity, scales=scaling, rotations=rot,
                cov3Ds_precomp=None, norm3Ds_precomp=None, extra_attrs=None,
                return_aux=False)
            o = opac.squeeze(0)                            # [res,res]
            d = depth.squeeze(0)                           # unnormalised D
            d_eff = jt.where(o > 1e-4, d / jt.maximum(o, 1e-6),
                             jt.zeros_like(d) + float(zfar))
        depth_faces.append(d_eff * ray_len.squeeze(-1))    # ray distance
        alpha_faces.append(o.clamp(0.0, 1.0))
        del o, d_eff
    return jt.stack(depth_faces), jt.stack(alpha_faces)  # [6,res,res], [6,res,res]


# def get_depth_cubemap_moving(get_xyz,get_opacity,get_scaling,get_rotation,get_features,rotations, position, res = 256
# ):
#     # get canonical ray and its norm to normalize depth
#     canonical_rays = get_canonical_rays(H=res, W=res, tan_fovx=1.0, tan_fovy=1.0)  # [HW, 3]
#     norm = jt.norm(canonical_rays, p=2, dim=-1).reshape(res, res, 1)  # [H, W]

#     bg_color = jt.zeros([3, res, res])
    
#     zfar = 100.0
#     znear = 0.01
#     projection_matrix = (
#         getProjectionMatrix(znear=znear, zfar=zfar, fovX=np.pi * 0.5, fovY=np.pi * 0.5)
#         .transpose(0, 1)
#         
#     )

#     depth_cubemap = []
#     opacity_cubemap = []
#     for r_idx, rotation in enumerate(rotations):
#         print(r_idx)
#          c2w = rotations[r_idx]
#         # print(c2w.shape,position.shape,type(c2w),type(position))
#         c2w[:3, 3] = position
#         w2c = jt.linalg.inv(c2w)
#         T = w2c[:3, 3]
#         R = w2c[:3, :3].T
#         world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
#         full_proj_transform = (
#             world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
#         ).squeeze(0)
#         camera_center = world_view_transform.inverse()[3, :3]

#         input_args = (
#             bg_color,
#             # bg_colors[r_idx],
#             get_xyz,
#             jt.Var([]),
#             get_opacity,
#             get_scaling,
#             get_rotation,
#             jt.Var([]),
#             get_features,
#             camera_center,  # campos,
#             world_view_transform,  # viewmatrix,
#             full_proj_transform,  # projmatrix,
#             1.0,  # scale_modifier
#             1.0,  # tanfovx,
#             1.0,  # tanfovy,
#             res,  # image_height,
#             res,  # image_width,
#             1,
#             False,  # prefiltered,
#             True,  # argmax_depth, 
#         )
#         (num_rendered, rendered_image, opacity_map, radii, depth_map) = _C.lite_rasterize_gaussians(*input_args)

#         # depth_cubemap.append(depth_map.permute(1, 2, 0) * norm)
#         depth_cubemap.append(depth_map.permute(1, 2, 0))
#         opacity_cubemap.append(opacity_map.permute(1, 2, 0))

#     return jt.stack(depth_cubemap), jt.stack(opacity_cubemap)

    


def turbo_cmap(gray: np.ndarray) -> np.ndarray:
    """
    Visualize a single-channel image using matplotlib's turbo color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.turbo(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)



def DistributionGGX(
    normals: jt.Var,  # [H, W, 3]
    half_dirs: jt.Var,  # [H, W, 3]
    roughness: jt.Var,  # [H, W, 1]
) -> jt.Var:
    a = roughness * roughness
    a2 = a * a
    NoH = saturate_dot(normals, half_dirs)
    
    NoH2 = NoH * NoH

    nom = a2
    denom = (NoH2 * (a2 - 1.0) + 1.0)
    denom = np.pi * denom * denom + 1e-4
    # print("nom",nom.max(),nom.min())
    # print("denom",denom.max(),denom.min())
    # print("NoH2",NoH2.max(),NoH2.min())

    return nom / denom

def saturate_dot(a: jt.Var, b: jt.Var) -> jt.Var:
    return (a * b).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)


def GeometrySchlickGGX(
    NoV: jt.Var, # [H, W, 1]
    roughness: jt.Var,  # [H, W, 1]
) -> jt.Var:
    r = roughness + 1.0
    k = (r * r) / 8.0
    nom = NoV
    denom = NoV * (1.0 - k) + k

    return nom / denom

def GeometrySmith(
    normals: jt.Var,  # [H, W, 3]
    view_dirs: jt.Var,  # [H, W, 3]
    light_dirs: jt.Var,  # [H, W, 3]
    roughness: jt.Var,  # [H, W, 1]
) -> jt.Var:
    NoV = saturate_dot(normals, view_dirs)
    NoL = saturate_dot(normals, light_dirs)
    ggx2 = GeometrySchlickGGX(NoV, roughness)
    ggx1 = GeometrySchlickGGX(NoL, roughness)

    return ggx1 * ggx2


def fresnelSchlick(
    HoV: jt.Var,  # [H, W, 1]
    F0: jt.Var,  # [H, W, 3]
) -> jt.Var:
    return F0 + (1.0 - F0) * jt.pow((1.0 - HoV).clamp(0.0, 1.0), 5)

def linear_to_srgb(linear: Union[np.ndarray, jt.Var]) -> Union[np.ndarray, jt.Var]:
    if isinstance(linear, jt.Var):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = jt.finfo(jt.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * jt.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return jt.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError


# https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/6.pbr/2.2.1.ibl_specular/2.2.1.pbr.fs
def light_pbr_shading(
    light_position: jt.Var,  # [3]
    light_intensity: jt.Var,  # [3]
    points: jt.Var,  # [H, W, 3]
    normals: jt.Var,  # [H, W, 3]
    view_dirs: jt.Var,  # [H, W, 3]
    albedo: jt.Var,  # [H, W, 3]
    roughness: jt.Var,  # [H, W, 1]
    mask: jt.Var,  # [H, W, 1]
    linear: bool = False,
    metallic: Optional[jt.Var] = None,
    shadow: Optional[jt.Var] = None,
    background: Optional[jt.Var] = None,
) -> Dict:
    if background is None:
        background = jt.zeros_like(normals)  # [H, W, 3]

    # preapre
    light_dirs = jt.normalize(light_position - points, p=2, dim=-1)  # [H, W, 3]
    half_dirs = (light_dirs + view_dirs) / 2.0  # [H, W, 3]
    distance = jt.norm(light_position - points, p=2, dim=-1, keepdim=True)  # [H, W, 1]
    attenuation = 1.0 / jt.pow(distance, 2)  # [H, W, 1]
    radiance = light_intensity * attenuation  # [H, W, 3]

    if metallic is None:
        F0 = jt.ones_like(albedo) * 0.04  # [H, W, 3]
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic  # [H, W, 3]

    # Cook-Torrance BRDF
    NoV = saturate_dot(normals, view_dirs)  # [H, W, 1]
    NoL = saturate_dot(normals, light_dirs)  # [H, W, 1]
    HoV = saturate_dot(half_dirs, view_dirs)  # [H, W, 1]
    NDF = DistributionGGX(normals=normals, half_dirs=half_dirs, roughness=roughness)  # [H, W, 1]
    G = GeometrySmith(normals=normals, view_dirs=view_dirs, light_dirs=light_dirs, roughness=roughness)  # [H, W, 1]
    fresnel = fresnelSchlick(HoV=HoV, F0=F0)  # [H, W, 3]

    numerator = NDF * G * fresnel  # [H, W, 3]
    denominator = 4.0 * NoV * NoL + 1e-4  # [H, W, 1]
    specular = numerator / denominator + 1e-4  # [H, W, 3]

    kd = 1.0 - fresnel  # [H, W, 3]
    if metallic is not None:
        kd *= (1.0 - metallic)
    
    render_rgb = (kd * albedo / np.pi + specular) * radiance # * NoL

    render_rgb = jt.where(mask, render_rgb, background)

    if shadow is not None:
        render_rgb = jt.where(shadow == 0.0, render_rgb*0.2, render_rgb)

    # if linear:
    render_rgb = linear_to_srgb(render_rgb.squeeze())

    results = {}
    results.update(
        {
            "render_rgb": render_rgb,
        }
    )

    return results
