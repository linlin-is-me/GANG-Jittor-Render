#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from __future__ import annotations
from pathlib import Path
import sys

_SUBMODULES = Path(__file__).resolve().parents[1] / "submodules"
if str(_SUBMODULES) not in sys.path:
    sys.path.insert(0, str(_SUBMODULES))

import jittor as jt

import math
# from depth_normal_gauss import GaussianRasterizationSettings,GaussianRasterizer
# from light_geo_gauss import GaussianRasterizationSettings,GaussianRasterizer,SurfaceAlign
from light_gaussian import GaussianRasterizationSettings,GaussianRasterizer,SurfaceAlign
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scene.gaussian_model import GaussianModel  # type hint only, avoids circular import
import numpy as np
import jittor.nn as F
from utils.sh_utils import eval_sh
from utils.graphics_utils import rgb_to_srgb
# from Baking import recon_occlusion
# import open3d as o3d
from utils.graphics_utils import normal_from_depth_image
from utils.loss_utils import _stable_full_mean, eikonal_loss
from utils.jt_safe import path_log


def release_training_rasterizer_refs():
    """Release diagnostic rasterizer owners after a completed training step.

    The custom backward has already consumed and synchronized its saved CUDA
    buffers when the trainers call this helper.  Keeping these module globals
    alive until the next forward makes a continuous run retain one more native
    rasterizer context than a fresh-process resume.
    """
    global _last_rasterize_func, _last_rasterizer
    _last_rasterize_func = None
    _last_rasterizer = None


def training_rasterizer_stats():
    """Return shape-only runtime statistics without copying device arrays."""
    function = globals().get("_last_rasterize_func")
    if function is None:
        return None
    stats = getattr(function, "runtime_stats", None)
    return dict(stats) if stats is not None else None

def debug_hook(module, input, output):
    if jt.isnan(output).any():
        print(f"NaN detected in {module.__class__.__name__}")
        print("Input range:", input[0].min(), input[0].max())
        print("Output range:", output.min(), output.max())
        raise ValueError("NaN encountered")
    

def build_rotation(r):
    norm = jt.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]


    row0 = jt.stack((1 - 2 * (y * y + z * z),
                     2 * (x * y - r * z),
                     2 * (x * z + r * y)), dim=1)
    row1 = jt.stack((2 * (x * y + r * z),
                     1 - 2 * (x * x + z * z),
                     2 * (y * z - r * x)), dim=1)
    row2 = jt.stack((2 * (x * z - r * y),
                     2 * (y * z + r * x),
                     1 - 2 * (x * x + y * y)), dim=1)
    return jt.stack((row0, row1, row2), dim=1)

def local_var(inputs):
    # input N M C.  Keep the population-variance contract of jt.var's
    # unbiased=False default, but perform both reductions in FP64.  The FP32
    # nested reductions were the sole differing forward value in the
    # controlled phase-two resume trace (2.62e-10 absolute at a 2.60e-4 loss).
    values = inputs.float64()
    centered = values - values.mean(dim=1, keepdims=True)
    per_channel = (centered * centered).mean(dim=1)
    return _stable_full_mean(per_channel.sum(dim=-1)).float32()

def local_var_normal(inputs,mask):
    # input N M C
    unique_classes = mask.unique()
    variances = []
    for cls in unique_classes:

        cls_mask = (mask == cls).squeeze()
        cls_data = inputs[cls_mask]

        if cls_data.size(0) > 1:  
            cls_variance = cls_data.var(dim=1, unbiased=False)
            variances.append(cls_variance)
    if variances:
        variances = jt.concat(variances, dim=0)  
        overall_mean_variance = variances.mean()  

    return overall_mean_variance



def _bool_to_indices(mask):
    """Normalize mask/indices to int indices via GPU jt.nonzero() — no CPU round-trip.

    Jittor 1.3.11: jt.nonzero() uses where_op.cc CUDA kernel (warp/block/CUB).
    Only numpy bool arrays are handled on CPU (no GPU round-trip needed).
    """
    if mask is None:
        return None
    # Already int indices (numpy)
    if isinstance(mask, np.ndarray) and mask.dtype in (np.int32, np.int64):
        return mask
    # Numpy bool mask — trivially convert on CPU (already on CPU)
    if isinstance(mask, np.ndarray):
        return np.nonzero(mask)[0]
    # Jittor boolean mask → GPU nonzero (where_op.cc CUDA kernel)
    try:
        idx = jt.nonzero(mask)  # → [M] or [M, 1] jt.Var on GPU
        if idx.ndim > 1:
            idx = idx.squeeze(1)
        return idx  # jt.Var — preserves autograd for downstream
    except Exception as exc:
        # Visibility is part of the renderer's numerical contract.  Treating a
        # failed nonzero operation as an all-visible mask silently changes both
        # the rendered image and the gradients, so the production path must
        # fail closed instead of substituting indices.
        raise RuntimeError(
            f"failed to materialize Jittor visibility indices for mask "
            f"shape={tuple(mask.shape)} dtype={mask.dtype}"
        ) from exc


_safe_index_logged = {"numpy": False, "jtcode": False, "arange": False, "jtidx": False}
_safe_index_stats = {"jtidx": 0, "jtterr": 0, "jtcode": 0, "numpy": 0}

# Cache for _gather_jt compiled kernels (by C value)
_gather_jt_cache = {}

def _gather_jt(tensor, idx):
    """Gradient-safe gather: tensor[idx] via jt.code CUDA kernel.

    Unlike jt.array(tensor.numpy()[idx]), jt.code with inputs= preserves
    the autograd connection. Jittor traces the kernel and generates
    scatter-add backward automatically.

    Args:
        tensor: jt.Var [M, C]
        idx: numpy int array [N] — indices to gather
    Returns:
        jt.Var [N, C] — gathered tensor with autograd preserved
    """
    N = len(idx)
    if N == 0:
        return tensor[:0]
    C = tensor.shape[1]
    idx_jt = jt.array(idx.astype(np.int32))

    # Use cached kernel template for each C value
    cache_key = (C, tensor.dtype)
    if cache_key not in _gather_jt_cache:
        kernel = f'''
        __global__ void gather_kernel_{C}(float* out, float* inp, int* idx, int N) {{
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= N) return;
            int src = idx[tid];
            for (int c = 0; c < {C}; c++) {{
                out[tid * {C} + c] = inp[src * {C} + c];
            }}
        }}
        '''
        _gather_jt_cache[cache_key] = kernel
    else:
        kernel = _gather_jt_cache[cache_key]

    # jt.code with inputs= registers tensor in autograd graph
    out = jt.code([N, C], tensor.dtype, [tensor, idx_jt],
        cuda_src=str(N) + ' ' + kernel,
    )
    return out

def _safe_index(tensor, idx):
    """Integer indexing: uses Jittor native (preserves autograd), jt.code gather as fallback.

    Jittor 1.3.11: integer indexing with jt.Var or numpy int arrays preserves the autograd
    graph via GPU gather kernel (GetitemOp). No numpy round-trip — never breaks gradients.

    _bool_to_indices already converts booleans to int indices, so boolean mask crash is avoided.
    """
    if isinstance(idx, jt.Var):
        n = idx.shape[0]
    else:
        n = len(idx)
    if n == 0:
        return tensor[:0]
    if isinstance(idx, np.ndarray) and n == tensor.shape[0] and idx[0] == 0 and idx[-1] == n - 1:
        if not _safe_index_logged["arange"]:
            path_log("[data_src] _safe_index: arange shortcut (no copy)")
            _safe_index_logged["arange"] = True
        return tensor

    # Tier 1: Jittor native integer indexing — preserves autograd graph (GetitemOp CUDA kernel)
    try:
        result = tensor[idx]
        _safe_index_stats["jtidx"] += 1
        if not _safe_index_logged["jtidx"]:
            path_log("[data_src] _safe_index: Jittor native int indexing (preserves grad)")
            _safe_index_logged["jtidx"] = True
        return result
    except Exception as _e_idx:
        _safe_index_stats["jtterr"] += 1
        _msg = str(_e_idx)[:100]
        if _safe_index_stats["jtterr"] <= 3:
            path_log(f"[WARN] _safe_index: Jittor int indexing FAILED for {tensor.shape}, "
                     f"falling back to jt.code gather: {_msg}")

    # Tier 2: jt.code gather (preserves autograd via jt.code inputs= mechanism)
    try:
        result = _gather_jt(tensor, idx)
        _safe_index_stats["jtcode"] += 1
        if not _safe_index_logged["jtcode"]:
            path_log("[data_src] _safe_index: jt.code gather (preserves grad)")
            _safe_index_logged["jtcode"] = True
        return result
    except Exception as _e_jt:
        _msg2 = str(_e_jt)[:100]
        _safe_index_stats["numpy"] += 1
        path_log(f"[ERROR] _safe_index: ALL methods FAILED: {_msg2}")
        raise RuntimeError(f"_safe_index: all fallback methods failed for tensor "
                          f"shape={tensor.shape}: {_msg2}")


_AXIS0_GATHER_HEADER = r'''
template <typename T>
__global__ void axis0_gather_forward_kernel(
        const T* source, const int* indices, T* output,
        int rows, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * width;
    if (tid >= total) return;
    int row = tid / width;
    int column = tid - row * width;
    output[tid] = source[indices[row] * width + column];
}

__global__ void axis0_gather_backward_kernel(
        const float* grad_output, const int* indices, float* grad_source,
        int rows, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * width;
    if (tid >= total) return;
    int row = tid / width;
    int column = tid - row * width;
    atomicAdd(grad_source + indices[row] * width + column, grad_output[tid]);
}
'''


class _Axis0GatherFunction(jt.Function):
    def execute(self, source, indices):
        if str(source.dtype) not in {"float32", "int32", "int64", "bool"}:
            raise TypeError(f"axis-0 gather does not support dtype {source.dtype}")
        self.source_shape = list(source.shape)
        self.rows = int(indices.shape[0])
        self.width = int(np.prod(source.shape[1:])) if source.ndim > 1 else 1
        self.indices = indices.stop_grad()
        output = jt.code(
            [self.rows, self.width], source.dtype, [source, self.indices],
            cuda_header=_AXIS0_GATHER_HEADER,
            cuda_src=r'''
@alias(source, in0)
@alias(indices, in1)
int rows = in1->num;
int width = in0->num / in0_shape0;
int total = rows * width;
int block = 256;
int grid = (total + block - 1) / block;
axis0_gather_forward_kernel<<<grid, block>>>(
    source_p, indices_p, out0_p, rows, width);
''',
        )
        return output.reshape([self.rows] + self.source_shape[1:])

    def grad(self, grad_output):
        if grad_output.dtype != jt.float32:
            return None, None
        grad_source = jt.code(
            self.source_shape, grad_output.dtype, [grad_output, self.indices],
            cuda_header=_AXIS0_GATHER_HEADER,
            cuda_src=r'''
@alias(grad_output, in0)
@alias(indices, in1)
cudaMemset(out0_p, 0, out0->size);
int rows = in1->num;
int width = in0->num / rows;
int total = rows * width;
int block = 256;
int grid = (total + block - 1) / block;
axis0_gather_backward_kernel<<<grid, block>>>(
    grad_output_p, indices_p, out0_p, rows, width);
''',
        )
        return grad_source, None


def _axis0_gather(tensor, idx):
    """Gather axis-0 rows without collapsing trailing dimensions.

    Jittor 1.3.11 ``tensor[idx]`` does not match the required PyTorch contract
    when ``tensor`` is three-dimensional and ``idx`` is a one-dimensional Var.
    Expanding the index to the output shape makes the axis explicit.  The
    underlying ``jt.gather`` implementation uses getitem and has a verified
    scatter-add backward.
    """
    n = int(idx.shape[0]) if isinstance(idx, jt.Var) else len(idx)
    if n == 0:
        return tensor[:0]
    if (isinstance(idx, np.ndarray) and n == tensor.shape[0]
            and idx[0] == 0 and idx[-1] == n - 1):
        return tensor
    idx_var = idx if isinstance(idx, jt.Var) else jt.array(np.asarray(idx, dtype=np.int32))
    if idx_var.dtype != jt.int32:
        idx_var = idx_var.int32()
    expected_shape = [n] + list(tensor.shape[1:])
    result = _Axis0GatherFunction()(tensor, idx_var)
    if list(result.shape) != expected_shape:
        raise RuntimeError(
            f"axis-0 gather shape mismatch: {list(result.shape)} != {expected_shape}")
    return result


# All production call sites below resolve this audited axis-0 implementation.
_safe_index = _axis0_gather



def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, iteration= 0, ape_code=-1, is_pbr=False, normal_smooth_weight=0.0, trace_intermediates=False):
    ## view frustum filtering for acceleration
    global roughness, albedo, matallic
    indices = _bool_to_indices(visible_mask)

    def _idx(tensor):
        return tensor if indices is None else _safe_index(tensor, indices)

    anchor = _idx(pc.get_anchor)
    feat = _idx(pc.get_anchor_feat)
    level = _idx(pc.get_level)
    # For offset-level tensors, indices are anchor-level — need expansion
    if indices is not None:
        K = pc.n_offsets
        # Phase 78: support both numpy int and jt.Var int indices
        if isinstance(indices, np.ndarray):
            offset_indices = np.repeat(indices * K, K) + np.tile(np.arange(K), len(indices))
        else:
            # jt.Var int indices from jt.nonzero — use GPU repeat (safe in inference)
            n = indices.shape[0]
            offset_indices = (indices * K).unsqueeze(1).repeat(1, K) + jt.arange(K)
            offset_indices = offset_indices.reshape(-1)
    else:
        offset_indices = None
    # Jittor stores offsets flattened as [N*K, 3].  Expand visible anchor
    # indices before gathering to reproduce PT's pc._offset[visible_mask].
    grid_offsets = pc._offset
    if indices is not None:
        grid_offsets = _safe_index(grid_offsets, offset_indices)
    grid_offsets = grid_offsets.reshape([anchor.shape[0], pc.n_offsets, 3])
    # _scaling is anchor-level [N, 6]
    grid_scaling = _idx(pc.get_scaling)

    sdf_loss = 0

    local_loss = 0
    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist_raw = ob_view.norm(dim=1, keepdim=True)
    ob_dist = ob_dist_raw
    # view direction (unit vector)
    ob_view = ob_view / ob_dist_raw

    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.add_level:
            cat_view = jt.concat([ob_view, level], dim=1)
        else:
            cat_view = ob_view
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    if pc.add_level:
        cat_local_view = jt.concat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
        cat_local_view_wodist = jt.concat([feat, ob_view, level], dim=1) # [N, c+3+1]
    else:
        cat_local_view = jt.concat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = jt.concat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        if ape_code < 0:
            camera_indicies = jt.ones(cat_local_view[:,0].shape, dtype=jt.int64) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            camera_indicies = jt.ones(cat_local_view[:,0].shape, dtype=jt.int64) * ape_code[0]
            appearance = pc.get_appearance(camera_indicies)


    # get offset's opacity
    if pc.add_opacity_dist:
        _opacity_input = cat_local_view
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        _opacity_input = cat_local_view_wodist
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    
    if pc.dist2level=="progressive":
        prog = _idx(pc._prog_ratio)
        transition_mask = _idx(pc.transition_mask)
        # prog[~transition_mask] = 1.0  # disabled: boolean setitem uses jt.where (CUDA-only)
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)  # Phase 42: matches PyTorch Tanh threshold (Tanh outputs [-1,1], ~50% > 0)
    mask = mask.view(-1)


    # select opacity
    mask_indices = _bool_to_indices(mask)
    opacity = _safe_index(neural_opacity, mask_indices)

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            _color_input = jt.concat([cat_local_view, appearance], dim=1)
            color = pc.get_color_mlp(_color_input)
        else:
            _color_input = jt.concat([cat_local_view_wodist, appearance], dim=1)
            color = pc.get_color_mlp(_color_input)
    else:
        if pc.add_color_dist:
            _color_input = cat_local_view
            color = pc.get_color_mlp(cat_local_view)
        else:
            _color_input = cat_local_view_wodist
            color = pc.get_color_mlp(cat_local_view_wodist)


    # offset's color: already in [N, K*3=30] → reshape to [N*K, 3]
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])

    # get offset's cov
    if pc.add_cov_dist:
        _cov_input = cat_local_view
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        _cov_input = cat_local_view_wodist
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7])

    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]

    grid_rotation = _idx(pc._rotation)

    # Build individual repeated tensors via index-based expansion.
    # Replace unsqueeze+repeat+reshape — Jittor 1.3.11's repeat backward
    # lowers to binary_op between [N,K,C] and [N,C], causing shape mismatch
    # (xshape(10) != yshape(1448)). Integer-index backward uses scatter-add, no 3D intermediate.
    K = pc.n_offsets
    N_anchor = anchor.shape[0]
    # Use numpy int array as index (NOT jt.Var): avoids autograd tracking of index.
    # Jittor's tensor[numpy_int_array] forward is a gather, backward is scatter-add
    # without intermediate 3D tensors (unlike repeat/gather with jt.Var index).
    _expand_np = np.arange(N_anchor, dtype=np.int32).repeat(K)
    scaling_expanded = grid_scaling[_expand_np]             # [N*K, 6]
    rotation_expanded = grid_rotation[_expand_np]           # [N*K, 4]
    repeat_anchor = anchor[_expand_np]                      # [N*K, 3]

    if pc.normal_detal and iteration>500:
        flag = 1
    elif iteration>3000:
        flag = 1
    else:
        flag=0

    if is_training and flag ==1:
        # GANG uses a custom CUDA backward here.  It treats mean_d as saved
        # state and therefore is not the ordinary derivative of the forward
        # expression; use the ported jt.Function boundary rather than native
        # autodiff reductions.
        K = pc.n_offsets
        scaling_3_repeat = grid_scaling[_expand_np, :3]
        rot_input = scale_rot[:, 3:7]
        offsets_all = offsets * scaling_3_repeat
        rot_all = pc.rotation_activation(rot_input)
        knn_index = jt.array(
            np.arange(N_anchor * K, dtype=np.int32).reshape(N_anchor, K))
        pair_d_loss, pair_normal_loss = SurfaceAlign()(
            anchor, offsets_all, rot_all, knn_index)
        local_loss += (
            0.05 * _stable_full_mean(pair_d_loss)
            + 0.01 * _stable_full_mean(pair_normal_loss)
        )


    # Filter each component individually via _safe_index (no concat, no split, no numpy).
    # _safe_index with arange indices is a no-op (returns tensor directly).
    # All tensors stay as jt.Var — no CPU conversion needed.
    # The rasterizer (jt.code CUDA kernel) receives these directly.
    scaling_repeat = _safe_index(scaling_expanded, mask_indices)
    rotation_repeat = _safe_index(rotation_expanded, mask_indices)
    repeat_anchor = _safe_index(repeat_anchor, mask_indices)
    color_filtered = _safe_index(color, mask_indices)      # [N*K, 3]
    scale_rot_filtered = _safe_index(scale_rot, mask_indices)  # [N*K, 7]
    offsets_filtered = _safe_index(offsets, mask_indices)     # [N*K, 3]
    

    # post-process cov (using filtered versions passed to rasterizer)
    # Match GANG-master/gaussian_renderer/__init__.py:210 exactly.  A former
    # migration-only [1e-8, 1.0] clamp changed valid scales and their normals;
    # it was not a classified profile difference.
    scaling = scaling_repeat[:,3:] * jt.sigmoid(scale_rot_filtered[:,:3])

    rot = pc.rotation_activation(rotation_repeat*scale_rot_filtered[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets_out = offsets_filtered * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets_out
    decoder_trace = None
    if trace_intermediates:
        decoder_trace = {
            "visible_indices": indices,
            "anchor": anchor,
            "anchor_feat": feat,
            "grid_offsets": grid_offsets,
            "grid_scaling": grid_scaling,
            "grid_rotation": grid_rotation,
            "ob_view": ob_view,
            "ob_dist": ob_dist,
            "mlp_input": cat_local_view_wodist,
            "neural_opacity": neural_opacity,
            "mask_indices": mask_indices,
            "scaling_expanded": scaling_expanded,
            "rotation_expanded": rotation_expanded,
            "repeat_anchor_filtered": repeat_anchor,
            "scale_rot_full": scale_rot,
            "offsets_full": offsets,
            "scaling_repeat": scaling_repeat,
            "rotation_repeat": rotation_repeat,
            "color_filtered": color_filtered,
            "scale_rot_filtered": scale_rot_filtered,
            "offsets_filtered": offsets_filtered,
            "offsets_out": offsets_out,
            "xyz": xyz,
            "activated_scaling": scaling,
            "activated_rotation": rot,
        }

    view_dir = xyz - viewpoint_camera.camera_center.repeat(xyz.shape[0], 1)
    view_dir_normal = (view_dir/view_dir.norm(dim=1, keepdim=True)).detach() # (N, 3)

    if pc.normal_detal:
        if pc.add_opacity_dist:
            delta_normal1 = pc.get_normal1_mlp(cat_local_view)  # [N, k]
            delta_normal2 = pc.get_normal2_mlp(cat_local_view)
        else:
            delta_normal1 = pc.get_normal1_mlp(cat_local_view_wodist)
            delta_normal2 = pc.get_normal2_mlp(cat_local_view_wodist)
        delta_normal1 =delta_normal1.reshape([anchor.shape[0]*pc.n_offsets, 3])
        delta_normal2 =delta_normal2.reshape([anchor.shape[0]*pc.n_offsets, 3])
        normal,delta_normal = pc.computeNorm(scaling, rot,view_dir_normal, delta_normal1,delta_normal2)
        delta_normal_norm = delta_normal.norm(dim=1, keepdim=True)*0.1
    else:
        normal = pc.computeNorm(scaling, rot,view_dir_normal)
        delta_normal_norm = None

    # Sharp relighting exposes constant per-Gaussian normals as isolated
    # highlights.  The K offsets emitted by one anchor describe one local
    # neighbourhood, so inference may blend their view-aligned unit normals
    # before SG evaluation.  Training and legacy rendering keep weight 0.
    if (not is_training) and normal_smooth_weight > 0.0 and normal.shape[0] > 0:
        group_idx = (mask_indices // pc.n_offsets).int32()
        group_sum = jt.scatter(
            jt.zeros((anchor.shape[0], 3), dtype=jt.float32),
            0, group_idx, normal, reduce='add')
        group_count = jt.scatter(
            jt.zeros((anchor.shape[0], 1), dtype=jt.float32),
            0, group_idx, jt.ones((normal.shape[0], 1), dtype=jt.float32),
            reduce='add').clamp(min_v=1.0)
        group_normal = jt.normalize(group_sum / group_count, p=2, dim=-1)
        local_normal = group_normal[group_idx]
        w = float(normal_smooth_weight)
        normal = jt.normalize(normal * (1.0 - w) + local_normal * w, p=2, dim=-1)


    # Phase 93: sync to materialize non-PBR MLP outputs (opacity+color+cov+normal)
    # before PBR MLPs (roughness+albedo+metallic) start building their lazy graph.
    # This mirrors PyTorch eager execution — each MLP group executes and frees
    # intermediates independently, reducing peak lazy-graph memory by ~40%.
    if is_pbr:
        matallic = None
        if pc.add_opacity_dist:
            roughness = pc.get_roughness_mlp(cat_local_view)  # [N, k]
            albedo = pc.get_albedo_mlp(cat_local_view)
            if pc.with_matallic:
                matallic = pc.get_matallic_mlp(cat_local_view)
        else:
            roughness = pc.get_roughness_mlp(cat_local_view_wodist)
            albedo = pc.get_albedo_mlp(cat_local_view_wodist)
            if pc.with_matallic:
                matallic = pc.get_matallic_mlp(cat_local_view_wodist)

        if is_training:
        
            albedo_loss = local_var(albedo.reshape([anchor.shape[0],pc.n_offsets, 3]))
            roughness_loss = local_var(roughness.reshape([anchor.shape[0],pc.n_offsets, 1]))
            if pc.with_matallic:
                metrics_loss = local_var(matallic.reshape([anchor.shape[0],pc.n_offsets, 1]))
            if pc.with_matallic:
                local_loss += albedo_loss+roughness_loss+metrics_loss
            else:
                local_loss += albedo_loss+roughness_loss
    
        albedo = albedo.reshape([anchor.shape[0]*pc.n_offsets, 3])
        roughness = roughness.reshape([-1, 1])
        if pc.with_matallic:
            matallic = matallic.reshape([-1,1])

        albedo = _safe_index(albedo, mask_indices)
        roughness = _safe_index(roughness, mask_indices)
        if pc.with_matallic:
            matallic = _safe_index(matallic, mask_indices)

        albedo =  jt.clamp(albedo, 0.0, 1.0)
        roughness =  jt.clamp(roughness, 0.001, 1.0)
        if pc.with_matallic:
            matallic = jt.clamp(matallic, 0.0, 1.0)

    else:
        albedo = None
        roughness = None
        matallic = None



    # Use filtered versions for rasterizer output
    color = color_filtered
    scale_rot = scale_rot_filtered

    result = (xyz, color, opacity, scaling, rot, neural_opacity, mask, albedo,
              roughness, matallic, normal, delta_normal_norm, local_loss, sdf_loss)
    if trace_intermediates:
        decoder_trace["normal"] = normal
        return result + (decoder_trace,)
    return result



def generate_shadow_gaussians(pc, light_position, anchor_indices=None,
                              viewdir_center=None, prog_ratio=None):
    """P4.4 (§15.3): light-space shadow-caster decode — xyz/opacity/scaling/rot ONLY.

    The transmit cubemap needs only geometry (xyz), opacity, scaling and rotation.
    Decoding color / albedo / roughness / metallic / normal would waste memory on
    the full-anchor light-space cloud, so this function skips them.

    Physical fix (guide §P4.4.B.2): the opacity/cov MLPs are fed a VIEW DIRECTION
    of `anchor - light_position` — the LIGHT is the shadow-ray origin — instead of
    the main camera center. This removes the per-view-caster coupling measured in
    §15.2 (the per-view caster used camera-visible gaussians + camera view dir).

    N2 (§16.4): `viewdir_center` overrides the MLP view direction for the opacity
    semantic A/B. `None` keeps the light-space direction (mode `ls`, the P4.4
    default); a camera centre `c` makes the opacity/cov MLPs see `anchor - c`
    (mode `canonical` / per-camera of `ensemble`) while the cubemap is still
    rasterised FROM the light position. ALL anchors are always decoded — no
    camera frustum prefilter is ever applied here.

    Semantics mirror `generate_neural_gaussians()` exactly:
      - opacity activation (Tanh), `> 0` mask (Phase 42), progressive ratio
      - offset expansion [N,K]->[N*K], scaling sigmoid + Phase 44 clamp, rotation
        normalize, xyz = anchor + offset*scaling
      - `anchor_indices=None` decodes ALL anchors (bypass camera `_anchor_mask` /
        `prefilter_voxel`); a light-space prefilter index set may be passed for
        the 6-face sequential fallback (§15.4).
    Returns (xyz, opacity, scaling, rot) each [G, *].
    """
    import numpy as np
    indices = _bool_to_indices(anchor_indices) if anchor_indices is not None else None

    def _idx(t):
        return t if indices is None else _safe_index(t, indices)

    anchor = _idx(pc.get_anchor)
    feat = _idx(pc.get_anchor_feat)
    level = _idx(pc.get_level)

    # view direction for the opacity/cov MLPs: light-space by default (shadow ray
    # origin = light position); N2 canonical/ensemble mode overrides to a camera
    # centre so the MLP sees a training-distribution direction.
    vc = jt.array(np.asarray(light_position, dtype=np.float32).reshape(-1))
    if viewdir_center is not None:
        vc = jt.array(np.asarray(viewdir_center, dtype=np.float32).reshape(-1))
    ob_view = anchor - vc.unsqueeze(0)                       # [N,3]
    ob_dist_raw = ob_view.norm(dim=1, keepdim=True)
    ob_dist = ob_dist_raw
    ob_view = ob_view / ob_dist_raw

    if pc.add_level:
        cat_local_view_wodist = jt.concat([feat, ob_view, level], dim=1)
    else:
        cat_local_view_wodist = jt.concat([feat, ob_view], dim=1)

    neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    # N2.5-B: progressive fade is an EXPLICIT input, never read from the mutable
    # `pc._prog_ratio` (which the last receiver view's set_anchor_mask may have
    # written). dist2level='round' -> prog_ratio=None -> no fade.
    if prog_ratio is not None:
        prog = _idx(prog_ratio)
        neural_opacity = neural_opacity * prog

    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity > 0.0).view(-1)                   # Phase 42 Tanh>0
    mask_indices = _bool_to_indices(mask)
    opacity = _safe_index(neural_opacity, mask_indices)

    scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])

    K = pc.n_offsets
    if indices is None:
        grid_offsets = pc._offset
    elif isinstance(indices, np.ndarray):
        shadow_offset_indices = (
            np.repeat(indices * K, K) + np.tile(np.arange(K), len(indices))
        )
        grid_offsets = _safe_index(pc._offset, shadow_offset_indices)
    else:
        shadow_offset_indices = (
            (indices * K).unsqueeze(1).repeat(1, K) + jt.arange(K)
        ).reshape(-1)
        grid_offsets = _safe_index(pc._offset, shadow_offset_indices)
    grid_offsets = grid_offsets.reshape([anchor.shape[0], K, 3])
    grid_rotation = _idx(pc._rotation)
    grid_scaling = _idx(pc.get_scaling)
    offsets = grid_offsets.view([-1, 3])

    N_anchor = anchor.shape[0]
    _expand_np = np.arange(N_anchor, dtype=np.int32).repeat(K)
    scaling_expanded = grid_scaling[_expand_np]              # [N*K, 6]
    rotation_expanded = grid_rotation[_expand_np]            # [N*K, 4]
    repeat_anchor = anchor[_expand_np]                       # [N*K, 3]

    scaling_repeat = _safe_index(scaling_expanded, mask_indices)
    rotation_repeat = _safe_index(rotation_expanded, mask_indices)
    repeat_anchor = _safe_index(repeat_anchor, mask_indices)
    scale_rot_filtered = _safe_index(scale_rot, mask_indices)
    offsets_filtered = _safe_index(offsets, mask_indices)

    scaling = scaling_repeat[:, 3:] * jt.sigmoid(scale_rot_filtered[:, :3])
    scaling = scaling.maximum(1e-8).minimum(1.0)             # Phase 44 clamp
    rot = pc.rotation_activation(rotation_repeat * scale_rot_filtered[:, 3:7])
    offsets_out = offsets_filtered * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets_out
    return xyz, opacity, scaling, rot


def scale_loss(scaling):

    _, sorted_scale = jt.argsort(scaling, dim=-1)
    min_scale_loss = sorted_scale[...,0]
    # This reduction spans every visible Gaussian.  Jittor 1.3.11's FP32 GPU
    # reduction produced different scalar values from identical arrays in
    # independent resume processes, so use the audited FP64 accumulator while
    # retaining a connected FP32 scalar for the training graph.
    loss_scale = 100.0 * _stable_full_mean(min_scale_loss)

    return loss_scale


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : jt.Var, scaling_modifier=1.0, visible_mask=None,is_pbr=False,light=None, retain_grad=False, is_training =True, Local_pkg=None,iteration = 0,ape_code=-1, normalize_for_light=False, return_aux=True, normal_smooth_weight=0.0, return_light_components=False, shadow_ctx=None, trace_intermediates=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # if is_training:
    generated = generate_neural_gaussians(
        viewpoint_camera, pc, visible_mask, is_training=is_training,
        is_pbr=is_pbr, iteration=iteration, ape_code=ape_code,
        normal_smooth_weight=normal_smooth_weight,
        trace_intermediates=trace_intermediates)
    if trace_intermediates:
        (xyz, color, opacity, scaling, rot, neural_opacity, mask, albedo,
         roughness, matallic, normal, delta_normal_norm, local_loss, sdf_loss,
         decoder_trace) = generated
    else:
        (xyz, color, opacity, scaling, rot, neural_opacity, mask, albedo,
         roughness, matallic, normal, delta_normal_norm, local_loss, sdf_loss) = generated

    # scale_loss sorts every Gaussian scale and is only consumed by training.
    loss_scale = scale_loss(scaling) if is_training else jt.float32(0.0)
    if pc.normal_detal:
        delta_normal_norm = delta_normal_norm.repeat(1, 3)


    screenspace_points = jt.zeros_like(xyz) + 0
    # M2 (P0): only the training path needs an autograd tape on screenspace_points;
    # in inference the rasterizer runs inference-only (no tape, no save_for_backward).
    if is_training:
        screenspace_points.requires_grad = True
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except Exception as exc:
            raise RuntimeError("failed to retain screen-space gradients") from exc

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)



    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # §30.5.2 E2-C identity vars are declared in the PBR branch below but read
    # unconditionally on every return path (L1006). Non-PBR inference (25K render)
    # needs them defined; the PBR branch overwrites these defaults.
    receiver_query_meta = None
    receiver_query_diag = None
    receiver_transmittance = None

    if is_pbr:

        viewdirs = jt.normalize(viewpoint_camera.camera_center - xyz, p=2, dim=-1)

        # Training must rebuild from the current optimized base so continuous
        # and resumed steps share the same derived light state. Inference can
        # retain the exact-base disk/process cache.
        if is_training:
            light.build_mips(training=True)
        elif not getattr(light, '_mips_built', False):
            light.build_mips(training=False)
        
        # Keep the historical training path unchanged, but allow relighting
        # inference to provide the unit world-space directions expected by SG
        # and cubemap sampling.  [0,1] encoding remains only for raster output.
        if normalize_for_light:
            normal_for_light = jt.normalize(normal, p=2, dim=-1)
        else:
            normal_for_light = normal * 0.5 + 0.5

        # P1-b shadow: build a 6-face depth cubemap from the lamp (reuses the
        # already-materialised xyz/opacity/scaling/rot, no MLP re-run). Only in
        # inference point-light mode when point_shadow_enabled.
        shadow_cube = None
        shadow_alpha = None
        transmit_cube = None
        transmit_bounds = None
        transmit_final = None
        receiver_query_meta = None       # §30.5.2 E2-C: exact query identity (pkg)
        receiver_query_diag = None       # §30.5.2 E2-C: full schema-v3 diag (pkg, non-raster)
        receiver_transmittance = None    # §30.5.2 E2-C: explicit [G,1] exact T -> light
        if (not is_training and getattr(light, 'point_light_enabled', False)
                and getattr(light, 'point_shadow_enabled', False)):
            from utils.light_utils import render_shadow_cubemap, render_transmit_cubemap
            if getattr(light, 'shadow_diag', False):
                print(f"  [shadow-dbg] xyz{tuple(xyz.shape)} op{tuple(opacity.shape)} "
                      f"sc{tuple(scaling.shape)} rot{tuple(rot.shape)}", flush=True)
            if (shadow_ctx is not None
                    and shadow_ctx.get('query_mode') == 'receiver_exact'):
                # §30.5.2 E2-C: per-receiver exact transmittance. The receiver
                # cloud is THIS view's xyz; the fixed caster tensors + identity
                # come from the exact context. NO transmit_cube / transmit_final /
                # transmit_bounds and NO depth cubemap. Metadata stays in the
                # renderer (pkg) — only the explicit T reaches lightRender.
                from utils.light_utils import render_receiver_transmittance
                _rmeta = ('full' if shadow_ctx.get('dump_transmit_diag', False)
                          else 'compact')
                _T, _cm, _fd = render_receiver_transmittance(
                    shadow_ctx['caster_xyz'], shadow_ctx['caster_opacity'],
                    shadow_ctx['caster_scaling'], shadow_ctx['caster_rotation'],
                    xyz, shadow_ctx['light_position'], shadow_ctx['shadow_res'],
                    bias=float(shadow_ctx.get('bias', 0.02)),
                    znear=float(shadow_ctx.get('znear', 0.01)),
                    return_meta=_rmeta)
                receiver_transmittance = _T
                receiver_query_meta = dict(_cm)
                receiver_query_meta['caster_sha256'] = shadow_ctx.get('caster_sha256')
                receiver_query_meta['implementation_manifest_sha256'] = shadow_ctx.get(
                    'source_manifest_sha256')
                receiver_query_meta['shadow_ctx_build_count'] = shadow_ctx.get(
                    'shadow_ctx_build_count', 1)
                if _rmeta == 'full':
                    receiver_query_diag = _fd
            elif getattr(light, 'shadow_transmit', False):
                # P4.4 (§15.5): reuse a pre-computed FIXED light-space shadow
                # context when provided (built once before the view loop, from
                # ALL anchors with view dir = anchor - light). Otherwise fall back
                # to the historical per-view caster (camera-visible gaussians).
                if shadow_ctx is not None:
                    if 'samples' in shadow_ctx:
                        # N4 (§21): per-emitter-sample ctx — no shared cube; each
                        # emitter sample carries its own (used via emitter_transmit).
                        transmit_cube = transmit_final = transmit_bounds = None
                    else:
                        transmit_cube = shadow_ctx['transmit_cube']
                        transmit_final = shadow_ctx['transmit_final']
                        transmit_bounds = shadow_ctx['transmit_bounds']
                else:
                    # P4.1: prefix-transmittance cubemap (semi-transparent shadow).
                    # Contract: N boundaries -> N T(boundary) maps + 1 final-T map.
                    transmit_cube, transmit_final, transmit_bounds = render_transmit_cubemap(
                        xyz, opacity, scaling, rot,
                        np.array(light.point_light_position.numpy(), dtype=np.float32),
                        res=int(getattr(light, 'shadow_res', 512)),
                        num_boundaries=int(getattr(light, 'shadow_buckets', 4)))
                # §30.7.3 / §〇.31 E3: the FIXED boundary control carries the SAME
                # per-view receiver identity the exact route computes, so
                # method_identity_check(boundary, exact) can validate receiver /
                # caster / manifest identity (§30.6.3). CPU layout + hash only — no
                # CUDA query, no change to the cubemap or the rendered HDR.
                if (shadow_ctx is not None
                        and shadow_ctx.get('carry_receiver_identity')):
                    from utils.light_utils import receiver_identity_meta
                    _rqi = receiver_identity_meta(
                        xyz,
                        np.array(light.point_light_position.numpy(), dtype=np.float32),
                        int(getattr(light, 'shadow_res', 512)),
                        bias=float(getattr(light, 'shadow_bias', 0.02)),
                        znear=float(getattr(light, 'shadow_znear', 0.01)))
                    receiver_query_meta = dict(_rqi)
                    receiver_query_meta['caster_sha256'] = shadow_ctx.get('caster_sha256')
                    receiver_query_meta['implementation_manifest_sha256'] = shadow_ctx.get(
                        'implementation_manifest_sha256')
                    receiver_query_meta['shadow_ctx_build_count'] = shadow_ctx.get(
                        'shadow_ctx_build_count', 1)
            else:
                shadow_cube, shadow_alpha = render_shadow_cubemap(
                    xyz, opacity, scaling, rot,
                    np.array(light.point_light_position.numpy(), dtype=np.float32),
                    res=int(getattr(light, 'shadow_res', 512)),
                    argmax_depth=(getattr(light, 'shadow_depth', 'do') == 'argmax'))
        # N4 (§21) / N4-R0 (§25.4): per-emitter-sample shadow contexts (real
        # penumbra). When the pre-built shadow_ctx carries a `samples` list, each
        # entry is a STRUCTURED context {index, position, position_hash,
        # transmit_cube, transmit_final, transmit_bounds, bounds_hash,
        # shadow_meta}; the area-light branch queries the SAMPLE's cubemap and
        # validates its position/index/bounds instead of the shared center.
        emitter_transmit = None
        if shadow_ctx is not None and 'samples' in shadow_ctx:
            emitter_transmit = shadow_ctx['samples']
        # A model without the optional metallic decoder is a dielectric model.
        # Keep metallic absent from the exported auxiliary contract, but provide
        # an explicit zero tensor to the BRDF instead of passing None into its
        # arithmetic.
        metallic_for_light = (
            matallic if pc.with_matallic else jt.zeros_like(roughness))
        light_color, light_extras = light.lightRender(
            xyz, normal_for_light, albedo, roughness, metallic_for_light, viewdirs,
            shadow_cube=shadow_cube, shadow_alpha=shadow_alpha,
            transmit_cube=transmit_cube, transmit_bounds=transmit_bounds,
            transmit_final=transmit_final, emitter_transmit=emitter_transmit,
            receiver_transmittance=receiver_transmittance)
        # M2 (point-light diag): only keep the per-component split when requested;
        # the raster colour path needs only light_color.
        if not return_light_components:
            del light_extras

        # viewdirs/normal_for_light are no longer needed after lightRender.
        del viewdirs, normal_for_light, metallic_for_light

        if return_aux or is_training:
            if is_training:
                normal = normal @ viewpoint_camera.world_view_transform[:3, :3]
            normal = normal * 0.5 + 0.5

            if pc.with_matallic:
                if pc.normal_detal:
                    features = jt.concat([normal,delta_normal_norm,albedo,roughness,matallic],dim=-1)
                else:
                    features = jt.concat([normal,albedo,roughness,matallic],dim=-1)
            else:
                if pc.normal_detal:
                    features = jt.concat([normal,delta_normal_norm,albedo,roughness],dim=-1)
                else:
                    features = jt.concat([normal,albedo,roughness],dim=-1)
        else:
            features = None

        color = light_color
    else:
        if return_aux or is_training:
            if is_training:
                normal = normal @ viewpoint_camera.world_view_transform[:3, :3]
            normal = normal * 0.5 + 0.5

            if pc.normal_detal:
                features = jt.concat([normal,delta_normal_norm],dim=-1)
            else:
                features = normal
        else:
            features = None


    # Full-resolution relighting only needs RGB. Omitting extra_attrs avoids an
    # 8-channel 5187x3361 auxiliary image (~532 MiB) and its retained package.
    raster_extra_attrs = features if (return_aux or is_training) else None
    # P0-b: pixel-space diffuse/specular split. In point-light diag mode, pack the
    # per-Gaussian PL diffuse+specular as 6 extra channels so they rasterize with
    # the exact same sort/opacity/alpha as the final RGB. P2 (REASSESSMENT §P2):
    # when --diag-gaussians also pack the front-face mask as a 7th channel.
    # Default path keeps extra_attrs=None (no extra res=1 memory).
    # §27.2.4 (components-off lifecycle fix): an explicit collection gate evaluated
    # HERE (before the probes) — when components are off, `light_extras` was already
    # `del`-eted above and probing it would NameError; the raster path needs only
    # RGB. `is_pbr` short-circuits first so the non-PBR branch never touches it.
    # Semantics are unchanged on every existing path: `not is_training` in the gate
    # matches the original `_has_*` guards, and the `del` above keeps its original
    # `not return_light_components` condition (training path untouched).
    _collect_pl_components = is_pbr and return_light_components and not is_training
    _has_fm = (_collect_pl_components and getattr(light, 'diag_gaussians', False)
               and 'front_mask' in light_extras)
    # N4-R0 (§25.4 item 7): effective shadow visibility as a 1-channel pixel tail
    # (penumbra line profile). ED-generic rasterizer, no CUDA change needed.
    _has_veff = (_collect_pl_components and getattr(light, 'point_light_enabled', False)
                 and 'veff' in light_extras)
    _n_pl = 6 + (1 if _has_fm else 0) + (1 if _has_veff else 0)
    if is_pbr and return_light_components and not is_training and getattr(light, 'point_light_enabled', False):
        _pl_comp = jt.concat([light_extras["diffuse_rgb_pl"], light_extras["specular_rgb_pl"]], dim=-1)  # [G,6]
        if _has_fm:
            _pl_comp = jt.concat([_pl_comp, light_extras["front_mask"]], dim=-1)  # [G,7]
        if _has_veff:
            _pl_comp = jt.concat([_pl_comp, light_extras["veff"]], dim=-1)  # [G,8] or [G,7]
        if features is not None:
            raster_extra_attrs = jt.concat([features, _pl_comp], dim=-1)   # aux 在前, PL 尾在后
        else:
            raster_extra_attrs = _pl_comp
    n_contri,rendered_image, rendered_depth,rendered_opacity, rendered_norm,depth_normal, rendered_alpha, radii, rendered_features = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3Ds_precomp=None,
        extra_attrs=raster_extra_attrs,
        inference_only=(not is_training),  # M2 (P0): no tape / no scratch retention in inference
        return_aux=return_aux
    )
    # Keep shape-only rasterizer statistics available until step evidence is written.
    # M2 (P0): only the training path holds these global diagnostic refs — they keep
    # the previous rasterizer (with its huge Geometry/Binning/Image buffers) alive,
    # which caused memory to accumulate across inference views.
    if is_training:
        import gaussian_renderer as _gr
        _gr._last_rasterize_func = getattr(rasterizer, '_last_rasterize_func', None)
        _gr._last_rasterizer = rasterizer

    # P0-b: split the PL channels off rendered_features. Early-return path
    # (return_aux=False) gives exactly [n_pl,H,W]; aux path gives [aux+n_pl,H,W].
    # P2: n_pl = 6 (diffuse+specular) or 7 (with front-face mask).
    diffuse_pl_image = None
    specular_pl_image = None
    front_pl_image = None
    veff_pl_image = None
    if is_pbr and return_light_components and not is_training:
        C = rendered_features.shape[0]
        if C >= _n_pl:
            if C > _n_pl:
                rendered_features, _pl_tail = rendered_features.split([C - _n_pl, _n_pl], dim=0)
            else:
                _pl_tail = rendered_features
            if _has_fm and _has_veff:
                diffuse_pl_image, specular_pl_image, front_pl_image, veff_pl_image = \
                    _pl_tail.split([3, 3, 1, 1], dim=0)
            elif _has_fm:
                diffuse_pl_image, specular_pl_image, front_pl_image = _pl_tail.split([3, 3, 1], dim=0)
            elif _has_veff:
                diffuse_pl_image, specular_pl_image, veff_pl_image = _pl_tail.split([3, 3, 1], dim=0)
            else:
                diffuse_pl_image, specular_pl_image = _pl_tail.split([3, 3], dim=0)

    # §30.5.2 E2-C: the compact exact-query metadata must reach the pkg on EVERY
    # return path (components on AND off — off tasks still write the receiver
    # identity) and be consistent with THIS view's receiver cloud G.
    if receiver_query_meta is not None:
        assert receiver_query_meta['G'] == xyz.shape[0], \
            f'exact receiver G {receiver_query_meta["G"]} != xyz G {xyz.shape[0]}'
    _rq = {'receiver_query_meta': receiver_query_meta,
           'receiver_query_diag': receiver_query_diag}

    if not return_aux and not is_training:
        if return_light_components and is_pbr:
            return {
                "render": rendered_image,
                "diffuse_rgb": light_extras.get("diffuse_rgb", 0) + light_extras.get("diffuse_rgb_pl", 0),
                "specular_rgb": light_extras.get("specular_rgb", 0) + light_extras.get("specular_rgb_pl", 0),
                "diffuse_rgb_pl": light_extras.get("diffuse_rgb_pl", 0),
                "specular_rgb_pl": light_extras.get("specular_rgb_pl", 0),
                "diffuse_pl_image": diffuse_pl_image,     # [3,H,W] pixel-space PL diffuse
                "specular_pl_image": specular_pl_image,   # [3,H,W] pixel-space PL specular
                "front_pl_image": front_pl_image,         # [1,H,W] pixel-space front-face mask (diag)
                "veff_pl_image": veff_pl_image,           # [1,H,W] pixel-space effective visibility (penumbra)
                "alpha": rendered_alpha,                  # [1,H,W] accumulated alpha (for normalising front mask)
                **_rq,
            }
        return {"render": rendered_image, **_rq}

    feature_dict = {}

    if is_pbr:
        if pc.with_matallic:
            if pc.normal_detal:
                precomput_normal,delta_normal_t,rendered_albedo,rendered_roughness,rendered_matallic = rendered_features.split([3,3,3,1,1], dim=0)
            else:
                precomput_normal,rendered_albedo,rendered_roughness,rendered_matallic = rendered_features.split([3,3,1,1], dim=0)
                delta_normal_t = None

            feature_dict.update({"albedo": rendered_albedo,
                            "roughness": rendered_roughness,
                            "matallic": rendered_matallic
                            })
        else:
            if pc.normal_detal:
                precomput_normal,delta_normal_t,rendered_albedo,rendered_roughness = rendered_features.split([3,3,3,1], dim=0)             
            else:
                precomput_normal,rendered_albedo,rendered_roughness = rendered_features.split([3,3,1], dim=0)
                delta_normal_t = None
            feature_dict.update({"albedo": rendered_albedo,
                            "roughness": rendered_roughness
                            })             
    else:
        if pc.normal_detal:
            precomput_normal,delta_normal_t = rendered_features.split([3, 3],dim=0)
        else:
            precomput_normal = rendered_features
            delta_normal_t = None
            

    precomput_normal = (precomput_normal - 0.5) * 2.0
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # if is_training:
    if is_pbr:
        results = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "neural_opacity": neural_opacity,
            "selection_mask": mask,
            "scaling": scaling,
            "normal": rendered_norm,
            "precomput_normal": precomput_normal,
            "delta_normal":delta_normal_t,
            "depth_normal":depth_normal,
            "depth": rendered_depth,
            "opacity": rendered_opacity,
            "alpha": rendered_alpha,
            "local_loss": local_loss,
            "scale_loss":loss_scale,
            "sdf_loss":sdf_loss,
            "points":xyz,
            "points_normal":normal,
            **_rq,
            }
        if return_light_components and is_pbr:
            results["diffuse_rgb"] = light_extras.get("diffuse_rgb", 0) + light_extras.get("diffuse_rgb_pl", 0)
            results["specular_rgb"] = light_extras.get("specular_rgb", 0) + light_extras.get("specular_rgb_pl", 0)
            results["diffuse_rgb_pl"] = light_extras.get("diffuse_rgb_pl", 0)
            results["specular_rgb_pl"] = light_extras.get("specular_rgb_pl", 0)
            if diffuse_pl_image is not None:
                results["diffuse_pl_image"] = diffuse_pl_image
                results["specular_pl_image"] = specular_pl_image
                results["front_pl_image"] = front_pl_image
                results["veff_pl_image"] = veff_pl_image
        results.update(feature_dict)
        if trace_intermediates:
            results["_decoder_intermediates"] = decoder_trace
            results["_trace_intermediates"] = {
                "means3D": xyz,
                "colors_precomp": color,
                "opacities": opacity,
                "scales": scaling,
                "rotations": rot,
                "extra_attrs": raster_extra_attrs,
            }
        # 2026-08-02: color/opacity/rot are not referenced by results (scaling,
        # xyz, normal, neural_opacity are). Release them to cut ~400MiB before
        # the caller processes the image. features is kept (rasterizer may hold it).
        del color, opacity, rot

        return results

    else:
        results = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
            "normal": rendered_norm,
            "precomput_normal": precomput_normal,
            "delta_normal":delta_normal_t,
            "depth_normal":depth_normal,
            "depth": rendered_depth,
            "opacity": rendered_opacity,
            "alpha": rendered_alpha,
            "local_loss": local_loss,
            "scale_loss":loss_scale,
            "sdf_loss":sdf_loss,
            "points":xyz,
            "points_normal":normal,
            **_rq,

            }
        if trace_intermediates:
            results["_decoder_intermediates"] = decoder_trace
            results["_trace_intermediates"] = {
                "means3D": xyz,
                "colors_precomp": color,
                "opacities": opacity,
                "scales": scaling,
                "rotations": rot,
                "extra_attrs": raster_extra_attrs,
            }
        return results


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : jt.Var,anchor_mask=None, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if anchor_mask is None:
        # Phase 78: GPU boolean indexing (aligns with PyTorch GANG).
        # jt.where(CUDA) + contrib.getitem handle boolean indexing on GPU correctly.
        anchor_mask = pc._anchor_mask  # GPU bool tensor

    # GPU boolean indexing — no numpy, no CPU roundtrip
    means3D = pc.get_anchor[anchor_mask]
    scales = pc.get_scaling[anchor_mask]
    rotations = pc.get_rotation[anchor_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # GPU boolean setitem (Phase 78: contrib.setitem handles bool masks on GPU)
    visible_mask = anchor_mask.clone()
    from jittor.contrib import setitem as _contrib_setitem
    _contrib_setitem(visible_mask, anchor_mask, radii_pure > 0)
    return visible_mask  # GPU bool tensor
