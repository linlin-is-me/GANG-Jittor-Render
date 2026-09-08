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

from typing import NamedTuple
from jittor import nn
import jittor as jt
# Import from sibling module in the same package (light_gaussian/)
# Import from sibling file in the parent light_gaussian/ package
import sys, os
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
import rasterize_points_jt as _rasterize_points_jt
from rasterize_points_jt import (
    RasterizeGaussiansCUDA, RasterizeGaussiansBackwardCUDA, RasterizeGaussiansFilterCUDA,
    markVisible, depthToNormal, SurfaceAlignCUDA, SurfaceAlignBackwardCUDA,
)


_MEDIAN_BLUR_CUDA_HEADER = r'''
__global__ void median_blur_3x3_forward(
        const float* input, float* output, int* winner, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = height * width;
    if (index >= count) return;
    int y = index / width;
    int x = index - y * width;
    float values[9];
    int sources[9];
    int n = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int yy = y + dy;
            int xx = x + dx;
            bool valid = yy >= 0 && yy < height && xx >= 0 && xx < width;
            values[n] = valid ? input[yy * width + xx] : 0.0f;
            sources[n] = valid ? yy * width + xx : -1;
            ++n;
        }
    }
    // Stable insertion sort matches the row-major 3x3 feature order used by
    // Kornia's zero-padded convolution implementation.
    for (int i = 1; i < 9; ++i) {
        float value = values[i];
        int source = sources[i];
        int j = i;
        while (j > 0 && values[j - 1] > value) {
            values[j] = values[j - 1];
            sources[j] = sources[j - 1];
            --j;
        }
        values[j] = value;
        sources[j] = source;
    }
    output[index] = values[4];
    winner[index] = sources[4];
}

__global__ void median_blur_3x3_backward(
        const float* grad_output, const int* winner, float* grad_input,
        int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int source = winner[index];
    if (source >= 0) atomicAdd(grad_input + source, grad_output[index]);
}
'''


class _MedianBlurDepth3x3(jt.Function):
    """Kornia-compatible 3x3 median for a single float32 depth plane."""

    def execute(self, depth):
        if depth.ndim != 2 or depth.dtype != jt.float32:
            raise ValueError(
                f"median depth expects float32 [H,W], got {depth.dtype} {depth.shape}")
        self.depth_shape = list(depth.shape)
        height, width = map(int, depth.shape)
        output = jt.zeros((height, width), dtype=jt.float32)
        winner = jt.zeros((height, width), dtype=jt.int32)
        output, winner = jt.code(
            outputs=[output, winner],
            inputs=[depth],
            data={'H': height, 'W': width},
            cuda_header=_MEDIAN_BLUR_CUDA_HEADER,
            cuda_src=r'''
@alias(input, in0) @alias(output, out0) @alias(winner, out1)
int count = data["H"] * data["W"];
median_blur_3x3_forward<<<(count + 255) / 256, 256>>>(
    input_p, output_p, winner_p, data["H"], data["W"]);
''',
        )
        self.winner = winner.stop_grad()
        return output

    def grad(self, grad_output):
        count = int(self.depth_shape[0]) * int(self.depth_shape[1])
        grad_input = jt.zeros(self.depth_shape, dtype=jt.float32)
        (grad_input,) = jt.code(
            outputs=[grad_input],
            inputs=[grad_output, self.winner],
            data={'COUNT': count},
            cuda_header=_MEDIAN_BLUR_CUDA_HEADER,
            cuda_src=r'''
@alias(grad_output, in0) @alias(winner, in1) @alias(grad_input, out0)
cudaMemset(grad_input_p, 0, (size_t)data["COUNT"] * sizeof(float));
median_blur_3x3_backward<<<(data["COUNT"] + 255) / 256, 256>>>(
    grad_output_p, winner_p, grad_input_p, data["COUNT"]);
''',
        )
        self.winner = None
        return grad_input


def median_blur_depth_3x3(depth):
    """Apply the source renderer's zero-padded 3x3 median to depth."""
    if depth.ndim == 3:
        if int(depth.shape[0]) != 1:
            raise ValueError(f"depth channel count must be one, got {depth.shape}")
        return _MedianBlurDepth3x3()(depth.squeeze(0)).unsqueeze(0)
    if depth.ndim == 2:
        return _MedianBlurDepth3x3()(depth)
    raise ValueError(f"depth must have shape [H,W] or [1,H,W], got {depth.shape}")

class _surface_align(jt.Function):

    def save_for_backward(self, *args):
        self.saved_tensors = args

    @staticmethod
    def _bucket_capacity(active_rows):
        if active_rows <= 0:
            raise ValueError("surface alignment requires at least one anchor")
        return 1 << (int(active_rows) - 1).bit_length()

    @staticmethod
    def _pad_rows(value, target_rows):
        current_rows = int(value.shape[0])
        if current_rows == int(target_rows):
            return value
        if current_rows > int(target_rows):
            raise ValueError("surface alignment padding target is too small")
        shape = [int(target_rows) - current_rows, *value.shape[1:]]
        return jt.concat([value, jt.zeros(shape, dtype=value.dtype)], dim=0)

    def execute(self, anchor, offsets_all, rotation, knn_index):
        """anchor (N,3), offsets_all (N*K,3), rotation (N*K,4), knn_index (N,K)"""
        active_rows = int(knn_index.shape[0])
        K = int(knn_index.shape[1])
        capacity_rows = self._bucket_capacity(active_rows)
        active_offsets = active_rows * K
        capacity_offsets = capacity_rows * K

        # jt.code specializes on output shapes.  Visibility produces a new N
        # for nearly every camera, while topology changes every 100 steps.
        # Power-of-two capacity buckets keep the custom CUDA op shape stable;
        # trailing rows are excluded from both returned losses and gradients.
        padded_anchor = self._pad_rows(anchor, capacity_rows)
        padded_offsets = self._pad_rows(offsets_all, capacity_offsets)
        padded_rotation = self._pad_rows(rotation, capacity_offsets)
        if capacity_rows == active_rows:
            padded_knn = knn_index
        else:
            padding_knn = jt.arange(
                active_offsets, capacity_offsets, dtype=jt.int32,
            ).reshape(capacity_rows - active_rows, K)
            padded_knn = jt.concat([knn_index, padding_knn], dim=0)

        from ..rasterize_points_jt import repeat_cuda
        with jt.no_grad():
            repeat_anchor_all = repeat_cuda(padded_anchor, K)
        xyz_all = repeat_anchor_all + padded_offsets

        loss_d, loss_normal, binning_buffer, mean_d = SurfaceAlignCUDA(
            xyz_all, padded_rotation, padded_knn)
        self.save_for_backward(
            padded_anchor, padded_offsets, padded_rotation,
            binning_buffer, padded_knn, mean_d,
            active_rows, active_offsets,
        )
        return loss_d[:active_offsets], loss_normal[:active_offsets]

    def grad(self, grad_out_loss_d, grad_out_loss_normal):
        (anchor, offsets_all, rotation, binning_buffer, knn_index, mean_d,
         active_rows, active_offsets) = self.saved_tensors
        K = int(knn_index.shape[1])
        capacity_offsets = int(offsets_all.shape[0])

        if grad_out_loss_d is None:
            grad_out_loss_d = jt.zeros([active_offsets], dtype=mean_d.dtype)
        if grad_out_loss_normal is None:
            grad_out_loss_normal = jt.zeros(
                [active_offsets], dtype=mean_d.dtype)
        grad_out_loss_d = self._pad_rows(
            grad_out_loss_d.reshape(-1, 1), capacity_offsets).reshape(-1)
        grad_out_loss_normal = self._pad_rows(
            grad_out_loss_normal.reshape(-1, 1), capacity_offsets).reshape(-1)

        from ..rasterize_points_jt import repeat_cuda, repeat_sum_bwd_cuda
        with jt.no_grad():
            repeat_anchor_all = repeat_cuda(anchor, K)
        xyz_all = repeat_anchor_all + offsets_all

        grad_xyz, grad_rotation = SurfaceAlignBackwardCUDA(
            xyz_all, rotation, mean_d, knn_index,
            grad_out_loss_d, grad_out_loss_normal)

        grad_anchor = repeat_sum_bwd_cuda(grad_xyz, K)

        # Production performs one coordinated jt.grad. Release the graph-owned
        # state after that backward so a completed step cannot retain its Vars.
        self.saved_tensors = None
        return (
            grad_anchor[:active_rows],
            grad_xyz[:active_offsets],
            grad_rotation[:active_offsets],
            None,
        )


class SurfaceAlign(nn.Module):
    def __init__(self):
        super().__init__()
        self.alignFunc = _surface_align()

    def execute(self, anchor, offsets_all, rotation, knn_index):
        return self.alignFunc(anchor, offsets_all, rotation, knn_index)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    norm3Ds_precomp,
    extra_attrs,
    raster_settings,
):
    num_contrib, color, depth, opacity, norm, alpha, radii, extra = _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        norm3Ds_precomp,
        extra_attrs,
        raster_settings,
    )

    norm = jt.normalize(norm, p=2, dim=0)

    focal_x = raster_settings.image_width / (2.0 * raster_settings.tanfovx)
    focal_y = raster_settings.image_height / (2.0 * raster_settings.tanfovy)
    depth_filter = median_blur_depth_3x3(depth)
    normal_from_depth = depthToNormal(
        depth_filter.squeeze(0) if depth_filter.ndim == 3 else depth_filter,
        raster_settings.viewmatrix,
        focal_x,
        focal_y,
    )
    return num_contrib, color, depth, opacity, norm, normal_from_depth, alpha, radii, extra

class _RasterizeGaussians(jt.Function):

    def save_for_backward(self, *args):
        self.saved_tensors = args

    def execute(
        self,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        norm3Ds_precomp,
        extra_attrs,
        raster_settings,
    ):
        assert extra_attrs.shape[0] == 0 or extra_attrs.shape[1] <= 34

        if raster_settings.debug:
            try:
                num_rendered, num_contrib, color, depth, opacity, norm, alpha, extra, radii, geomBuffer, binningBuffer, imgBuffer = RasterizeGaussiansCUDA(
                    raster_settings.bg, means3D, colors_precomp, opacities, scales, rotations,
                    raster_settings.scale_modifier, cov3Ds_precomp, norm3Ds_precomp, extra_attrs,
                    raster_settings.viewmatrix, raster_settings.projmatrix,
                    raster_settings.tanfovx, raster_settings.tanfovy,
                    raster_settings.image_height, raster_settings.image_width,
                    sh, raster_settings.sh_degree, raster_settings.campos,
                    raster_settings.prefiltered, raster_settings.debug)
            except Exception as ex:
                raise RuntimeError(
                    "rasterizer forward failed in debug mode; legacy object "
                    "snapshots are disabled, use the NPZ diagnostic exporter") from ex
        else:
            num_rendered, num_contrib, color, depth, opacity, norm, alpha, extra, radii, geomBuffer, binningBuffer, imgBuffer = RasterizeGaussiansCUDA(
                raster_settings.bg, means3D, colors_precomp, opacities, scales, rotations,
                raster_settings.scale_modifier, cov3Ds_precomp, norm3Ds_precomp, extra_attrs,
                raster_settings.viewmatrix, raster_settings.projmatrix,
                raster_settings.tanfovx, raster_settings.tanfovy,
                raster_settings.image_height, raster_settings.image_width,
                sh, raster_settings.sh_degree, raster_settings.campos,
                raster_settings.prefiltered, raster_settings.debug)

        self.raster_settings = raster_settings
        self.num_rendered = num_rendered
        point_count = int(means3D.shape[0])
        extra_dim = int(extra_attrs.shape[1]) if extra_attrs.ndim >= 2 and extra_attrs.shape[0] else 0
        geom_bytes = int(geomBuffer.numel())
        binning_bytes = int(binningBuffer.numel())
        image_bytes = int(imgBuffer.numel())
        self.runtime_stats = {
            "P": point_count,
            "R": int(num_rendered),
            "ED": extra_dim,
            "forward_buffer_bytes": {
                "geometry": geom_bytes,
                "binning": binning_bytes,
                "image": image_bytes,
                "total": geom_bytes + binning_bytes + image_bytes,
            },
            "backward_estimated_bytes": {
                "fp64_accumulators": 8 * point_count * (15 + extra_dim),
                "mean_conic_partials": 8 * int(num_rendered) * 5,
                "second_sort_keys": 8 * int(num_rendered) * 2,
                "cub_sort_scratch": None,
            },
        }
        self.save_for_backward(colors_precomp, means3D, scales, rotations,
                               cov3Ds_precomp, norm3Ds_precomp, radii, extra_attrs,
                               sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        return num_contrib, color, depth, opacity, norm, alpha, radii, extra

    def grad(self, grad_out_contrib, grad_out_color, grad_out_depth, grad_out_opacity, grad_out_norm, grad_out_alpha, _, grad_out_extra):
        num_rendered = self.num_rendered
        raster_settings = self.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, norm3Ds_precomp, \
            radii, extra_attrs, sh, geomBuffer, binningBuffer, imgBuffer, alpha = self.saved_tensors

        if grad_out_color is None:
            grad_out_color = jt.zeros((3, grad_out_depth.shape[1], grad_out_depth.shape[2]))
        if grad_out_depth is None:
            grad_out_depth = jt.zeros((1, grad_out_color.shape[1], grad_out_color.shape[2]))
        if grad_out_norm is None:
            grad_out_norm = jt.zeros((3, grad_out_color.shape[1], grad_out_color.shape[2]))
        if grad_out_alpha is None:
            grad_out_alpha = jt.zeros((1, grad_out_color.shape[1], grad_out_color.shape[2]))
        if grad_out_extra is None:
            if extra_attrs.shape[0] != 0:
                grad_out_extra = jt.zeros((extra_attrs.shape[1], grad_out_color.shape[1], grad_out_color.shape[2]))
            else:
                grad_out_extra = jt.zeros([1])

        if raster_settings.debug:
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_norm3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_extra_attrs = RasterizeGaussiansBackwardCUDA(
                    raster_settings.bg, means3D, radii, colors_precomp, scales, rotations, extra_attrs,
                    raster_settings.scale_modifier, cov3Ds_precomp, norm3Ds_precomp,
                    raster_settings.viewmatrix, raster_settings.projmatrix,
                    raster_settings.tanfovx, raster_settings.tanfovy,
                    grad_out_color, grad_out_depth, grad_out_norm, grad_out_alpha, grad_out_extra,
                    sh, raster_settings.sh_degree, raster_settings.campos,
                    geomBuffer, num_rendered, binningBuffer, imgBuffer, alpha, raster_settings.debug)
            except Exception as ex:
                raise RuntimeError(
                    "rasterizer backward failed in debug mode; legacy object "
                    "snapshots are disabled, use the NPZ diagnostic exporter") from ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_norm3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_extra_attrs = RasterizeGaussiansBackwardCUDA(
                raster_settings.bg, means3D, radii, colors_precomp, scales, rotations, extra_attrs,
                raster_settings.scale_modifier, cov3Ds_precomp, norm3Ds_precomp,
                raster_settings.viewmatrix, raster_settings.projmatrix,
                raster_settings.tanfovx, raster_settings.tanfovy,
                grad_out_color, grad_out_depth, grad_out_norm, grad_out_alpha, grad_out_extra,
                sh, raster_settings.sh_degree, raster_settings.campos,
                geomBuffer, num_rendered, binningBuffer, imgBuffer, alpha, raster_settings.debug)

        gradients = (
            grad_means3D,         # means3D
            grad_means2D,         # means2D
            grad_sh,              # sh
            grad_colors_precomp,  # colors_precomp
            grad_opacities,       # opacities
            grad_scales,          # scales
            grad_rotations,       # rotations
            grad_cov3Ds_precomp,  # cov3Ds_precomp
            grad_norm3Ds_precomp, # norm3Ds_precomp
            grad_extra_attrs,     # extra_attrs
            None                  # raster_settings
        )
        # The production coordinator calls jt.grad exactly once.  Native
        # reverse outputs are synchronized before returning, so the large
        # Geometry/Binning/Image buffers no longer need Python ownership.
        del self.saved_tensors
        return gradients

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : jt.Var
    scale_modifier : float
    viewmatrix : jt.Var
    projmatrix : jt.Var
    sh_degree : int
    campos : jt.Var
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings
        # Persistent Function instance: Jittor 1.3.11's tape_together may lose
        # the Python instance reference during C++→Python callback, causing
        # "saved_tensors missing" on a fresh instance. Persistent instance avoids GC.
        self._rasterize_func = _RasterizeGaussians()

    def markVisible(self, positions):
        with jt.no_grad():
            raster_settings = self.raster_settings
            visible = markVisible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
        return visible

    def _execute_inference(self, means3D, means2D, shs, colors_precomp, opacities,
                           scales, rotations, cov3Ds_precomp, norm3Ds_precomp, extra_attrs,
                           return_aux=True):
        """M2 (P0): inference-only rasterize — direct RasterizeGaussiansCUDA call,
        NO jt.Function tape, NO save_for_backward, NO diagnostic refs (_last_*_input,
        _last_rasterize_func). Scratch buffers are released right after forward_1 so
        multi-view inference does not accumulate the previous view's Geometry/Binning/
        Image buffers (previously retained via _gr._last_rasterizer)."""
        raster_settings = self.raster_settings

        if shs is None:
            shs = jt.array([])
        if colors_precomp is None:
            colors_precomp = jt.array([])
        if scales is None:
            raise ValueError('To support norm and depth prediction, scales == None is not allowed')
        if rotations is None:
            raise ValueError('To support norm and depth prediction, rotations == None is not allowed')
        if cov3Ds_precomp is None:
            cov3Ds_precomp = jt.array([])
        if norm3Ds_precomp is None:
            norm3Ds_precomp = jt.array([])
        if extra_attrs is None:
            extra_attrs = jt.array([])

        num_rendered, num_contrib, color, depth, opacity, norm, alpha, extra, radii, \
            geomBuffer, binningBuffer, imgBuffer = RasterizeGaussiansCUDA(
                raster_settings.bg, means3D, colors_precomp, opacities, scales, rotations,
                raster_settings.scale_modifier, cov3Ds_precomp, norm3Ds_precomp, extra_attrs,
                raster_settings.viewmatrix, raster_settings.projmatrix,
                raster_settings.tanfovx, raster_settings.tanfovy,
                raster_settings.image_height, raster_settings.image_width,
                shs, raster_settings.sh_degree, raster_settings.campos,
                raster_settings.prefiltered, raster_settings.debug)

        # M2 (P0, bisect F): drop the Python refs to scratch buffers; the caller's
        # frame cleanup (del pkg + sync_all(True) + gc) handles device-sync/collect.
        # (Removed the in-rasterizer sync_all(True)+gc — redundant with the frame
        # cleanup and was a suspect for premature SFRL free between views.)
        del geomBuffer, binningBuffer, imgBuffer

        # Post-processing (inlined from rasterize_gaussians / execute)
        if return_aux:
            norm = jt.normalize(norm, p=2, dim=0)
            focal_x = raster_settings.image_width / (2.0 * raster_settings.tanfovx)
            focal_y = raster_settings.image_height / (2.0 * raster_settings.tanfovy)
            depth_filter = median_blur_depth_3x3(depth)
            normal_from_depth = depthToNormal(
                depth_filter.squeeze(0) if depth_filter.ndim == 3 else depth_filter,
                raster_settings.viewmatrix,
                focal_x,
                focal_y,
            )
        else:
            # RGB-only inference does not need the additional full-resolution
            # depth-derived normal image (~200 MiB at 5187x3361).
            normal_from_depth = jt.array([])
        return num_contrib, color, depth, opacity, norm, normal_from_depth, alpha, radii, extra

    def execute(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3Ds_precomp = None, norm3Ds_precomp=None, extra_attrs=None, inference_only=False, return_aux=True):

        # M2 (P0): inference-only path — no tape, no save_for_backward, no retention.
        if inference_only:
            return self._execute_inference(
                means3D, means2D, shs, colors_precomp, opacities, scales, rotations,
                cov3Ds_precomp, norm3Ds_precomp, extra_attrs, return_aux=return_aux)

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or ((scales is not None or rotations is not None) and cov3Ds_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = jt.array([])
        if colors_precomp is None:
            colors_precomp = jt.array([])

        if scales is None:
            raise ValueError('To support norm and depth prediction, scales == None is not allowed')
        if rotations is None:
            raise ValueError('To support norm and depth prediction, rotations == None is not allowed')
        if cov3Ds_precomp is None:
            cov3Ds_precomp = jt.array([])
        if norm3Ds_precomp is None:
            norm3Ds_precomp = jt.array([])
        if extra_attrs is None:
            extra_attrs = jt.array([])
        # Keep the Function instance owned by this module while Jittor's tape is
        # live.  The backward callback releases its saved tensors immediately.
        self._last_rasterize_func = self._rasterize_func
        num_contrib, color, depth, opacity, norm, alpha, radii, extra = \
            self._last_rasterize_func(
                means3D,
                means2D,
                shs,
                colors_precomp,
                opacities,
                scales,
                rotations,
                cov3Ds_precomp,
                norm3Ds_precomp,
                extra_attrs,
                raster_settings,
            )

        # Post-processing (inlined from rasterize_gaussians)
        norm = jt.normalize(norm, p=2, dim=0)
        focal_x = raster_settings.image_width / (2.0 * raster_settings.tanfovx)
        focal_y = raster_settings.image_height / (2.0 * raster_settings.tanfovy)
        depth_filter = median_blur_depth_3x3(depth)
        normal_from_depth = depthToNormal(
            depth_filter.squeeze(0) if depth_filter.ndim == 3 else depth_filter,
            raster_settings.viewmatrix,
            focal_x,
            focal_y,
        )
        return num_contrib, color, depth, opacity, norm, normal_from_depth, alpha, radii, extra

    def visible_filter(self, means3D, scales = None, rotations = None, cov3D_precomp = None):

        raster_settings = self.raster_settings

        if scales is None:
            scales = jt.array([])
        if rotations is None:
            rotations = jt.array([])
        if cov3D_precomp is None:
            cov3D_precomp = jt.array([])

        with jt.no_grad():
            radii = RasterizeGaussiansFilterCUDA(
                means3D, scales, rotations,
                raster_settings.scale_modifier,
                cov3D_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.prefiltered,
                raster_settings.debug)
        return radii
