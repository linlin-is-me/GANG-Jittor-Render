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

import jittor as jt
import jittor.nn as F
from math import exp
from utils.graphics_utils import fov2focal
import numpy as np
# Jittor replacements for kornia functions
from utils.kornia_ops_jt import (
    jt_erosion2d, jt_opencv_binary_erosion, jt_spatial_gradient,
)

def _stable_full_mean(value):
    """Accumulate large FP32 loss maps in FP64 on Jittor 1.3.11.

    Its CUDA FP32 full reduction has a measured 9.1e-5 absolute error on the
    controlled 3x416x634 L1 map, while FP64 agrees with NumPy to 1.1e-9.
    Casting the scalar back keeps the public loss dtype and the gradient path.
    """
    return value.float64().mean().float32()


def l1_loss(network_output, gt):
    return _stable_full_mean(jt.abs(network_output - gt))

def l2_loss(network_output, gt):
    return _stable_full_mean((network_output - gt) ** 2)

def gaussian(window_size, sigma):
    gauss = jt.array([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = (_1D_window @ _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    # NOTE: .sync() removed — in Jittor 1.3.11, .sync() on CUDA-only tensors
    # raises RuntimeError and may corrupt internal tensor state, producing NaN.
    # The SSIM computation works fine on GPU without explicit sync.

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return _stable_full_mean(ssim_map)
    else:
        return ssim_map.mean(1).mean(1).mean(1)




def get_tv_loss(gt_image,prediction,pad = 1,step = 1):
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = jt.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = jt.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = jt.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = jt.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = jt.exp(
                -(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            rgb_grad_w = jt.exp(
                -(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            tv_h = jt.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  # [C, H-1, W]
            tv_w = jt.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)  # [C, H, W-1]
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss


def depth2normal(depth, mask, camera):
    # conver to camera position
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    h, w, _ = jt.meshgrid(jt.arange(0, shape[0]), jt.arange(0, shape[1]), jt.arange(0, shape[2]),
                             indexing='ij')
    h = h.float32()
    w = w.float32()
    p = jt.concat([w, h], axis=-1)

    p[..., 0:1] -= 0.5 * camera.image_width
    p[..., 1:2] -= 0.5 * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = jt.array([K00, 0, 0, K11]).reshape([2, 2])
    Kinv = jt.linalg.inv(K)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = jt.concat([p, camD], -1)

    # padded = mod.contour_padding(camPos.contiguous(), mask.contiguous(), jt.zeros_like(camPos), filter_size // 2)
    # camPos = camPos + padded
    p = F.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask = F.pad(mask[None].float32(), [0, 0, 1, 1, 1, 1], mode='replicate').bool()

    p_c = (p[:, 1:-1, 1:-1, :]) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:, :-2, 1:-1, :] - p_c) * mask[:, :-2, 1:-1, :]
    p_l = (p[:, 1:-1, :-2, :] - p_c) * mask[:, 1:-1, :-2, :]
    p_b = (p[:, 2:, 1:-1, :] - p_c) * mask[:, 2:, 1:-1, :]
    p_r = (p[:, 1:-1, 2:, :] - p_c) * mask[:, 1:-1, 2:, :]

    n_ul = jt.cross(p_u, p_l)
    n_ur = jt.cross(p_r, p_u)
    n_br = jt.cross(p_b, p_r)
    n_bl = jt.cross(p_l, p_b)

    # n_ul = jt.normalize(jt.cross(p_u, p_l), dim=-1)
    # n_ur = jt.normalize(jt.cross(p_r, p_u), dim=-1)
    # n_br = jt.normalize(jt.cross(p_b, p_r), dim=-1)
    # n_bl = jt.normalize(jt.cross(p_l, p_b), dim=-1)

    # n_ul = jt.normalize(jt.cross(p_l, p_u), dim=-1)
    # n_ur = jt.normalize(jt.cross(p_u, p_r), dim=-1)
    # n_br = jt.normalize(jt.cross(p_r, p_b), dim=-1)
    # n_bl = jt.normalize(jt.cross(p_b, p_l), dim=-1)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]

    # n *= -jt.sum(camVDir * camN, -1, True).sign() # no cull back

    mask = mask[0, 1:-1, 1:-1, :]

    # n = gaussian_blur(n, filter_size, 1) * mask

    n = jt.normalize(n, dim=-1)
    # n[..., 1] *= -1
    # n *= -1

    n = (n * mask).permute([2, 0, 1])
    return n

def cos_loss(output, gt, thrsh=0, weight=1):
    cos = jt.sum(output * gt * weight, 0)
    return (1 - cos[cos < np.cos(thrsh)]).mean()

def normal2curv(normal, mask):
    # normal = normal.detach()
    n = normal.permute([1, 2, 0])
    m = mask.permute([1, 2, 0])
    n = F.pad(n[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    m = F.pad(m[None].float32(), [0, 0, 1, 1, 1, 1], mode='replicate').bool()
    n_c = (n[:, 1:-1, 1:-1, :]      ) * m[:, 1:-1, 1:-1, :]
    n_u = (n[:,  :-2, 1:-1, :] - n_c) * m[:,  :-2, 1:-1, :]
    n_l = (n[:, 1:-1,  :-2, :] - n_c) * m[:, 1:-1,  :-2, :]
    n_b = (n[:, 2:  , 1:-1, :] - n_c) * m[:, 2:  , 1:-1, :]
    n_r = (n[:, 1:-1, 2:  , :] - n_c) * m[:, 1:-1, 2:  , :]
    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1]) * mask
    curv = curv.norm(1, 0, True)
    return curv


def normal_loss(render_normal,render_depth,render_opacity,viewpoint_cam,opt):
    mask_vis = (render_opacity.detach() > 1e-5)
    # gt_normal = viewpoint_cam.normal
    render_normal = jt.normalize(render_normal, dim=-1)  * mask_vis
    # gt_normal = jt.normalize(gt_normal, dim=-1)  * mask_vis
    dep2nor = depth2normal(render_depth, mask_vis, viewpoint_cam)

    loss_pseudo = cos_loss(render_normal, dep2nor, thrsh=np.pi * 1 / 10000, weight=1)
    curv_n = normal2curv(render_normal, mask_vis)
    loss_curv = l1_loss(curv_n * 1, 0)

    # loss_N = cos_loss(render_normal, gt_normal)
    # loss_N = l2_loss(render_normal, gt_normal)

    loss_normal = opt.pseudo*loss_pseudo + opt.curv*loss_curv
    # loss_normal = opt.normal*loss_N

    return loss_normal




def get_masked_tv_loss(
    mask: jt.Var,  # [1, H, W]
    gt_image: jt.Var,  # [3, H, W]
    prediction: jt.Var,  # [C, H, W]
    erosion: bool = False,
) -> jt.Var:
    rgb_grad_h = jt.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = jt.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = jt.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = jt.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]

    # erode mask (Jittor replacement for kornia.morphology.erosion)
    mask = mask.float()
    if erosion:
        mask = jt_erosion2d(mask[None, ...], kernel_size=7)[0]
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]
    # mask_h = mask[1:, :] * mask[:-1, :]  # [1, H-1, W]
    # mask_w = mask[:, 1:] * mask[:, :-1]  # [1, H, W-1]


    tv_loss = _stable_full_mean(tv_h * rgb_grad_h * mask_h) + _stable_full_mean(
        tv_w * rgb_grad_w * mask_w
    )

    return tv_loss


def predicted_normal_loss(normal, normal_ref, alpha=None, threshold=0.05):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: ( H, W)
    if alpha is not None:
        # The source path thresholds to a surface mask and applies OpenCV
        # erosion with a 4x4 kernel.  Keep the mask on device while preserving
        # its even-kernel anchor and border contract.
        weight = alpha.detach().float()
        weight = jt.where(weight < threshold, jt.zeros_like(weight), weight)
        weight = jt_opencv_binary_erosion(
            weight[None, None, ...], kernel_size=4)[0, 0]
        weight = weight[None, ...].repeat(3, 1, 1)
    else:
        weight = jt.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    n = normal_ref.permute(1,2,0).reshape(-1,3)
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = _stable_full_mean(w * (1.0 - jt.sum(n * n_pred, dim=-1)))

    return loss



def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = jt.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = jt.mean(jt.log(val) + jt.log(1 - val))
    return loss


def delta_normal_loss(delta_normal_norm, alpha=None):
    # delta_normal_norm: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        # GPU erosion via jt_erosion2d (Jittor native) — no CPU round-trip
        weight = alpha.detach().float()
        weight = jt_erosion2d(weight[None, None, ...], kernel_size=7)[0, 0]  # [H, W]
        weight = weight[None, ...].repeat(3, 1, 1)
    else:
        weight = jt.ones_like(delta_normal_norm)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    l = delta_normal_norm.permute(1,2,0).reshape(-1,3)[...,0]
    loss = (w * l).mean()

    return loss

def first_order_edge_aware_loss(data, img):
    return (jt_spatial_gradient(data[None], order=1)[0].abs() * jt.exp(-jt_spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()


def eikonal_loss(sdf_gradients):
    gradient_error = (jt.norm(sdf_gradients.reshape(-1, 3), p=2, dim=-1) - 1.0) ** 2
    return gradient_error.mean()


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = jt.array([[W - 1, H - 1]])
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    # get_calib_matrix_nerf supplies the pinhole matrix
    # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].  Jittor 1.3.11 implements
    # linalg.inv through numpy_code and imports optional CuPy when CUDA is on,
    # so the normal-loss backward failed exactly when it first became active.
    # Apply the closed-form inverse to preserve the PyTorch row-vector result
    # while keeping the depth path entirely in Jittor autograd.
    matrix = intrinsic[0, ...]
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    cam_x = (cam_xy[..., 0:1] - cx * cam_z) / fx
    cam_y = (cam_xy[..., 1:2] - cy * cam_z) / fy
    return jt.concat([cam_x, cam_y, cam_z], dim=-1)


def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = jt.arange(W, dtype=jt.float32) / (W - 1)
    valid_y = jt.arange(H, dtype=jt.float32) / (H - 1)
    valid_y, valid_x = jt.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = jt.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth_pcd2normal(xyz):
    hd, wd,_ = xyz.shape 
    bottom_point = xyz[..., 2:hd,   1:wd-1, :]
    top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
    right_point  = xyz[..., 1:hd-1, 2:wd,   :]
    left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = jt.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = jt.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = F.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal


class _DepthNormalFunction(jt.Function):
    """Pinhole depth normals with a fixed-order gather backward."""

    def execute(self, depth, intrinsic):
        if depth.ndim != 2 or depth.dtype != jt.float32:
            raise ValueError(
                f"depth normal expects a float32 two-dimensional depth map, "
                f"got {depth.dtype} {depth.shape}")
        if tuple(intrinsic.shape) != (3, 3) or intrinsic.dtype != jt.float32:
            raise ValueError(
                f"depth normal expects a float32 [3,3] intrinsic matrix, "
                f"got {intrinsic.dtype} {intrinsic.shape}")
        self.saved_tensors = (depth, intrinsic)
        h, w = map(int, depth.shape)
        return jt.code(
            [h, w, 3], jt.float32, inputs=[depth, intrinsic], cuda_header=f'''
            __global__ void depth_normal_forward_kernel(
                    const float* in0_p, const float* in1_p, float* out0_p) {{
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int H = {h}, W = {w};
            if (index >= H * W) return;
            const int y = index / W, x = index - y * W;
            float* output_value = out0_p + index * 3;
            output_value[0] = output_value[1] = output_value[2] = 0.0f;
            if (x <= 0 || x >= W - 1 || y <= 0 || y >= H - 1) return;
            const float fx = in1_p[0], fy = in1_p[4];
            const float cx = in1_p[2], cy = in1_p[5];
            const float dl = in0_p[y * W + x - 1];
            const float dr = in0_p[y * W + x + 1];
            const float dt = in0_p[(y - 1) * W + x];
            const float db = in0_p[(y + 1) * W + x];
            const float3 pl = make_float3((x - 1 - cx) * dl / fx, (y - cy) * dl / fy, dl);
            const float3 pr = make_float3((x + 1 - cx) * dr / fx, (y - cy) * dr / fy, dr);
            const float3 pt = make_float3((x - cx) * dt / fx, (y - 1 - cy) * dt / fy, dt);
            const float3 pb = make_float3((x - cx) * db / fx, (y + 1 - cy) * db / fy, db);
            const float3 a = make_float3(pr.x-pl.x, pr.y-pl.y, pr.z-pl.z);
            const float3 b = make_float3(pt.x-pb.x, pt.y-pb.y, pt.z-pb.z);
            float3 n = make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
            const float length = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
            if (length > 1.0e-12f) {{
                output_value[0] = n.x / length; output_value[1] = n.y / length; output_value[2] = n.z / length;
            }}
            }}
            ''', cuda_src=f'''
            const int total = {h * w};
            depth_normal_forward_kernel<<<(total + 255) / 256, 256>>>(
                in0_p, in1_p, out0_p);
            ''')

    def grad(self, grad_normal):
        depth, intrinsic = self.saved_tensors
        h, w = map(int, depth.shape)
        grad_depth = jt.code(
            [h, w], jt.float32, inputs=[depth, intrinsic, grad_normal], cuda_header=f'''
            __global__ void depth_normal_backward_kernel(
                    const float* in0_p, const float* in1_p,
                    const float* in2_p, float* out0_p) {{
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int H = {h}, W = {w};
            if (index >= H * W) return;
            const int py = index / W, px = index - py * W;
            const float fx = in1_p[0], fy = in1_p[4];
            const float cx0 = in1_p[2], cy0 = in1_p[5];
            double result = 0.0;
            for (int role = 0; role < 4; ++role) {{
                int x = px, y = py;
                if (role == 0) x = px - 1;
                if (role == 1) x = px + 1;
                if (role == 2) y = py + 1;
                if (role == 3) y = py - 1;
                if (x <= 0 || x >= W - 1 || y <= 0 || y >= H - 1) continue;
                const float dl = in0_p[y * W + x - 1];
                const float dr = in0_p[y * W + x + 1];
                const float dt = in0_p[(y - 1) * W + x];
                const float db = in0_p[(y + 1) * W + x];
                const float3 pl = make_float3((x - 1 - cx0) * dl / fx, (y - cy0) * dl / fy, dl);
                const float3 pr = make_float3((x + 1 - cx0) * dr / fx, (y - cy0) * dr / fy, dr);
                const float3 pt = make_float3((x - cx0) * dt / fx, (y - 1 - cy0) * dt / fy, dt);
                const float3 pb = make_float3((x - cx0) * db / fx, (y + 1 - cy0) * db / fy, db);
                const float3 a = make_float3(pr.x-pl.x, pr.y-pl.y, pr.z-pl.z);
                const float3 b = make_float3(pt.x-pb.x, pt.y-pb.y, pt.z-pb.z);
                const float3 c = make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
                const float length = sqrtf(c.x*c.x + c.y*c.y + c.z*c.z);
                if (length <= 1.0e-12f) continue;
                const float3 n = make_float3(c.x/length, c.y/length, c.z/length);
                const float* gout = in2_p + (y * W + x) * 3;
                const float dot = n.x*gout[0] + n.y*gout[1] + n.z*gout[2];
                const float3 gc = make_float3((gout[0]-n.x*dot)/length,
                                              (gout[1]-n.y*dot)/length,
                                              (gout[2]-n.z*dot)/length);
                const float3 ga = make_float3(b.y*gc.z-b.z*gc.y,
                                              b.z*gc.x-b.x*gc.z,
                                              b.x*gc.y-b.y*gc.x);
                const float3 gb = make_float3(gc.y*a.z-gc.z*a.y,
                                              gc.z*a.x-gc.x*a.z,
                                              gc.x*a.y-gc.y*a.x);
                float3 gp;
                if (role == 0) gp = ga;
                else if (role == 1) gp = make_float3(-ga.x,-ga.y,-ga.z);
                else if (role == 2) gp = gb;
                else gp = make_float3(-gb.x,-gb.y,-gb.z);
                result += (double)gp.x * (px - cx0) / fx
                        + (double)gp.y * (py - cy0) / fy
                        + (double)gp.z;
            }}
            out0_p[index] = static_cast<float>(result);
            }}
            ''', cuda_src=f'''
            const int total = {h * w};
            depth_normal_backward_kernel<<<(total + 255) / 256, 256>>>(
                in0_p, in1_p, in2_p, out0_p);
            ''')
        self.saved_tensors = ()
        return grad_depth, None

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    matrix = intrinsic_matrix.float32().contiguous()
    depth_2d = depth[0] if depth.ndim == 3 else depth
    return _DepthNormalFunction()(depth_2d, matrix)


def render_normal_from_depth(viewpoint_cam, depth):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_ref = normal_from_depth_image(depth, 
                                        intrinsic_matrix,  # Jittor: no .to(device)
                                        extrinsic_matrix)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref
