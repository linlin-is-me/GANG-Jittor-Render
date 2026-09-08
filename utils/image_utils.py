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
import numpy as np
import cv2


def reference_metric_inputs_numpy(img1, img2):
    """Return FP32 metric inputs using GANG training-evaluation clipping.

    Raw render artifacts remain untouched; only PSNR/SSIM inputs are clipped.
    """
    lhs = np.asarray(img1, dtype=np.float32)
    rhs = np.asarray(img2, dtype=np.float32)
    if lhs.shape != rhs.shape or lhs.ndim != 3:
        raise ValueError(
            f"metric inputs must be same-shape CHW arrays, got {lhs.shape} and {rhs.shape}")
    return np.clip(lhs, 0.0, 1.0), np.clip(rhs, 0.0, 1.0)


def channelwise_psnr_numpy(img1, img2):
    """Match GANG-master ``psnr(...).mean()`` for CHW images.

    The reference computes one MSE and PSNR per channel before averaging the
    channel scores.  Computing a single MSE over all channels is generally not
    equivalent.
    """
    lhs = np.asarray(img1, dtype=np.float64)
    rhs = np.asarray(img2, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 3:
        raise ValueError(
            f"PSNR inputs must be same-shape CHW arrays, got {lhs.shape} and {rhs.shape}")
    channel_mse = ((lhs - rhs) ** 2).reshape(lhs.shape[0], -1).mean(axis=1)
    channel_psnr = np.where(
        channel_mse == 0.0,
        99.0,
        -10.0 * np.log10(np.maximum(channel_mse, np.finfo(np.float64).tiny)),
    )
    return float(channel_psnr.mean())

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * (jt.log(1.0 / jt.sqrt(mse)) / jt.log(10.0))

def linear_to_srgb(linear,key=0.18):
        # key controls the overall exposure
    # img = linear / (1 + linear)  # Simple Reinhard tonemapping
    # img *= key  # Adjust key to increase exposure
    # return jt.clamp(img, 0, 1)

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
    

def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out
