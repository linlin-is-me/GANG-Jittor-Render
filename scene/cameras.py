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

from collections import OrderedDict
import os
from threading import Lock

import jittor as jt
from jittor import nn
import numpy as np
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix,fov2focal


_DEFAULT_HOST_IMAGE_CACHE_BYTES = 512 * 1024 ** 2


def _host_image_cache_capacity() -> int:
    raw = os.environ.get(
        "GANG_CAMERA_HOST_CACHE_BYTES", str(_DEFAULT_HOST_IMAGE_CACHE_BYTES))
    try:
        capacity = int(raw)
    except ValueError as exc:
        raise ValueError(
            "GANG_CAMERA_HOST_CACHE_BYTES must be an integer") from exc
    return max(capacity, 0)


class _HostImageCache:
    """Bounded process-local cache of resized CPU image arrays.

    GPU tensors keep the existing on-demand lifetime.  Only deterministic PIL
    decode/resize results are retained, so revisiting a camera avoids disk I/O
    without pinning Jittor device allocations.
    """

    def __init__(self):
        self._entries = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._lock = Lock()

    @staticmethod
    def _key(path, resolution):
        return (os.path.realpath(path), tuple(int(value) for value in resolution))

    def get(self, path, resolution):
        key = self._key(path, resolution)
        capacity = _host_image_cache_capacity()
        with self._lock:
            cached = self._entries.pop(key, None)
            if cached is not None:
                self._entries[key] = cached
                self._hits += 1
                return cached

        with Image.open(path) as source:
            source.load()
            decoded = np.array(source.resize(key[1]), copy=True)

        with self._lock:
            self._misses += 1
            if capacity <= 0 or decoded.nbytes > capacity:
                return decoded
            replaced = self._entries.pop(key, None)
            if replaced is not None:
                self._bytes -= int(replaced.nbytes)
            self._entries[key] = decoded
            self._bytes += int(decoded.nbytes)
            while self._bytes > capacity and self._entries:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= int(evicted.nbytes)
        return decoded

    def stats(self):
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": int(self._bytes),
                "hits": int(self._hits),
                "misses": int(self._misses),
                "capacity_bytes": _host_image_cache_capacity(),
            }

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self._hits = 0
            self._misses = 0


_HOST_IMAGE_CACHE = _HostImageCache()


def host_image_cache_stats():
    """Return counters suitable for performance evidence."""
    return _HOST_IMAGE_CACHE.stats()


def _reset_host_image_cache_for_tests():
    _HOST_IMAGE_CACHE.clear()


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image,normal,albedo,roughness,
                  metal,irradiance, gt_alpha_mask,
                 image_name, resolution_scale, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 camera_residency="device_all", image_path=None,
                 target_resolution=None
                 ):
        super(Camera, self).__init__()
        

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.resolution_scale = resolution_scale

        self.data_device = data_device
        self.camera_residency = str(camera_residency)
        if self.camera_residency not in {"device_all", "on_demand"}:
            raise ValueError(
                f"unsupported camera residency policy: {self.camera_residency!r}")
        self._image_path = str(image_path) if image_path is not None else None
        self._target_resolution = (
            tuple(int(value) for value in target_resolution)
            if target_resolution is not None else None)
        if self.camera_residency == "on_demand":
            if image is not None:
                raise ValueError("on-demand camera must not receive an eager image tensor")
            if self._image_path is None or self._target_resolution is None:
                raise ValueError("on-demand camera requires an image path and target resolution")
            if normal is not None or gt_alpha_mask is not None:
                raise ValueError(
                    "on-demand camera residency currently supports RGB supervision without "
                    "material metadata or alpha masks")
            self.original_image = None
            self.image_width, self.image_height = self._target_resolution
        else:
            if image is None:
                raise ValueError("device-all camera requires an eager image tensor")
            self.original_image = image.clamp(0.0, 1.0)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

        if normal is not None:
            self.normal = normal
            self.albedo = albedo
            self.roughness = roughness
            self.metal = metal
            self.irradiance = irradiance 
        else:
            self.normal = self.albedo = self.roughness = self.metal = self.irradiance = None
    
 
        if self.camera_residency == "on_demand":
            self.mask = None
        elif gt_alpha_mask is not None:
            self.mask = gt_alpha_mask.clamp(0.0, 1.0)
            self.original_image *= self.mask
            if normal is not None:
                self.normal *= self.mask
                self.albedo *= self.mask
                self.roughness *= self.mask
                self.metal *= self.mask
                self.irradiance *= self.mask

        else:
            self.mask = None
            self.original_image *= jt.ones((1, self.image_height, self.image_width))
            if normal is not None:
                self.normal *= jt.ones((1, self.image_height, self.image_width))
                self.albedo *= jt.ones((1, self.image_height, self.image_width))
                self.roughness *= jt.ones((1, self.image_height, self.image_width))
                self.metal *= jt.ones((1, self.image_height, self.image_width))
                self.irradiance *= jt.ones((1, self.image_height, self.image_width))

        prcppoint = np.array([0.5, 0.5])
        self.prcppoint = jt.array(prcppoint).float32()  # 

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        world_view_numpy = getWorld2View2(R, T, trans, scale).astype(np.float32)
        self.world_view_transform = jt.array(world_view_numpy).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0) @ self.projection_matrix.unsqueeze(0)).squeeze(0)
        # Jittor 1.3.11 implements linalg.inv with numpy_code, which imports
        # optional CuPy under CUDA. This rigid transform is already available as
        # a host matrix, so invert it once with NumPy and upload the three values.
        self.camera_center_numpy = np.linalg.inv(world_view_numpy.T)[3, :3].astype(np.float32)
        self.camera_center = jt.array(self.camera_center_numpy)

    def materialize_original_image(self):
        """Return the RGB supervision tensor, loading one camera when requested."""
        if self.original_image is None:
            resized = _HOST_IMAGE_CACHE.get(
                self._image_path, self._target_resolution)
            image = jt.array(resized).float32() / 255.0
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)
            else:
                image = image.unsqueeze(dim=-1).permute(2, 0, 1)
            image = image[:3, ...]
            self.original_image = image.clamp(0.0, 1.0)
        return self.original_image

    def release_image_data(self):
        """Release only tensors created by the explicit on-demand policy."""
        if self.camera_residency == "on_demand":
            self.original_image = None
        
    def get_calib_matrix_nerf(self):
        focal = fov2focal(self.FoVx, self.image_width)  # original focal length
        intrinsic_matrix = jt.array([[focal, 0, self.image_width / 2], [0, focal, self.image_height / 2], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = jt.linalg.inv(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
