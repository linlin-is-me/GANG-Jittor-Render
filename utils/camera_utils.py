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

import numpy as np
from utils.general_utils import PILtoTorch,NumpytoTorch
from utils.graphics_utils import fov2focal
import jittor as jt
import jittor.nn as F

from typing import List
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
from PIL import Image

WARNED = False


def _source_image_size(image):
    if isinstance(image, Image.Image):
        width, height = image.size
        return int(width), int(height)
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError(f"camera image must have at least two dimensions, got {array.shape}")
    return int(array.shape[1]), int(array.shape[0])


def _resize_image_tensor(image, resolution):
    if isinstance(image, Image.Image):
        return PILtoTorch(image, resolution)
    return NumpytoTorch(image, resolution)

def loadCam(args, id, cam_info, resolution_scale):
    # Import lazily so this utility remains importable on its own.  Importing
    # scene.cameras at module load time re-enters scene.__init__, which itself
    # imports camera_utils.
    from scene.cameras import Camera

    orig_w, orig_h = _source_image_size(cam_info.image)

    if args.resolution == -1:
        if orig_w > 1600:
            global WARNED
            if not WARNED:
                print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                    "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                WARNED = True
            global_down = orig_w / 1600
        else:
            global_down = 1
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    else:
        resolution = (round(orig_w / (resolution_scale * args.resolution)),
                      round(orig_h / (resolution_scale * args.resolution)))
    
    # if args.resolution in [1, 2, 4, 8]:
    #     try:
    #         resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    #     except:
    #         resized_image_rgb = NumpytoTorch(cam_info.image, resolution)
    # else:
    #     resized_image_rgb = jt.array(np.array(cam_info.image))
    #     if len(resized_image_rgb.shape) == 3:
    #         resized_image_rgb =  resized_image_rgb.permute(2, 0, 1)
    #     else:
    #         resized_image_rgb = resized_image_rgb.unsqueeze(dim=-1).permute(2, 0, 1)

    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError(f"camera resolution must be positive, got {resolution}")
    camera_residency = str(getattr(args, "camera_residency", "device_all"))
    if camera_residency not in {"device_all", "on_demand"}:
        raise ValueError(f"unsupported camera residency policy: {camera_residency!r}")
    if camera_residency == "on_demand":
        gt_image = None
    else:
        resized_image_rgb = _resize_image_tensor(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]

    resized_image_mask = None
    if camera_residency == "on_demand" and (
            cam_info.image_mask is not None or cam_info.normal is not None):
        raise ValueError(
            "on-demand camera residency currently supports RGB supervision without "
            "material metadata or alpha masks")
    if cam_info.image_mask is not None:
        image_mask = jt.array(np.array(cam_info.image_mask)).float().unsqueeze(0)
        # The effective size also changes for the -1 auto-cap policy, so resize
        # according to the computed dimensions rather than the CLI token.
        resized_image_mask = F.interpolate(
            image_mask.unsqueeze(0), size=(resolution[1], resolution[0]),
            mode='nearest').squeeze(0)
    # else:

    resize_normal = resize_albedo = resize_roughness = resize_metal = resize_irradiance = None
    if cam_info.normal is not None:

        resize_normal = _resize_image_tensor(cam_info.normal, resolution)
        resize_albedo = NumpytoTorch(cam_info.albedo, resolution)
        resize_roughness = NumpytoTorch(cam_info.roughness, resolution)
        resize_metal = NumpytoTorch(cam_info.metal, resolution)
        resize_irradiance = NumpytoTorch(cam_info.irradiance, resolution)

        # normal_m = jt.array(cam_info.normal).float().permute(2,0,1)    
        # albedo_m = jt.array(cam_info.albedo).float().permute(2,0,1)
        # roughness_m = jt.array(cam_info.roughness).float().permute(2,0,1)
        # metal_m = jt.array(cam_info.metal).float().permute(2,0,1)
        # irradiance_m =  jt.array(cam_info.irradiance).float().permute(2,0,1)
        # if args.resolution in [1, 2, 4, 8]:
        #     resize_normal = torchvision.transforms.Resize(
        #         (resolution[1],resolution[0]), interpolation=InterpolationMode.NEAREST)(normal_m)
        #     resize_albedo = torchvision.transforms.Resize(
        #         (resolution[1],resolution[0]), interpolation=InterpolationMode.NEAREST)(albedo_m)
        #     resize_roughness = torchvision.transforms.Resize(
        #         (resolution[1],resolution[0]), interpolation=InterpolationMode.NEAREST)(roughness_m)
        #     resize_metal = torchvision.transforms.Resize(
        #         (resolution[1],resolution[0]), interpolation=InterpolationMode.NEAREST)(metal_m)
        #     resize_irradiance = torchvision.transforms.Resize(
        #         (resolution[1],resolution[0]), interpolation=InterpolationMode.NEAREST)(irradiance_m)
        # else:
        #     resize_normal = normal_m
        #     resize_albedo = albedo_m
        #     resize_roughness = roughness_m
        #     resize_metal = metal_m
        #     resize_irradiance = irradiance_m

    # resized_normal = None
    # if cam_info.normal is not None:
    #     normal = jt.array(cam_info.normal).float().permute(2, 0, 1)
    #     resized_normal = torchvision.transforms.Resize(
    #         resolution, interpolation=InterpolationMode.NEAREST)(normal)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, normal=resize_normal,albedo=resize_albedo,roughness=resize_roughness,
                  metal=resize_metal,irradiance=resize_irradiance,gt_alpha_mask=resized_image_mask,
                  image_name=cam_info.image_name, resolution_scale=resolution_scale, 
                  uid=id, data_device=args.data_device,
                  camera_residency=camera_residency,
                  image_path=cam_info.image_path,
                  target_resolution=resolution)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry





def trajectory_from_c2ws(c2ws: List[np.ndarray], frames: int) -> List[np.ndarray]:
    """generate trajector from given c2ws

    Args:
        c2ws (List[np.ndarray]): list of c2ws
        frames (int): the number of output frames

    Returns:
        List[np.ndarray]: the interpolated c2ws of trajectory from given c2ws
    """
    # store key frames and rotation for slerp
    rots = []
    key_times = []
    pos_x = []
    pos_y = []
    pos_z = []
    for key_id, c2w in enumerate(c2ws):
        pos_x.append(c2w[0, 3])
        pos_y.append(c2w[1, 3])
        pos_z.append(c2w[2, 3])
        rots.append(c2w[:3, :3])
        key_times.append(key_id)
    key_rots = R.from_matrix(np.stack(rots))
    slerp = Slerp(key_times, key_rots)
    lerp_x = interp1d(key_times, np.array(pos_x), "cubic")
    lerp_y = interp1d(key_times, np.array(pos_y), "cubic")
    lerp_z = interp1d(key_times, np.array(pos_z), "cubic")

    # get the times for interpolation
    times = []
    for i in range(frames):
        curr = i / frames * (len(c2ws) - 1)
        times.append(curr)
    
    # interpolation generation
    rots_inter = slerp(times).as_matrix()
    x_inter = lerp_x(times)
    y_inter = lerp_y(times)
    z_inter = lerp_z(times)

    # pose
    c2ws_inter = []
    for i in range(frames):
        c2w_inter = np.eye(4)
        c2w_inter[:3, :3] = rots_inter[i]
        c2w_inter[:3, 3] = np.array([x_inter[i], y_inter[i], z_inter[i]])
        c2ws_inter.append(c2w_inter)

    return c2ws_inter







def trajectory_from_c2ws_moving(c2ws: List[np.ndarray], frames: int) -> List[np.ndarray]:
    """generate trajector from given c2ws

    Args:
        c2ws (List[np.ndarray]): list of c2ws
        frames (int): the number of output frames

    Returns:
        List[np.ndarray]: the interpolated c2ws of trajectory from given c2ws
    """
    # store key frames and rotation for slerp
    rots = []
    key_times = []
    pos_x = []
    pos_y = []
    pos_z = []
    for key_id, c2w in enumerate(c2ws):
        pos_x.append(c2w[0, 3])
        pos_y.append(c2w[1, 3])
        pos_z.append(c2w[2, 3])
        rots.append(c2w[:3, :3])
        key_times.append(key_id)
    key_rots = R.from_matrix(np.stack(rots))
    slerp = Slerp(key_times, key_rots)
    lerp_x = interp1d(key_times, np.array(pos_x), "cubic")
    lerp_y = interp1d(key_times, np.array(pos_y), "cubic")
    lerp_z = interp1d(key_times, np.array(pos_z), "cubic")

    # get the times for interpolation
    times = []
    for i in range(frames):
        curr = i / frames * (len(c2ws) - 1)
        times.append(curr)
    
    # interpolation generation
    rots_inter = slerp(times).as_matrix()
    x_inter = lerp_x(times)
    y_inter = lerp_y(times)
    z_inter = lerp_z(times)

    # pose
    c2ws_inter = []
    positions_list = []
    for i in range(frames):
        c2w_inter = np.eye(4)
        c2w_inter[:3, :3] = rots_inter[i]
        c2w_inter[:3, 3] = np.array([x_inter[i], y_inter[i], z_inter[i]])
        positions_list.append(np.array([x_inter[i], y_inter[i], z_inter[i]]))
        c2ws_inter.append(c2w_inter)

    return c2ws_inter,positions_list

