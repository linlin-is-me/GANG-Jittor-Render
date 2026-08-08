# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import jittor as jt
from jittor import nn

# Jittor texture module — replaces nvdiffrast.torch
from scene.NVDIFFREC.texture import texture as _jt_tex_fn

class _dr_compat:
    """Compatibility wrapper: dr.texture(...) → _jt_tex_fn(...)"""
    @staticmethod
    def texture(*args, **kwargs):
        return _jt_tex_fn(*args, **kwargs)

dr = _dr_compat()

from . import util
from scene.NVDIFFREC.renderutils import specular_cubemap, diffuse_cubemap
from utils.general_utils import get_expon_lr_func
import cv2

import jittor.nn as F
import imageio
from utils.light_utils import DistributionGGX,GeometrySmith,fresnelSchlick


TINY_NUMBER = 1e-6


######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(jt.Function):
    def execute(self, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    def grad(self, dout):
        res = dout.shape[1] * 2
        out = jt.zeros(6, res, res, dout.shape[-1], dtype=jt.float32)
        for s in range(6):
            gy, gx = jt.meshgrid(jt.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res),
                                    jt.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res),
                                    )
                                    # indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

def compute_energy(lgtSGs):
    lgtLambda = jt.abs(lgtSGs[:, 3:4])       # [M, 1]
    lgtMu = jt.abs(lgtSGs[:, 4:])               # [M, 3]
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - jt.exp(-2.0 * lgtLambda))
    return energy

def fibonacci_sphere(samples=1):
    '''
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points

class Hybridlight(nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base_res = 256,
                 scale = 0.5,
                 bias = 0.25,
                 num_sg = 16,
                 numBrdfSGs = 1,
                 inital_position = None,
                 upper_hemi = False,
                 is_white_light = False,
                 cache_dir = None):
        super(Hybridlight, self).__init__()
        self.mtx = None
        self._cache_dir = cache_dir or '.'
        # Preserve the historical training behaviour by default.  Relighting
        # scripts may opt into a distant SG, a safer roughness floor and an
        # unclipped linear-HDR diffuse term without changing checkpoint replay.
        self.sg_distance_attenuation = True
        self.sg_min_roughness = 1e-5
        self.sg_clamp_diffuse = True

        # Experimental point-light path for relighting inference. It is disabled
        # by default and is not part of checkpoint training or baseline replay.
        self.point_light_enabled = False
        self.point_light_position = None   # [3] world-space lamp position
        self.point_light_color = None      # [3] RGB warm bulb colour
        self.point_light_intensity = 1.0   # scalar, lamp "wattage"
        self.point_light_radius = 0.5      # scalar, finite glowing radius (softens near-field)

        # Phase 35: Use numpy RNG for deterministic cubemap (cacheable across runs).
        # jt.rand() is non-deterministic → different cache key each run.
        np.random.seed(42)
        base_np = (np.random.rand(6, base_res, base_res, 3).astype(np.float32) * scale + bias)
        base = jt.array(base_np)
        self.base = base
        # self.register_parameter('env_base', self.base)

        self.numLgtSGs = num_sg

        self.white_light = is_white_light
        # Initialize SG params in numpy (Jittor .data assignment doesn't work)
        if is_white_light:
            print("SG is white light!!")
            lgt_np = np.random.randn(num_sg, 8).astype(np.float32)  # pos(3)+lobe(3)+lambda(1)+mu(1)
            spec_np = np.random.randn(numBrdfSGs, 1).astype(np.float32)
            energy_offset = 6  # lambda at col 6, mu at col 7
        else:
            lgt_np = np.random.randn(num_sg, 10).astype(np.float32)  # pos(3)+lobe(3)+lambda(1)+mu(3)
            lgt_np[:, -2:] = lgt_np[:, -3:-2]  # copy to last 2 cols
            spec_np = np.random.randn(numBrdfSGs, 3).astype(np.float32)
            energy_offset = 6  # lambda at col 6, mu at cols 7-9
        spec_np = np.abs(spec_np)

        self.get_env = None

        if inital_position is not None:
            lgt_np[:, :3] = np.array(inital_position)
        else:
            print("random SG light position inital!!!")

        # make sure lambda is not too close to zero
        lgt_np[:, energy_offset:energy_offset+1] = 20. + np.abs(lgt_np[:, energy_offset:energy_offset+1] * 100.)
        # make sure total energy is around 1
        energy_np = compute_energy(jt.array(lgt_np[:, energy_offset-3:])).numpy()
        lgt_np[:, energy_offset+1:] = np.abs(lgt_np[:, energy_offset+1:]) / np.sum(energy_np, axis=0, keepdims=True) * 2. * np.pi
        energy_check = compute_energy(jt.array(lgt_np[:, energy_offset-3:])).numpy()
        print('init envmap energy: ', np.sum(energy_check))

        lobes = fibonacci_sphere(self.numLgtSGs).astype(np.float32)
        lgt_np[:, 3:6] = lobes

        self.upper_hemi = upper_hemi
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: jt.concat(
                (lgtSGs[..., :1], jt.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)
            lgt_np[:, 1:2] = np.abs(lgt_np[:, 1:2])

        # Convert to JT at the end (avoids .data assignment issues)
        self.lgtSGs = jt.array(lgt_np)
        self.specular_reflectance = jt.array(spec_np)

        # optimize
        roughness = [np.random.uniform(1.5, 2.0) for i in range(numBrdfSGs)]           # big roughness
        roughness = np.array(roughness).astype(dtype=np.float32).reshape((numBrdfSGs, 1))  # [K, 1]
        print('init SG roughness: ', 1.0 / (1.0 + np.exp(-roughness)))
        self.roughness = jt.array(roughness)


    def training_setup(self,training_args):

        l = [
            {'params': self.base, 'lr': training_args.env_map_init, "name": "Envmap"},
            {'params': self.lgtSGs, 'lr': training_args.sg_init, "name": "SGLight"},
            {'params': self.roughness,'lr': training_args.sg_init, "name": "sg_roughness"},
            {'params': self.specular_reflectance, 'lr': training_args.sg_init, "name": "specular_reflectance"},
        ]

        self.optimizer = jt.optim.Adam(l, lr=0.0, eps=1e-8)

        self.env_light_scheduler = get_expon_lr_func(lr_init=training_args.env_map_init,
                                                         lr_final=training_args.env_map_final,
                                                         lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                         max_steps=training_args.iterations)

        self.sg_light_scheduler = get_expon_lr_func(lr_init=training_args.sg_init,
                                                         lr_final=training_args.sg_final,
                                                         lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                         max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "Envmap":
                lr = self.env_light_scheduler(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "SGLight" or param_group["name"] == "specular_reflectance" or param_group["name"] == "sg_roughness":
                lr = self.sg_light_scheduler(iteration)
                param_group['lr'] = lr


    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return Hybridlight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        mask = (roughness < self.MAX_ROUGHNESS).float()
        branch_a = (jt.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
        branch_b = (jt.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2
        return mask * branch_a + (1.0 - mask) * branch_b


    def load_from_numpy(self, light_dict):
        """Phase 64: Load light state from numpy dict (from checkpoint npz)."""
        if light_dict is None:
            return
        self.base = jt.array(light_dict['base'].astype(np.float32))
        self.lgtSGs = jt.array(light_dict['lgtSGs'].astype(np.float32))
        self.specular_reflectance = jt.array(light_dict['specular_reflectance'].astype(np.float32))
        self.roughness = jt.array(light_dict['roughness'].astype(np.float32))
        self.numLgtSGs = self.lgtSGs.shape[0]
        print("  Light state restored from checkpoint")

    def load_light(self, filepath,is_training=False):
        assert(filepath.endswith('.npy'))

        print("load Light paramer!!")


        light_dict = np.load(filepath, allow_pickle=True)
        lgtSG = light_dict.item()["lgtSGs"]
        base = light_dict.item()["base"]
        specular_reflectance = light_dict.item()["specular_reflectance"]
        sg_roughness = light_dict.item()["sg_roughness"]

        self.lgtSGs = jt.array(lgtSG)
        self.base = jt.array(base)
        self.specular_reflectance = jt.array(specular_reflectance)
        self.roughness = jt.array(sg_roughness)
        self.numLgtSGs = self.lgtSGs.shape[0]

    def save_light(self,path):
        result = {}
        result["lgtSGs"] = self.lgtSGs.detach().numpy()
        result["base"] = self.base.detach().numpy()
        result["specular_reflectance"] = self.specular_reflectance.detach().numpy()
        result["sg_roughness"] = self.roughness.detach().numpy()
        np.save(path,result)

    def build_mips(self, cutoff=0.99):
        # Phase 35: Check disk cache first to avoid slow numpy specular_cubemap.
        # specular_cubemap at res=256 takes hours in pure numpy.
        # The cubemap depends only on self.base (SG initialization), which is
        # deterministic given the same random seed. Cache to disk for reuse.
        import os, hashlib
        cache_dir = getattr(self, '_cache_dir', '.')
        os.makedirs(os.path.join(cache_dir, 'pbr_cache'), exist_ok=True)
        base_hash = hashlib.md5(self.base.numpy().tobytes()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, 'pbr_cache', f'mips_{self.base.shape[1]}_{base_hash}.npz')

        if os.path.exists(cache_path):
            data = dict(np.load(cache_path, allow_pickle=True))
            n_mips = int(data['n_mips'])
            self.specular = [jt.array(data[f'specular_{i}']) for i in range(n_mips)]
            self.diffuse = jt.array(data['diffuse'])
            self._mips_built = True   # M5 (P1): set on the disk-cache branch too
            return

        # Compute mip chain (Jittor ops — fast with no_grad)
        with jt.no_grad():
            self.specular = [self.base]
            while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
                self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = diffuse_cubemap(self.specular[-1])

        # Phase 38: specular_cubemap now CUDA-accelerated via jt.code — no performance issue.
        n_levels = len(self.specular)
        for idx in range(n_levels - 1):
            if n_levels > 2:
                roughness = (idx / (n_levels - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            else:
                roughness = self.MIN_ROUGHNESS
            self.specular[idx] = specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = specular_cubemap(self.specular[-1], 1.0, cutoff)

        # Save to cache
        try:
            cache_data = {'n_mips': len(self.specular),
                          'diffuse': self.diffuse.numpy()}
            for i, s in enumerate(self.specular):
                cache_data[f'specular_{i}'] = s.numpy()
            np.savez_compressed(cache_path, **cache_data)
        except Exception:
            pass  # Cache write failure is non-fatal

        # M5 (P1): mark built on the compute path too — avoids re-running build_mips
        # from render() (which checks _mips_built) on every render call.
        self._mips_built = True

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return jt.mean(jt.abs(self.base - white))

    def compute_env_envmap(self,filename=None,res=[512, 1024],return_img = False):
        # cubemap_to_latlong
        gy, gx = jt.meshgrid(
            jt.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0]),
            jt.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1]),
            indexing="ij",
        )

        sintheta, costheta = jt.sin(gy * np.pi), jt.cos(gy * np.pi)
        sinphi, cosphi = jt.sin(gx * np.pi), jt.cos(gx * np.pi)

        reflvec = jt.stack(
            (sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1
        )  # [H, W, 3]
        color = dr.texture(
            self.base[None, ...],
            reflvec[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 3]
        if return_img:
            return color
        else:
            cv2.imwrite(filename, color.clamp(0.0).numpy()[..., ::-1])

    def compute_SG_envmap(self,SGs= None, filename=None, res=[512, 1024],return_img = False, upper_hemi=False):
        H,W = res
        # exactly same convetion as Mitsuba, check envmap_convention.png
        if upper_hemi:
            phi, theta = jt.meshgrid(
                [jt.linspace(0., np.pi / 2., H), jt.linspace(-0.5 * np.pi, 1.5 * np.pi, W)])
        else:
            phi, theta = jt.meshgrid([jt.linspace(0., np.pi, H), jt.linspace(-0.5 * np.pi, 1.5 * np.pi, W)])

        viewdirs = jt.stack([jt.cos(theta) * jt.sin(phi), jt.cos(phi), jt.sin(theta) * jt.sin(phi)],
                               dim=-1)  # [H, W, 3]

        if SGs is None:
            lgtSGs = self.lgtSGs.clone().detach()
        else:
            lgtSGs = SGs

        viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
        # [M, 7] ---> [..., M, 7]
        dots_sh = list(viewdirs.shape[:-2])
        M = lgtSGs.shape[0]
        lgtSGs = lgtSGs.view([1, ] * len(dots_sh) + [M, 10]).expand(dots_sh + [M, 10])
        # sanity
        # [..., M, 3]
        lgtSGLobes = lgtSGs[..., 3:6] / (jt.norm(lgtSGs[..., 3:6], dim=-1, keepdim=True))
        lgtSGLambdas = jt.abs(lgtSGs[..., 6:7])
        lgtSGMus = jt.abs(lgtSGs[..., -3:])  # positive values
        # [..., M, 3]
        rgb = lgtSGMus * jt.exp(lgtSGLambdas * (jt.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
        rgb = jt.sum(rgb, dim=-2)  # [..., 3]
        envmap = rgb.reshape((H, W, 3))
        if return_img:
            return envmap
        else:
            cv2.imwrite(filename, envmap.clamp(0.0).numpy())

    def lightRender(self, points, normal, albedo, roughness, metallic, viewdirs, is_env = True, shadow_cube=None, shadow_alpha=None, transmit_cube=None, transmit_bounds=None, transmit_final=None, emitter_transmit=None, receiver_transmittance=None):

        N,_ = normal.shape

        # Point-light path (relight inference only). Computed first so its [G,3]
        # intermediates are separate from the [N,M,*] SG expansion. Falls back to
        # zero if disabled, so the SG/env path below is unchanged.
        if self.point_light_enabled:
            specular_rgb_pl, diffuse_rgb_pl = self.point_light_render(
                points, normal, albedo, roughness, metallic, viewdirs,
                shadow_cube=shadow_cube, shadow_alpha=shadow_alpha,
                transmit_cube=transmit_cube, transmit_bounds=transmit_bounds,
                transmit_final=transmit_final, emitter_transmit=emitter_transmit,
                receiver_transmittance=receiver_transmittance)
        else:
            specular_rgb_pl = jt.zeros_like(albedo)
            diffuse_rgb_pl = jt.zeros_like(albedo)

        #specular

        if self.numLgtSGs > 0:
            specular_rgb_sg, diffuse_rgb_sg = self.sg_render(normal,viewdirs,points,albedo,roughness,metallic)
        else:
            specular_rgb_sg = jt.zeros_like(albedo)
            diffuse_rgb_sg = jt.zeros_like(albedo)
        # Phase 91: force sync+gc to free sg_render intermediates (~8GB [N,16,3] tensors)
        # before envmap cubemap rendering allocates more.
        # 2026-08-02: jt.sync_all(True) waits for device before SFRL reclaim.
        if self.numLgtSGs > 0:
            jt.sync_all(True); jt.gc()

        if is_env:
            normals = normal.reshape(1, N, 3)
            view_dirs = viewdirs.reshape(1, N, 3)
            albedo = albedo.reshape(1, N, 3)
            roughness = roughness.reshape(1, N, 1)
            
            metallic = metallic.reshape(1, N, 1)
            
            diff_col  = albedo * (1.0 - metallic)

            ref_dirs = (2.0 * (normals * view_dirs).sum(-1, keepdims=True).clamp(0.0) * normals - view_dirs)

            diffuse_light = dr.texture(self.diffuse[None, ...], normals[None, ...].contiguous(), filter_mode='linear',
                                    boundary_mode='cube')
            diffuse_rgb = diffuse_light * diff_col
            # 2026-08-02: diff_col/diffuse_light are only used here; release before
            # specular allocates more. diffuse_rgb is still needed at render_rgb.
            del diff_col, diffuse_light
            jt.sync_all(True); jt.gc()  # free diffuse cubemap intermediates

            # specular
            NoV = jt.clamp(util.dot(view_dirs, normals), 1e-4, 1.0)
            fg_uv = jt.concat((NoV, roughness), dim=-1)  # [1, N, 2]
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = jt.array(
                    np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256,2),
                    dtype=jt.float32)
            fg_lookup = dr.texture(
                self._FG_LUT,  # [1, 256, 256, 2]
                fg_uv[None,...].contiguous(),  # [1, N, 2]
                filter_mode="linear",
                boundary_mode="clamp",
            )  # [1, N, 2]
            
            miplevel = self.get_mip(roughness)

            spec = dr.texture(self.specular[0][None, ...], ref_dirs[None,...].contiguous(),
                                mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel.permute(0,2,1),
                                filter_mode='linear-mipmap-linear', boundary_mode='cube')
            # 2026-08-02: ref_dirs/fg_uv/NoV/miplevel no longer needed after spec
            # sampled. fg_lookup (L422) and spec (L425) are still used — keep.
            del ref_dirs, fg_uv, NoV, miplevel
            jt.sync_all(True); jt.gc()  # free specular mipmap cubemap intermediates

            F0 = (1.0 - metallic) * 0.04 + albedo * metallic
           

            reflectance = F0 * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]  # [1,N, 3]


            specular_rgb = spec * reflectance  # [1, N, 3]

            extras = {"specular_rgb": specular_rgb[0,0],"diffuse_rgb": diffuse_rgb[0,0],
                      "specular_rgb_sg": specular_rgb_sg, "diffuse_rgb_sg": diffuse_rgb_sg,
                      "specular_rgb_pl": specular_rgb_pl, "diffuse_rgb_pl": diffuse_rgb_pl}
            # P2 (REASSESSMENT §P2): expose the front-face mask as a pixel channel.
            if getattr(self, 'diag_gaussians', False) and hasattr(self, '_last_front_mask'):
                extras['front_mask'] = self._last_front_mask
            # N4-R0 (§25.4 item 7): expose the effective shadow visibility as the
            # pixel-space Veff channel (penumbra line profile -> `_vis.npy`).
            if getattr(self, 'point_light_enabled', False) and hasattr(self, '_last_veff'):
                extras['veff'] = self._last_veff


            render_rgb = (diffuse_rgb[0,0] + diffuse_rgb_sg + specular_rgb[0,0] + specular_rgb_sg
                          + diffuse_rgb_pl + specular_rgb_pl)  # [N, 3]
            # 2026-08-02: release remaining envmap [1,N,*] intermediates after render_rgb.
            # diff_col/diffuse_light already del'd after diffuse (L396);
            # ref_dirs/fg_uv/NoV/miplevel already del'd after spec (L417).
            # extras already extracted [0,0] slices; del originals is safe.
            del normals, view_dirs, albedo, roughness, metallic, diffuse_rgb, \
                fg_lookup, spec, F0, reflectance, specular_rgb
            jt.sync_all(True); jt.gc()
        
        else:
            extras = {"specular_rgb": specular_rgb_sg,"diffuse_rgb": diffuse_rgb_sg,
                      "specular_rgb_sg": specular_rgb_sg, "diffuse_rgb_sg": diffuse_rgb_sg,
                      "specular_rgb_pl": specular_rgb_pl, "diffuse_rgb_pl": diffuse_rgb_pl}
            # P2 (REASSESSMENT §P2): expose the front-face mask as a pixel channel.
            if getattr(self, 'diag_gaussians', False) and hasattr(self, '_last_front_mask'):
                extras['front_mask'] = self._last_front_mask
            # N4-R0 (§25.4 item 7): expose the effective shadow visibility as the
            # pixel-space Veff channel (penumbra line profile -> `_vis.npy`).
            if getattr(self, 'point_light_enabled', False) and hasattr(self, '_last_veff'):
                extras['veff'] = self._last_veff
            render_rgb = diffuse_rgb_sg + specular_rgb_sg + diffuse_rgb_pl + specular_rgb_pl  # [N, 3]

        return render_rgb, extras
    
    def sg_render(self,normal,viewdirs,points,albedo,roughness,metallic):
        N, _ = normal.shape
        M = self.lgtSGs.shape[0]

        roughness = jt.clamp(roughness, float(self.sg_min_roughness), 1.0)

        normal_sg = normal.unsqueeze(-2).expand([N, M, 3])
        viewdirs_sg = viewdirs.unsqueeze(-2).expand([N, M, 3])
        point_sg = None
        if self.sg_distance_attenuation:
            point_sg = points.unsqueeze(-2).expand([N, M, 3])
        roughness_sg = roughness.unsqueeze(-2).expand([N, M, 1])
        albedo_sg = albedo.unsqueeze(-2).expand([N, M, 3])
 
        metallic_sg = metallic.unsqueeze(-2).expand([N, M, 3])
        lgtSGs = self.lgtSGs.unsqueeze(0).expand([N, M, 10])  # # [N, M, 10]

        #### note: sanity
        lgtSGPosition = lgtSGs[..., :3]  # [N, M, 3]
        lgtSGLobes = lgtSGs[..., 3:6] / (
                jt.norm(lgtSGs[..., 3:6], dim=-1, keepdim=True) + TINY_NUMBER)  # [N, M, 3]
        lgtSGLambdas = jt.abs(lgtSGs[..., 6:7])
        lgtSGMus = jt.abs(lgtSGs[..., -3:])  # positive values

        if self.sg_distance_attenuation:
            decay_weight = compute_weight(point_sg, lgtSGPosition)  # [N, M]
        else:
            decay_weight = jt.ones([N, M], dtype=jt.float32)

        # NDF
        brdfSGLobes = normal_sg  # use normal as the brdf SG lobes
        inv_roughness_pow4 = 1. / (roughness_sg * roughness_sg * roughness_sg * roughness_sg)  # [N, M, 1]

        brdfSGLambdas = (2. * inv_roughness_pow4)  # [N, M, 1]
        brdfSGMus = (inv_roughness_pow4 / np.pi)  # [N, M, 3]
       
        # perform spherical warping
        v_dot_lobe = jt.sum(brdfSGLobes * viewdirs_sg, dim=-1, keepdim=True)  # [N, M, 1]
        ### note: for numeric stability
        v_dot_lobe = v_dot_lobe.clamp(0.0)  # [N, M, 1]
        warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs_sg  # [N, M, 3]
        warpBrdfSGLobes = warpBrdfSGLobes / (
                    jt.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)  # [N, M, 3]
        # warpBrdfSGLambdas = brdfSGLambdas / (4 * jt.abs(jt.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)) + TINY_NUMBER)
       
        warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)  # # [N, M, 1] can be huge
        warpBrdfSGMus = brdfSGMus  # [N, M, 3]

        # add fresnel and geometric terms; apply the smoothness assumption in SG paper
        new_half = warpBrdfSGLobes + viewdirs_sg  # [N, M, 3]
        new_half = new_half / (jt.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)  # [N, M, 3]
        v_dot_h = jt.sum(viewdirs_sg * new_half, dim=-1, keepdim=True)  # [N, M, 1]
        ### note: for numeric stability
        v_dot_h = v_dot_h.clamp(0.0)  # [N, M, 1]
        specular_reflectance = albedo_sg

        F = specular_reflectance + (1. - specular_reflectance) * jt.pow(2.0, -(
                5.55473 * v_dot_h + 6.8316) * v_dot_h)  # [N, M, 1]
        

        dot1 = jt.sum(warpBrdfSGLobes * normal_sg, dim=-1, keepdim=True)  # [N, M, 1]
        ### note: for numeric stability
        dot1 = dot1.clamp(0.0)  # [N, M, 1]
        dot2 = jt.sum(viewdirs_sg * normal_sg, dim=-1, keepdim=True)  # [N, M, 1]
        ### note: for numeric stability
        dot2 = dot2.clamp(0.0)
        k = (roughness_sg + 1.) * (roughness_sg + 1.) / 8.  # [N, M, 1]
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # [N, M, 1]
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)  # [N, M, 1]
        G = G1 * G2  # [N, M, 1]

        Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [N, M, 1]
        warpBrdfSGMus = warpBrdfSGMus * Moi  # [N, M, 3]

        # multiply with light sg
        final_lobes, final_lambdas, final_mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus,
                                                                warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)
        mu_cos = 32.7080
        lambda_cos = 0.0315
        alpha_cos = 31.7003
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal_sg, lambda_cos, mu_cos,
                                                            final_lobes, final_lambdas, final_mus)
        # print("lobe_prime",lobe_prime.max(), lambda_prime.max(), mu_prime.max(),)
        dot1 = jt.sum(lobe_prime * normal_sg, dim=-1, keepdim=True)  # [N, M, 1]
        dot2 = jt.sum(final_lobes * normal_sg, dim=-1, keepdim=True)  # [N, M, 1]

        # Phase 91: float64 subtraction to avoid 115× cancellation amplification.
        # term1 ≈ term2 → tiny float32 errors in each term get magnified.
        H1 = hemisphere_int(lambda_prime, dot1)
        H2 = hemisphere_int(final_lambdas, dot2)
        specular_rgb_sg = (mu_prime.float64() * H1.float64() -
                           final_mus.float64() * (alpha_cos * H2.float64())).float32()

        specular_rgb_sg = (specular_rgb_sg * decay_weight.unsqueeze(-1)).sum(dim=-2)  # [N, 3]
        specular_rgb_sg = specular_rgb_sg.clamp(0.0)  # [N, 3]

        # 2026-08-02: release ~24GB of specular-only [N,M,*] intermediates before
        # allocating diffuse ones. Only specular-path vars are deleted; the diffuse
        # path re-binds final_lobes/final_mus/final_lambdas/lobe_prime/mu_prime/
        # dot1/dot2 below, and needs normal_sg/albedo_sg/metallic_sg/lgtSGLobes/
        # lgtSGLambdas/lgtSGMus/decay_weight/specular_rgb_sg.
        # jt.sync_all(True) waits for the device before SFRL reclaims blocks.
        del viewdirs_sg, point_sg, roughness_sg, lgtSGs, lgtSGPosition, inv_roughness_pow4, \
            brdfSGLambdas, brdfSGMus, v_dot_lobe, warpBrdfSGLobes, warpBrdfSGLambdas, \
            warpBrdfSGMus, new_half, v_dot_h, F, k, G1, G2, G, Moi, H1, H2, \
            final_lobes, final_lambdas, final_mus, lobe_prime, lambda_prime, mu_prime, \
            dot1, dot2, specular_reflectance
        jt.sync_all(True); jt.gc()

        # diffuse color
        diffuse = (1-metallic_sg)*albedo_sg / np.pi  # [N, M, 3]
       
        # multiply with light sg
        # .narrow(dim=-2, start=0, length=1) → Jittor slicing
        final_lobes = lgtSGLobes[..., :1, :]  # [N, M, 3]
        final_mus = lgtSGMus[..., :1, :] * diffuse
        final_lambdas = lgtSGLambdas[..., :1, :]

        # now multiply with clamped cosine, and perform hemisphere integral
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal_sg, lambda_cos, mu_cos,
                                                            final_lobes, final_lambdas, final_mus)

        dot1 = jt.sum(lobe_prime * normal_sg, dim=-1, keepdim=True)
        dot2 = jt.sum(final_lobes * normal_sg, dim=-1, keepdim=True)
        diffuse_rgb_sg = (mu_prime.float64() * hemisphere_int(lambda_prime, dot1).float64() -
                           final_mus.float64() * (alpha_cos * hemisphere_int(final_lambdas, dot2).float64())).float32()
        
        diffuse_rgb_sg = (diffuse_rgb_sg * decay_weight.unsqueeze(-1)).sum(dim=-2)  # [N, 3]
        if self.sg_clamp_diffuse:
            diffuse_rgb_sg = diffuse_rgb_sg.clamp(0.0, 1.0)
        else:
            diffuse_rgb_sg = diffuse_rgb_sg.clamp(min_v=0.0)
        return specular_rgb_sg, diffuse_rgb_sg

    def _query_transmittance(self, points, light_pos, transmit_cube, transmit_final,
                             transmit_bounds, summary=None, diag=None):
        """N4 (§21) / N4-R0 (§25.4): unified P4.1 receiver transmittance query —
        dist-bias + optical-depth interpolation + T_lo/T_interp/T_hi + optional
        tau-domain filter + opacity-scale. Shared by the point-light and EVERY
        emitter sample of the area light (per-sample visibility).

        `light_pos` is a [3] jt.Var or np.ndarray. `summary` (optional dict) is
        filled with per-sample stats (T means + bounds hash) for the area-light
        two-level diagnostics.

        N3-E (§26.4): `diag` (optional dict) is filled with the ADDITIVE per-receiver
        diagnostics — T_prev (the boundary map BEFORE the hit bin, = T_lo for the
        first interval), hit bin index, bracketing boundary distances b_lo/b_hi and
        the self-cross marker (b_hi >= receiver dist). These never alter the interp
        T_lo/T_hi/T_interp values above, so the default path stays pixel-identical.

        Returns (vis, T_lo, T_interp, T_hi, bounds, bias).
        """
        G = points.shape[0]
        if isinstance(light_pos, jt.Var):
            lpv = light_pos
        else:
            lpv = jt.array(np.asarray(light_pos, dtype=np.float32))
        L_vec = lpv.unsqueeze(0) - points
        dist = jt.norm(L_vec, p=2, dim=-1, keepdim=True)
        dirs = -jt.normalize(L_vec, p=2, dim=-1)                       # lamp -> point
        B = int(transmit_cube.shape[0])
        bounds = np.asarray(transmit_bounds, dtype=np.float32).reshape(-1)
        if transmit_final is None:
            transmit_final = transmit_cube[-1]
        bias = float(getattr(self, 'shadow_bias', 0.02))
        znear = float(getattr(self, 'shadow_znear', 0.01))
        _eps = 1e-6
        q = jt.maximum(dist - bias, znear)
        log_q = jt.log(jt.maximum(q, _eps))
        log_b = np.log(np.maximum(bounds, _eps))
        _tfilter = str(getattr(self, 'shadow_filter', 'nn'))
        tau_domain = (_tfilter == 'tau')
        _teps = 1e-4
        trans_all = []
        for b in range(B):
            if tau_domain:
                tau_map = -jt.log(jt.clamp(transmit_cube[b], _teps, 1.0))
                tau_s = dr.texture(tau_map[None, ..., None], dirs.unsqueeze(0).contiguous(),
                                   filter_mode='linear', boundary_mode='cube')[0]
                t = jt.exp(-tau_s)
            else:
                t = dr.texture(transmit_cube[b][None, ..., None], dirs.unsqueeze(0).contiguous(),
                               filter_mode='nearest', boundary_mode='cube')[0]
            trans_all.append(t)
        if tau_domain:
            tau_f = -jt.log(jt.clamp(transmit_final, _teps, 1.0))
            final_t = jt.exp(-dr.texture(tau_f[None, ..., None], dirs.unsqueeze(0).contiguous(),
                                         filter_mode='linear', boundary_mode='cube')[0])
        else:
            final_t = dr.texture(transmit_final[None, ..., None], dirs.unsqueeze(0).contiguous(),
                                 filter_mode='nearest', boundary_mode='cube')[0]
        est = str(getattr(self, 'transmit_estimator', 'interp'))
        # N3-E (§26.4): left_tau / self_guarded need the boundary map BEFORE the
        # hit bin + the self-cross marker (b_hi >= receiver dist), computed even
        # when no diag dump is requested; interp/lo need neither (default path
        # stays minimal and pixel-identical).
        _need_prev = (diag is not None) or (est in ('left_tau', 'self_guarded'))
        T_lo = jt.ones([G, 1]); T_hi = jt.ones([G, 1]); T_interp = jt.ones([G, 1])
        if _need_prev:
            T_prev = jt.ones([G, 1])
            bin_i = jt.zeros([G, 1])
            b_lo_v = jt.zeros([G, 1])
            b_hi_v = jt.zeros([G, 1])
            b_prev_v = jt.zeros([G, 1])
            scross = jt.zeros([G, 1])
        for b in range(B - 1):
            w = (log_q - float(log_b[b])) / float(log_b[b + 1] - log_b[b])
            in_int = jt.logical_and(q >= float(bounds[b]), q < float(bounds[b + 1]))
            tau_lo = -jt.log(jt.maximum(trans_all[b], _eps))
            tau_hi = -jt.log(jt.maximum(trans_all[b + 1], _eps))
            tau = tau_lo + w * (tau_hi - tau_lo)
            T_lo = jt.where(in_int, trans_all[b], T_lo)
            T_hi = jt.where(in_int, trans_all[b + 1], T_hi)
            T_interp = jt.where(in_int, jt.exp(-tau), T_interp)
            if _need_prev:
                # N3-E (§26.4): per-receiver diagnostics — T_prev = boundary map
                # BEFORE the hit bin (first interval falls back to T_lo = lo),
                # bin index, bracketing distances + their T_prev boundary distance
                # and the self-cross marker (b_hi >= receiver dist). Additive;
                # never touches the interp T_lo/T_hi/T_interp values above.
                _prev = trans_all[b - 1] if b > 0 else trans_all[0]
                T_prev = jt.where(in_int, _prev, T_prev)
                bin_i = jt.where(in_int, jt.array(float(b), dtype='float32'), bin_i)
                b_lo_v = jt.where(in_int, jt.array(float(bounds[b]), dtype='float32'), b_lo_v)
                b_hi_v = jt.where(in_int, jt.array(float(bounds[b + 1]), dtype='float32'), b_hi_v)
                _bprev = float(bounds[b - 1]) if b > 0 else float(bounds[b])
                b_prev_v = jt.where(in_int, jt.array(_bprev, dtype='float32'), b_prev_v)
                scross = jt.where(in_int, (dist <= float(bounds[b + 1])).float32(), scross)
        after = (q >= float(bounds[-1])).float32()
        T_lo = jt.where(after > 0, final_t, T_lo)
        T_hi = jt.where(after > 0, final_t, T_hi)
        T_interp = jt.where(after > 0, final_t, T_interp)
        # N3-E (§26.4): receiver transmittance estimator A/B — only the QUERY-side
        # combination of the same boundary maps changes; cubemap generation and CUDA
        # are untouched. The default `interp` is the regression baseline.
        if est == 'lo':
            vis = T_lo.clamp(0.0, 1.0)
        elif est in ('left_tau', 'self_guarded'):
            # §27.2.1 DEPRECATED experimental modes: T_prev->T_lo LEFT optical-depth
            # slope extrapolated to q then clamped to [T_hi, T_lo]. This is only a
            # *bounded* left extrapolation, NOT an explicit self-contribution
            # exclusion — the clamp below still reads the right boundary T_hi, so the
            # §26.4.1 strict left contract is not met. Kept for reading back old
            # N3-E artifacts only; NOT part of the N192/N256 route. If revived, unify
            # on a T_hi-free contract and re-run synthetic ground-truth + full matrix.
            if not getattr(self, '_warned_deprecated', False):
                print(f"[WARN] light._query_transmittance: estimator {est!r} is "
                      "DEPRECATED (§27.2.1, bounded-left only, not explicit self "
                      "exclusion); N192/N256 route uses interp/lo.", flush=True)
                self._warned_deprecated = True
            # left_tau: extrapolate the T_prev->T_lo LEFT optical-depth slope to q,
            # clamped to [T_hi, T_lo] (guide §26.4.1 candidate contract) and never
            # reading the right boundary T_hi for the extrapolation. The first
            # interval has T_prev = T_lo (slope 0) -> falls back to `lo`.
            _e = 1e-6
            _t_prev = -jt.log(jt.maximum(T_prev, _e))
            _t_lo = -jt.log(jt.maximum(T_lo, _e))
            _lg_prev = jt.log(jt.maximum(b_prev_v, _e))
            _lg_lo = jt.log(jt.maximum(b_lo_v, _e))
            _slope = (_t_lo - _t_prev) / jt.maximum(_lg_lo - _lg_prev, 1e-6)
            _tau_left = _t_lo + _slope * (jt.log(jt.maximum(q, _e)) - _lg_lo)
            T_left = jt.where(b_lo_v > 0, jt.exp(-_tau_left), T_lo)
            T_left = jt.minimum(jt.maximum(T_left, T_hi), T_lo)     # [T_hi, T_lo]
            # self_guarded: interp while the interval stays BEFORE the receiver
            # depth (b_hi < dist); left_tau when b_hi >= dist (receiver-depth
            # self-contamination); first interval reduces to lo via left_tau.
            if est == 'self_guarded':
                T_self = jt.where(scross > 0, T_left, T_interp)
                vis = jt.where(after > 0, final_t, T_self).clamp(0.0, 1.0)
            else:
                vis = jt.where(after > 0, final_t, T_left).clamp(0.0, 1.0)
        else:   # interp (default regression)
            vis = T_interp.clamp(0.0, 1.0)
        scale = float(getattr(self, 'shadow_opacity_scale', 1.0))
        if scale != 1.0:
            vis = jt.pow(vis, scale)
        if summary is not None:
            import hashlib as _hl
            summary['T_lo_mean'] = float(T_lo.float32().mean())
            summary['T_interp_mean'] = float(T_interp.float32().mean())
            summary['T_hi_mean'] = float(T_hi.float32().mean())
            summary['bounds_hash'] = _hl.md5(bounds.tobytes()).hexdigest()[:12]
        if diag is not None:
            diag['T_prev'] = T_prev
            diag['bin'] = bin_i
            diag['b_lo'] = b_lo_v
            diag['b_hi'] = b_hi_v
            diag['self_cross'] = scross
            diag['dist'] = dist
            diag['q'] = q
        return vis, T_lo, T_interp, T_hi, bounds, bias

    def point_light_render(self, points, normal, albedo, roughness, metallic, viewdirs, shadow_cube=None, shadow_alpha=None, transmit_cube=None, transmit_bounds=None, transmit_final=None, emitter_transmit=None, receiver_transmittance=None):
        """Evaluate a point light independently at each Gaussian centre.

        Unlike the positional-SG approximation (fixed lobe + exp(-0.4d) weight),
        this computes a per-Gaussian incident direction to a lamp at
        self.point_light_position with radius-protected inverse-square falloff.
        Cook-Torrance BRDF reuses the pure-Jittor helpers from light_utils.py.

        Args (all [G, *], G = visible Gaussians):
            points   [G,3] world-space Gaussian centres
            normal   [G,3] unit world-space normal (normalize_for_light=True)
            albedo   [G,3] in [0,1]
            roughness[G,1] in [0,1]
            metallic [G,1] in [0,1]
            viewdirs [G,3] unit vector Gaussian->camera
            receiver_transmittance [G,1] explicit per-receiver exact T (E2, §30.5.3).
                When present it is the ONLY visibility source — it must NOT be
                combined with shadow_cube / transmit_cube / emitter_transmit (the
                exact/boundary exclusion is enforced here) and it is rejected for
                the area-light path (per-sample exact is N4-F2, later).
        Returns:
            specular [G,3], diffuse [G,3]
        """
        # §30.5.3 E2-C: exact/boundary mutual exclusion + T contract FIRST (before
        # the G==0 early return and before the point/area dispatch). Explicit T is
        # mutually exclusive with every boundary/shadow input; must be float32,
        # [G,1] (or [0,1] when G==0), finite and in [0,1].
        if receiver_transmittance is not None:
            _T = receiver_transmittance
            for _a, _v in (('shadow_cube', shadow_cube),
                           ('transmit_cube', transmit_cube),
                           ('emitter_transmit', emitter_transmit)):
                if _v is not None:
                    raise RuntimeError(
                        f'explicit receiver_transmittance with {_a} is forbidden '
                        f'(exact/boundary exclusion, §30.5.3)')
            if str(_T.dtype) != 'float32':
                raise RuntimeError(
                    f'explicit receiver_transmittance dtype {_T.dtype} != float32')
            _sh = tuple(_T.shape)
            if len(_sh) != 2 or _sh[1] != 1:
                raise RuntimeError(
                    f'explicit receiver_transmittance shape {_sh} != [G,1]')
            if points.shape[0] > 0 and _sh[0] != points.shape[0]:
                raise RuntimeError(
                    f'explicit receiver_transmittance shape[0] {_sh[0]} != G '
                    f'{points.shape[0]}')
            _t_np = np.asarray(_T.numpy()).reshape(-1)
            if not np.all(np.isfinite(_t_np)):
                raise RuntimeError('explicit receiver_transmittance not finite')
            if not np.all((_t_np >= 0.0) & (_t_np <= 1.0)):
                raise RuntimeError('explicit receiver_transmittance out of [0,1]')

        G = points.shape[0]
        if G == 0:
            zero = jt.zeros_like(albedo)
            return zero, zero

        # N4-R0 (§25.4 item 5): clear cross-view diagnostic residue so a frame can
        # never inherit the previous view's _diag_pl / _transmit_diag /
        # _last_front_mask / _last_veff / _area_sample_stats.
        for _a in ('_diag_pl', '_transmit_diag', '_last_front_mask', '_last_veff',
                   '_area_sample_stats'):
            if hasattr(self, _a):
                delattr(self, _a)

        S = int(getattr(self, 'point_emitter_samples', 0))
        eradius = float(getattr(self, 'point_emitter_radius', 0.0))
        if receiver_transmittance is not None and (S > 1 and eradius > 0):
            raise RuntimeError('explicit receiver_transmittance with the area-light '
                               'path is forbidden (§30.5.3; per-sample exact is N4-F2)')
        if S > 1 and eradius > 0:
            return self._point_light_render_area(
                points, normal, albedo, roughness, metallic, viewdirs,
                S, eradius, transmit_cube, transmit_final, transmit_bounds,
                emitter_transmit)
        return self._point_light_render_point(
            points, normal, albedo, roughness, metallic, viewdirs,
            shadow_cube, shadow_alpha, transmit_cube, transmit_bounds, transmit_final,
            receiver_transmittance=receiver_transmittance)

    def _point_direct_one_sample(self, points, normal, albedo, roughness, metallic,
                                 viewdirs, light_pos, sample_intensity):
        """N4-R0 (§25.4 item 4): single emitter-sample direct light — the ONLY
        Cook-Torrance implementation shared by the point light (S=1) and every
        emitter sample of the area light.

        `light_pos` is a [3] jt.Var or np.ndarray; `sample_intensity` is this
        sample's share of the total flux (full `intensity` for the point light,
        `intensity/S` per area sample).

        Returns (specular, diffuse, L, front_f, NoL_raw, NoV_raw, dist, atten) —
        all PRE-visibility (the caller multiplies by the transmittance).
        """
        if isinstance(light_pos, jt.Var):
            _lp = light_pos
        else:
            _lp = jt.array(np.asarray(light_pos, dtype=np.float32))
        L = jt.normalize(_lp.unsqueeze(0) - points, p=2, dim=-1)              # [G,3]
        dist = jt.norm(_lp.unsqueeze(0) - points, p=2, dim=-1, keepdim=True)  # [G,1]
        # radius-protected inverse-square: prevents /0 and softens near-field lamp.
        atten = sample_intensity / jt.maximum(dist * dist, float(self.point_light_radius) ** 2)
        radiance = self.point_light_color.unsqueeze(0) * atten                # [G,3]
        # Match sg_render's roughness floor so low-roughness Gaussians cannot
        # produce extreme GGX specular spikes near the lamp.
        rough = roughness.clamp(float(self.sg_min_roughness), 1.0)
        # Use explicit cosine terms and a front-face mask. Do not reuse the
        # module-level saturate_dot: its 1e-4 floor keeps back-facing
        # Gaussians alive and inflates grazing-angle specular. SG/env paths
        # are untouched.
        NoL_raw = jt.sum(normal * L, dim=-1, keepdim=True)                    # [G,1]
        NoV_raw = jt.sum(normal * viewdirs, dim=-1, keepdim=True)             # [G,1]
        front_f = jt.logical_and(NoL_raw > 0.0, NoV_raw > 0.0).float32()      # [G,1]
        NoL = NoL_raw.clamp(0.0, 1.0)
        NoV = NoV_raw.clamp(0.0, 1.0)
        H = jt.normalize(L + viewdirs, p=2, dim=-1)                           # [G,3]
        HoV = jt.sum(H * viewdirs, dim=-1, keepdim=True).clamp(0.0, 1.0)      # [G,1]
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic                      # [G,3]
        NDF = DistributionGGX(normals=normal, half_dirs=H, roughness=rough)
        Gs = GeometrySmith(normals=normal, view_dirs=viewdirs, light_dirs=L, roughness=rough)
        F = fresnelSchlick(HoV=HoV, F0=F0)                                    # [G,3]
        # P0 bug1: Cook-Torrance needs the rendering-equation cosine term *NoL;
        # bug3: zero out back-facing (light or camera) Gaussians.
        specular = NDF * Gs * F / jt.maximum(4.0 * NoV * NoL, 1e-4) * radiance * NoL * front_f
        kd = (1.0 - F) * (1.0 - metallic)
        diffuse = (kd * albedo / np.pi) * NoL * radiance * front_f
        return specular, diffuse, L, front_f, NoL_raw, NoV_raw, dist, atten

    def _point_light_render_point(self, points, normal, albedo, roughness, metallic,
                                  viewdirs, shadow_cube=None, shadow_alpha=None,
                                  transmit_cube=None, transmit_bounds=None,
                                  transmit_final=None, receiver_transmittance=None):
        """S<=1 pure point-light path (N4-R0: reuses the shared single-sample
        direct helper + the unified _query_transmittance). Behaviour preserved so
        the R0 S=1 regression stays pixel-identical (max|Δ| <= 1e-5). §30.5.3 E2-C:
        an explicit `receiver_transmittance` [G,1] is the per-receiver exact T and
        bypasses _query_transmittance entirely (boundary stays untouched)."""
        G = points.shape[0]
        specular, diffuse, L, front_f, NoL_raw, NoV_raw, dist, atten = \
            self._point_direct_one_sample(
                points, normal, albedo, roughness, metallic, viewdirs,
                self.point_light_position, float(self.point_light_intensity))
        # N4-R0 (§25.4 item 6): keep the UNOCCLUDED (shadow-off) direct light for
        # the N3-D error-driven bounds weights (pre-vis, captured before multiply).
        spec_off, diff_off = specular, diffuse
        vis = jt.ones([G, 1])
        if receiver_transmittance is not None:
            # §30.5.3 E2-C: explicit per-receiver exact transmittance. Keep the raw
            # T_exact; T_effective = T_exact**scale is only non-identity in the
            # non-验收 visual experiments (E3 uses scale=1 so they are identical).
            # Both diffuse and specular are scaled by the SAME T.
            T_exact = receiver_transmittance
            _scale = float(getattr(self, 'shadow_opacity_scale', 1.0))
            T_eff = T_exact if _scale == 1.0 else jt.pow(T_exact, _scale)
            vis = T_eff.clamp(0.0, 1.0)
            specular = specular * vis
            diffuse = diffuse * vis
            if getattr(self, 'shadow_diag', False):
                import numpy as _np
                _v = vis.float32().numpy().reshape(-1)
                print(f"[exact-T] G={G} vis[mean={_v.mean():.4f} <0.5="
                      f"{float((_v < 0.5).mean())*100:.1f}% >0.9="
                      f"{float((_v > 0.9).mean())*100:.1f}%] scale={_scale:g}",
                      flush=True)
        elif transmit_cube is not None:
            # P4.1 (transmittance shadow, FIXED receiver query): N4-R0 replaces the
            # point branch's 3rd inline copy with the unified _query_transmittance.
            # N3-E (§26.4): the per-receiver diagnostic capture (T_prev/bin/b_lo/
            # b_hi/self-cross/dist/q) is gated on `light.transmit_diag` so the
            # default production path adds no per-interval jt ops; the estimator
            # identity string is always recorded (zero cost).
            _cap = bool(getattr(self, 'transmit_diag', False))
            _diag = {} if _cap else None
            vis, T_lo, T_interp, T_hi, bounds, bias = self._query_transmittance(
                points, self.point_light_position, transmit_cube, transmit_final,
                transmit_bounds, diag=_diag)
            vis = vis.clamp(0.0, 1.0)
            specular = specular * vis
            diffuse = diffuse * vis
            # P4.1: keep the raw per-receiver bounds for the physical-ROI analysis
            # scripts (T_lo/T_interp/T_hi + boundary metadata). `points` (world
            # xyz) lets P4.2's LOS scanner tag receivers by tabletop/other.
            self._transmit_diag = {
                'T_lo': T_lo, 'T_interp': T_interp, 'T_hi': T_hi,
                'bounds': bounds, 'bias': bias,
                'points': points.float32(),
                'estimator': str(getattr(self, 'transmit_estimator', 'interp')),
            }
            if _cap:
                self._transmit_diag.update({
                    'T_prev': _diag['T_prev'], 'bin': _diag['bin'],
                    'b_lo': _diag['b_lo'], 'b_hi': _diag['b_hi'],
                    'self_cross': _diag['self_cross'],
                    'dist': _diag['dist'], 'q': _diag['q'],
                })
            if getattr(self, 'shadow_diag', False):
                import numpy as _np
                _v = vis.numpy().reshape(-1)
                _lo = T_lo.float32().numpy().reshape(-1)
                _hi = T_hi.float32().numpy().reshape(-1)
                _it = T_interp.float32().numpy().reshape(-1)
                _gap = float((_lo - _hi).mean())
                print(f"[transmit-stats] G={G} N={int(transmit_cube.shape[0])} "
                      f"vis[mean={_v.mean():.4f} <0.5={float((_v < 0.5).mean())*100:.1f}% "
                      f">0.9={float((_v > 0.9).mean())*100:.1f}%] "
                      f"T_lo[mean={_lo.mean():.4f}] T_interp[mean={_it.mean():.4f}] "
                      f"T_hi[mean={_hi.mean():.4f}] gap={_gap:.4f} scale="
                      f"{float(getattr(self, 'shadow_opacity_scale', 1.0))} "
                      f"bounds={np.round(bounds, 3).tolist()}", flush=True)
        elif shadow_cube is not None:
            # P1-b soft depth shadow (unchanged semantics).
            dirs = -L                                   # lamp -> point (cubemap outward dir)
            shd = dr.texture(shadow_cube[None, ..., None], dirs.unsqueeze(0).contiguous(),
                             filter_mode='nearest', boundary_mode='cube')[0]  # [G,1]
            bias = float(getattr(self, 'shadow_bias', 0.05))
            soft = float(getattr(self, 'shadow_soft', 0.5))
            over = dist - shd - bias                                        # [G,1]
            vis = (1.0 - over / max(soft, 1e-6)).clamp(0.0, 1.0).float32()  # [G,1]
            specular = specular * vis
            diffuse = diffuse * vis
            if getattr(self, 'shadow_diag', False):
                import numpy as _np
                _sh, _v, _ds = shd.numpy(), vis.numpy(), dist.numpy()
                _sd = shd - _ds
                print(f"[shadow-stats] G={G} shd[med={_np.median(_sh):.3f} "
                      f"p99={_np.percentile(_sh,99):.3f} max={_sh.max():.3f}] "
                      f"dist[med={_np.median(_ds):.3f} p99={_np.percentile(_ds,99):.3f}] "
                      f"shd-dist[med={_np.median(_sd):.3f}] "
                      f"vis[<0.5={float((_v<0.5).mean())*100:.1f}% mean={_v.mean():.3f}]",
                      flush=True)
        # N4-R0 (§25.4 item 7): expose the effective shadow visibility as the
        # pixel-space Veff channel so `_vis.npy` gives the penumbra line profile.
        self._last_veff = vis
        # P2 (REASSESSMENT §P2): gated Gaussian-level point-light diagnostics for
        # verifying the shadow-off lighting distribution (tabletop direct light,
        # view-to-view normal consistency). Default path is bit-identical / zero
        # extra work. `_last_front_mask` is kept as a live Var so lightRender can
        # expose it to the pixel-space raster path (7th extra channel).
        if getattr(self, 'diag_gaussians', False):
            import numpy as _np
            _nol = NoL_raw.float32().numpy().reshape(-1)
            _nov = NoV_raw.float32().numpy().reshape(-1)
            _ds = dist.float32().numpy().reshape(-1)
            _at = atten.float32().numpy().reshape(-1)
            _rg = roughness.float32().numpy().reshape(-1)
            _fm = front_f.numpy().reshape(-1)
            _vs = vis.float32().numpy().reshape(-1)
            self._last_front_mask = front_f                      # [G,1] Var (live)
            self._diag_pl = {
                # N3-E (§26.4/§26.2.6): explicit schema so analysis tools can
                # branch — schema 1 = per-receiver raw fields (point light).
                'schema_version': 1,
                'NoL_raw': _nol, 'NoV_raw': _nov, 'dist': _ds, 'atten': _at,
                'roughness': _rg, 'front_mask': _fm, 'vis': _vs,
                'diffuse': diffuse.float32().numpy().reshape(-1, 3),
                'specular': specular.float32().numpy().reshape(-1, 3),
                # N3-C (§21): world positions for Gaussian top-K highlight
                # attribution (view3's peaks live in vegetation, not subject).
                # N4-R0 (§25.4 item 6): UNOCCLUDED direct light (pre-vis) for the
                # N3-D error-driven bounds weights (guide §25.5.2).
                'diffuse_off': diff_off.float32().numpy().reshape(-1, 3),
                'specular_off': spec_off.float32().numpy().reshape(-1, 3),
                'points': points.float32().numpy().reshape(-1, 3),
            }
            print(f"[diag-pl] G={G} "
                  f"NoL_raw[>0={float((_nol > 0).mean())*100:.1f}% med={_np.median(_nol):.3f}] "
                  f"NoV_raw[>0={float((_nov > 0).mean())*100:.1f}%] "
                  f"front={float(_fm.mean())*100:.1f}% "
                  f"dist[med={_np.median(_ds):.3f} p99={_np.percentile(_ds,99):.3f}] "
                  f"atten[med={_np.median(_at):.3f} max={_at.max():.3f}] "
                  f"rough[med={_np.median(_rg):.3f}] vis[mean={_vs.mean():.3f}]", flush=True)

        return specular, diffuse

    def _point_light_render_area(self, points, normal, albedo, roughness, metallic,
                                 viewdirs, S, eradius, transmit_cube, transmit_final,
                                 transmit_bounds, emitter_transmit):
        """Finite-emitter area light. `emitter_transmit` is a list
        of STRUCTURED per-sample contexts {index, position, position_hash,
        transmit_cube, transmit_final, transmit_bounds, bounds_hash, shadow_meta};
        when None the shared-center visibility (stage-1 sharedvis) is used.

        The BRDF sample positions always come from the shared
        `point_emitter_positions`, populated with `make_emitter_samples`; they
        are never resampled here. Total flux stays
        `intensity` (each sample carries intensity/S). Direct light is
        accumulated independently for every emitter sample."""
        G = points.shape[0]
        per_sample = (emitter_transmit is not None)
        _lpos = np.asarray(self.point_light_position.numpy(), np.float32).reshape(-1)
        _positions = getattr(self, 'point_emitter_positions', None)
        if per_sample:
            if len(emitter_transmit) != S:
                raise ValueError(
                    f"area-light sample/context mismatch: emitter_samples={S} but "
                    f"context has {len(emitter_transmit)} samples")
            positions = [np.asarray(c['position'], np.float32).reshape(3)
                         for c in emitter_transmit]
            if _positions is not None:
                _exp = np.asarray(_positions, np.float32)
                for s, _p in enumerate(positions):
                    if not np.allclose(_exp[s], _p, atol=1e-6):
                        raise ValueError(
                            f"area-light sample {s} context position {_p.tolist()} "
                            f"!= shared emitter position {_exp[s].tolist()}")
        else:
            if _positions is None:
                raise ValueError(
                    "area-light BRDF sampling requires `point_emitter_positions` "
                    "(set once by the caller from the SHARED emitter sampler)")
            positions = list(np.asarray(_positions, np.float32))
        center_res = None
        if not per_sample and transmit_cube is not None:            # shared-vis stage 1
            center_res = self._query_transmittance(
                points, _lpos, transmit_cube, transmit_final, transmit_bounds)
        spec_acc = jt.zeros_like(albedo)
        diff_acc = jt.zeros_like(albedo)
        spec_off_acc = jt.zeros_like(albedo)      # pre-vis sum (shadow-off direct)
        diff_off_acc = jt.zeros_like(albedo)
        _LUM = jt.float32([0.2126, 0.7152, 0.0722])
        ET_acc = None; E_Tlo = None; E_Thi = None; E_front = None
        _sres = [None] * S                          # per-sample (vis, T_lo, T_int, T_hi)
        # N3-E (§26.4): energy-weighted aggregate diagnostics (only when
        # requested) + optional per-sample unoccluded energy E_full[S,G] for the
        # area-light adaptive-refinement path. Guarded so the default path keeps
        # its minimal lazy graph.
        _want_ew = bool(getattr(self, 'diag_gaussians', False))
        _want_full = bool(getattr(self, 'area_full_t', False))
        E_NoL = E_NoV = E_dist = E_atten = None
        _E_full = [] if _want_full else None
        import hashlib as _hl_mod
        _pos_arr = np.asarray(positions, np.float32)
        _pos_hash = _hl_mod.md5(_pos_arr.tobytes()).hexdigest()[:12]
        for s in range(S):
            _sp = positions[s]
            intensity_s = float(self.point_light_intensity) / S
            spec_s, diff_s, L_s, front_f_s, NoL_r_s, NoV_r_s, dist_s, atten_s = \
                self._point_direct_one_sample(
                    points, normal, albedo, roughness, metallic, viewdirs,
                    _sp, intensity_s)
            if per_sample:
                vs = self._query_transmittance(
                    points, _sp, emitter_transmit[s]['transmit_cube'],
                    emitter_transmit[s]['transmit_final'],
                    emitter_transmit[s]['transmit_bounds'])
                _sres[s] = vs
                vis_s, lo_s, it_s, hi_s = vs[0], vs[1], vs[2], vs[3]
            elif center_res is not None:
                vis_s = center_res[0]
            else:
                vis_s = jt.ones([G, 1])
            spec_acc = spec_acc + spec_s * vis_s
            diff_acc = diff_acc + diff_s * vis_s
            spec_off_acc = spec_off_acc + spec_s
            diff_off_acc = diff_off_acc + diff_s
            # energy-weighted effective visibility (§25.4 item 6):
            #   Veff = Σ_s (E_s·T_s) / Σ_s E_s,  E_s = unoccluded direct energy
            E_s = ((diff_s + spec_s) * _LUM).sum(dim=-1, keepdim=True)     # [G,1]
            ET_acc = (E_s * vis_s) if ET_acc is None else ET_acc + E_s * vis_s
            E_front = (E_s * front_f_s) if E_front is None else E_front + E_s * front_f_s
            if per_sample:
                E_Tlo = (E_s * lo_s) if E_Tlo is None else E_Tlo + E_s * lo_s
                E_Thi = (E_s * hi_s) if E_Thi is None else E_Thi + E_s * hi_s
            if _want_ew:
                # N3-E (§26.2.6): energy-weighted geometry fields — the per-receiver
                # raw NoL/NoV/dist/atten of the LAST sample are meaningless next to
                # the aggregate diffuse/specular, so aggregate them by unoccluded
                # energy instead of recording a single sample's values.
                E_NoL = (E_s * NoL_r_s) if E_NoL is None else E_NoL + E_s * NoL_r_s
                E_NoV = (E_s * NoV_r_s) if E_NoV is None else E_NoV + E_s * NoV_r_s
                E_dist = (E_s * dist_s) if E_dist is None else E_dist + E_s * dist_s
                E_atten = (E_s * atten_s) if E_atten is None else E_atten + E_s * atten_s
            if _want_full:
                _E_full.append(E_s)
        E_sum = ((diff_off_acc + spec_off_acc) * _LUM).sum(dim=-1, keepdim=True)
        E_safe = jt.maximum(E_sum, 1e-9)
        veff = (ET_acc / E_safe).clamp(0.0, 1.0)
        self._last_veff = veff
        if getattr(self, 'shadow_diag', False):
            print(f"[area-stats] S={S} r={eradius} per_sample={per_sample} "
                  f"spec_mean={spec_acc.mean():.5f} diff_mean={diff_acc.mean():.5f} "
                  f"veff_mean={veff.mean():.5f}", flush=True)
        # Gaussian-level diagnostics (aggregated; per-sample only in _transmit_diag).
        if getattr(self, 'diag_gaussians', False):
            import numpy as _np
            _veff = veff.float32().numpy().reshape(-1)
            _fm_ew = (E_front / E_safe).float32().numpy().reshape(-1)
            # N3-E (§26.4/§26.2.6): schema_version=2 distinguishes the aggregated
            # area fields from the per-receiver point schema (v1); geometry fields
            # are ENERGY-WEIGHTED (NoL_ew/NoV_ew/dist_ew/atten_ew) so top-K cannot
            # mistake a single emitter sample for the aggregate. Per-sample raw
            # values + unoccluded energy E_full[S,G] only in --area-full-t mode.
            self._diag_pl = {
                'schema_version': 2,
                'lighting_geometry': str(getattr(self, 'point_emitter_geometry', 'disk')),
                'S': int(S),
                'positions_hash': _pos_hash,
                'vis': _veff, 'Veff': _veff,
                'front_mask': _fm_ew,
                'NoL_ew': (E_NoL / E_safe).float32().numpy().reshape(-1),
                'NoV_ew': (E_NoV / E_safe).float32().numpy().reshape(-1),
                'dist_ew': (E_dist / E_safe).float32().numpy().reshape(-1),
                'atten_ew': (E_atten / E_safe).float32().numpy().reshape(-1),
                'roughness': roughness.float32().numpy().reshape(-1),
                'diffuse': diff_acc.float32().numpy().reshape(-1, 3),
                'specular': spec_acc.float32().numpy().reshape(-1, 3),
                'diffuse_off': diff_off_acc.float32().numpy().reshape(-1, 3),
                'specular_off': spec_off_acc.float32().numpy().reshape(-1, 3),
                'points': points.float32().numpy().reshape(-1, 3),
            }
            if per_sample:
                self._diag_pl['T_lo'] = (E_Tlo / E_safe).float32().numpy().reshape(-1)
                self._diag_pl['T_hi'] = (E_Thi / E_safe).float32().numpy().reshape(-1)
            if _want_full:
                self._diag_pl['E_full'] = _np.asarray(
                    [e.float32().numpy().reshape(-1) for e in _E_full], _np.float32)
            print(f"[diag-pl-area] S={S} Veff[mean={_veff.mean():.4f} "
                  f"<0.5={float((_veff < 0.5).mean())*100:.1f}%] "
                  f"front_ew={float(_fm_ew.mean())*100:.1f}% "
                  f"off_energy[mean={float((diff_off_acc + spec_off_acc).float32().mean()):.4f}]",
                  flush=True)
        # Two-level transmit diagnostics (default = per-sample summary; full T only
        # when the `area_full_t` debug flag is set).
        if per_sample:
            stats = []
            for s in range(S):
                _lo, _it, _hi = _sres[s][1], _sres[s][2], _sres[s][3]
                stats.append({
                    'index': int(emitter_transmit[s]['index']),
                    'position': positions[s].tolist(),
                    'bounds_hash': emitter_transmit[s].get('bounds_hash'),
                    'T_lo_mean': float(_lo.float32().mean()),
                    'T_interp_mean': float(_it.float32().mean()),
                    'T_hi_mean': float(_hi.float32().mean()),
                })
            self._area_sample_stats = stats
            td = {'per_sample': True, 'S': S,
                  # N3-E (§26.4): estimator identity + emitter positions hash so
                  # analysis can reproduce the per-sample bounds/q from points.
                  'estimator': str(getattr(self, 'transmit_estimator', 'interp')),
                  'positions_hash': _pos_hash,
                  'T_lo_mean': [st['T_lo_mean'] for st in stats],
                  'T_interp_mean': [st['T_interp_mean'] for st in stats],
                  'T_hi_mean': [st['T_hi_mean'] for st in stats],
                  'position': np.asarray(positions, np.float32),
                  'bounds': np.asarray(transmit_bounds, np.float32),
                  'bias': float(getattr(self, 'shadow_bias', 0.02)),
                  'points': points.float32()}
            if getattr(self, 'area_full_t', False):
                td['T_lo_full'] = jt.stack([_sres[s][1].float32() for s in range(S)], dim=0)
                td['T_interp_full'] = jt.stack([_sres[s][2].float32() for s in range(S)], dim=0)
                td['T_hi_full'] = jt.stack([_sres[s][3].float32() for s in range(S)], dim=0)
            self._transmit_diag = td
        elif center_res is not None:
            self._transmit_diag = {'T_lo': center_res[1], 'T_interp': center_res[2],
                                   'T_hi': center_res[3], 'bounds': center_res[4],
                                   'bias': center_res[5], 'points': points.float32(),
                                   'estimator': str(getattr(self, 'transmit_estimator', 'interp'))}
        return spec_acc, diff_acc



######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file

def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.  

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def load_env_hdr(fn, sg_pth = None,res=256,numLgtSGs=16, is_sg=True, scale=1.0):
    
    if fn[-4:] == ".hdr":
        with open(fn, "rb") as h:
            buffer_ = np.frombuffer(h.read(), np.uint8)
        bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = imageio.imread(fn)[:,:,:3]

    rgb = np.clip(rgb/255.0,0.0,1.0)*255.0
    latlong_img =jt.array(rgb)


    cubemap = util.latlong_to_cubemap(latlong_img, [res, res])

    l = Hybridlight(res)

    if sg_pth is not None:
        print("Load Pretrain Light Param!")
        l.load_light(sg_pth)
    
    l.base = cubemap
    l.build_mips()

    if is_sg:
        latlong_img_sg = jt.array(normalize_hdr(rgb,method="exposure",alpha=1.0))
        lgtSGs = fit_sg_envmap(numLgtSGs,latlong_img_sg)
        # Jittor: reconstruct lgtSGs tensor (no .data slice assignment)
        l.lgtSGs = jt.concat([l.lgtSGs[:, :3], lgtSGs], dim=1)

    return l




def load_env(fn, sg_path, res,numLgtSGs,is_sg=True, scale=1.0):
    return load_env_hdr(fn, sg_path, res, numLgtSGs,is_sg,scale)


def save_env_map(fn, light):
    assert isinstance(light, Hybridlight), "Can only save EnvironmentLight currently"
    if isinstance(light, Hybridlight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    util.save_image_raw(fn, color.detach().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    base = jt.rand(6, base_res, base_res, 3, dtype=jt.float32) * scale + bias
    return Hybridlight(base)

def extract_env_map(light, resolution=[512, 1024]):
    assert isinstance(light, Hybridlight), "Can only save EnvironmentLight currently"
    color = util.cubemap_to_latlong(light.base, resolution)
    return color




def saturate_dot(a: jt.Var, b: jt.Var) -> jt.Var:
    return (a * b).sum(dim=-1, keepdim=True).clamp(1e-4, 1.0)





def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER

    inv_lambda_val = 1. / lambda_val
    t = jt.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

    ### note: for numeric stability
    inv_a = jt.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = jt.exp(-t * cos_beta.clamp(0.0))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
    b = jt.exp(t * cos_beta.clamp(-float('inf'), 0.0))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (jt.exp(-lambda_val) - jt.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - jt.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s



def compute_weight(point_sg,lgtSGPosition):
    diff = (lgtSGPosition-point_sg)
    squared_diff = diff ** 2  # 每个维度的差值平方
    distance = jt.sqrt(squared_diff.sum(dim=-1)) #(N, K)
    return jt.exp(-0.4*distance)


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    dot = jt.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = jt.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    tmp = jt.minimum(tmp, ratio + 1.)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1. / tmp
    # Phase 91: float64 for the entire cancellation chain.
    # tmp ≈ ratio+1.0 → subtraction loses ~7 bits in f32 → λ2(>20000)× amplification.
    # When both lambdas are tensors (BRDF×light SG path), compute ratio/dot/tmp/diff/exp in float64.
    if hasattr(lambda1, 'float64') and hasattr(lambda2, 'float64'):
        r_f64 = lambda1.float64() / lambda2.float64()
        d_f64 = jt.sum(lobe1.float64() * lobe2.float64(), dim=-1, keepdim=True)
        t_f64 = jt.sqrt(r_f64 * r_f64 + 1.0 + 2.0 * r_f64 * d_f64)
        t_f64 = jt.minimum(t_f64, r_f64 + 1.0)
        diff = lambda2.float64() * (t_f64 - r_f64 - 1.0)

        tmp_f32 = t_f64.float32()
        ratio_f32 = r_f64.float32()
        lambda3 = lambda2 * tmp_f32
        lambda1_over_lambda3 = ratio_f32 / tmp_f32
        lambda2_over_lambda3 = 1.0 / tmp_f32
    else:
        # Scalar lambdas (cosine lobe: λ=0.0315, μ=32.7080).
        # lambda1 or lambda2 is a Python float → ratio is a jt.Var broadcast from scalar.
        # diff = lambda2 * (tmp - ratio - 1.0) — still cancellation-prone if lambda2 is large.
        ratio = lambda1 / lambda2
        dot = jt.sum(lobe1 * lobe2, dim=-1, keepdim=True)
        tmp = jt.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
        tmp = jt.minimum(tmp, ratio + 1.)
        lambda3 = lambda2 * tmp
        lambda1_over_lambda3 = ratio / tmp
        lambda2_over_lambda3 = 1. / tmp
        # Use float64 for diff if lambda2 is a tensor (cosine lobe × final_lambdas)
        if hasattr(lambda2, 'float64'):
            diff = lambda2.float64() * (tmp.float64() - ratio.float64() - 1.0)
        else:
            diff = lambda2 * (tmp - ratio - 1.0)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * jt.exp(diff).float32()

    return final_lobes, final_lambdas, final_mus


def get_envmap_dirs(res = [512, 1024]):
    # Phase 54c: jt.meshgrid in Jittor 1.3.11 doesn't support indexing= kwarg.
    # Use numpy meshgrid then convert to jt.Var (same pattern as SDF/utils.py).
    gy_np, gx_np = np.meshgrid(
        np.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0]),
        np.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1]),
        indexing="ij",
    )
    gy = jt.array(gy_np, dtype=jt.float32)
    gx = jt.array(gx_np, dtype=jt.float32)

    sintheta, costheta = jt.sin(gy * np.pi), jt.cos(gy * np.pi)
    sinphi, cosphi = jt.sin(gx * np.pi), jt.cos(gx * np.pi)

    reflvec = jt.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]
    return reflvec


def fit_sg_envmap(numLgtSGs,hdri):

    # Initialize in numpy (avoids Jittor .data assignment issues)
    lgt_np = np.random.randn(numLgtSGs, 7).astype(np.float32)
    lgt_np[:, 3:4] = 20. + np.abs(lgt_np[:, 3:4] * 100.)
    energy_np = compute_energy(jt.array(lgt_np)).numpy()
    lgt_np[:, 4:] = np.abs(lgt_np[:, 4:]) / np.sum(energy_np, axis=0, keepdims=True) * 2. * np.pi
    lobes = fibonacci_sphere(numLgtSGs).astype(np.float32)
    lgt_np[:, :3] = lobes
    lgtSGs = jt.array(lgt_np)

    optimizer = jt.optim.Adam([lgtSGs,], lr=1e-2)

    H, W = hdri.shape[:2]
    
    N_iter = 2000
    
    for step in range(N_iter):
        env_map = SG2Envmap(lgtSGs, H, W)
        loss = jt.mean((env_map - hdri) * (env_map - hdri))
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()

        if step % 200 == 0:
            try:
                loss_val = loss.item()
            except RuntimeError:
                loss_val = float('nan')  # CUDA-only tensor fallback
            print('step: {}, loss: {}'.format(step, loss_val))
    
    return lgtSGs
       


def SG2Envmap(lgtSGs, H=512, W=1024, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = jt.meshgrid([jt.linspace(0., np.pi/2., H), jt.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = jt.meshgrid([jt.linspace(0., np.pi, H), jt.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = jt.stack([jt.cos(theta) * jt.sin(phi), jt.cos(phi), jt.sin(theta) * jt.sin(phi)],
                           dim=-1)    # [H, W, 3]
    # print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    # print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    # print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    # lgtSGs = lgtSGs.clone().detach()
    # Jittor: no .to(device) needed, all tensors share same device
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (jt.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = jt.abs(lgtSGs[..., 3:4])
    lgtSGMus = jt.abs(lgtSGs[..., -3:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * jt.exp(lgtSGLambdas * (jt.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = jt.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    
    return envmap


def normalize_hdr(envmap, method="max", alpha=1.0, exposure=1.0, gamma=2.2):
    """
    Normalize HDR environment map.
    
    Parameters:
    - envmap: HDR environment map as a numpy array (H, W, C).
    - method: Normalization method ("max", "log", "exposure").
    - alpha: Parameter for log normalization (used when method="log").
    - exposure: Exposure parameter (used when method="exposure").
    - gamma: Gamma correction value.

    Returns:
    - norm_envmap: Normalized HDR environment map.
    """
    # Convert to luminance (optional, depends on use case)
    luminance = np.mean(envmap, axis=-1)
    # luminance = envmap
    
    if method == "max":
        # Maximum value normalization
        max_val = np.max(luminance)
        # max_val = np.percentile(luminance,80)
        norm_envmap = envmap / max_val if max_val > 0 else envmap
    elif method == "log":
        # Log-based normalization
        max_val = np.max(luminance)
        # max_val = np.percentile(luminance,80)
        norm_envmap = np.log(1 + alpha * envmap) / np.log(1 + alpha * max_val)
    elif method == "exposure":
        # Exposure adjustment normalization
        norm_envmap = envmap / (2 ** exposure)
    else:
        raise ValueError("Invalid normalization method. Choose 'max', 'log', or 'exposure'.")
    
    # Apply gamma correction
    if gamma > 0:
        norm_envmap = np.clip(norm_envmap, 0, 1) ** (1.0 / gamma)
    
    return norm_envmap
