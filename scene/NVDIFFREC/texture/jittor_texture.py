"""
Pure Jittor texture sampling implementation (Phase 35).

Replaces nvdiffrast dr.texture() with F.grid_sample.
No CUDA compilation required — pure Jittor autograd compatible.

Supports:
- 2D textures: F.grid_sample directly
- Cubemap textures: cube face projection + F.grid_sample per face
- Cubemap mipmap: trilinear interpolation between mip levels
"""

from functools import lru_cache

import jittor as jt
import jittor.nn as F
import numpy as np


# Exact tables from GANG-master/submodules/nvdiffrast-main/nvdiffrast/common/
# texture.cu.  They define seamless bilinear wrapping across cube faces and
# the three-face average used for a missing corner texel.
_CUBE_WRAP_MASK1 = (
    0x1530A440, 0x1133A550, 0x6103A110, 0x1515AA44, 0x6161AA11, 0x40154A04, 0x44115A05, 0x04611A01,
    0x2630A440, 0x2233A550, 0x5203A110, 0x2626AA44, 0x5252AA11, 0x40264A04, 0x44225A05, 0x04521A01,
    0x32608064, 0x3366A055, 0x13062091, 0x32328866, 0x13132299, 0x50320846, 0x55330A55, 0x05130219,
    0x42508064, 0x4455A055, 0x14052091, 0x42428866, 0x14142299, 0x60420846, 0x66440A55, 0x06140219,
    0x5230A044, 0x5533A055, 0x1503A011, 0x5252AA44, 0x1515AA11, 0x40520A44, 0x44550A55, 0x04150A11,
    0x6130A044, 0x6633A055, 0x2603A011, 0x6161AA44, 0x2626AA11, 0x40610A44, 0x44660A55, 0x04260A11,
)
_CUBE_WRAP_MASK2 = (
    0x26, 0x33, 0x11, 0x05, 0x00, 0x09, 0x0C, 0x04, 0x04, 0x00, 0x00, 0x05, 0x00, 0x81, 0xC0, 0x40,
    0x02, 0x03, 0x09, 0x00, 0x0A, 0x00, 0x00, 0x02, 0x64, 0x30, 0x90, 0x55, 0xA0, 0x99, 0xCC, 0x64,
    0x24, 0x30, 0x10, 0x05, 0x00, 0x01, 0x00, 0x00, 0x06, 0x03, 0x01, 0x05, 0x00, 0x89, 0xCC, 0x44,
)
_PAD_INDEX_VARS = {}


def _wrap_cube_quad(face, ix0, ix1, iy0, iy1, width):
    cx = 0 if ix0 < 0 else 2 if ix1 >= width else 1
    cy = 0 if iy0 < 0 else 6 if iy1 >= width else 3
    case = cx + cy
    if case >= 5:
        case -= 1
    case = (face << 3) + case
    mask = _CUBE_WRAP_MASK1[case]
    selectors = (
        (mask >> 0) & 3, (mask >> 2) & 3, (mask >> 4) & 3, (mask >> 6) & 3,
        (mask >> 8) & 3, (mask >> 10) & 3, (mask >> 12) & 3, (mask >> 14) & 3,
    )
    pairs = (
        (ix0, iy0), (ix1, iy0), (ix0, iy1), (ix1, iy1),
        (ix0, iy0), (ix1, iy0), (ix0, iy1), (ix1, iy1),
    )
    values = [
        0 if selector == 0 else first if selector == 1 else second
        for selector, (first, second) in zip(selectors, pairs)
    ]
    xs, ys = values[:4], values[4:]
    faces = [
        ((mask >> 16) & 15) - 1,
        ((mask >> 20) & 15) - 1,
        ((mask >> 24) & 15) - 1,
        ((mask >> 28) & 15) - 1,
    ]
    flips = _CUBE_WRAP_MASK2[case]
    maximum = width - 1
    for index in range(4):
        if flips & (1 << index):
            xs[index] = maximum - xs[index]
        if flips & (1 << (index + 4)):
            ys[index] = maximum - ys[index]
    return [
        xs[index] + (ys[index] + faces[index] * width) * width
        for index in range(4)
    ]


@lru_cache(maxsize=16)
def _cube_padding_table(width):
    if width <= 0:
        raise ValueError("cubemap width must be positive")
    padded = width + 2
    indices = np.zeros((6, padded, padded, 3), dtype=np.int32)
    weights = np.zeros((6, padded, padded, 3), dtype=np.float32)
    for face in range(6):
        for py in range(padded):
            y = py - 1
            for px in range(padded):
                x = px - 1
                if 0 <= x < width and 0 <= y < width:
                    address = x + width * (y + face * width)
                    indices[face, py, px, 0] = address
                    weights[face, py, px, 0] = 1.0
                    continue
                if x < 0:
                    ix0, ix1, xslot = -1, 0, 0
                elif x >= width:
                    ix0, ix1, xslot = width - 1, width, 1
                elif x == width - 1:
                    ix0, ix1, xslot = width - 2, width - 1, 1
                else:
                    ix0, ix1, xslot = x, x + 1, 0
                if y < 0:
                    iy0, iy1, yslot = -1, 0, 0
                elif y >= width:
                    iy0, iy1, yslot = width - 1, width, 1
                elif y == width - 1:
                    iy0, iy1, yslot = width - 2, width - 1, 1
                else:
                    iy0, iy1, yslot = y, y + 1, 0
                quad = _wrap_cube_quad(face, ix0, ix1, iy0, iy1, width)
                selected = quad[yslot * 2 + xslot]
                if selected >= 0:
                    indices[face, py, px, 0] = selected
                    weights[face, py, px, 0] = 1.0
                else:
                    valid = [address for address in quad if address >= 0]
                    if len(valid) != 3:
                        raise RuntimeError("nvdiffrast cube-corner contract is invalid")
                    indices[face, py, px] = valid
                    weights[face, py, px] = 1.0 / 3.0
    return indices, weights


def _cube_padding_vars(width):
    cached = _PAD_INDEX_VARS.get(int(width))
    if cached is None:
        indices, weights = _cube_padding_table(int(width))
        cached = (
            jt.array(indices, dtype=jt.int32).stop_grad(),
            jt.array(weights, dtype=jt.float32).stop_grad(),
        )
        _PAD_INDEX_VARS[int(width)] = cached
    return cached


def _pad_cubemap(tex):
    if len(tex.shape) != 5 or tex.shape[0] != 1 or tex.shape[1] != 6:
        raise ValueError("cubemap must have shape [1, 6, H, W, C]")
    width = int(tex.shape[2])
    if int(tex.shape[3]) != width:
        raise ValueError("cubemap faces must be square")
    indices, weights = _cube_padding_vars(width)
    channels = int(tex.shape[-1])
    flat = tex.reshape(6 * width * width, channels)
    gathered = flat[indices.reshape(-1)].reshape(6, width + 2, width + 2, 3, channels)
    padded = (gathered * weights.unsqueeze(-1)).sum(dim=3)
    return padded.unsqueeze(0)


def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None,
            filter_mode='auto', boundary_mode='wrap', max_mip_level=None):
    """Texture sampling — pure Jittor F.grid_sample replacement for dr.texture().

    Args:
        tex: [1, H, W, C] for 2D, [1, 6, H, W, C] for cubemap
        uv:  [B, H, W, 2] for 2D, [B, H, W, 3] for cubemap, or [N, D] flat
        uv_da: (unused, accepted for compatibility)
        mip_level_bias: per-pixel mip bias [B, H, W] or [N]
        mip: list of tensors [1, 6, H_i, W_i, C] for custom mip stack
        filter_mode: 'nearest', 'linear', 'linear-mipmap-linear', 'linear-mipmap-nearest', 'auto'
        boundary_mode: 'cube', 'clamp', 'wrap', 'zero'
        max_mip_level: (unused, accepted for compatibility)

    Returns:
        Sampled tensor [B, H, W, C] or [1, N, C]
    """
    # Default filter mode
    if filter_mode == 'auto':
        filter_mode = 'linear-mipmap-linear' if (uv_da is not None or mip_level_bias is not None) else 'linear'
    if max_mip_level == 0 and 'mipmap' in filter_mode:
        filter_mode = 'linear'

    is_cube = (boundary_mode == 'cube')
    use_mip = ('mipmap' in filter_mode)

    # ---- Dispatch ----
    if use_mip:
        return _texture_cube_mip(tex, uv, mip, mip_level_bias, filter_mode)
    elif is_cube:
        return _texture_cube(tex, uv, filter_mode)
    else:
        return _texture_2d(tex, uv, filter_mode, boundary_mode)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _texture_2d(tex, uv, filter_mode, boundary_mode):
    """Sample 2D texture [1, H, W, C] at UV [..., 2] in [0,1] range."""
    spatial_shape = uv.shape[:-1]  # [B, H, W] or [N]
    uv_flat = uv.reshape(-1, 2)  # [N, 2]

    # tex [1, H, W, C] → [1, C, H, W]
    tex_nchw = tex.permute(0, 3, 1, 2)
    mode = 'bilinear' if 'linear' in filter_mode else 'nearest'

    if boundary_mode == 'wrap':
        # nvdiffrast's default 2D boundary mode takes the fractional part of
        # both texture coordinates.  Sample the centre tile of a 3x3 periodic
        # texture so bilinear footprints crossing any edge see the opposite
        # edge instead of F.grid_sample's border clamp.
        wrapped = uv_flat - jt.floor(uv_flat)
        tiled_x = jt.concat([tex_nchw, tex_nchw, tex_nchw], dim=3)
        tex_nchw = jt.concat([tiled_x, tiled_x, tiled_x], dim=2)
        tiled_uv = (wrapped + 1.0) / 3.0
        grid = (tiled_uv * 2.0 - 1.0).reshape(1, uv_flat.shape[0], 1, 2)
        padding = 'border'
    else:
        if boundary_mode not in {'zero', 'clamp'}:
            raise ValueError(f"unsupported 2D texture boundary mode: {boundary_mode}")
        grid = (uv_flat * 2.0 - 1.0).reshape(1, uv_flat.shape[0], 1, 2)
        padding = 'zeros' if boundary_mode == 'zero' else 'border'

    out = F.grid_sample(tex_nchw, grid, mode=mode, padding_mode=padding, align_corners=False)
    # [1, C, N, 1] → [N, C]
    result = out[0, :, :, 0].permute(1, 0)

    # Restore spatial dims
    if len(spatial_shape) >= 2:
        result = result.reshape(*(list(spatial_shape) + [-1]))
    else:
        result = result.reshape(1, spatial_shape[0], -1)
    return result


def _texture_cube(tex, uv, filter_mode):
    """Sample cubemap [1, 6, H, W, C] at 3D directions [..., 3]."""
    spatial_shape = uv.shape[:-1]
    uv_flat = uv.reshape(-1, 3)
    N = uv_flat.shape[0]
    C = tex.shape[-1]
    mode = 'bilinear' if 'linear' in filter_mode else 'nearest'

    # nvdiffrast indexes a cubemap directly from the original lookup vector.
    # Normalizing first is redundant in the forward pass because face
    # coordinates are component ratios, but it adds a numerically different
    # normalization Jacobian to the coordinate backward.  Keep the original
    # vector so Jittor differentiates the same ratios as indexCubeMapGrad().
    dx, dy, dz = uv_flat[..., 0], uv_flat[..., 1], uv_flat[..., 2]

    result = jt.zeros((N, C), dtype=tex.dtype)
    padded_tex = _pad_cubemap(tex)
    padded_scale = float(tex.shape[2]) / float(tex.shape[2] + 2)

    # P4.5.B (fix): MUTUALLY-EXCLUSIVE dominant-axis face selection. The previous
    # six `>=` masks all fired on cube edges/corners (e.g. (0,±.707,∓.707)) and
    # summed 2-3 faces -> up to ~3x brightness (transmit probes showed T>1).
    # Priority on equal |component|: X > Y > Z (same order as nvdiffrast's
    # if-else chain), so every direction maps to EXACTLY ONE face.
    adx, ady, adz = jt.abs(dx), jt.abs(dy), jt.abs(dz)
    safe_x = jt.where(adx > 0.0, adx, jt.ones_like(adx))
    safe_y = jt.where(ady > 0.0, ady, jt.ones_like(ady))
    safe_z = jt.where(adz > 0.0, adz, jt.ones_like(adz))
    valid = jt.logical_or(jt.logical_or(adx > 0.0, ady > 0.0), adz > 0.0)
    x_dom = jt.logical_and(adx >= ady, adx >= adz)
    y_dom = jt.logical_and(jt.logical_not(x_dom), ady >= adz)
    z_dom = jt.logical_and(jt.logical_not(x_dom), jt.logical_not(y_dom))
    face_masks = [
        jt.logical_and(x_dom, dx >= 0),   # +X
        jt.logical_and(x_dom, dx < 0),    # -X
        jt.logical_and(y_dom, dy >= 0),   # +Y
        jt.logical_and(y_dom, dy < 0),    # -Y
        jt.logical_and(z_dom, dz >= 0),   # +Z
        jt.logical_and(z_dom, dz < 0),    # -Z
    ]

    for s in range(6):
        # Project to face UV (same as _cubemap_sample_jt in util.py)
        if s == 0:   # +X
            u, v = -dz / safe_x, -dy / safe_x
        elif s == 1: # -X
            u, v = dz / safe_x, -dy / safe_x
        elif s == 2: # +Y
            u, v = dx / safe_y, dz / safe_y
        elif s == 3: # -Y
            u, v = dx / safe_y, -dz / safe_y
        elif s == 4: # +Z
            u, v = dx / safe_z, -dy / safe_z
        else:        # -Z
            u, v = -dx / safe_z, -dy / safe_z
        in_face = jt.logical_and(face_masks[s], valid)

        # The one-texel gutter implements nvdiffrast's cross-face bilinear
        # wrapping.  Scale the original face grid so texel centers remain at
        # identical coordinates after padding.
        grid = (jt.stack([u, v], dim=-1) * padded_scale).reshape(1, N, 1, 2)

        face_tex = padded_tex[0, s:s+1]  # [1, H+2, W+2, C]
        face_nchw = face_tex.permute(0, 3, 1, 2)
        sampled = F.grid_sample(face_nchw, grid, mode=mode, padding_mode='border', align_corners=False)
        sampled = sampled[0, :, :, 0].permute(1, 0)  # [N, C]

        mask = in_face.float().unsqueeze(-1)
        result = result + sampled * mask

    # Restore spatial dims
    if len(spatial_shape) >= 2:
        result = result.reshape(*(list(spatial_shape) + [C]))
    else:
        result = result.reshape(1, spatial_shape[0], C)
    return result


def _texture_cube_mip(tex, uv, mip, mip_level_bias, filter_mode):
    """Sample cubemap with trilinear mipmap interpolation."""
    spatial_shape = uv.shape[:-1]
    uv_flat = uv.reshape(-1, 3)
    N = uv_flat.shape[0]
    C = tex.shape[-1]

    # Build mip list
    if mip is None or not isinstance(mip, list):
        mip_list = []
    else:
        mip_list = list(mip)

    num_mips = len(mip_list)
    if num_mips == 0:
        result = _sample_single(tex, uv_flat, 'linear')
        if len(spatial_shape) >= 2:
            return result.reshape(*(list(spatial_shape) + [C]))
        return result.reshape(1, spatial_shape[0], C)

    # Mip level bias: [B, H, W] or [B, 1, H, W] or [N] → [N]
    if mip_level_bias is None:
        lvl = jt.zeros(N)
    else:
        lvl = mip_level_bias.reshape(N)

    max_lvl = float(num_mips)
    lvl = lvl.clamp(0.0, max_lvl)
    lvl_lo = lvl.floor().int()
    lvl_hi = (lvl_lo + 1).clamp(0, int(max_lvl))
    frac = (lvl - lvl_lo.float()).reshape(N, 1)

    # Sample base level (tex = level 0)
    c0 = _sample_single(tex, uv_flat, 'linear')

    # Accumulate per-mip samples
    c_lo = jt.zeros((N, C))
    c_hi = jt.zeros((N, C))

    for li in range(num_mips):
        in_lo = (lvl_lo == (li + 1)).float().unsqueeze(-1)
        in_hi = (lvl_hi == (li + 1)).float().unsqueeze(-1)
        sampled = _sample_single(mip_list[li], uv_flat, 'linear')
        c_lo = c_lo + sampled * in_lo
        c_hi = c_hi + sampled * in_hi

    # Blend: level 0 uses c0, others use c_lo/c_hi
    is_lo_zero = (lvl_lo == 0).float().unsqueeze(-1)
    lo_sample = c0 * is_lo_zero + c_lo * (1.0 - is_lo_zero)
    hi_sample = c_hi
    result = lo_sample * (1.0 - frac) + hi_sample * frac

    # Restore spatial dims
    if len(spatial_shape) >= 2:
        result = result.reshape(*(list(spatial_shape) + [C]))
    else:
        result = result.reshape(1, spatial_shape[0], C)
    return result


def _sample_single(tex, uv_flat, filter_mode):
    """Sample a single cubemap level (no mip blending)."""
    N = uv_flat.shape[0]
    C = tex.shape[-1]
    # Match nvdiffrast's indexCubeMap()/indexCubeMapGrad() contract by using
    # the original lookup vector.  See the non-mip path above.
    dx, dy, dz = uv_flat[..., 0], uv_flat[..., 1], uv_flat[..., 2]
    result = jt.zeros((N, C), dtype=tex.dtype)
    mode = 'bilinear' if 'linear' in filter_mode else 'nearest'
    padded_tex = _pad_cubemap(tex)
    padded_scale = float(tex.shape[2]) / float(tex.shape[2] + 2)

    # Match _texture_cube's mutually-exclusive dominant-axis contract.  The
    # previous six independent >= predicates selected two faces on cube edges
    # and three faces at corners, then added those samples together.  Mipmapped
    # PBR lookups call this helper, so a uniform-one cubemap could return 2 or 3.
    # Equal-component priority is X > Y > Z, matching the non-mip path.
    adx, ady, adz = jt.abs(dx), jt.abs(dy), jt.abs(dz)
    safe_x = jt.where(adx > 0.0, adx, jt.ones_like(adx))
    safe_y = jt.where(ady > 0.0, ady, jt.ones_like(ady))
    safe_z = jt.where(adz > 0.0, adz, jt.ones_like(adz))
    valid = jt.logical_or(jt.logical_or(adx > 0.0, ady > 0.0), adz > 0.0)
    x_dom = jt.logical_and(adx >= ady, adx >= adz)
    y_dom = jt.logical_and(jt.logical_not(x_dom), ady >= adz)
    z_dom = jt.logical_and(jt.logical_not(x_dom), jt.logical_not(y_dom))
    face_masks = [
        jt.logical_and(x_dom, dx >= 0),
        jt.logical_and(x_dom, dx < 0),
        jt.logical_and(y_dom, dy >= 0),
        jt.logical_and(y_dom, dy < 0),
        jt.logical_and(z_dom, dz >= 0),
        jt.logical_and(z_dom, dz < 0),
    ]

    for s in range(6):
        if s == 0:
            u, v = -dz / safe_x, -dy / safe_x
        elif s == 1:
            u, v = dz / safe_x, -dy / safe_x
        elif s == 2:
            u, v = dx / safe_y, dz / safe_y
        elif s == 3:
            u, v = dx / safe_y, -dz / safe_y
        elif s == 4:
            u, v = dx / safe_z, -dy / safe_z
        else:
            u, v = -dx / safe_z, -dy / safe_z
        in_face = jt.logical_and(face_masks[s], valid)

        grid = (jt.stack([u, v], dim=-1) * padded_scale).reshape(1, N, 1, 2)
        face_nchw = padded_tex[0, s:s+1].permute(0, 3, 1, 2)
        sampled = F.grid_sample(face_nchw, grid, mode=mode, padding_mode='border', align_corners=False)
        sampled = sampled[0, :, :, 0].permute(1, 0)
        result = result + sampled * in_face.float().unsqueeze(-1)
    return result


# ---------------------------------------------------------------------------
# Legacy compatibility stubs (unused, kept for import compatibility)
# ---------------------------------------------------------------------------

class TextureMipWrapper:
    """Stub for compatibility with old code that references this class."""
    def __init__(self, **kwargs):
        pass


def texture_construct_mip(texin, max_mip_level=None, cube_mode=False):
    """Stub: mip construction is handled by build_mips() in light.py."""
    return TextureMipWrapper()
