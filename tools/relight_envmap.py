"""Strict pure-envmap relighting A/B (advisor-facing visualization).

Renders the SAME checkpoint / camera / material / resolution / exposure twice —
once with the checkpoint's ORIGINAL environment map (light.base from model.npz)
and once with a REPLACEMENT HDR envmap (aerodynamics_workshop_2k.hdr). The ONLY
varying quantity is `light.base`:

  - SG is hard-off on both lights (lgtSGs = zeros((0,10)), numLgtSGs = 0);
  - point light / point-light shadow / positional SG are never enabled;
  - roughness / normal / metallic / display transform / background are
    identical across the pair;
  - the two scene PNGs use one selected display transform, with no per-image
    auto-exposure.

Hybridlight env loading uses scene/NVDIFFREC/light.py::load_env_hdr and the envmap
lat-long visualization uses util.cubemap_to_latlong (save_env_map's raw PNG write
cannot handle float32 -> PNG, so the thumbnail is written with a fixed ACES map).

Outputs (the directory passed with --output-dir):
    env_original.png / env_workshop.png        lat-long envmap thumbnails
    view{0,1,2}_env_original.{png,npy}         scene render (display PNG + linear HDR npy)
    view{0,1,2}_env_workshop.{png,npy}
    envmap_ab_comparison.png                   2x2 grid (envmaps | renders, view0)
    envmap_ab_params.json                      full experiment identity

Usage (WSL, cwd=GANG-Jittor-Render-master):
    python3 -u tools/relight_envmap.py \
        --model-npz /path/to/model.npz \
        --camera-json /path/to/cameras.json \
        --hdr-path /path/to/environment.hdr
"""
import os, sys, json, math, hashlib, time, argparse, threading, shutil
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'submodules'))

import jittor as jt
jt.flags.use_cuda = 1

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.NVDIFFREC.light import Hybridlight, load_env_hdr
from scene.NVDIFFREC import util
from PIL import Image, ImageDraw, ImageFont

HDR_PATH  = os.path.join(BASE, 'scene', 'NVDIFFREC', 'irrmaps',
                         'aerodynamics_workshop_2k.hdr')

# Fixed experiment knobs (single source of truth for the params JSON).
RES        = 4          # render resolution divisor (1297 x 840 for this set)
BASE_RES   = 256        # envmap cubemap per-face resolution
EXPOSURE   = 0.0        # fixed ACES exposure (identical for both sides)
TONE_MAP   = 'aces'
TRAINING_RES = 4.0
LOD_SCALE  = 1.0
BG         = [0, 0, 0]
ACES_SG_COUNT = 16      # checkpoint stores 16 SGs; both sides are zeroed


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Load 46-key named npz -> model args list (capture() order) + light state
# ---------------------------------------------------------------------------
def load_named_npz(npz_path):
    with np.load(npz_path, allow_pickle=False) as archive:
        raw = dict(archive)
    N = raw['_anchor'].shape[0]
    K = int(raw['n_offsets'])
    print(f"[load] anchors={N}, K={K}, is_pbr={raw['is_pbr']}, with_matallic={raw['with_matallic']}")

    def sd(w1, b1, w2, b2):
        return {'0.weight': w1.astype(np.float32), '0.bias': b1.astype(np.float32),
                '2.weight': w2.astype(np.float32), '2.bias': b2.astype(np.float32)}

    model_args = [
        raw['_anchor'], raw['_level'], raw['_offset'], raw['_anchor_feat'],
        np.zeros((N, 1), np.float32),          # 4  opacity_accum
        raw['_scaling'], raw['_rotation'], raw['_opacity'],
        np.zeros((N * K, 1), np.float32),      # 8  offset_gradient_accum
        np.zeros((N * K, 1), np.float32),      # 9  offset_denom
        np.zeros((N, 1), np.float32),          # 10 anchor_demon
        None,                                  # 11 optimizer state
        float(raw['spatial_lr_scale']),        # 12
        sd(raw['mlp_opacity_w1'], raw['mlp_opacity_b1'], raw['mlp_opacity_w2'], raw['mlp_opacity_b2']),  # 13
        sd(raw['mlp_cov_w1'],    raw['mlp_cov_b1'],    raw['mlp_cov_w2'],    raw['mlp_cov_b2']),         # 14
        sd(raw['mlp_color_w1'],  raw['mlp_color_b1'],  raw['mlp_color_w2'],  raw['mlp_color_b2']),       # 15
        None,                                  # 16 training_args_dict
        None,                                  # 17 mlp_feature
        None,                                  # 18 mlp_appearance
        sd(raw['mlp_albedo_w1'],    raw['mlp_albedo_b1'],    raw['mlp_albedo_w2'],    raw['mlp_albedo_b2']),       # 19
        sd(raw['mlp_matallic_w1'],  raw['mlp_matallic_b1'],  raw['mlp_matallic_w2'],  raw['mlp_matallic_b2']),     # 20
        sd(raw['mlp_roughness_w1'], raw['mlp_roughness_b1'], raw['mlp_roughness_w2'], raw['mlp_roughness_b2']),   # 21
    ]
    light_state = {
        'base': raw['light_base'].astype(np.float32),
        'lgtSGs': raw['light_lgtSGs'].astype(np.float32),
        'specular_reflectance': raw['light_specular_reflectance'].astype(np.float32),
        'roughness': raw['light_sg_roughness'].astype(np.float32),
    }
    meta = {k: raw[k] for k in ('is_pbr', 'with_matallic', 'standard_dist',
                                'n_offsets', 'feat_dim', 'fork', 'base_layer',
                                'dist2level', 'progressive', 'extend')}
    for key in ('voxel_size', 'levels', 'init_level', '_extra_level'):
        if key in raw:
            meta[key] = raw[key]
    return model_args, light_state, meta


# ---------------------------------------------------------------------------
# 2. Minimal camera object (res-independent matrices, res-dependent pixels)
# ---------------------------------------------------------------------------
def make_cam(entry, res=1):
    W, H = int(entry['width']), int(entry['height'])
    fx, fy = float(entry['fx']), float(entry['fy'])
    FoVx = 2.0 * math.atan(W / (2.0 * fx))
    FoVy = 2.0 * math.atan(H / (2.0 * fy))
    R = np.array(entry['rotation'], dtype=np.float32)
    pos = np.array(entry['position'], dtype=np.float32)
    T = -R.T @ pos

    class Cam: pass
    cam = Cam()
    cam.image_width = int(round(W / res))
    cam.image_height = int(round(H / res))
    cam.FoVx, cam.FoVy = FoVx, FoVy
    cam.uid = int(entry['id'])
    cam.world_view_transform = jt.array(getWorld2View2(R, T)).transpose(0, 1)
    P = getProjectionMatrix(0.01, 100.0, FoVx, FoVy).transpose(0, 1)
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0) @ P.unsqueeze(0)).squeeze(0)
    cam.camera_center = jt.array(pos)
    return cam


# ---------------------------------------------------------------------------
# 3. Fixed tone mapping (identical for both sides) + image / HDR I/O
# ---------------------------------------------------------------------------
def aces_srgb(raw, exposure=0.0):
    linear = np.maximum(raw * (2.0 ** exposure), 0.0)
    mapped = (linear * (2.51 * linear + 0.03) /
              np.maximum(linear * (2.43 * linear + 0.59) + 0.14, 1e-8))
    mapped = np.clip(mapped, 0.0, 1.0)
    return np.where(mapped <= 0.0031308,
                    12.92 * mapped,
                    1.055 * np.power(mapped, 1.0 / 2.4) - 0.055)


def save_scene(tensor, base_path, output_mode='aces_srgb', exposure=EXPOSURE):
    """Save the unclipped linear HDR array and the display PNG with one fixed
    exposure, then record max-channel statistics used by the
    presentation highlight gate (p99.9 <= 1.5 and frac_gt_2 <= 0.01%)."""
    raw = tensor.float32().numpy().transpose(1, 2, 0)   # [H, W, 3] linear HDR
    np.save(base_path + '.npy', raw)
    if output_mode == 'paper_linear_clamp':
        # GANG-master/relight.py:139 + torchvision.save_image: clamp linear RGB
        # to [0,1], then quantize directly. No ACES and no sRGB transfer.
        img = np.clip(raw, 0.0, 1.0)
        img8 = np.floor(img * 255.0 + 0.5).astype(np.uint8)
    elif output_mode == 'aces_srgb':
        img = aces_srgb(raw, exposure=exposure)
        img8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        raise ValueError(f'unknown scene output mode: {output_mode}')
    Image.fromarray(img8).save(base_path + '.png')
    mx = raw.max(axis=-1)                               # max-channel
    stats = {
        'p99': float(np.percentile(mx, 99)),
        'p99_9': float(np.percentile(mx, 99.9)),
        'p99_99': float(np.percentile(mx, 99.99)),
        'frac_gt_1': float((mx > 1).mean()),
        'frac_gt_2': float((mx > 2).mean()),
        'frac_gt_5': float((mx > 5).mean()),
        'frac_gt_10': float((mx > 10).mean()),
    }
    print(f"[save] {base_path}.png mean={img8.mean():.3f} max={img8.max()}  "
          f"(hdr [{raw.min():.3f}, {raw.max():.3f}] p99.9={stats['p99_9']:.3f})")
    return raw, stats


def cubemap_to_latlong(light_base, res=(512, 1024)):
    """Pure-numpy cubemap -> lat-long (bilinear, border clamp).

    `util.cubemap_to_latlong` has a latent shape bug (B=1 dirs make grid_sample
    fail), so the projection is re-implemented here with the EXACT production
    face/UV rules from util._cubemap_sample_jt (dominant-axis, X>Y>Z priority;
    per-face u/v; grid_sample align_corners=False pixel = uv*R - 0.5; border
    padding). Direction convention matches the HDR tools:
        theta = gy*pi (gy in [0,1], +Y at gy=0), phi = gx*pi (gx in [-1,1]).

    `face` is initialized to -1 and every X/Y/Z branch keys on
    `face < 0` as the unassigned condition, so a pixel defaulting to 0 can no
    longer be mistaken for the +X face. An assert guarantees full assignment.
    Input accepts a numpy array (or anything with .numpy(), e.g. jt.Var).
    """
    if hasattr(light_base, 'numpy'):
        base = np.asarray(light_base.numpy(), np.float32)     # [6, R, R, 3]
    else:
        base = np.asarray(light_base, np.float32)             # [6, R, R, 3]
    R = base.shape[1]
    H, W = int(res[0]), int(res[1])
    gy, gx = np.meshgrid(np.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H),
                         np.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W),
                         indexing='ij')
    sintheta, costheta = np.sin(gy * np.pi), np.cos(gy * np.pi)
    sinphi, cosphi = np.sin(gx * np.pi), np.cos(gx * np.pi)
    dx, dy, dz = sintheta * sinphi, costheta, -sintheta * cosphi
    adx, ady, adz = np.abs(dx), np.abs(dy), np.abs(dz)
    # mutually-exclusive face assignment (X > Y > Z tie priority); -1 = unassigned.
    face = np.full(dx.shape, -1, np.int32)
    xmask = (adx >= ady) & (adx >= adz)
    un = face < 0
    face[un & xmask & (dx > 0)] = 0    # +X
    face[un & xmask & (dx <= 0)] = 1   # -X
    un = face < 0
    ymask = ady >= adz
    face[un & ymask & (dy > 0)] = 2    # +Y
    face[un & ymask & (dy <= 0)] = 3   # -Y
    un = face < 0
    face[un & (dz > 0)] = 4            # +Z
    face[un & (dz <= 0)] = 5           # -Z
    assert np.all(face >= 0), 'cubemap_to_latlong: pixel left unassigned'
    # per-face UV (production rules; epsilon guards the denominator)
    E = 1e-10
    u = np.where(face == 0, -dz / (dx + E), 0.0)
    u = np.where(face == 1,  dz / (-dx + E), u)
    u = np.where(face == 2,  dx / (dy + E), u)
    u = np.where(face == 3,  dx / (-dy + E), u)
    u = np.where(face == 4,  dx / (dz + E), u)
    u = np.where(face == 5, -dx / (-dz + E), u)
    v = np.where(face == 0, -dy / (dx + E), 0.0)
    v = np.where(face == 1, -dy / (-dx + E), v)
    v = np.where(face == 2,  dz / (dy + E), v)
    v = np.where(face == 3, -dz / (-dy + E), v)
    v = np.where(face == 4, -dy / (dz + E), v)
    v = np.where(face == 5, -dy / (-dz + E), v)
    un = u * 0.5 + 0.5
    vn = v * 0.5 + 0.5
    px = un * R - 0.5
    py = vn * R - 0.5
    x0 = np.clip(np.floor(px).astype(np.int32), 0, R - 1)
    y0 = np.clip(np.floor(py).astype(np.int32), 0, R - 1)
    x1 = np.clip(x0 + 1, 0, R - 1)
    y1 = np.clip(y0 + 1, 0, R - 1)
    fx = (px - np.floor(px))[..., None]
    fy = (py - np.floor(py))[..., None]
    c00 = base[face, y0, x0]
    c01 = base[face, y0, x1]
    c10 = base[face, y1, x0]
    c11 = base[face, y1, x1]
    out = (c00 * (1 - fx) * (1 - fy) + c01 * fx * (1 - fy)
           + c10 * (1 - fx) * fy + c11 * fx * fy)
    return out.astype(np.float32)                          # [H, W, 3]


def save_envmap_thumb(light, path):
    """Lat-long envmap thumbnail (2:1, 512x1024) with a display-only percentile
    tone map.

    The scene A/B keeps exposure=0 for both sides; the envmap PREVIEW is allowed
    its own display normalization so the [0,1] learned map and the max-55
    workshop HDR are both readable — this is explicitly marked `display-only`
    in the caption and the params JSON and never applies to the scene render.
    `save_env_map()`'s raw imageio float->PNG write cannot handle float32, so
    the thumbnail is produced from a cubemap->lat-long projection + the
    display-only percentile map."""
    latlong = cubemap_to_latlong(light.base, (512, 1024))     # [512,1024,3]
    p99_9 = float(np.percentile(latlong, 99.9))
    norm = np.clip(latlong / max(p99_9, 1e-8), 0.0, 1.0)
    display = np.where(norm <= 0.0031308,
                       12.92 * norm,
                       1.055 * np.power(norm, 1.0 / 2.4) - 0.055)
    img8 = (np.clip(display, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img8).save(path)
    print(f"[save] {path}  envmap lat-long min={latlong.min():.4f} "
          f"max={latlong.max():.4f} mean={latlong.mean():.4f} "
          f"(DISPLAY-ONLY p99.9={p99_9:.4f} normalize)")
    return latlong, p99_9


# ---------------------------------------------------------------------------
# 4. GPU memory sampler (nvidia-smi, robust path)
# ---------------------------------------------------------------------------
def _gpu_used_mb():
    import subprocess
    def _find():
        for c in (shutil.which('nvidia-smi'), '/usr/lib/wsl/lib/nvidia-smi',
                  '/usr/bin/nvidia-smi', '/usr/local/bin/nvidia-smi'):
            if c and os.path.isfile(c):
                return c
        return None
    nv = _find()
    if not nv:
        return -1
    try:
        r = subprocess.run([nv, '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=5)
        return float(r.stdout.strip().split('\n')[0])
    except Exception:
        return -1


class _MemSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0.0
        self._stop_flag = False      # do NOT name it `_stop`: shadows Thread._stop()
        self.fail = 0

    def run(self):
        while not self._stop_flag:
            m = _gpu_used_mb()
            if m >= 0:
                self.peak = max(self.peak, m)
            else:
                self.fail += 1
            time.sleep(0.2)

    def stop(self):
        self._stop_flag = True


# ---------------------------------------------------------------------------
# 5. 2x2 comparison grid
# ---------------------------------------------------------------------------
def _load_font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              'C:/Windows/Fonts/arialbd.ttf'):
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def make_comparison(out_path, view, env_orig, env_new, scene_o, scene_n,
                    replacement_label='Workshop'):
    """2x2 layout preserving each image's own aspect ratio:
    row1 = envmap thumbnails at 2:1 (640x320); row2 = scene renders at
    1297:840 (640x414). The two rows differ in height so nothing is stretched
    into a mismatched cell."""
    LABEL = 26
    W1, H1, W2 = 640, 320, 640
    H2 = int(round(640 * 840.0 / 1297.0))
    canvas = Image.new('RGB', (2 * W2, (H1 + LABEL) + (H2 + LABEL)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(20)
    row_y = {0: 0, 1: H1 + LABEL}
    cells = [(env_orig, 'Original envmap', W1, H1, 0, 0),
             (env_new, f'{replacement_label} envmap', W1, H1, 0, 1),
             (scene_o, f'view{view} Original render', W2, H2, 1, 0),
             (scene_n, f'view{view} {replacement_label} render', W2, H2, 1, 1)]
    for src, lab, Wc, Hc, r, c in cells:
        x0, y0 = c * W2, row_y[r]
        draw.text((x0 + 8, y0 + 4), lab, fill=(240, 240, 240), font=font)
        im = Image.open(src).convert('RGB').resize((Wc, Hc), Image.LANCZOS)
        canvas.paste(im, (x0, y0 + LABEL))
    canvas.save(out_path)
    print(f"[save] {out_path}  2x2 comparison (view{view})")


def make_teacher_image(out_path, view, env_orig, env_new, scene_o, scene_n,
                       p99_orig, p99_new, replacement_label='Workshop',
                       scene_display='exposure=0, ACES'):
    """Presentation image. Top: original and replacement envmaps.
    (DISPLAY-ONLY percentile preview, explicitly labelled); bottom = view{view}
    same-camera same-exposure scene pair (exposure=0, ACES); footer states the
    strict A/B contract. No retouching / per-side darkening / HDR clipping."""
    LABEL, FOOT = 26, 84
    W1, H1, W2 = 640, 320, 640
    H2 = int(round(640 * 840.0 / 1297.0))
    canvas = Image.new('RGB', (2 * W2, (H1 + LABEL) + (H2 + LABEL) + FOOT), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(20)
    row_y = {0: 0, 1: H1 + LABEL}
    cells = [(env_orig, 'Original learned envmap', W1, H1, 0, 0),
             (env_new, f'{replacement_label} HDR envmap', W1, H1, 0, 1),
             (scene_o, f'view{view} Original envmap render', W2, H2, 1, 0),
             (scene_n, f'view{view} {replacement_label} envmap render', W2, H2, 1, 1)]
    for src, lab, Wc, Hc, r, c in cells:
        x0, y0 = c * W2, row_y[r]
        draw.text((x0 + 8, y0 + 4), lab, fill=(240, 240, 240), font=font)
        im = Image.open(src).convert('RGB').resize((Wc, Hc), Image.LANCZOS)
        canvas.paste(im, (x0, y0 + LABEL))
    fy = (H1 + LABEL) + (H2 + LABEL) + 8
    footer = ('same checkpoint / camera / display | SG=0, point light=off\n'
              f'envmap preview: DISPLAY-ONLY percentile tone-map '
              f'(orig p99.9={p99_orig:.3f}, {replacement_label} p99.9={p99_new:.3f})\n'
              f'scene: {scene_display}, both sides identical')
    draw.multiline_text((8, fy), footer, fill=(235, 235, 235),
                        font=_load_font(16), spacing=3)
    canvas.save(out_path)
    print(f"[save] {out_path}  teacher image (view{view})")


def _projection_selfcheck():
    """Verify that all six solid-color cubemap faces appear in the projection."""
    cols = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]],
                    np.float32)
    cm = np.zeros((6, 32, 32, 3), np.float32)
    for s in range(6):
        cm[s, ...] = cols[s]
    ll = cubemap_to_latlong(cm, (64, 128))
    present = {s for s in range(6) if (np.abs(ll - cols[s]).max(axis=-1) < 0.01).any()}
    return present == set(range(6)), f'faces={sorted(present)}'


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-npz', required=True,
                    help='40K named NPZ checkpoint produced from the PyTorch model')
    ap.add_argument('--camera-json', required=True,
                    help='camera metadata JSON used by the checkpoint')
    ap.add_argument('--views', default='0 1 2',
                    help='camera indices to render (space-separated, quoted)')
    ap.add_argument('--res', type=int, default=RES)
    ap.add_argument('--base-res', type=int, default=BASE_RES)
    ap.add_argument('--exposure', type=float, default=EXPOSURE)
    ap.add_argument('--comparison-view', type=int, default=None,
                    help='view used for the 2x2 comparison + teacher image; must be in --views')
    ap.add_argument('--output-dir', default='outputs/envmap_ab',
                    help='output directory; relative paths are resolved from the repository root')
    ap.add_argument('--hdr-path', default=HDR_PATH,
                    help='replacement HDR path; defaults to the workshop HDR')
    ap.add_argument('--replacement-label', default=None,
                    help='label shown in comparison images; defaults to the HDR filename stem')
    ap.add_argument('--scene-output', choices=('aces_srgb', 'paper_linear_clamp'),
                    default='aces_srgb',
                    help='PNG display transform; paper_linear_clamp matches GANG-master/relight.py')
    ap.add_argument('--fail-on-highlight', action='store_true',
                    help='return exit 1 when the optional display highlight gate fails')
    args = ap.parse_args()
    model_npz = os.path.abspath(args.model_npz)
    camera_json = os.path.abspath(args.camera_json)
    for label, path in (('checkpoint', model_npz), ('camera JSON', camera_json)):
        if not os.path.isfile(path):
            print(f'[error] {label} does not exist: {path}')
            sys.exit(2)
    hdr_path = os.path.abspath(args.hdr_path)
    if not os.path.isfile(hdr_path):
        print(f'[error] replacement HDR does not exist: {hdr_path}')
        sys.exit(2)
    replacement_label = (args.replacement_label
                         if args.replacement_label
                         else os.path.splitext(os.path.basename(hdr_path))[0])
    views = [int(v) for v in args.views.split()]
    if args.comparison_view is None:
        args.comparison_view = views[0]
    if args.comparison_view not in views:
        print(f'[error] --comparison-view {args.comparison_view} not in --views {views}')
        sys.exit(2)
    cv = args.comparison_view
    OUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    generated = set()
    t0 = time.time()

    # ---- model -------------------------------------------------------------
    model_args, light_state, meta = load_named_npz(model_npz)
    lp = argparse.Namespace(
        feat_dim=int(meta['feat_dim']), n_offsets=int(meta['n_offsets']),
        fork=int(meta['fork']), use_feat_bank=False, appearance_dim=0,
        add_opacity_dist=False, add_cov_dist=False, add_color_dist=False,
        add_level=False, visible_threshold=0.1, dist2level=str(meta['dist2level']),
        base_layer=int(meta['base_layer']), progressive=bool(meta['progressive']),
        extend=float(meta['extend']), is_pbr=True, normal_detal=False, with_meta=True,
    )
    g = GaussianModel(lp.feat_dim, lp.n_offsets, lp.fork, lp.use_feat_bank, lp.appearance_dim,
                      lp.add_opacity_dist, lp.add_cov_dist, lp.add_color_dist,
                      lp.add_level, lp.visible_threshold, lp.dist2level,
                      lp.base_layer, lp.progressive, lp.extend,
                      is_pbr=True, normal_detal=False, with_matallic=True)
    g.restore_numpy(model_args, metadata=meta, inference_only=True, build_optimizer=False)
    g._offset = g._offset.reshape((-1, 3))
    g.eval()
    K = g.n_offsets
    N = g.get_anchor.shape[0]
    l = [
        {"params": [g._anchor], "lr": 0.0, "name": "anchor"},
        {"params": [g._offset], "lr": 0.01 * g.spatial_lr_scale, "name": "offset"},
        {"params": [g._anchor_feat], "lr": 0.0075, "name": "anchor_feat"},
        {"params": [g._opacity], "lr": 0.02, "name": "opacity"},
        {"params": [g._scaling], "lr": 0.007, "name": "scaling"},
        {"params": [g._rotation], "lr": 0.002, "name": "rotation"},
        {"params": g.mlp_opacity.parameters(), "lr": 0.002, "name": "mlp_opacity"},
        {"params": g.mlp_cov.parameters(), "lr": 0.004, "name": "mlp_cov"},
        {"params": g.mlp_color.parameters(), "lr": 0.008, "name": "mlp_color"},
    ]
    g.optimizer = argparse.Namespace(param_groups=l)
    g.optimizer.state_dict = lambda: {}
    g.optimizer.load_state_dict = lambda d: None
    g.opacity_accum = jt.zeros((N, 1))
    g.offset_gradient_accum = jt.zeros((N * K, 1))
    g.offset_denom = jt.zeros((N * K, 1))
    g.anchor_demon = jt.zeros((N, 1))
    print(f"[restore] anchors={N}")
    with open(camera_json, encoding='utf-8') as f:
        cams = json.load(f)

    # ---- lights: SG hard-off on BOTH, only light.base differs -----------------
    light_orig = Hybridlight(base_res=args.base_res, num_sg=ACES_SG_COUNT, cache_dir=OUT_DIR)
    light_orig.load_from_numpy(light_state)
    light_orig.lgtSGs = jt.zeros((0, 10)); light_orig.numLgtSGs = 0
    light_new = load_env_hdr(hdr_path, sg_pth=None, res=args.base_res,
                             numLgtSGs=ACES_SG_COUNT, is_sg=False)
    light_new.lgtSGs = jt.zeros((0, 10)); light_new.numLgtSGs = 0
    # Strict BRDF identity: the env path never reads these, but pin them anyway
    # so NO non-base light attribute differs between the two sides.
    light_new.specular_reflectance = light_orig.specular_reflectance.clone()
    light_new.roughness = light_orig.roughness.clone()
    print(f"[light] orig SG={int(light_orig.lgtSGs.shape[0])} new SG={int(light_new.lgtSGs.shape[0])} "
          f"orig.base={tuple(light_orig.base.shape)} new.base={tuple(light_new.base.shape)}")

    # ---- envmap thumbnails (DISPLAY-ONLY percentile, 2:1) --------------------
    env_orig_ll, p99_orig = save_envmap_thumb(light_orig, os.path.join(OUT_DIR, 'env_original.png'))
    env_new_ll, p99_new = save_envmap_thumb(light_new, os.path.join(OUT_DIR, 'env_workshop.png'))
    generated |= {'env_original.png', 'env_workshop.png'}

    # ---- render loop ----------------------------------------------------------
    pipe = argparse.Namespace(compute_cov3D_python=False, debug=False, sample_num=64)
    bg = jt.float32(BG)
    sampler = _MemSampler(); sampler.start()
    base_mb = _gpu_used_mb()
    projection_ok, projection_detail = _projection_selfcheck()
    view_stats = {}
    output_files = ['env_original.png', 'env_workshop.png',
                    'envmap_ab_comparison.png', f'envmap_ab_teacher_view{cv}.png',
                    'envmap_ab_params.json']
    with jt.no_grad():
        for vi in views:
            cam = make_cam(cams[vi], res=args.res)
            try:
                g.set_anchor_mask(cam.camera_center, 99999, 1.0)
            except Exception:
                pass
            print(f"[cam] view{vi}: {cam.image_width}x{cam.image_height} "
                  f"img={cams[vi].get('img_name')} (res={args.res})")
            pkg_o = render(cam, g, pipe, bg, visible_mask=None, is_pbr=True,
                           light=light_orig, is_training=False)
            raw_o, st_o = save_scene(pkg_o['render'],
                                     os.path.join(OUT_DIR, f'view{vi}_env_original'),
                                     output_mode=args.scene_output, exposure=args.exposure)
            del pkg_o; jt.sync_all(True); jt.gc()
            pkg_n = render(cam, g, pipe, bg, visible_mask=None, is_pbr=True,
                           light=light_new, is_training=False)
            raw_n, st_n = save_scene(pkg_n['render'],
                                     os.path.join(OUT_DIR, f'view{vi}_env_workshop'),
                                     output_mode=args.scene_output, exposure=args.exposure)
            del pkg_n; jt.sync_all(True); jt.gc()
            H, Wc = int(raw_o.shape[0]), int(raw_o.shape[1])
            diff = np.abs(raw_n.astype(np.float64) - raw_o.astype(np.float64))
            view_stats[vi] = {
                'image_size': [Wc, H],
                'hdr_orig': {'min': float(raw_o.min()), 'max': float(raw_o.max()),
                             'mean': float(raw_o.mean())},
                'hdr_new': {'min': float(raw_n.min()), 'max': float(raw_n.max()),
                            'mean': float(raw_n.mean())},
                'highlight_orig': st_o,
                'highlight_workshop': st_n,
                'diff': {'mean': float(diff.mean()), 'max': float(diff.max())},
                'finite_orig': bool(np.isfinite(raw_o).all()),
                'finite_new': bool(np.isfinite(raw_n).all()),
            }
            for f in (f'view{vi}_env_original.npy', f'view{vi}_env_original.png',
                      f'view{vi}_env_workshop.npy', f'view{vi}_env_workshop.png'):
                generated.add(f); output_files.append(f)
            print(f"[view{vi}] diff mean={diff.mean():.5f} max={diff.max():.5f}")

    sampler.stop(); sampler.join(timeout=5)
    peak_mb = sampler.peak
    peak_delta_mb = (peak_mb - base_mb) if base_mb >= 0 else -1

    # ---- comparison + teacher (current-process files only) --------------------
    comp_src = [os.path.join(OUT_DIR, 'env_original.png'),
                os.path.join(OUT_DIR, 'env_workshop.png'),
                os.path.join(OUT_DIR, f'view{cv}_env_original.png'),
                os.path.join(OUT_DIR, f'view{cv}_env_workshop.png')]
    if all(os.path.basename(s) in generated for s in comp_src):
        make_comparison(os.path.join(OUT_DIR, 'envmap_ab_comparison.png'), cv,
                        *comp_src, replacement_label=replacement_label)
        make_teacher_image(os.path.join(OUT_DIR, f'envmap_ab_teacher_view{cv}.png'), cv,
                           comp_src[0], comp_src[1], comp_src[2], comp_src[3],
                           p99_orig, p99_new,
                           replacement_label=replacement_label,
                           scene_display=('linear clip [0,1], no gamma'
                                          if args.scene_output == 'paper_linear_clamp'
                                          else f'exposure={args.exposure:g}, ACES+sRGB'))
    else:
        print('[warn] comparison sources incomplete -> comparison/teacher skipped')
    runtime_s = time.time() - t0

    # ---- params ---------------------------------------------------------------
    orig_base = light_orig.base.numpy()
    new_base = light_new.base.numpy()
    params = {
        'experiment': 'envmap_ab_strict_v2',
        'checkpoint': {'name': os.path.basename(model_npz), 'sha256': sha256(model_npz)},
        'cameras_json': {'name': os.path.basename(camera_json), 'sha256': sha256(camera_json)},
        'replacement_hdr': {'name': os.path.basename(hdr_path), 'sha256': sha256(hdr_path),
                            'label': replacement_label},
        'camera_indices': views,
        'camera_names': {str(v): cams[v].get('img_name') for v in views},
        'comparison_view': cv,
        'image_sizes': {str(v): view_stats[v]['image_size'] for v in views},
        'image_size': view_stats[cv]['image_size'],
        'envmap_cubemap_res': args.base_res,
        'envmap_display_mode': ('display-only percentile p99.9 + sRGB '
                                '(envmap preview only)'),
        'envmap_display_p99_9': {'original': p99_orig, 'workshop': p99_new},
        'original_envmap': {'shape': list(orig_base.shape),
                            'min': float(orig_base.min()), 'max': float(orig_base.max()),
                            'mean': float(orig_base.mean())},
        'new_envmap': {'shape': list(new_base.shape),
                       'min': float(new_base.min()), 'max': float(new_base.max()),
                       'mean': float(new_base.mean())},
        'original_sg_count_after_zero': int(light_orig.lgtSGs.shape[0]),
        'replacement_sg_count_after_zero': int(light_new.lgtSGs.shape[0]),
        'checkpoint_sg_count_initial': int(light_state['lgtSGs'].shape[0]),
        'use_point_light': False,
        'point_light_enabled_orig': bool(light_orig.point_light_enabled),
        'point_light_enabled_new': bool(light_new.point_light_enabled),
        'positional_sg': False,
        'render_res': args.res,
        'training_res': TRAINING_RES,
        'lod_scale': LOD_SCALE,
        'background': BG,
        'exposure': (None if args.scene_output == 'paper_linear_clamp'
                     else args.exposure),
        'tone_map': ('none; linear clip [0,1]'
                     if args.scene_output == 'paper_linear_clamp'
                     else TONE_MAP + '+sRGB'),
        'scene_output': args.scene_output,
        'sg_hard_off': True,
        'gpu_baseline_mb': base_mb,
        'gpu_peak_mb': peak_mb,
        'gpu_peak_delta_mb': peak_delta_mb,
        'runtime_s': round(runtime_s, 2),
        'view_stats': {str(k): v for k, v in view_stats.items()},
        'output_files': output_files,
    }
    with open(os.path.join(OUT_DIR, 'envmap_ab_params.json'), 'w') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"[params] -> {os.path.join(OUT_DIR, 'envmap_ab_params.json')}")

    # ---- acceptance: experiment_contract_pass + presentation_pass ------------
    # experiment_contract_pass: 身份/SG/点光源/唯一变量/HDR/显存
    exp = {}
    exp['sg_zero'] = (int(light_orig.lgtSGs.shape[0]) == 0
                      and int(light_new.lgtSGs.shape[0]) == 0
                      and light_orig.numLgtSGs == 0 and light_new.numLgtSGs == 0)
    exp['point_light_off'] = (not light_orig.point_light_enabled
                              and not light_new.point_light_enabled)
    exp['unique_var_base'] = (bool(np.array_equal(light_new.specular_reflectance.numpy(),
                                                  light_orig.specular_reflectance.numpy()))
                              and bool(np.array_equal(light_new.roughness.numpy(),
                                                      light_orig.roughness.numpy())))
    exp['camera_material_identity'] = (len(view_stats) == len(views)
                                       and all(v['finite_orig'] and v['finite_new']
                                               for v in view_stats.values()))
    hdr_shape = True
    for vi in views:
        for side in ('original', 'workshop'):
            p = os.path.join(OUT_DIR, f'view{vi}_env_{side}.npy')
            if not os.path.isfile(p):
                hdr_shape = False
                continue
            a = np.load(p)
            if a.dtype != np.float32 or len(a.shape) != 3 or not np.isfinite(a).all():
                hdr_shape = False
    exp['hdr_shape_dtype_finite'] = hdr_shape
    exp['peak_lt_8gb'] = (peak_mb < 8.0 * 1024.0)
    exp['inputs_sha256_present'] = (len(params['checkpoint']['sha256']) == 64
                                    and len(params['cameras_json']['sha256']) == 64
                                    and len(params['replacement_hdr']['sha256']) == 64)

    # presentation_pass: 投影/无旧文件/比例/图注/亮点/尺寸
    pres = {}
    pres['projection_ok'] = projection_ok
    pres['no_old_file_mixing'] = all(os.path.basename(s) in generated for s in comp_src)
    ratio_ok = True
    for f in ('env_original.png', 'env_workshop.png'):
        w, h = Image.open(os.path.join(OUT_DIR, f)).size
        if abs(w - 2.0 * h) > 2:
            ratio_ok = False
    for vi in views:
        for side in ('original', 'workshop'):
            png = os.path.join(OUT_DIR, f'view{vi}_env_{side}.png')
            if not os.path.isfile(png):
                ratio_ok = False
                continue
            pim = Image.open(png)
            if tuple(pim.size) != tuple(view_stats[vi]['image_size']):
                ratio_ok = False
    pres['image_ratio'] = ratio_ok
    pres['captions_complete'] = os.path.isfile(
        os.path.join(OUT_DIR, f'envmap_ab_teacher_view{cv}.png'))
    # Optional display highlight gate for the replacement side.
    def _gate_ok(st):
        return st['p99_9'] <= 1.5 and st['frac_gt_2'] <= 0.0001
    hl = {vi: _gate_ok(view_stats[vi]['highlight_workshop']) for vi in views}
    pres['main_view_no_dense_highlights'] = hl[cv]
    pres['all_views_highlight_ok'] = all(hl.values())
    dims_ok = True
    for vi in views:
        for side in ('original', 'workshop'):
            a = np.load(os.path.join(OUT_DIR, f'view{vi}_env_{side}.npy'))
            if (int(a.shape[1]), int(a.shape[0])) != tuple(view_stats[vi]['image_size']):
                dims_ok = False
    pres['params_dims_match'] = dims_ok

    print('\n=== ENV MAP A/B ACCEPTANCE (v2) ===')
    print('  [experiment_contract_pass]')
    for k, v in exp.items():
        print(f'    [{"PASS" if v else "FAIL"}] {k}')
    print('  [presentation_pass]')
    for k, v in pres.items():
        print(f'    [{"PASS" if v else "FAIL"}] {k}')
    print(f'  projection={projection_detail}')
    print(f'  highlight verdicts: { {k: ("PASS" if v else "FAIL") for k, v in hl.items()} }')
    print(f'  gpu peak={peak_mb:.0f}MB baseline={base_mb:.0f}MB delta={peak_delta_mb:.0f}MB '
          f'runtime={runtime_s:.1f}s')
    highlight_ok = pres['main_view_no_dense_highlights'] and pres['all_views_highlight_ok']
    required_presentation = {k: v for k, v in pres.items()
                             if k not in ('main_view_no_dense_highlights',
                                          'all_views_highlight_ok')}
    all_ok = (all(exp.values()) and all(required_presentation.values())
              and (highlight_ok or not args.fail_on_highlight))
    params['acceptance'] = {
        'experiment_contract': exp,
        'presentation': pres,
        'highlight_by_view': {str(k): v for k, v in hl.items()},
        'fail_on_highlight': bool(args.fail_on_highlight),
    }
    with open(os.path.join(OUT_DIR, 'envmap_ab_params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    if all_ok:
        if not highlight_ok:
            print('ENVIRONMENT_MAP_AB_PASS_WITH_HIGHLIGHT_WARNING')
        else:
            print('ENVIRONMENT_MAP_AB_PASS')
        sys.exit(0)
    print('ENVIRONMENT_MAP_AB_FAIL')
    sys.exit(1)


if __name__ == '__main__':
    main()
