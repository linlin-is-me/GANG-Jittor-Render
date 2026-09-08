"""Render with the checkpoint's TRAINED light (envmap base + learned SGs).

Question this answers: can we load the SG learned at training time and render a
scene with it?

The checkpoint npz stores the fully-trained Hybridlight state:
    light_base                    [6, 256, 256, 3]  learned envmap cubemap
    light_lgtSGs                  [16, 10]          learned Spherical Gaussians
    light_specular_reflectance    [1, 3]
    light_sg_roughness            [1, 1]

This script restores that exact state on a Hybridlight and runs the SAME
inference path used at training time — is_pbr=True, SG enabled (numLgtSGs=16),
no point light, fixed ACES exposure. It does NOT zero the SGs and does NOT
substitute a synthetic directional SG, so the output is the model's own
training-time appearance (the envmap A/B task hard-off SG by design; this is
the SG-on complement).

Model, camera and render helpers are shared with tools/relight_envmap.py.

Outputs (the directory passed with --output-dir):
    env_original.png         trained envmap lat-long thumbnail (display-only p99.9)
    sg_lobes.png             learned 16 SG energy over the sphere (display-only p99.9)
    view{N}_sg.png / .npy    scene render (fixed-exposure ACES png + linear HDR npy)
    trained_sg_params.json   full render identity

Usage (WSL, cwd=GANG-Jittor-Render-master):
    python3 -u tools/render_learned_light.py \
        --model-npz /path/to/model.npz \
        --camera-json /path/to/cameras.json \
        --views "0 1 2" --res 4
"""
import os, sys, json, time, argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'submodules'))

import jittor as jt
jt.flags.use_cuda = 1

from scene.NVDIFFREC.light import Hybridlight
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from PIL import Image

# Reuse the tested model/camera/light helpers from the envmap A/B tool.
from relight_envmap import (load_named_npz, make_cam, aces_srgb,
                            cubemap_to_latlong, save_envmap_thumb,
                            _MemSampler, _gpu_used_mb, sha256)

TRAINING_SG_COUNT = 16      # the checkpoint was trained with 16 SGs
BG = [0, 0, 0]
TONE_MAP = 'aces'


def save_scene(tensor, base_path, exposure=0.0):
    """Linear HDR .npy (unclipped) + fixed-exposure ACES PNG + max-channel
    highlight stats (p99.9 / frac_gt_2 …), matching the envmap A/B contract."""
    raw = tensor.float32().numpy().transpose(1, 2, 0)      # [H, W, 3] linear HDR
    np.save(base_path + '.npy', raw)
    img = aces_srgb(raw, exposure=exposure)
    img8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img8).save(base_path + '.png')
    mx = raw.max(axis=-1)
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


def save_sg_lobes(light, path, res=(256, 512)):
    """Learned SG energy over the full sphere (display-only percentile PNG).
    Pure diagnostic — never affects the scene render."""
    if os.path.isfile(path):
        os.remove(path)
    try:
        env = light.compute_SG_envmap(return_img=True, res=list(res)).float32().numpy()
        p99_9 = float(np.percentile(env, 99.9))
        norm = np.clip(env / max(p99_9, 1e-8), 0.0, 1.0)
        img8 = (norm * 255).astype(np.uint8)
        Image.fromarray(img8).save(path)
        print(f"[save] {path}  SG energy lat-long (DISPLAY-ONLY p99.9={p99_9:.4f})")
        return p99_9
    except Exception as e:
        print(f"[warn] sg_lobes skipped: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-npz', required=True,
                    help='40K named NPZ checkpoint produced from the PyTorch model')
    ap.add_argument('--camera-json', required=True,
                    help='camera metadata JSON used by the checkpoint')
    ap.add_argument('--views', default='0 1 2',
                    help='camera indices to render (space-separated, quoted)')
    ap.add_argument('--res', type=int, default=4)
    ap.add_argument('--sg-reduce-backend', choices=('native', 'vector3_cuda'),
                    default='native', help='Inference-only SG reduction backend')
    ap.add_argument('--base-res', type=int, default=256)
    ap.add_argument('--exposure', type=float, default=0.0)
    ap.add_argument('--output-dir', default='outputs/learned_light',
                    help='output directory; relative paths are resolved from the repository root')
    ap.add_argument('--no-sg-lobes', action='store_true',
                    help='skip the SG energy visualization')
    args = ap.parse_args()
    model_npz = os.path.abspath(args.model_npz)
    camera_json = os.path.abspath(args.camera_json)
    for label, path in (('checkpoint', model_npz), ('camera JSON', camera_json)):
        if not os.path.isfile(path):
            print(f'[error] {label} does not exist: {path}')
            sys.exit(2)
    views = [int(v) for v in args.views.split()]
    OUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # ---- model (same restore as envmap A/B) ----------------------------------
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
    N = g.get_anchor.shape[0]
    print(f"[restore] anchors={N}")
    with open(camera_json, encoding='utf-8') as f:
        cams = json.load(f)

    # ---- light: restore the TRAINED light, KEEP the learned SGs ---------------
    light = Hybridlight(base_res=args.base_res, num_sg=TRAINING_SG_COUNT,
                        cache_dir=OUT_DIR)
    light.sg_reduce_backend = args.sg_reduce_backend
    light.load_from_numpy(light_state)              # base + lgtSGs + BRDF params
    sg = light.lgtSGs.numpy()
    mu = np.abs(sg[:, -3:])
    lam = np.abs(sg[:, 6:7])
    light.build_mips()
    print(f"[light] SG count={int(light.numLgtSGs)} (trained {TRAINING_SG_COUNT}) "
          f"base={tuple(light.base.shape)} finite={bool(np.isfinite(sg).all())}")
    print(f"[light] SG lambda [{lam.min():.2f}, {lam.max():.2f}] "
          f"mu-total={np.sum(mu):.4f} mu-max={np.max(mu):.4f}")

    # ---- envmap thumbnail + (optional) SG energy ------------------------------
    env_ll, p99_env = save_envmap_thumb(light, os.path.join(OUT_DIR, 'env_original.png'))
    p99_sg = None if args.no_sg_lobes else save_sg_lobes(
        light, os.path.join(OUT_DIR, 'sg_lobes.png'))

    # ---- render loop (SG enabled, no point light) ------------------------------
    pipe = argparse.Namespace(compute_cov3D_python=False, debug=False, sample_num=64)
    bg = jt.float32(BG)
    sampler = _MemSampler(); sampler.start()
    base_mb = _gpu_used_mb()
    view_stats = {}
    output_files = ['env_original.png', 'trained_sg_params.json']
    if p99_sg is not None:
        output_files.append('sg_lobes.png')
    with jt.no_grad():
        for vi in views:
            cam = make_cam(cams[vi], res=args.res)
            try:
                g.set_anchor_mask(cam.camera_center, 99999, 1.0)
            except Exception:
                pass
            print(f"[cam] view{vi}: {cam.image_width}x{cam.image_height} "
                  f"img={cams[vi].get('img_name')} (res={args.res})")
            pkg = render(cam, g, pipe, bg, visible_mask=None, is_pbr=True,
                         light=light, is_training=False)
            raw, st = save_scene(pkg['render'], os.path.join(OUT_DIR, f'view{vi}_sg'),
                                 exposure=args.exposure)
            del pkg; jt.sync_all(True); jt.gc()
            H, Wc = int(raw.shape[0]), int(raw.shape[1])
            view_stats[vi] = {
                'image_size': [Wc, H],
                'hdr': {'min': float(raw.min()), 'max': float(raw.max()),
                        'mean': float(raw.mean())},
                'highlight': st,
                'finite': bool(np.isfinite(raw).all()),
            }
            for f in (f'view{vi}_sg.npy', f'view{vi}_sg.png'):
                output_files.append(f)

    sampler.stop(); sampler.join(timeout=5)
    peak_mb = sampler.peak
    peak_delta_mb = (peak_mb - base_mb) if base_mb >= 0 else -1

    # ---- params ----------------------------------------------------------------
    params = {
        'experiment': 'trained_sg_render',
        'checkpoint': {'name': os.path.basename(model_npz), 'sha256': sha256(model_npz)},
        'cameras_json': {'name': os.path.basename(camera_json), 'sha256': sha256(camera_json)},
        'camera_indices': views,
        'image_sizes': {str(v): view_stats[v]['image_size'] for v in views},
        'sg_count': int(light.numLgtSGs),
        'sg_reduce_backend': args.sg_reduce_backend,
        'sg_trained': TRAINING_SG_COUNT,
        'sg_stats': {'lambda_min': float(lam.min()), 'lambda_max': float(lam.max()),
                     'mu_total': float(np.sum(mu)), 'mu_max': float(np.max(mu)),
                     'finite': bool(np.isfinite(sg).all())},
        'envmap': {'shape': list(light.base.shape),
                   'min': float(light.base.numpy().min()),
                   'max': float(light.base.numpy().max()),
                   'mean': float(light.base.numpy().mean())},
        'envmap_display_p99_9': p99_env,
        'sg_lobes_display_p99_9': p99_sg,
        'use_point_light': bool(light.point_light_enabled),
        'positional_sg': bool(getattr(light, 'sg_distance_attenuation', True)),
        'sg_min_roughness': float(getattr(light, 'sg_min_roughness', 1e-5)),
        'render_res': args.res,
        'training_res': 4.0,
        'exposure': args.exposure,
        'tone_map': TONE_MAP,
        'background': BG,
        'gpu_baseline_mb': base_mb,
        'gpu_peak_mb': peak_mb,
        'gpu_peak_delta_mb': peak_delta_mb,
        'runtime_s': round(time.time() - t0, 2),
        'view_stats': {str(k): v for k, v in view_stats.items()},
        'output_files': output_files,
    }
    with open(os.path.join(OUT_DIR, 'trained_sg_params.json'), 'w') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"[params] -> {os.path.join(OUT_DIR, 'trained_sg_params.json')}")

    # ---- acceptance -------------------------------------------------------------
    checks = {}
    checks['sg_enabled'] = (int(light.numLgtSGs) == TRAINING_SG_COUNT
                            and int(sg.shape[0]) == TRAINING_SG_COUNT
                            and bool(np.isfinite(sg).all()))
    checks['point_light_off'] = not light.point_light_enabled
    checks['all_views_finite'] = all(v['finite'] for v in view_stats.values())
    dims_ok = True
    for vi in views:
        a = np.load(os.path.join(OUT_DIR, f'view{vi}_sg.npy'))
        if (int(a.shape[1]), int(a.shape[0])) != tuple(view_stats[vi]['image_size']):
            dims_ok = False
    checks['image_size_consistent'] = dims_ok
    checks['inputs_sha256_present'] = (len(params['checkpoint']['sha256']) == 64
                                       and len(params['cameras_json']['sha256']) == 64)
    checks['peak_lt_8gb'] = (peak_mb < 8.0 * 1024.0)

    print('\n=== TRAINED SG RENDER ACCEPTANCE ===')
    for k, v in checks.items():
        print(f'  [{"PASS" if v else "FAIL"}] {k}')
    print(f'  gpu peak={peak_mb:.0f}MB baseline={base_mb:.0f}MB delta={peak_delta_mb:.0f}MB '
          f'runtime={params["runtime_s"]:.1f}s')
    all_ok = all(checks.values())
    print('TRAINED_SG_RENDER_PASS' if all_ok else 'TRAINED_SG_RENDER_FAIL')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
