"""Independent-process 40K inference worker; select source root before imports.

All variants use the same named model, camera arrays and synchronized timing.
PyTorch additionally verifies its original PTH and light against that NPZ.
"""
from argparse import ArgumentParser, Namespace
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import inspect
import numpy as np

from inference_named_checkpoint import load_named_checkpoint, restore_jittor_model
from inference_raster_benchmark import RasterCapture, run_raster, save_image


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError('timings must be finite and positive')
    return {'count': len(a), 'mean_ms': float(a.mean()),
            'p50_ms': float(np.percentile(a, 50)),
            'p95_ms': float(np.percentile(a, 95)), 'fps': float(1000 / a.mean())}


def camera_arrays(entry):
    """One CPU representation distributed unchanged to both backends."""
    rotation = np.asarray(entry['rotation'], dtype=np.float64)
    position = np.asarray(entry['position'], dtype=np.float64)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = rotation, position
    view = np.linalg.inv(c2w).astype(np.float32).T.copy()
    fovx = 2 * math.atan(entry['width'] / (2 * entry['fx']))
    fovy = 2 * math.atan(entry['height'] / (2 * entry['fy']))
    near, far = .01, 100.
    p = np.zeros((4, 4), np.float32)
    p[0, 0], p[1, 1] = 1 / math.tan(fovx / 2), 1 / math.tan(fovy / 2)
    p[3, 2], p[2, 2], p[2, 3] = 1, far / (far-near), -far*near/(far-near)
    return {'world_view_transform': view,
            'full_proj_transform': np.ascontiguousarray(view @ p.T),
            'camera_center': position.astype(np.float32)}, fovx, fovy


def assert_equal(actual, expected, name):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
        raise ValueError(f'checkpoint mismatch: {name}; shapes {a.shape}/{b.shape}, dtypes {a.dtype}/{b.dtype}')


def verify_pt_state(state, payload, numpy):
    if len(state) != 21:
        raise ValueError(f'expected original 21-item PBR capture, got {len(state)}')
    for i in (0, 2, 3, 5, 6, 7):
        assert_equal(numpy(state[i]), payload[i], f'geometry[{i}]')
    # The historical PTH stores integral LOD labels as FP32; named NPZ uses INT32.
    level = numpy(state[1])
    if not np.isfinite(level).all() or not np.array_equal(level, payload[1]):
        raise ValueError('checkpoint mismatch: LOD level values')
    if float(state[12]) != payload[12]:
        raise ValueError('spatial_lr_scale mismatch')
    # Historical capture order (documented Phase 94): albedo, roughness, metallic.
    for pt, jt in ((13, 13), (14, 14), (15, 15), (18, 19), (19, 21), (20, 20)):
        if set(state[pt]) != set(payload[jt]):
            raise ValueError(f'MLP keys mismatch: {pt}')
        for key in state[pt]:
            assert_equal(numpy(state[pt][key]), payload[jt][key], f'mlp[{pt}].{key}')


class MemorySampler:
    """Driver memory sampling; explicitly not an exact instantaneous peak."""
    def __init__(self):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        if ',' in device:
            raise ValueError('select one GPU with CUDA_VISIBLE_DEVICES')
        self.handle = (pynvml.nvmlDeviceGetHandleByIndex(int(device)) if device.isdigit()
                       else pynvml.nvmlDeviceGetHandleByUUID(device))
        self.samples = []
        self.stop = threading.Event()
        self.error = None
        self.name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(self.name, bytes):
            self.name = self.name.decode()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            while not self.stop.is_set():
                procs = self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle)
                if any(p.pid != os.getpid() for p in procs):
                    raise RuntimeError('external GPU compute process detected')
                self.samples.append(int(self.nv.nvmlDeviceGetMemoryInfo(self.handle).used))
                self.stop.wait(.1)
        except Exception as exc:
            self.error = str(exc)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.nv.nvmlShutdown()


def measure(render, sync, host):
    sync()
    start = time.perf_counter()
    package = render()
    sync()
    forward_end = time.perf_counter()
    image = host(package['render'])
    sync()
    copy_end = time.perf_counter()
    if not np.isfinite(image).all():
        raise FloatingPointError('nonfinite rendered image')
    return image, (forward_end-start)*1000, (copy_end-start)*1000


def run(args):
    root = args.source_root.resolve()
    weights = args.weights.resolve()
    contract = json.loads(args.contract.read_text())
    views = contract['views'][:3] if args.smoke else contract['views']
    if args.view_name:
        views = [v for v in contract['views'] if v['name'] == args.view_name]
        if len(views) != 1:
            raise ValueError('requested view is not unique in contract')
    if contract['resolution'] != 4 or contract['anchor_count'] != 597027:
        raise ValueError('unexpected benchmark contract')
    os.environ['GANG_RASTER_BACKWARD_VARIANT'] = 'stable'
    os.chdir(root)
    # Remove the worktree root and script directory from module resolution so
    # imported renderer/scene/utils exclusively belong to the selected source.
    own_root = Path(__file__).resolve().parents[1]
    source_paths = [str(root)]
    if args.framework == 'jittor':
        source_paths.append(str(root / 'submodules'))
    sys.path[:] = source_paths + [p for p in sys.path
                     if p and Path(p).resolve() not in (own_root, own_root / 'tools', Path.cwd())]
    if args.framework == 'pytorch':
        import torch as backend
        backend.manual_seed(42)
        backend.backends.cuda.matmul.allow_tf32 = False
        backend.backends.cudnn.allow_tf32 = False
        backend.backends.cudnn.benchmark = False
        tensor = lambda a: backend.as_tensor(np.ascontiguousarray(a), device='cuda')
        host = lambda t: t.detach().cpu().numpy()
        sync = backend.cuda.synchronize
    else:
        import jittor as backend
        backend.flags.use_cuda = 1
        backend.set_global_seed(42)
        tensor = backend.array
        host = lambda t: t.numpy()
        sync = lambda: backend.sync_all(True)
    import gaussian_renderer as renderer
    if args.mode == 'raster':
        return run_raster(args, renderer, backend, tensor, host, sync, views,
                          MemorySampler, measure, summary)
    if args.mode == 'rgb' and 'return_aux' not in inspect.signature(renderer.render).parameters:
        report = {'status': 'unsupported', 'mode': 'rgb', 'label': args.label,
                  'framework': args.framework,
                  'reason': 'No native RGB-only switch; original renderer remains unchanged. Use full path as available baseline.'}
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return
    from scene.gaussian_model import GaussianModel
    from scene.NVDIFFREC.light import Hybridlight
    for module in (renderer, sys.modules[GaussianModel.__module__], sys.modules[Hybridlight.__module__]):
        Path(module.__file__).resolve().relative_to(root)
    payload, light_state, metadata, config = load_named_checkpoint(weights / 'model.npz', weights / 'outputs.log')
    common = {k: config[k] for k in ('feat_dim', 'n_offsets', 'fork', 'base_layer',
                                    'dist2level', 'progressive', 'extend', 'is_pbr', 'with_matallic')}
    common.update(use_feat_bank=False, appearance_dim=0, add_opacity_dist=False,
                  add_cov_dist=False, add_color_dist=False, add_level=False,
                  visible_threshold=.1, normal_detal=False)
    model = GaussianModel(**common)
    with backend.no_grad():
        if args.framework == 'pytorch':
            if getattr(args, 'sg_reduce_backend', 'native') != 'native':
                raise ValueError('SG optimization is local Jittor only')
            state, iteration = backend.load(weights / 'chkpnt40000.pth', map_location='cuda', weights_only=False)
            if iteration != 40000:
                raise ValueError('not a 40K checkpoint')
            if args.verify_weights or args.load_only:
                verify_pt_state(state, payload, host)
            # Adapt the historical capture layout to this pinned repository's
            # restore layout; never rewrite checkpoint or learned values.
            restore_state = list(state)
            restore_state[19], restore_state[20] = state[20], state[19]
            model.restore(restore_state)
            for key, value in metadata.items():
                setattr(model, key, tensor(value) if isinstance(value, np.ndarray) else value)
            model.eval()
            del state, restore_state
            light = Hybridlight(base_res=256, num_sg=16)
            light.load_light(str(weights / 'Hybridlight40000.npy'))
        else:
            restore_jittor_model(model, payload, metadata)
            light = Hybridlight(base_res=256, num_sg=16, cache_dir=str(args.output))
            light.load_from_numpy(light_state)
            sg_backend = getattr(args, 'sg_reduce_backend', 'native')
            if sg_backend != 'native' and not hasattr(light, 'sg_reduce_backend'):
                raise ValueError('selected source does not support the SG backend')
            light.sg_reduce_backend = sg_backend
        for key, value in (light_state.items() if args.verify_weights or args.load_only else []):
            assert_equal(host(getattr(light, key)), value, f'light.{key}')
        for i, key in ([(0, '_anchor'), (1, '_level'), (2, '_offset'), (3, '_anchor_feat'),
                       (5, '_scaling'), (6, '_rotation'), (7, '_opacity')] if args.verify_weights or args.load_only else []):
            actual = host(getattr(model, key))
            if key == '_level' and args.framework == 'pytorch':
                if not np.array_equal(actual, payload[i]):
                    raise ValueError('loaded LOD level values differ')
            else:
                assert_equal(actual, payload[i], key)
        for i, name in ([(13, 'opacity'), (14, 'cov'), (15, 'color'),
                        (19, 'albedo'), (20, 'matallic'), (21, 'roughness')] if args.verify_weights or args.load_only else []):
            actual = getattr(model, 'mlp_' + name).state_dict()
            for key, value in payload[i].items():
                assert_equal(host(actual[key]), value, f'{name}.{key}')
        if args.offset_layout == 'flattened':
            if args.framework != 'jittor':
                raise ValueError('flattened offsets are only supported by the local Jittor adapter')
            model._offset = model._offset.reshape((-1, 3))
            if args.verify_weights or args.load_only:
                assert_equal(host(model._offset).reshape(payload[2].shape), payload[2], 'offset layout adaptation')
        if args.load_only:
            sync()
            report = {'status': 'gpu_load_checked', 'framework': args.framework,
                      'anchor_count': 597027, 'geometry_mlp_light_exact': True,
                      'pytorch_pth_equality_verified': args.framework == 'pytorch',
                      'historical_pth_layout': 'albedo,roughness,metallic; integral float32 LOD',
                      'render_executed': False}
            (args.output / 'load_report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report))
            return
        light.build_mips()
        sync()
        pipe = Namespace(compute_cov3D_python=False, debug=False, sample_num=64)
        background = tensor(np.zeros(3, np.float32))
        cameras, saved_arrays = [], {}
        for view in views:
            arrays, fovx, fovy = camera_arrays(view['camera'])
            camera = Namespace(image_width=view['width'], image_height=view['height'],
                               FoVx=fovx, FoVy=fovy, uid=view['camera']['id'],
                               image_name=view['name'], resolution_scale=1.)
            for key, value in arrays.items():
                setattr(camera, key, tensor(value))
                assert_equal(host(getattr(camera, key)), value, key)
                saved_arrays[f"{view['name']}.{key}"] = value
            cameras.append(camera)
        np.savez(args.output / 'camera_inputs.npz', **saved_arrays)
        del payload

        gaussian_counts = {}
        def draw(camera):
            model.set_anchor_mask(camera.camera_center, 40000, 1.)
            visible = renderer.prefilter_voxel(camera, model, pipe, background)
            package = renderer.render(camera, model, pipe, background, visible_mask=visible,
                                   is_pbr=True, light=light, is_training=False, iteration=40000,
                                   **({'return_aux': False} if args.mode == 'rgb' else {}))
            if 'radii' in package:
                gaussian_counts[camera.image_name] = int(package['radii'].shape[0])
            return package

        if args.mode == 'export-raster':
            if args.framework != 'pytorch':
                raise ValueError('canonical raster inputs must come from PyTorch')
            capture = RasterCapture(renderer, host)
            for camera in cameras:
                image, _, _ = measure(lambda: draw(camera), sync, host)
                capture.save(args.output, camera.image_name)
                print(json.dumps({'stage': 'export', 'view': camera.image_name}), flush=True)
            (args.output / 'summary.json').write_text(json.dumps({'status': 'complete', 'mode': args.mode,
                 'view_names': [c.image_name for c in cameras], 'source': str(root)}, indent=2))
            return

        warmup_start = time.perf_counter()
        warmup_rows = []
        for warmup_round in range(args.warmup):
            for camera in cameras:
                image, ms, _ = measure(lambda: draw(camera), sync, host)
                row = {'stage': 'warmup', 'round': warmup_round, 'view': camera.image_name, 'ms': ms}
                warmup_rows.append(row)
                print(json.dumps(row), flush=True)
                if image.shape != (3, camera.image_height, camera.image_width):
                    raise ValueError('output size differs from contract')
        warmup_seconds = time.perf_counter() - warmup_start
        records = []
        with MemorySampler() as memory, (args.output / 'timings.jsonl').open('x') as log:
            for repeat in range(args.rounds):
                for camera in cameras:
                    image, forward, e2e = measure(lambda: draw(camera), sync, host)
                    if memory.error:
                        raise RuntimeError(memory.error)
                    record = {'round': repeat, 'view': camera.image_name,
                              'gaussian_count': gaussian_counts.get(camera.image_name),
                              'forward_ms': forward, 'forward_and_download_ms': e2e}
                    records.append(record)
                    log.write(json.dumps(record) + '\n')
                    log.flush()
                    if repeat == 0:
                        save_image(args.output, camera.image_name, image)
            if memory.error:
                raise RuntimeError(memory.error)
        if memory.error:
            raise RuntimeError(memory.error)
        report = {'status': 'timing_complete_quality_comparison_pending', 'mode': args.mode,
                  'label': args.label, 'framework': args.framework, 'source_root': str(root),
                  'sg_reduce_backend': getattr(args, 'sg_reduce_backend', 'native'),
                  'gaussian_counts': gaussian_counts,
                  'framework_version': str(backend.__version__), 'gpu': memory.name,
                  'offset_layout': args.offset_layout,
                  'driver_memory_peak_sampled_bytes': max(memory.samples) if memory.samples else None,
                  'memory_sampling_seconds': .1, 'resolution': 4, 'anchor_count': 597027,
                  'extra_level_policy': 'historical zero convention; original state not captured',
                  'camera_policy': 'common CPU matrices; historical output comparison pending',
                  'view_names': [c.image_name for c in cameras], 'warmup_seconds': warmup_seconds,
                  'warmup_rounds': args.warmup, 'rounds': args.rounds,
                  'warmup_steps': warmup_rows,
                  'weight_verification': 'current run' if args.verify_weights else 'reused successful single-view verification',
                  'mipmap_policy': 'prebuild once; any additional renderer-internal calls remain timed',
                  'forward': summary([r['forward_ms'] for r in records]),
                  'forward_and_download': summary([r['forward_and_download_ms'] for r in records]),
                  'per_view': {c.image_name: summary([r['forward_ms'] for r in records
                              if r['view'] == c.image_name]) for c in cameras},
                  'pytorch_checkpoint_equality': args.framework == 'pytorch'}
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        if getattr(args, 'audit_outputs', False):
            audits = []
            for camera in [c for c in cameras if c.image_name in ('DSC07956','DSC07957','DSC07966','DSC08021')]:
                model.set_anchor_mask(camera.camera_center, 40000, 1.)
                visible = renderer.prefilter_voxel(camera, model, pipe, background)
                package = renderer.render(camera, model, pipe, background, visible_mask=visible,
                    is_pbr=True, light=light, is_training=False, iteration=40000)
                sync()  # Exactly the same completion boundary as normal timing.
                tensor_type = backend.Tensor if args.framework == 'pytorch' else backend.Var
                leaves = {}
                def collect(value, name):
                    if isinstance(value, tensor_type): leaves[name] = value
                    elif isinstance(value, dict):
                        for key,item in value.items(): collect(item,name+'.'+key)
                    elif isinstance(value, (list,tuple)):
                        for key,item in enumerate(value): collect(item,name+'.'+str(key))
                collect(package,'output')
                started = time.perf_counter()
                profile = []
                if args.framework == 'jittor':
                    with backend.profile_scope() as profile:
                        for value in leaves.values(): value.sync()
                        sync()
                else: sync()
                finish_ms = (time.perf_counter()-started)*1000
                started = time.perf_counter()
                arrays = {key: host(value) for key,value in leaves.items()}
                read_ms = (time.perf_counter()-started)*1000
                audits.append({'view': camera.image_name, 'explicit_finish_ms_including_profiler_overhead':finish_ms,
                    'all_output_readback_ms':read_ms,'jittor_post_boundary_profile':profile,
                    'outputs':{key:{'shape':list(a.shape),'dtype':str(a.dtype),'finite':bool(np.isfinite(a).all())}
                               for key,a in arrays.items()}})
                if camera.image_name == 'DSC08021':
                    np.savez(args.output/'selection_boundary.npz',
                        visible=host(visible),neural_opacity=arrays['output.neural_opacity'],
                        selection=arrays['output.selection_mask'])
                del arrays,leaves,package
            (args.output/'output_completion_audit.json').write_text(json.dumps(audits,indent=2))
        print(json.dumps(report))


def main():
    parser = ArgumentParser()
    parser.add_argument('--framework', choices=('pytorch', 'jittor'), required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--view-name', help='render only this contract view for visual review')
    parser.add_argument('--offset-layout', choices=('anchor', 'flattened'), default='anchor')
    parser.add_argument('--mode', choices=('full', 'rgb', 'export-raster', 'raster'), default='full')
    parser.add_argument('--raster-inputs', type=Path)
    parser.add_argument('--verify-weights', action='store_true')
    parser.add_argument('--sg-reduce-backend', choices=('native', 'vector3_cuda'), default='native')
    parser.add_argument('--load-only', action='store_true')
    parser.add_argument('--audit-outputs', action='store_true', help='post-timing auxiliary-output and selection diagnosis')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 1 or args.rounds < 1:
        parser.error('warmup and rounds must be positive')
    for name in ('source_root', 'weights', 'contract', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if args.raster_inputs:
        args.raster_inputs = args.raster_inputs.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception:
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
