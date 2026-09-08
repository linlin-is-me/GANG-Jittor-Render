"""Diagnostic-only fixed-input raster capture and replay; no renderer edits."""
import json
import time
import numpy as np


class RasterCapture:
    def __init__(self, renderer, host):
        original = renderer.GaussianRasterizer
        self.packet = None
        owner = self

        class Proxy:
            def __init__(self, raster_settings):
                self.settings = raster_settings
                self.impl = original(raster_settings=raster_settings)

            def __getattr__(self, key):
                return getattr(self.impl, key)

            def __call__(self, **kwargs):
                owner.packet = (self.settings, kwargs)
                return self.impl(**kwargs)

        self.original = original
        self.renderer = renderer
        self.host = host
        renderer.GaussianRasterizer = Proxy

    def save(self, directory, view):
        if self.packet is None:
            raise RuntimeError('no raster call captured')
        settings, kwargs = self.packet
        metadata = {'settings': {}, 'kwargs': {}, 'view': view}
        arrays = {}
        for section, values in [('settings', settings._asdict()), ('kwargs', kwargs)]:
            for key, value in values.items():
                if value is None or isinstance(value, (bool, int, float, str)):
                    metadata[section][key] = value
                else:
                    a = np.ascontiguousarray(self.host(value))
                    if a.dtype.kind == 'f' and not np.isfinite(a).all():
                        raise FloatingPointError(key)
                    name = section + '.' + key
                    arrays[name] = a
                    metadata[section][key] = {'array': name}
        metadata['gaussian_count'] = int(arrays['kwargs.means3D'].shape[0])
        np.savez(directory / (view + '.npz'), **arrays)
        (directory / (view + '.json')).write_text(json.dumps(metadata, indent=2))
        self.packet = None


def load_packet(directory, view, renderer, tensor, framework):
    metadata = json.loads((directory / (view + '.json')).read_text())
    with np.load(directory / (view + '.npz'), allow_pickle=False) as arrays:
        def decode(values):
            return {k: tensor(arrays[v['array']]) if isinstance(v, dict) else v
                    for k, v in values.items()}
        settings, kwargs = decode(metadata['settings']), decode(metadata['kwargs'])
    if framework == 'jittor':
        kwargs.update(inference_only=True, return_aux=True)
    rasterizer = renderer.GaussianRasterizer(
        raster_settings=renderer.GaussianRasterizationSettings(**settings))
    return rasterizer, kwargs, metadata['gaussian_count']


def save_image(directory, view, image):
    from PIL import Image
    np.save(directory / (view + '.npy'), image)
    pixels = np.rint(np.clip(image.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(directory / (view + '.png'))


def run_raster(args, renderer, backend, tensor, host, sync, views, sampler, measure, summary):
    if not args.raster_inputs:
        raise ValueError('--raster-inputs required')
    rows, warmups = [], []
    started = time.perf_counter()
    with backend.no_grad():
        for repeat in range(args.warmup + args.rounds):
            for view in views:
                rasterizer, kwargs, count = load_packet(args.raster_inputs, view['name'], renderer, tensor, args.framework)
                # All upload/packet IO precedes the synchronized timer.
                def draw():
                    return {'render': rasterizer(**kwargs)[1]}
                if repeat < args.warmup:
                    _, ms, _ = measure(draw, sync, host)
                    warmups.append({'round': repeat, 'view': view['name'], 'ms': ms})
                    print(json.dumps({'stage': 'warmup', **warmups[-1]}), flush=True)
                else:
                    # Sampling starts before the raster call and includes its workspace.
                    with sampler() as memory:
                        image, ms, d2h = measure(draw, sync, host)
                    if memory.error:
                        raise RuntimeError(memory.error)
                    row = {'round': repeat-args.warmup, 'view': view['name'], 'gaussian_count': count,
                           'forward_ms': ms, 'forward_and_download_ms': d2h,
                           'sampled_memory_bytes': max(memory.samples) if memory.samples else None}
                    rows.append(row)
                    with (args.output / 'timings.jsonl').open('a') as log:
                        log.write(json.dumps(row) + '\n')
                    if repeat == args.warmup:
                        save_image(args.output, view['name'], image)
                del rasterizer, kwargs
    report = {'status': 'complete', 'mode': 'raster', 'framework': args.framework,
              'label': args.label, 'gpu': memory.name, 'view_names': [v['name'] for v in views],
              'rounds': args.rounds, 'warmup_rounds': args.warmup,
              'resolution': 4, 'anchor_count': 597027,
              'input_policy': 'same PyTorch-exported per-Gaussian attributes; full auxiliary channels',
              'forward': summary([r['forward_ms'] for r in rows]),
              'driver_memory_peak_sampled_bytes': max(r['sampled_memory_bytes'] or 0 for r in rows),
              'elapsed_seconds': time.perf_counter()-started,
              'warmup_steps': warmups}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
