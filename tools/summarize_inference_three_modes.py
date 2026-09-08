"""Compact inference tables and per-view quality; no hashing or source mutation."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmark_inference_versions import summary
from compare_inference_benchmarks import image_metrics


def compare(a, b):
    x, y = np.load(a).astype(np.float64), np.load(b).astype(np.float64)
    if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f'invalid image {b}')
    return {'max_abs_raw': float(np.abs(x-y).max()), **image_metrics(x, y)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--families', nargs='+', default=['pytorch', 'repository', 'local'])
    args = p.parse_args()
    rows, quality = [], []
    for mode in ('full', 'raster', 'rgb'):
        baseline = None
        for family in args.families:
            folders = [d for suffix in ('A', 'B') if (d := args.root / f'{mode}-{family}-{suffix}').exists()]
            reports = [json.loads((d/'summary.json').read_text()) for d in folders]
            if not reports:
                raise ValueError(f'missing {mode} {family}')
            if reports[0]['status'] == 'unsupported':
                rows.append({'mode': mode, 'version': family, 'status': 'unsupported'})
                continue
            for r in reports:
                if 'forward' not in r:
                    raise ValueError(f'incomplete {mode} {family}')
                for key in ('view_names', 'rounds', 'warmup_rounds', 'gpu', 'resolution', 'anchor_count'):
                    if r[key] != reports[0][key]:
                        raise ValueError(f'inconsistent {key}')
            times = [json.loads(line)['forward_ms'] for d in folders for line in (d/'timings.jsonl').read_text().splitlines()]
            metrics = summary(times)
            if family == 'pytorch':
                baseline = metrics['mean_ms']
            means = [r['forward']['mean_ms'] for r in reports]
            row = {'mode': mode, 'version': family, **metrics,
                   'sampled_peak_gib': max(r['driver_memory_peak_sampled_bytes'] for r in reports)/2**30,
                   'speedup_vs_same_mode_pytorch': baseline/metrics['mean_ms'] if baseline else None,
                   'repeat_drift': (max(means)-min(means))/min(means),
                   'independent_runs': len(reports)}
            rows.append(row)
            reference = args.root / (f'full-{family}-A' if mode == 'rgb' else f'{mode}-pytorch-A')
            if mode != 'raster':
                with np.load(reference/'camera_inputs.npz') as a, np.load(folders[0]/'camera_inputs.npz') as b:
                    if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                        raise ValueError(f'camera mismatch {mode} {family}')
            for view in reports[0]['view_names']:
                quality.append({'mode': mode, 'version': family, 'view': view,
                                'reference': reference.name,
                                **compare(reference/f'{view}.npy', folders[0]/f'{view}.npy')})
    fastest = []
    for family in args.families:
        available = [r for r in rows if r['version'] == family and r['mode'] in ('full', 'rgb') and 'mean_ms' in r]
        fastest.append(dict(min(available, key=lambda r: r['mean_ms']), selection='lowest measured two-run mean among existing full and RGB-only entries'))
    result = {'timings': rows, 'fastest_available_rgb': fastest, 'quality': quality,
              'notes': ['Synchronized wall-clock inference latency, excludes image download and file saving.',
                        'Raster uses shared PyTorch-exported inputs, includes wrapper and GPU raster forward.',
                        'Memory is driver usage sampled at 100ms, not exact instantaneous allocator peak.',
                        'RGB PyTorch lacks a native switch. Its full path remains available; unsupported is not zero FPS.']}
    target = args.root/'results.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2)
    lines = ['| Mode | Version | ms/frame | FPS | P95 ms | Sampled VRAM GiB | A/B drift |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r.get('status') == 'unsupported':
            lines.append(f"| {r['mode']} | {r['version']} | N/A | N/A | N/A | N/A | N/A |")
        else:
            lines.append(f"| {r['mode']} | {r['version']} | {r['mean_ms']:.3f} | {r['fps']:.2f} | {r['p95_ms']:.3f} | {r['sampled_peak_gib']:.3f} | {r['repeat_drift']:.1%} |")
    (args.root/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
