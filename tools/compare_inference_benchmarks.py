"""Compare camera inputs, raw renders and timings; do not invent quality gates."""
import argparse
import json
from pathlib import Path
import numpy as np


def image_metrics(a, b):
    from scipy.ndimage import convolve1d
    a, b = np.clip(a, 0, 1), np.clip(b, 0, 1)
    weights = np.exp(-np.arange(-5, 6, dtype=np.float64)**2/(2*1.5**2))
    weights /= weights.sum()
    def blur(x):
        return convolve1d(convolve1d(x, weights, axis=1, mode='constant'),
                          weights, axis=2, mode='constant')
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a)-ma*ma, blur(b*b)-mb*mb, blur(a*b)-ma*mb
    ssim = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    mse = float(np.mean((a-b)**2))
    return {'psnr_clamped_db': -10*float(np.log10(max(mse, 1e-30))),
            'ssim_clamped': float(ssim.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [json.loads((p / 'summary.json').read_text()) for p in args.runs]
    ref, refdir = reports[0], args.runs[0]
    if ref['framework'] != 'pytorch' or not ref['pytorch_checkpoint_equality']:
        raise ValueError('first run must be the verified original PyTorch reference')
    comparisons = []
    with np.load(refdir / 'camera_inputs.npz') as cameras:
        for folder, report in zip(args.runs[1:], reports[1:]):
            for key in ('view_names', 'gpu', 'resolution', 'anchor_count', 'rounds', 'warmup_rounds'):
                if report[key] != ref[key]:
                    raise ValueError(f'incompatible benchmark input: {key}, {folder}')
            with np.load(folder / 'camera_inputs.npz') as candidate:
                if set(candidate.files) != set(cameras.files) or any(
                        not np.array_equal(candidate[k], cameras[k]) for k in cameras.files):
                    raise ValueError(f'camera matrix mismatch: {folder}')
            rows = []
            for view in ref['view_names']:
                a = np.load(refdir / f'{view}.npy').astype(np.float64)
                b = np.load(folder / f'{view}.npy').astype(np.float64)
                if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
                    raise ValueError(f'invalid render: {view}')
                diff = b-a
                rows.append({'view': view, 'max_abs_raw': float(np.abs(diff).max()),
                             'relative_l2_raw': float(np.linalg.norm(diff)/max(np.linalg.norm(a), 1e-30)),
                             **image_metrics(a, b)})
            comparisons.append({'label': report['label'], 'quality': rows,
                                'reference_time_over_candidate': ref['forward']['mean_ms']/report['forward']['mean_ms']})
    drift = {}
    for family in ('pytorch', 'repository', 'local'):
        samples = [r['forward']['mean_ms'] for r in reports
                   if r['label'] in (family+'-A', family+'-B')]
        if len(samples) == 2:
            ratio = abs(samples[0]-samples[1])/min(samples)
            drift[family] = {'relative_drift': ratio, 'within_5_percent': ratio <= .05}
    gt_results = {}
    if args.dataset:
        from PIL import Image
        for folder, report in zip(args.runs, reports):
            rows = []
            for view in ref['view_names']:
                candidates = [p for p in (args.dataset / 'images').iterdir() if p.stem == view]
                if len(candidates) != 1:
                    raise ValueError(f'GT image not unique: {view}')
                rendered = np.load(folder / f'{view}.npy').astype(np.float64)
                with Image.open(candidates[0]) as image:
                    gt = np.asarray(image.convert('RGB').resize(
                        (rendered.shape[2], rendered.shape[1]), Image.Resampling.LANCZOS),
                        dtype=np.float64).transpose(2, 0, 1)/255.
                rows.append({'view': view, **image_metrics(rendered, gt)})
            gt_results[report['label']] = rows
    args.output.write_text(json.dumps({'reference': str(refdir), 'comparisons': comparisons,
                                      'ground_truth': gt_results,
                                      'metric_convention': 'clamp01; global MSE PSNR; 11x11 sigma1.5 Gaussian SSIM zero padding; GT PIL RGB LANCZOS',
                                      'repeat_drift': drift,
                                      'status': 'quality_review_required_no_automatic_acceptance'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
