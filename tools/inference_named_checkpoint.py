"""Shared named-NPZ loading for both Jittor inference implementations.

No renderer imports, casts, checkpoint rewrites or content hashes.
"""
import inspect
import re
from pathlib import Path
import numpy as np


def load_named_checkpoint(path, training_log):
    with np.load(path, allow_pickle=False) as archive:
        raw = {k: archive[k] for k in archive.files}
    for key, value in raw.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'nonfinite checkpoint field: {key}')
    n = raw['_anchor'].shape[0]
    k = int(raw['n_offsets'])
    if raw['_anchor'].shape != (n, 3) or raw['_offset'].shape != (n, k, 3):
        raise ValueError('invalid anchor/offset layout')

    def mlp(name):
        return {target: raw[f'mlp_{name}_{source}'] for target, source in
                [('0.weight', 'w1'), ('0.bias', 'b1'),
                 ('2.weight', 'w2'), ('2.bias', 'b2')]}

    args = [raw['_anchor'], raw['_level'], raw['_offset'], raw['_anchor_feat'],
            np.zeros((n, 1), np.float32), raw['_scaling'], raw['_rotation'],
            raw['_opacity'], np.zeros((n*k, 1), np.float32),
            np.zeros((n*k, 1), np.float32), np.zeros((n, 1), np.float32),
            None, float(raw['spatial_lr_scale']), mlp('opacity'), mlp('cov'),
            mlp('color'), None, None, None, mlp('albedo'), mlp('matallic'),
            mlp('roughness')]
    log = Path(training_log).read_text(encoding='utf-8')

    def number(label, cast):
        matches = re.findall(re.escape(label) + r':\s*([0-9.eE+-]+)', log)
        if not matches or len(set(matches)) != 1:
            raise ValueError(f'missing or ambiguous metadata: {label}')
        return cast(matches[0])

    metadata = {'standard_dist': float(raw['standard_dist']),
                'voxel_size': number('Max Voxel Size', float),
                'levels': number('LOD Levels', int),
                'init_level': number('Initial Levels', int)}
    # This state was omitted by the original capture. Report the historical
    # inference convention explicitly; do not claim it was recovered exactly.
    metadata['_extra_level'] = np.zeros(n, np.float32)
    light = {target: raw[source] for target, source in
             [('base', 'light_base'), ('lgtSGs', 'light_lgtSGs'),
              ('specular_reflectance', 'light_specular_reflectance'),
              ('roughness', 'light_sg_roughness')]}
    config = {key: raw[key].item() for key in
              ('n_offsets', 'feat_dim', 'fork', 'base_layer', 'dist2level',
               'progressive', 'extend', 'is_pbr', 'with_matallic')}
    return args, light, metadata, config


def restore_jittor_model(model, args, metadata):
    """One payload; use optimizer-free option where the implementation has it."""
    parameters = inspect.signature(model.restore_numpy).parameters
    kwargs = {'metadata': metadata}
    if 'inference_only' in parameters:
        kwargs.update(inference_only=True, build_optimizer=False)
    model.restore_numpy(list(args), **kwargs)
    model.eval()
    return {'extra_level_origin': 'zeros: historical inference convention, not captured',
            'optimizer_free_option': 'inference_only' in parameters,
            'payload_items': len(args)}
