"""Convert original GANG PBR capture checkpoints to renderer-only named NPZ.

PyTorch is required only by the CLI, not by Jittor inference. See
docs/checkpoint_conversion.md for the supported schema and trust boundary.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def array(value, name):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in 'fiu' or not np.isfinite(result).all():
        raise ValueError(f'{name}: expected finite numeric data')
    return result


def convert_payload(checkpoint, light, metadata, pbr_order='albedo-metallic-roughness'):
    if not isinstance(checkpoint, (list, tuple)) or len(checkpoint) != 2:
        raise ValueError('expected (capture_list, iteration)')
    items, iteration = checkpoint
    if not isinstance(items, (list, tuple)) or len(items) != 21:
        raise ValueError('only original 21-item GANG PBR captures are supported')
    if items[16] is not None or items[17] is not None:
        raise ValueError('feature-bank and appearance embeddings are unsupported')
    if pbr_order not in ('albedo-metallic-roughness', 'albedo-roughness-metallic'):
        raise ValueError('unsupported PBR ordering')
    iteration = array(iteration, 'iteration')
    if iteration.size != 1 or float(iteration.reshape(-1)[0]) != int(iteration.reshape(-1)[0]) or int(iteration.reshape(-1)[0]) < 0:
        raise ValueError('iteration must be a nonnegative integer')
    out = {'iteration': np.asarray(int(iteration.reshape(-1)[0])),
           'format_version': np.asarray('gang-render-named-v1'),
           'source_pbr_order': np.asarray(pbr_order),
           'is_pbr': np.asarray(True), 'with_matallic': np.asarray(True)}
    names = {0: '_anchor', 1: '_level', 2: '_offset', 3: '_anchor_feat',
             5: '_scaling', 6: '_rotation', 7: '_opacity'}
    for index, name in names.items():
        value = array(items[index], name)
        if name != '_level' and value.dtype != np.float32:
            raise ValueError(f'{name}: expected FP32; refusing implicit precision conversion')
        out[name] = value
    anchors = out['_anchor']
    if anchors.ndim != 2 or anchors.shape[1] != 3 or not len(anchors):
        raise ValueError('_anchor must be nonempty [N,3]')
    n = len(anchors)
    offsets, features = out['_offset'], out['_anchor_feat']
    if offsets.ndim != 3 or offsets.shape[0] != n or offsets.shape[2] != 3 or offsets.shape[1] < 1:
        raise ValueError('_offset must be [N,K,3]')
    if features.ndim != 2 or features.shape[0] != n or features.shape[1] < 1:
        raise ValueError('_anchor_feat must be [N,F]')
    k, f = offsets.shape[1], features.shape[1]
    for name, shape in (('_level', (n,1)), ('_scaling', (n,6)),
                        ('_rotation', (n,4)), ('_opacity', (n,1))):
        if out[name].shape != shape:
            raise ValueError(f'{name}: expected {shape}, got {out[name].shape}')
    if (out['_level'] < 0).any() or not np.equal(out['_level'], np.floor(out['_level'])).all():
        raise ValueError('_level must contain nonnegative integers')
    out.update(n_offsets=np.asarray(k), feat_dim=np.asarray(f))
    spatial = array(items[12], 'spatial_lr_scale')
    if spatial.size != 1 or float(spatial.reshape(-1)[0]) <= 0:
        raise ValueError('spatial_lr_scale must be positive')
    out['spatial_lr_scale'] = spatial.reshape(())
    mlps = [('opacity',13,k), ('cov',14,7*k), ('color',15,3*k), ('albedo',18,3*k)]
    scalar_names = ('matallic','roughness') if pbr_order == 'albedo-metallic-roughness' else ('roughness','matallic')
    mlps += [(scalar_names[0],19,k), (scalar_names[1],20,k)]
    for name, index, outputs in mlps:
        state = items[index]
        expected = {'0.weight': ('w1',(f,f+3)), '0.bias': ('b1',(f,)),
                    '2.weight': ('w2',(outputs,f)), '2.bias': ('b2',(outputs,))}
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError(f'mlp_{name}: unsupported state keys')
        for key, (suffix, shape) in expected.items():
            value = array(state[key], f'mlp_{name}.{key}')
            if value.dtype != np.float32 or value.shape != shape:
                raise ValueError(f'mlp_{name}.{key}: expected FP32 {shape}')
            out[f'mlp_{name}_{suffix}'] = value
    for key in ('base','lgtSGs','specular_reflectance','sg_roughness'):
        if key not in light:
            raise ValueError(f'missing light field: {key}')
        value = array(light[key], key)
        if value.dtype != np.float32:
            raise ValueError(f'{key}: expected FP32')
        out['light_'+key] = value
    base, sg = out['light_base'], out['light_lgtSGs']
    if base.ndim != 4 or base.shape[0] != 6 or base.shape[-1] != 3 or base.shape[1] != base.shape[2] or base.shape[1] < 16:
        raise ValueError('light base must be [6,R,R,3], R >= 16')
    if base.shape[1] & (base.shape[1]-1):
        raise ValueError('cubemap resolution must be a power of two')
    if sg.ndim != 2 or sg.shape[1] != 10:
        raise ValueError('lgtSGs must be [M,10] RGB SGs')
    if out['light_specular_reflectance'].shape != (1,3) or out['light_sg_roughness'].shape != (1,1):
        raise ValueError('only one RGB BRDF SG is supported')
    required = ('standard_dist','voxel_size','levels','init_level','fork',
                'base_layer','dist2level','progressive','extend')
    for key in required:
        if key not in metadata:
            raise ValueError(f'missing metadata: {key}; recover it from the training configuration/log')
    for key in ('standard_dist','voxel_size','extend'):
        value = metadata[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('levels','init_level','fork','base_layer'):
        if type(metadata[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    if metadata['fork'] < 2 or metadata['levels'] < 1 or not 0 <= metadata['init_level'] < metadata['levels'] or metadata['base_layer'] < 0:
        raise ValueError('invalid LOD configuration')
    if metadata['dist2level'] not in ('round','floor','ceil','progressive') or type(metadata['progressive']) is not bool:
        raise ValueError('invalid dist2level/progressive configuration')
    if out['_level'].max() >= metadata['levels']:
        raise ValueError('checkpoint levels exceed metadata levels')
    for key in required:
        out[key] = np.asarray(metadata[key])
    # Original capture omits adaptive extra levels; this is the historical
    # inference convention, not recovered training state.
    out['_extra_level'] = np.zeros(n, np.float32)
    out['extra_level_origin'] = np.asarray('zeros: omitted by original capture')
    return out


def write_npz(output, arrays):
    output = Path(output)
    if output.suffix.lower() != '.npz':
        raise ValueError('output must end with .npz')
    # Exclusive creation preserves existing checkpoints, including on races.
    with output.open('xb') as handle:
        np.savez(handle, **arrays)


def load_light(path, trust_pickle=False):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            return dict(data)
    if path.suffix.lower() != '.npy' or not trust_pickle:
        raise ValueError('original object .npy light requires --trust-pickle; use only trusted files')
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError('light file must contain a dictionary')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--light', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True, help='JSON with explicit LOD configuration')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pbr-order', choices=('albedo-metallic-roughness','albedo-roughness-metallic'),
                        default='albedo-metallic-roughness')
    parser.add_argument('--trust-pickle', action='store_true',
                        help='Allow unsafe pickle loading for trusted legacy PTH/NPY files only')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new path')
    try:
        import torch
    except ImportError as exc:
        raise SystemExit('Conversion requires a CPU PyTorch environment; Jittor inference does not.') from exc
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=not args.trust_pickle)
    light = load_light(args.light, args.trust_pickle)
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    arrays = convert_payload(checkpoint, light, metadata, args.pbr_order)
    write_npz(args.output, arrays)
    print(json.dumps({'output': str(args.output), 'iteration': int(arrays['iteration']),
                      'anchors': len(arrays['_anchor']), 'arrays': len(arrays),
                      'pbr_order': args.pbr_order, 'optimizer_exported': False}))


if __name__ == '__main__':
    main()
