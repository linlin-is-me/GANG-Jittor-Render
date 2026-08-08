"""Compose a garden envmap-relighting contact sheet.

The renderer writes one controlled A/B directory per TensoIR environment map.
This utility collects the original learned-light render and selected envmap
renders, with the corresponding lat-long map as an inset. It does not alter
scene pixels beyond presentation resizing.
"""
import argparse
import hashlib
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont


NAMES = ('bridge', 'city', 'courtyard', 'fireplace', 'forest', 'interior',
         'museum', 'night', 'snow', 'square', 'studio', 'sunrise', 'sunset',
         'tunnel')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 'C:/Windows/Fonts/arialbd.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def tile(scene_path, env_path, label, width=600):
    scene = Image.open(scene_path).convert('RGB')
    height = int(round(width * scene.height / scene.width))
    scene = scene.resize((width, height), Image.Resampling.LANCZOS)

    header = 42
    out = Image.new('RGB', (width, height + header), (16, 16, 16))
    out.paste(scene, (0, header))
    draw = ImageDraw.Draw(out)
    draw.text((12, 8), label, fill=(245, 245, 245), font=load_font(24))

    env = Image.open(env_path).convert('RGB')
    inset_w = 190
    inset_h = inset_w // 2
    border = 3
    env = env.resize((inset_w, inset_h), Image.Resampling.LANCZOS)
    outer_w = inset_w + border * 2
    outer_h = inset_h + border * 2
    outer_x = width - outer_w
    outer_y = header + height - outer_h
    draw.rectangle((outer_x, outer_y, width - 1, header + height - 1),
                   fill=(245, 245, 245))
    out.paste(env, (outer_x + border, outer_y + border))
    return out


def control_signature(params, view):
    """Fields that must stay identical across an envmap-only comparison."""
    key = str(view)
    if view not in params.get('camera_indices', []):
        raise ValueError(f'view {view} is absent from camera_indices')
    return {
        'checkpoint_sha256': params['checkpoint']['sha256'],
        'cameras_sha256': params['cameras_json']['sha256'],
        'camera_name': params.get('camera_names', {}).get(key),
        'image_size': params['image_sizes'][key],
        'render_res': params['render_res'],
        'training_res': params.get('training_res'),
        'lod_scale': params.get('lod_scale'),
        'cubemap_res': params['envmap_cubemap_res'],
        'scene_output': params['scene_output'],
        'tone_map': params['tone_map'],
        'exposure': params.get('exposure'),
        'background': params['background'],
        'sg_hard_off': params['sg_hard_off'],
        'original_sg_count': params['original_sg_count_after_zero'],
        'replacement_sg_count': params['replacement_sg_count_after_zero'],
        'use_point_light': params['use_point_light'],
        'point_light_enabled_orig': params['point_light_enabled_orig'],
        'point_light_enabled_new': params['point_light_enabled_new'],
    }


def signature_diff(expected, actual):
    return {key: {'expected': expected.get(key), 'actual': actual.get(key)}
            for key in expected.keys() | actual.keys()
            if expected.get(key) != actual.get(key)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='directory containing one envmap A/B output subdirectory per name')
    ap.add_argument('--view', type=int, default=1)
    ap.add_argument('--names', default=' '.join(NAMES),
                    help='space-separated replacement envmap directory names')
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--original-position', choices=('first', 'last'),
                    default='first')
    ap.add_argument('--no-footer', action='store_true')
    ap.add_argument('--output-name', default=None,
                    help='grid PNG filename; defaults to paper_fig9_garden_view{view}_grid.png')
    args = ap.parse_args()
    names = tuple(args.names.split())
    if not names or len(set(names)) != len(names):
        raise ValueError('--names must contain one or more unique names')
    if args.cols < 1:
        raise ValueError('--cols must be positive')
    if args.output_name is not None:
        if (not args.output_name.lower().endswith('.png')
                or os.path.basename(args.output_name) != args.output_name
                or '/' in args.output_name or '\\' in args.output_name):
            raise ValueError('--output-name must be a PNG filename without directory components')

    root = os.path.abspath(args.root)
    first = os.path.join(root, names[0])
    with open(os.path.join(first, 'envmap_ab_params.json'), encoding='utf-8') as f:
        first_params = json.load(f)
    expected_control = control_signature(first_params, args.view)
    if not expected_control['sg_hard_off'] or expected_control['use_point_light']:
        raise ValueError('the first result is not an envmap-only comparison')
    camera_name = first_params.get('camera_names', {}).get(str(args.view),
                                                            f'view{args.view}')
    scene_output = first_params.get('scene_output', 'unknown')
    original_entry = (
        'original learned envmap',
        os.path.join(first, f'view{args.view}_env_original.png'),
        os.path.join(first, 'env_original.png'), None)
    original_scene_sha256 = sha256(original_entry[1])
    original_env_sha256 = sha256(original_entry[2])
    entries = [original_entry] if args.original_position == 'first' else []
    records = []
    for name in names:
        directory = os.path.join(root, name)
        params_path = os.path.join(directory, 'envmap_ab_params.json')
        with open(params_path, encoding='utf-8') as f:
            params = json.load(f)
        actual_control = control_signature(params, args.view)
        differences = signature_diff(expected_control, actual_control)
        if differences:
            raise ValueError(f'control mismatch in {name}: {differences}')
        if params['replacement_hdr'].get('label') != name:
            raise ValueError(
                f'replacement label mismatch in {name}: '
                f"{params['replacement_hdr'].get('label')!r}")
        scene = os.path.join(directory, f'view{args.view}_env_workshop.png')
        env = os.path.join(directory, 'env_workshop.png')
        original_scene = os.path.join(directory, f'view{args.view}_env_original.png')
        original_env = os.path.join(directory, 'env_original.png')
        if sha256(original_scene) != original_scene_sha256:
            raise ValueError(f'original scene render differs in {name}')
        if sha256(original_env) != original_env_sha256:
            raise ValueError(f'original envmap preview differs in {name}')
        stats = params['view_stats'][str(args.view)]['highlight_workshop']
        records.append({
            'name': name,
            'hdr_file': (params['replacement_hdr'].get('name')
                         or os.path.basename(params['replacement_hdr'].get('path', ''))),
            'hdr_sha256': params['replacement_hdr']['sha256'],
            'scene_png': os.path.relpath(scene, root).replace(os.sep, '/'),
            'scene_png_sha256': sha256(scene),
            'linear_hdr': os.path.relpath(
                os.path.join(directory, f'view{args.view}_env_workshop.npy'), root
            ).replace(os.sep, '/'),
            'p99_9': stats['p99_9'],
            'frac_gt_2': stats['frac_gt_2'],
            'gpu_peak_mb': params['gpu_peak_mb'],
            'finite': params['view_stats'][str(args.view)]['finite_new'],
            'strict_highlight_gate': (stats['p99_9'] <= 1.5
                                      and stats['frac_gt_2'] <= 0.0001),
        })
        entries.append((name, scene, env, records[-1]))
    if args.original_position == 'last':
        entries.append(original_entry)

    tiles = [tile(scene, env, label) for label, scene, env, _ in entries]
    cols = args.cols
    rows = math.ceil(len(tiles) / cols)
    tw, th = tiles[0].size
    footer = 0 if args.no_footer else 70
    grid = Image.new('RGB', (cols * tw, rows * th + footer), (12, 12, 12))
    for idx, item in enumerate(tiles):
        grid.paste(item, ((idx % cols) * tw, (idx // cols) * th))
    if not args.no_footer:
        draw = ImageDraw.Draw(grid)
        draw.text((12, rows * th + 10),
                  f'garden 40k | {camera_name} (view{args.view}) | TensoIR 1K HDR | {scene_output} | SG=0 | point light=off',
                  fill=(235, 235, 235), font=load_font(22))

    grid_filename = (args.output_name
                     if args.output_name
                     else f'paper_fig9_garden_view{args.view}_grid.png')
    grid_path = os.path.join(root, grid_filename)
    grid.save(grid_path)
    manifest = {
        'experiment': 'garden_paper_fig9_envmap_relighting',
        'view': args.view,
        'camera_name': camera_name,
        'scene_output': scene_output,
        'control': ('same garden 40k checkpoint, camera, material, resolution and '
                    'display transform; only envmap changes; SG and point light disabled'),
        'control_signature': expected_control,
        'original_scene_sha256': original_scene_sha256,
        'original_envmap_preview_sha256': original_env_sha256,
        'envmap_source': 'TensoIR envmap archive linked by the GANG README',
        'layout': {
            'columns': cols,
            'rows': rows,
            'original_position': args.original_position,
            'footer': not args.no_footer,
            'envmap_inset_outer_edge': 'flush with scene right and bottom edges',
        },
        'envmap_order': list(names),
        'envmaps': records,
        'grid_png': grid_filename,
        'grid_png_sha256': sha256(grid_path),
        'composer_sha256': sha256(__file__),
    }
    manifest_filename = ('paper_fig9_garden_manifest.json'
                         if args.output_name is None
                         else os.path.splitext(grid_filename)[0] + '_manifest.json')
    manifest_path = os.path.join(root, manifest_filename)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(grid_path)
    print(manifest_path)


if __name__ == '__main__':
    main()
