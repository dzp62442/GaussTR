#!/usr/bin/env python
"""Build sharded GaussTR dataset files.

This script repacks GaussTR's nuScenes-style data into large `.torch` shard
files. It is intentionally limited to preprocessing; the sharded dataloader is
implemented separately after the preprocessing output is validated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


CAMERA_ORDER = (
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
    'CAM_BACK_LEFT',
)

GROUP_KINDS = {'raw', 'depth', 'feats', 'sem_seg', 'occ_gt'}
DERIVED_KINDS = {'depth', 'feats', 'sem_seg'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build large .torch shards for GaussTR datasets.')
    parser.add_argument(
        '--preset',
        choices=('all', 'talk2dino', 'featup'),
        default='all',
        help='Group preset. Defaults to all, the union needed by featup and talk2dino.')
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('data/nuscenes'),
        help='nuScenes root. Used to resolve image paths and occ gt paths.')
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=None,
        help='Path to nuscenes_infos_{train,val}.pkl. Defaults from --split.')
    parser.add_argument(
        '--split',
        default='train',
        choices=('train', 'val', 'test'),
        help='Dataset split name written under out-root.')
    parser.add_argument(
        '--out-root',
        type=Path,
        default=Path('data/gausstr_shards'),
        help='Output root. Defaults to data/gausstr_shards, which can be a symlink to a TOS mount path.')
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=None,
        help=(
            'Override preset groups. Supported: raw_nuscenes, occ_gt, '
            'depth:<method>, feats:<method>, sem_seg:<method>. '
            'Legacy meta/images values are treated as raw_nuscenes.'))
    parser.add_argument(
        '--ratio',
        type=float,
        default=1.0,
        help='Scene-level subset ratio in (0, 1].')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional maximum number of samples after scene sampling.')
    parser.add_argument(
        '--seed', type=int, default=2026, help='Random seed for train subset.')
    parser.add_argument(
        '--target-shard-size',
        '--target-control-shard-size',
        dest='target_shard_size',
        default='256MB',
        help='Target size for derived groups, e.g. 256MB or 512MB.')
    parser.add_argument(
        '--raw-target-shard-size',
        default='100MB',
        help='Target size for raw_nuscenes when --raw-samples-per-shard is not set.')
    parser.add_argument(
        '--base-block-size',
        type=int,
        default=8,
        help='Base alignment block size. Group shard sizes are multiples of this.')
    parser.add_argument(
        '--min-blocks-per-shard',
        type=int,
        default=1,
        help='Lower bound in base blocks for inferred group shard sizes.')
    parser.add_argument(
        '--max-blocks-per-shard',
        type=int,
        default=8,
        help='Upper bound in base blocks for inferred derived group shard sizes.')
    parser.add_argument(
        '--raw-samples-per-shard',
        type=int,
        default=None,
        help='Override samples per raw_nuscenes shard. Must be a multiple of --base-block-size.')
    parser.add_argument(
        '--samples-per-shard',
        type=int,
        default=None,
        help='Override all group shard sizes. Must be a multiple of --base-block-size.')
    parser.add_argument(
        '--size-estimate-samples',
        type=int,
        default=1000,
        help='Maximum samples used to estimate per-sample group size.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of shard build workers per group. Use 1 for serial processing.')
    parser.add_argument(
        '--depth-root',
        type=Path,
        default=Path('data/nuscenes_metric3d'),
        help='Root for depth .npy files.')
    parser.add_argument(
        '--feat-root',
        type=Path,
        default=Path('data/nuscenes_featup'),
        help='Root for feature .npy files.')
    parser.add_argument(
        '--sem-seg-root',
        type=Path,
        default=Path('data/nuscenes_grounded_sam2'),
        help='Root for semantic pseudo-label .npy files.')
    parser.add_argument(
        '--occ-root',
        type=Path,
        default=None,
        help='Root containing gts/{scene_idx}/{token}/labels.npz. '
        'Defaults to data-root.')
    parser.add_argument(
        '--image-prefix',
        type=Path,
        default=None,
        help='Root containing samples/CAM_* images. Defaults to data-root.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing shard files and manifests.')
    parser.add_argument(
        '--allow-missing',
        action='store_true',
        help='Write None for missing source files instead of failing.')
    parser.add_argument(
        '--allow-nonstandard-cameras',
        action='store_true',
        help='Allow missing or extra camera keys instead of requiring 6 views.')
    parser.add_argument(
        '--no-sanity-load',
        action='store_true',
        help='Skip torch.load sanity check after writing each shard.')
    parser.add_argument(
        '--skip-file-sha256',
        action='store_true',
        help='Skip full-file sha256 computation for shard .torch files.')
    parser.add_argument(
        '--compression',
        choices=('none',),
        default='none',
        help='Reserved for future compression variants.')
    return parser.parse_args()


def preset_modalities(preset: str, split: str) -> List[str]:
    if preset == 'all':
        modalities = [
            'raw_nuscenes', 'depth:metric3d', 'feats:featup',
            'sem_seg:grounded_sam2'
        ]
    elif preset == 'talk2dino':
        modalities = ['raw_nuscenes', 'depth:metric3d']
    elif preset == 'featup':
        modalities = [
            'raw_nuscenes', 'depth:metric3d', 'feats:featup',
            'sem_seg:grounded_sam2'
        ]
    else:
        raise ValueError(f'Unsupported preset: {preset}')
    if split in {'val', 'test'}:
        modalities = [*modalities, 'occ_gt']
    return modalities


def load_pickle(path: Path) -> Any:
    try:
        import mmengine

        return mmengine.load(path)
    except Exception:
        with path.open('rb') as f:
            return pickle.load(f)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def dump_json(path: Path, data: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        try:
            with path.open('r', encoding='utf-8') as f:
                existing = json.load(f)
            if existing == data:
                print(f'Skipping unchanged json: {path}')
                return
        except Exception:
            pass
        raise FileExistsError(f'{path} already exists. Pass --overwrite.')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    tmp_path.replace(path)


def dump_json_or_skip_existing(path: Path, data: Mapping[str, Any],
                               overwrite: bool) -> None:
    if path.exists() and not overwrite:
        try:
            with path.open('r', encoding='utf-8') as f:
                existing = json.load(f)
            if existing == data:
                print(f'Skipping unchanged json: {path}')
            else:
                print(f'Skipping existing json: {path}')
            return
        except Exception:
            print(f'Skipping existing json: {path}')
            return
    dump_json(path, data, overwrite=overwrite)


def dump_json_replace_existing(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    tmp_path.replace(path)


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def parse_size(value: str) -> int:
    text = value.strip().upper()
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'KIB': 1024,
        'MIB': 1024**2,
        'GIB': 1024**3,
    }
    for unit, multiplier in sorted(units.items(), key=lambda item: -len(item[0])):
        if text.endswith(unit):
            number = text[:-len(unit)].strip()
            return int(float(number) * multiplier)
    return int(float(text))


def normalize_modalities(raw_modalities: Iterable[str]) -> List[Tuple[str, Optional[str], str]]:
    normalized = []
    for raw in raw_modalities:
        if raw in {'raw', 'raw_nuscenes', 'meta', 'images'}:
            kind, method = 'raw', None
            name = 'raw_nuscenes'
            normalized.append((kind, method, name))
            continue
        if ':' in raw:
            kind, method = raw.split(':', 1)
        else:
            kind, method = raw, None
        if kind not in GROUP_KINDS:
            raise ValueError(f'Unsupported group: {raw}')
        if kind in DERIVED_KINDS and not method:
            raise ValueError(f'Modality {kind} requires a method, e.g. {kind}:metric3d')
        if kind == 'occ_gt' and method:
            raise ValueError(f'Modality {kind} does not accept a method: {raw}')
        name = group_name(kind, method)
        normalized.append((kind, method, name))
    seen = set()
    deduped = []
    for item in normalized:
        if item[2] not in seen:
            deduped.append(item)
            seen.add(item[2])
    return deduped


def group_name(kind: str, method: Optional[str]) -> str:
    if kind == 'raw':
        return 'raw_nuscenes'
    if method:
        return f'{kind}_{method}'
    return kind


def group_rel_dir(kind: str, method: Optional[str]) -> Path:
    return Path(group_name(kind, method))


def sample_identifier(info: Mapping[str, Any]) -> str:
    if 'sample_idx' in info:
        return str(info['sample_idx'])
    if 'token' in info:
        return str(info['token'])
    raise KeyError('Sample info has neither sample_idx nor token.')


def scene_identifier(info: Mapping[str, Any]) -> str:
    scene = info.get('scene_idx') or info.get('scene_token')
    if scene is None:
        raise KeyError(
            f"Sample {info.get('sample_idx', info.get('token', '<unknown>'))} "
            'has neither scene_idx nor scene_token.')
    return str(scene)


def validate_infos(
    infos: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    require_occ: bool,
    allow_nonstandard_cameras: bool,
) -> None:
    for idx in selected_indices:
        info = infos[idx]
        ident = info.get('sample_idx', info.get('token', idx))
        for key in ('token', 'scene_idx', 'scene_token', 'images'):
            if key not in info:
                raise KeyError(f'Sample {ident} missing required key: {key}')
        if not isinstance(info['images'], Mapping):
            raise TypeError(f"Sample {ident} has non-mapping 'images'.")
        missing = [cam for cam in CAMERA_ORDER if cam not in info['images']]
        extra = [cam for cam in info['images'] if cam not in CAMERA_ORDER]
        if (missing or extra) and not allow_nonstandard_cameras:
            raise ValueError(
                f'Sample {ident} camera keys are not the expected 6 views. '
                f'Missing={missing}, extra={extra}. Pass '
                '--allow-nonstandard-cameras to bypass.')
        for cam_name, cam_item in iter_cameras(
                info, allow_nonstandard_cameras=allow_nonstandard_cameras):
            for key in ('img_path', 'cam2ego', 'cam2img', 'lidar2cam'):
                if key not in cam_item:
                    raise KeyError(
                        f'Sample {ident} camera {cam_name} missing {key}.')
        if require_occ:
            for key in ('scene_idx', 'token'):
                if key not in info:
                    raise KeyError(f'Sample {ident} missing occ key: {key}')


def image_stem(img_path: str) -> str:
    return Path(img_path).stem


def resolve_image_path(data_root: Path, image_prefix: Path, cam_name: str,
                       img_path: str) -> Path:
    candidates = []
    raw = Path(img_path)
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend([
        data_root / img_path,
        image_prefix / img_path,
        data_root / 'samples' / cam_name / raw.name,
        image_prefix / 'samples' / cam_name / raw.name,
        image_prefix / cam_name / raw.name,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def modality_file_path(args: argparse.Namespace, kind: str,
                       method: Optional[str], info: Mapping[str, Any],
                       cam_item: Optional[Mapping[str, Any]] = None) -> Path:
    if kind in {'depth', 'feats', 'sem_seg'}:
        assert cam_item is not None
        stem = image_stem(str(cam_item['img_path']))
        root = {
            'depth': args.depth_root,
            'feats': args.feat_root,
            'sem_seg': args.sem_seg_root,
        }[kind]
        return root / f'{stem}.npy'
    if kind == 'occ_gt':
        occ_root = args.occ_root or args.data_root
        if occ_root.name == 'gts':
            return occ_root / str(info['scene_idx']) / str(info['token']) / 'labels.npz'
        return occ_root / 'gts' / str(info['scene_idx']) / str(info['token']) / 'labels.npz'
    raise ValueError(f'No standalone file path for modality {kind}:{method}')


def file_size(path: Path, allow_missing: bool) -> int:
    if path.exists():
        return path.stat().st_size
    if allow_missing:
        return 0
    raise FileNotFoundError(path)


def stable_json_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    return int(np.percentile(np.asarray(values, dtype=np.float64), q))


def group_by_scene(infos: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    scenes: Dict[str, List[int]] = {}
    for idx, info in enumerate(infos):
        scenes.setdefault(scene_identifier(info), []).append(idx)
    return scenes


def select_indices(infos: Sequence[Mapping[str, Any]], split: str, ratio: float,
                   max_samples: Optional[int], seed: int) -> List[int]:
    if ratio <= 0 or ratio > 1:
        raise ValueError('--ratio must be in (0, 1].')
    if ratio >= 1 and max_samples is None:
        indices = list(range(len(infos)))
        if split == 'train':
            rng = random.Random(seed)
            rng.shuffle(indices)
        return indices

    target = max(1, math.ceil(len(infos) * ratio))

    scenes = group_by_scene(infos)
    scene_ids = list(scenes)
    if split == 'train':
        rng = random.Random(seed)
        rng.shuffle(scene_ids)
    else:
        scene_ids.sort()

    selected: List[int] = []
    for scene_id in scene_ids:
        if max_samples is not None and selected and (
                len(selected) + len(scenes[scene_id]) > max_samples):
            break
        selected.extend(scenes[scene_id])
        if len(selected) >= target:
            break
        if max_samples is not None and len(selected) >= max_samples:
            break
    if split == 'train':
        rng = random.Random(seed)
        rng.shuffle(selected)
    else:
        selected.sort()
    return selected


def estimate_meta_size(info: Mapping[str, Any]) -> int:
    return len(pickle.dumps(make_meta_sample(info), protocol=pickle.HIGHEST_PROTOCOL))


def estimate_group_sizes(
    args: argparse.Namespace,
    infos: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    groups: Sequence[Tuple[str, Optional[str], str]],
) -> Dict[str, Dict[str, int]]:
    sampled_indices = list(indices[:args.size_estimate_samples])
    values: Dict[str, List[int]] = {name: [] for _, _, name in groups}
    image_prefix = args.image_prefix or args.data_root
    for idx in sampled_indices:
        info = infos[idx]
        for kind, method, name in groups:
            if kind == 'raw':
                total = estimate_meta_size(info)
                for cam_name, cam_item in iter_cameras(
                        info, args.allow_nonstandard_cameras):
                    path = resolve_image_path(
                        args.data_root, image_prefix, cam_name,
                        str(cam_item['img_path']))
                    total += file_size(path, args.allow_missing)
                values[name].append(total)
            elif kind in DERIVED_KINDS:
                total = 0
                for _, cam_item in iter_cameras(
                        info, args.allow_nonstandard_cameras):
                    total += file_size(
                        modality_file_path(args, kind, method, info, cam_item),
                        args.allow_missing)
                values[name].append(total)
            elif kind == 'occ_gt':
                values[name].append(
                    file_size(
                        modality_file_path(args, kind, method, info),
                        args.allow_missing))

    stats: Dict[str, Dict[str, int]] = {}
    for name, modality_values in values.items():
        stats[name] = {
            'num_samples': len(modality_values),
            'p50': percentile(modality_values, 50),
            'p90': percentile(modality_values, 90),
            'p99': percentile(modality_values, 99),
            'mean': int(np.mean(modality_values)) if modality_values else 0,
            'max': max(modality_values) if modality_values else 0,
        }
    return stats


def align_samples_per_shard(samples: int, base_block_size: int) -> int:
    if base_block_size <= 0:
        raise ValueError('--base-block-size must be positive.')
    if samples <= base_block_size:
        return base_block_size
    return max(base_block_size, (samples // base_block_size) * base_block_size)


def infer_group_samples_per_shard(
    args: argparse.Namespace,
    stats: Mapping[str, Mapping[str, int]],
    groups: Sequence[Tuple[str, Optional[str], str]],
) -> Dict[str, int]:
    if args.base_block_size <= 0:
        raise ValueError('--base-block-size must be positive.')
    if args.samples_per_shard is not None:
        if args.samples_per_shard <= 0:
            raise ValueError('--samples-per-shard must be positive.')
        if args.samples_per_shard % args.base_block_size != 0:
            raise ValueError('--samples-per-shard must be a multiple of --base-block-size.')
        return {name: args.samples_per_shard for _, _, name in groups}

    if args.raw_samples_per_shard is not None:
        if args.raw_samples_per_shard <= 0:
            raise ValueError('--raw-samples-per-shard must be positive.')
        if args.raw_samples_per_shard % args.base_block_size != 0:
            raise ValueError('--raw-samples-per-shard must be a multiple of --base-block-size.')

    target_bytes = parse_size(args.target_shard_size)
    raw_target_bytes = parse_size(args.raw_target_shard_size)
    min_samples = args.base_block_size * args.min_blocks_per_shard
    max_samples = args.base_block_size * args.max_blocks_per_shard
    if min_samples <= 0 or max_samples < min_samples:
        raise ValueError('Invalid --min-blocks-per-shard/--max-blocks-per-shard.')

    result: Dict[str, int] = {}
    for kind, _, name in groups:
        p90 = max(stats.get(name, {}).get('p90', 0), 1)
        if kind == 'raw':
            inferred = max(1, raw_target_bytes // p90)
            result[name] = args.raw_samples_per_shard or align_samples_per_shard(
                int(inferred), args.base_block_size)
        else:
            inferred = int(max(1, target_bytes // p90))
            aligned = align_samples_per_shard(inferred, args.base_block_size)
            result[name] = max(min_samples, min(max_samples, aligned))
    return result


def build_group_shard_plan(infos: Sequence[Mapping[str, Any]],
                           indices: Sequence[int],
                           samples_per_shard: int, split: str, seed: int,
                           group: str,
                           sample_order_hash: str) -> Dict[str, Any]:
    shards = []
    for shard_index, start in enumerate(range(0, len(indices), samples_per_shard)):
        shard_indices = list(indices[start:start + samples_per_shard])
        shard_infos = [infos[i] for i in shard_indices]
        shards.append({
            'shard_id': f'{shard_index:06d}',
            'start': start,
            'end': start + len(shard_indices),
            'global_offsets': list(range(start, start + len(shard_indices))),
            'indices': shard_indices,
            'sample_idx': [sample_identifier(info) for info in shard_infos],
            'scene_idx': [scene_identifier(info) for info in shard_infos],
            'token': [str(info.get('token', '')) for info in shard_infos],
        })
    return {
        'schema_version': 2,
        'split': split,
        'group': group,
        'seed': seed,
        'sample_order_sha256': sample_order_hash,
        'samples_per_shard': samples_per_shard,
        'num_samples': len(indices),
        'num_shards': len(shards),
        'shards': shards,
    }


def iter_cameras(
    info: Mapping[str, Any],
    allow_nonstandard_cameras: bool = False,
) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    images = info.get('images')
    if not isinstance(images, Mapping):
        raise KeyError('Sample info missing images mapping.')
    if not allow_nonstandard_cameras:
        missing = [cam for cam in CAMERA_ORDER if cam not in images]
        extra = [cam for cam in images if cam not in CAMERA_ORDER]
        if missing or extra:
            raise ValueError(
                f'Expected exactly {list(CAMERA_ORDER)} cameras, '
                f'got missing={missing}, extra={extra}.')
        for cam_name in CAMERA_ORDER:
            yield cam_name, images[cam_name]
        return
    yielded = set()
    for cam_name in CAMERA_ORDER:
        if cam_name in images:
            yielded.add(cam_name)
            yield cam_name, images[cam_name]
    for cam_name, cam_item in images.items():
        if cam_name not in yielded:
            yield cam_name, cam_item


def make_meta_sample(info: Mapping[str, Any]) -> Dict[str, Any]:
    meta = copy.deepcopy(dict(info))
    meta['occ_path'] = f"gts/{info.get('scene_idx')}/{info.get('token')}"
    return meta


def read_file_bytes(path: Path, allow_missing: bool) -> Optional[bytes]:
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(path)
    return path.read_bytes()


def load_npy_tensor(path: Path, allow_missing: bool) -> Optional[torch.Tensor]:
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(path)
    array = np.load(path)
    return torch.from_numpy(np.asarray(array))


def load_occ_npz(path: Path, allow_missing: bool) -> Optional[Dict[str, torch.Tensor]]:
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(path)
    with np.load(path) as occ:
        for key in ('semantics', 'mask_lidar', 'mask_camera'):
            if key not in occ:
                raise KeyError(f'{path} missing key {key}.')
        return {
            'semantics': torch.from_numpy(np.asarray(occ['semantics'])),
            'mask_lidar': torch.from_numpy(np.asarray(occ['mask_lidar'])),
            'mask_camera': torch.from_numpy(np.asarray(occ['mask_camera'])),
        }


def build_group_payload(
    args: argparse.Namespace,
    infos: Sequence[Mapping[str, Any]],
    shard: Mapping[str, Any],
    group: Tuple[str, Optional[str], str],
    sample_order_hash: str,
) -> Dict[str, Any]:
    kind, method, name = group
    image_prefix = args.image_prefix or args.data_root
    samples = []
    for idx in shard['indices']:
        info = infos[int(idx)]
        sample = {
            'sample_idx': sample_identifier(info),
            'scene_idx': scene_identifier(info),
            'token': str(info.get('token', '')),
        }
        if kind == 'raw':
            sample['meta'] = make_meta_sample(info)
            images = {}
            for cam_name, cam_item in iter_cameras(
                    info, args.allow_nonstandard_cameras):
                path = resolve_image_path(
                    args.data_root, image_prefix, cam_name,
                    str(cam_item['img_path']))
                images[cam_name] = {
                    'img_path': str(cam_item['img_path']),
                    'source_path': str(path),
                    'bytes': read_file_bytes(path, args.allow_missing),
                }
            sample['images'] = images
        elif kind in DERIVED_KINDS:
            views = {}
            for cam_name, cam_item in iter_cameras(
                    info, args.allow_nonstandard_cameras):
                path = modality_file_path(args, kind, method, info, cam_item)
                views[cam_name] = {
                    'img_path': str(cam_item['img_path']),
                    'source_path': str(path),
                    'tensor': load_npy_tensor(path, args.allow_missing),
                }
            sample[kind] = views
        elif kind == 'occ_gt':
            path = modality_file_path(args, kind, method, info)
            sample['occ_gt'] = {
                'source_path': str(path),
                'arrays': load_occ_npz(path, args.allow_missing),
            }
        samples.append(sample)

    sample_idx = list(shard['sample_idx'])
    return {
        'schema_version': 2,
        'format': 'gausstr-sharded-v2',
        'split': args.split,
        'shard_id': shard['shard_id'],
        'group': name,
        'kind': kind,
        'method': method,
        'global_offsets': list(shard['global_offsets']),
        'sample_idx': sample_idx,
        'sample_idx_sha256': stable_json_hash(sample_idx),
        'sample_order_sha256': sample_order_hash,
        'camera_order': list(CAMERA_ORDER),
        'samples': samples,
    }


def summarize_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    kind = str(payload['kind'])
    shape_counts: Dict[str, int] = {}
    dtype_counts: Dict[str, int] = {}
    missing_samples = []

    def add_tensor(tensor: Optional[torch.Tensor], sample_idx: str) -> None:
        if tensor is None:
            missing_samples.append(sample_idx)
            return
        shape = 'x'.join(str(dim) for dim in tensor.shape)
        dtype = str(tensor.dtype).replace('torch.', '')
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

    for sample in payload['samples']:
        sample_idx = str(sample['sample_idx'])
        if kind == 'raw':
            for view in sample['images'].values():
                if view['bytes'] is None:
                    missing_samples.append(sample_idx)
        elif kind in DERIVED_KINDS:
            for view in sample[kind].values():
                add_tensor(view['tensor'], sample_idx)
        elif kind == 'occ_gt':
            arrays = sample['occ_gt']['arrays']
            if arrays is None:
                missing_samples.append(sample_idx)
            else:
                for tensor in arrays.values():
                    add_tensor(tensor, sample_idx)

    return {
        'shape_summary': dict(sorted(shape_counts.items())),
        'dtype_summary': dict(sorted(dtype_counts.items())),
        'missing_count': len(missing_samples),
        'missing_samples': sorted(set(missing_samples)),
    }


def torch_save_atomic(path: Path,
                      payload: Mapping[str, Any],
                      overwrite: bool,
                      sanity_load: bool,
                      compute_sha256: bool) -> Dict[str, Any]:
    success_path = path.with_suffix('.SUCCESS')
    sha_path = path.with_suffix(path.suffix + '.sha256')
    if path.exists() and success_path.exists() and not overwrite:
        return {
            'path': str(path),
            'bytes': path.stat().st_size,
            'sha256': sha_path.read_text().strip() if sha_path.exists() else None,
            'skipped': True,
        }
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} already exists. Pass --overwrite.')

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    success_tmp = path.with_suffix('.SUCCESS.tmp')
    sha_tmp = path.with_suffix(path.suffix + '.sha256.tmp')
    for stale in (tmp_path, success_tmp, sha_tmp):
        if stale.exists():
            stale.unlink()

    torch.save(payload, tmp_path)
    if sanity_load:
        loaded = torch_load(tmp_path)
        if loaded.get('sample_idx') != payload.get('sample_idx'):
            raise RuntimeError(f'Sanity load sample_idx mismatch: {tmp_path}')

    digest = None
    if compute_sha256:
        digest = file_sha256(tmp_path)
        sha_tmp.write_text(digest + '\n', encoding='utf-8')

    tmp_path.replace(path)
    if digest is not None:
        sha_tmp.replace(sha_path)
    elif overwrite and sha_path.exists():
        sha_path.unlink()
    success_tmp.write_text(
        json.dumps({
            'path': str(path),
            'bytes': path.stat().st_size,
            'sha256': digest,
            'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        },
                   sort_keys=True) + '\n',
        encoding='utf-8')
    success_tmp.replace(success_path)
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': digest,
        'skipped': False,
    }


def write_group_shards(
    args: argparse.Namespace,
    infos: Sequence[Mapping[str, Any]],
    shard_plan: Mapping[str, Any],
    group: Tuple[str, Optional[str], str],
    sample_order_hash: str,
) -> Dict[str, Any]:
    kind, method, name = group
    split_root = args.out_root / args.split
    rel_dir = group_rel_dir(kind, method)
    group_manifest = {
        'schema_version': 2,
        'format': 'gausstr-sharded-v2',
        'split': args.split,
        'group': name,
        'kind': kind,
        'method': method,
        'relative_dir': str(rel_dir),
        'sample_order_sha256': sample_order_hash,
        'shard_plan_sha256': stable_json_hash(make_public_shard_plan(shard_plan)),
        'samples_per_shard': shard_plan['samples_per_shard'],
        'producer': producer_info(args),
        'shards': {},
    }
    shards = list(shard_plan['shards'])
    manifest_path = split_root / rel_dir / 'group_manifest.json'
    existing_manifest = None if args.overwrite else load_json_if_exists(manifest_path)
    existing_entries = {}
    if existing_manifest is not None:
        existing_entries = existing_manifest.get('shards', {})

    def existing_entry(shard: Mapping[str, Any], shard_path: Path) -> Optional[Dict[str, Any]]:
        success_path = shard_path.with_suffix('.SUCCESS')
        if args.overwrite or not shard_path.exists() or not success_path.exists():
            return None
        shard_id = str(shard['shard_id'])
        entry = existing_entries.get(shard_id)
        if entry is not None:
            return entry
        sample_idx = list(shard['sample_idx'])
        sha_path = shard_path.with_suffix(shard_path.suffix + '.sha256')
        return {
            'path': str((rel_dir / f'{shard_id}.torch').as_posix()),
            'success_path': str((rel_dir / f'{shard_id}.SUCCESS').as_posix()),
            'num_samples': len(sample_idx),
            'sample_idx_sha256': stable_json_hash(sample_idx),
            'bytes': shard_path.stat().st_size,
            'sha256': sha_path.read_text().strip() if sha_path.exists() else None,
            'shape_summary': {},
            'dtype_summary': {},
            'missing_count': None,
            'missing_samples': [],
        }

    def write_one(shard: Mapping[str, Any]) -> Tuple[str, Dict[str, Any], Path, bool]:
        shard_path = split_root / rel_dir / f"{shard['shard_id']}.torch"
        entry = existing_entry(shard, shard_path)
        if entry is not None:
            return str(shard['shard_id']), entry, shard_path, True

        payload = build_group_payload(args, infos, shard, group,
                                      sample_order_hash)
        payload_summary = summarize_payload(payload)
        write_info = torch_save_atomic(
            shard_path,
            payload,
            overwrite=args.overwrite,
            sanity_load=not args.no_sanity_load,
            compute_sha256=not args.skip_file_sha256)
        sample_idx = list(shard['sample_idx'])
        entry = {
            'path': str((rel_dir / f"{shard['shard_id']}.torch").as_posix()),
            'success_path': str((rel_dir / f"{shard['shard_id']}.SUCCESS").as_posix()),
            'num_samples': len(sample_idx),
            'sample_idx_sha256': stable_json_hash(sample_idx),
            'bytes': write_info['bytes'],
            'sha256': write_info['sha256'],
            **payload_summary,
        }
        return str(shard['shard_id']), entry, shard_path, write_info['skipped']

    if args.num_workers <= 1 or len(shards) <= 1:
        for index, shard in enumerate(shards, 1):
            shard_id, entry, shard_path, skipped = write_one(shard)
            group_manifest['shards'][shard_id] = entry
            action = 'skipped' if skipped else 'wrote'
            print(f'[{name}] {index}/{len(shards)} {action} shard {shard_id} -> {shard_path}')
    else:
        max_workers = min(args.num_workers, len(shards))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(write_one, shard) for shard in shards]
            for index, future in enumerate(as_completed(futures), 1):
                shard_id, entry, shard_path, skipped = future.result()
                group_manifest['shards'][shard_id] = entry
                action = 'skipped' if skipped else 'wrote'
                print(f'[{name}] {index}/{len(shards)} {action} shard {shard_id} -> {shard_path}')

    group_manifest['shards'] = dict(sorted(group_manifest['shards'].items()))

    dump_json_replace_existing(manifest_path, group_manifest)
    dump_json_replace_existing(split_root / rel_dir / 'shard_plan.json',
                               make_public_shard_plan(shard_plan))
    return group_manifest


def producer_info(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        'script': 'tools/build_sharded_dataset.py',
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'numpy': np.__version__,
        'command': ' '.join(sys.argv),
        'target_shard_size': args.target_shard_size,
        'raw_target_shard_size': args.raw_target_shard_size,
        'base_block_size': args.base_block_size,
        'num_workers': args.num_workers,
        'compression': args.compression,
    }


def write_manifests(args: argparse.Namespace,
                    ann_data: Mapping[str, Any],
                    groups: Sequence[Tuple[str, Optional[str], str]],
                    stats: Mapping[str, Any],
                    sample_order_hash: str,
                    group_plan_hashes: Mapping[str, str]) -> None:
    root_manifest = {
        'schema_version': 2,
        'format': 'gausstr-sharded-v2',
        'dataset': 'nuscenes',
        'source_info_version': 'openmmlab-v2',
        'num_views': len(CAMERA_ORDER),
        'camera_order': list(CAMERA_ORDER),
        'metainfo': ann_data.get('metainfo', {}),
        'producer': producer_info(args),
    }
    root_manifest_path = args.out_root / 'manifest.json'
    dump_json_or_skip_existing(
        root_manifest_path, root_manifest, overwrite=args.overwrite)

    split_manifest = {
        **root_manifest,
        'split': args.split,
        'ann_file': str(args.ann_file),
        'data_root': str(args.data_root),
        'source_roots': {
            'image_prefix': str(args.image_prefix or args.data_root),
            'depth_root': str(args.depth_root),
            'feat_root': str(args.feat_root),
            'sem_seg_root': str(args.sem_seg_root),
            'occ_root': str(args.occ_root or args.data_root),
        },
        'required_groups': [name for _, _, name in groups],
        'stats': stats,
        'sample_order_sha256': sample_order_hash,
        'group_plan_sha256': dict(group_plan_hashes),
    }
    dump_json_or_skip_existing(
        args.out_root / args.split / 'manifest.json',
        split_manifest,
        overwrite=args.overwrite)


def write_sample_order(split_root: Path, infos: Sequence[Mapping[str, Any]],
                       indices: Sequence[int], split: str, seed: int,
                       overwrite: bool) -> str:
    samples = []
    for offset, idx in enumerate(indices):
        info = infos[idx]
        samples.append({
            'global_offset': offset,
            'source_index': int(idx),
            'sample_idx': sample_identifier(info),
            'token': str(info.get('token', '')),
            'scene_idx': scene_identifier(info),
        })
    sample_order = {
        'schema_version': 2,
        'split': split,
        'seed': seed,
        'num_samples': len(samples),
        'samples': samples,
    }
    sample_order_hash = stable_json_hash(sample_order)
    sample_order['sample_order_sha256'] = sample_order_hash
    dump_json(split_root / 'sample_order.json', sample_order, overwrite=overwrite)
    return sample_order_hash


def write_index(split_root: Path, infos: Sequence[Mapping[str, Any]],
                indices: Sequence[int],
                group_plans: Mapping[str, Mapping[str, Any]],
                sample_order_hash: str,
                overwrite: bool) -> None:
    index = {
        'schema_version': 2,
        'split': next(iter(group_plans.values()))['split'] if group_plans else '',
        'num_samples': len(indices),
        'sample_order_sha256': sample_order_hash,
        'samples': [],
        'groups': {},
        'by_sample_idx': {},
    }
    for offset, idx in enumerate(indices):
        info = infos[idx]
        sample_idx = sample_identifier(info)
        item = {
            'global_offset': offset,
            'sample_idx': sample_idx,
            'token': str(info.get('token', '')),
            'scene_idx': scene_identifier(info),
        }
        index['samples'].append(item)
        index['by_sample_idx'][sample_idx] = {
            'global_offset': offset,
            'groups': {},
        }

    for group_name_, plan in group_plans.items():
        index['groups'][group_name_] = {}
        for shard in plan['shards']:
            index['groups'][group_name_][shard['shard_id']] = {
                'start': shard['start'],
                'end': shard['end'],
            }
            for offset, sample_idx in enumerate(shard['sample_idx']):
                index['by_sample_idx'][sample_idx]['groups'][group_name_] = {
                    'shard_id': shard['shard_id'],
                    'offset': offset,
                    'global_offset': shard['global_offsets'][offset],
                }
    dump_json(split_root / 'index.json', index, overwrite=overwrite)


def write_build_success(split_root: Path, summary: Mapping[str, Any]) -> None:
    success_path = split_root / '_SUCCESS'
    success_tmp = split_root / '_SUCCESS.tmp'
    payload = dict(summary)
    payload['time'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    with success_tmp.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')
    success_tmp.replace(success_path)


def make_public_shard_plan(shard_plan: Mapping[str, Any]) -> Dict[str, Any]:
    public_plan = copy.deepcopy(dict(shard_plan))
    for shard in public_plan['shards']:
        shard.pop('indices', None)
    return public_plan


def copy_ann_subset(args: argparse.Namespace, ann_data: Mapping[str, Any],
                    infos: Sequence[Mapping[str, Any]],
                    indices: Sequence[int]) -> None:
    subset = dict(ann_data)
    subset['data_list'] = [infos[i] for i in indices]
    path = args.out_root / args.split / 'sample_manifest.pkl'
    if path.exists() and not args.overwrite:
        print(f'Skipping existing sample manifest: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('wb') as f:
        pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def write_sample_manifest_json(args: argparse.Namespace,
                               infos: Sequence[Mapping[str, Any]],
                               indices: Sequence[int],
                               groups: Sequence[Tuple[str, Optional[str], str]]) -> None:
    image_prefix = args.image_prefix or args.data_root
    samples = []
    for idx in indices:
        info = infos[idx]
        item = {
            'sample_idx': sample_identifier(info),
            'token': str(info.get('token', '')),
            'scene_idx': scene_identifier(info),
            'images': {},
            'groups': {},
        }
        for cam_name, cam_item in iter_cameras(
                info, args.allow_nonstandard_cameras):
            item['images'][cam_name] = str(
                resolve_image_path(args.data_root, image_prefix, cam_name,
                                   str(cam_item['img_path'])))
        for kind, method, name in groups:
            if kind in DERIVED_KINDS:
                item['groups'][name] = {
                    cam_name: str(modality_file_path(args, kind, method, info,
                                                     cam_item))
                    for cam_name, cam_item in iter_cameras(
                        info, args.allow_nonstandard_cameras)
                }
            elif kind == 'occ_gt':
                item['groups'][name] = str(
                    modality_file_path(args, kind, method, info))
        samples.append(item)
    dump_json(
        args.out_root / args.split / 'sample_manifest.json', {
            'schema_version': 2,
            'split': args.split,
            'num_samples': len(samples),
            'samples': samples,
        },
        overwrite=args.overwrite)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser()
    args.out_root = args.out_root.expanduser()
    args.depth_root = args.depth_root.expanduser()
    args.feat_root = args.feat_root.expanduser()
    args.sem_seg_root = args.sem_seg_root.expanduser()
    args.occ_root = args.occ_root.expanduser() if args.occ_root else None
    args.image_prefix = args.image_prefix.expanduser() if args.image_prefix else None

    if args.ann_file is None:
        args.ann_file = args.data_root / f'nuscenes_infos_{args.split}.pkl'
    else:
        args.ann_file = args.ann_file.expanduser()
    raw_modalities = args.modalities or preset_modalities(args.preset, args.split)
    groups = normalize_modalities(raw_modalities)
    group_names = [name for _, _, name in groups]
    print(f'Loading annotation file: {args.ann_file}')
    ann_data = load_pickle(args.ann_file)
    infos = ann_data['data_list']
    print(f'Loaded {len(infos)} samples.')

    indices = select_indices(infos, args.split, args.ratio, args.max_samples,
                             args.seed)
    print(f'Selected {len(indices)} samples for split {args.split}.')
    validate_infos(
        infos,
        indices,
        require_occ='occ_gt' in group_names,
        allow_nonstandard_cameras=args.allow_nonstandard_cameras)

    stats = estimate_group_sizes(args, infos, indices, groups)
    group_samples_per_shard = infer_group_samples_per_shard(
        args, stats, groups)
    print('Using group samples_per_shard:')
    for name in group_names:
        print(f'  {name}: {group_samples_per_shard[name]}')

    split_root = args.out_root / args.split
    sample_order_hash = write_sample_order(split_root, infos, indices,
                                           args.split, args.seed,
                                           args.overwrite)
    group_plans = {}
    group_plan_hashes = {}
    for _, _, name in groups:
        plan = build_group_shard_plan(
            infos, indices, group_samples_per_shard[name], args.split,
            args.seed, name, sample_order_hash)
        public_plan = make_public_shard_plan(plan)
        group_plans[name] = plan
        group_plan_hashes[name] = stable_json_hash(public_plan)

    dump_json(split_root / 'stats.json', stats, overwrite=args.overwrite)
    write_index(split_root, infos, indices, group_plans, sample_order_hash,
                args.overwrite)
    copy_ann_subset(args, ann_data, infos, indices)
    write_sample_manifest_json(args, infos, indices, groups)
    write_manifests(args, ann_data, groups, stats, sample_order_hash,
                    group_plan_hashes)

    group_manifests = {}
    for group in groups:
        group_manifest = write_group_shards(args, infos, group_plans[group[2]],
                                            group, sample_order_hash)
        group_manifests[group[2]] = {
            'relative_dir': group_manifest['relative_dir'],
            'num_shards': len(group_manifest['shards']),
            'samples_per_shard': group_plans[group[2]]['samples_per_shard'],
        }

    summary = {
        'schema_version': 2,
        'split': args.split,
        'num_samples': len(indices),
        'sample_order_sha256': sample_order_hash,
        'groups': group_manifests,
        'group_plan_sha256': group_plan_hashes,
    }
    dump_json(split_root / 'build_summary.json', summary, overwrite=True)
    write_build_success(split_root, summary)
    print(f'Done. Summary written to {split_root / "build_summary.json"}')


if __name__ == '__main__':
    main()
