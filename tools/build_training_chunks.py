#!/usr/bin/env python
"""Build PixelSplat-style fused training chunks for GaussTR.

This script materializes complete training/evaluation samples directly from the
nuScenes annotation and source data folders. It does not read or write the old
grouped-shard format.
"""

from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import io
import json
import math
import pickle
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


CAMERA_ORDER = (
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
    'CAM_BACK_LEFT',
)

PROFILE_DEFS = {
    'talk2dino_metric3d': {
        'depth': 'metric3d',
        'feats': None,
        'sem_seg': None,
        'image_size': (504, 896),
        'resize_scale': 0.56,
        'patch_size': 14,
    },
    'featup_metric3d_sam2': {
        'depth': 'metric3d',
        'feats': 'featup',
        'sem_seg': 'grounded_sam2',
        'image_size': (432, 768),
        'resize_scale': 0.48,
        'patch_size': 16,
    },
}

RETRYABLE_FS_ERRNOS = {
    errno.EAGAIN,
    errno.EBUSY,
    errno.EINTR,
    getattr(errno, 'ESTALE', 116),
}
RETRYABLE_TORCH_LOAD_MESSAGES = (
    'PytorchStreamReader failed reading zip archive',
    'failed finding central directory',
    'failed locating file',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build fused .torch training chunks for GaussTR.')
    parser.add_argument(
        '--profile',
        choices=tuple(PROFILE_DEFS),
        default='featup_metric3d_sam2',
        help='Training profile to materialize.')
    parser.add_argument(
        '--split',
        choices=('train', 'val', 'test'),
        default='train',
        help='Dataset split written under out-root.')
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('data/nuscenes'),
        help='nuScenes root containing annotation, samples and gts.')
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=None,
        help='Path to nuscenes_infos_{split}.pkl. Defaults from --data-root.')
    parser.add_argument(
        '--image-prefix',
        type=Path,
        default=None,
        help='Root containing samples/CAM_* images. Defaults to data-root.')
    parser.add_argument(
        '--depth-root',
        type=Path,
        default=Path('data/nuscenes_metric3d'),
        help='Root for metric3d depth .npy files.')
    parser.add_argument(
        '--feat-root',
        type=Path,
        default=Path('data/nuscenes_featup'),
        help='Root for featup .npy files.')
    parser.add_argument(
        '--sem-seg-root',
        type=Path,
        default=Path('data/nuscenes_grounded_sam2'),
        help='Root for grounded-sam2 semantic pseudo-label .npy files.')
    parser.add_argument(
        '--occ-root',
        type=Path,
        default=None,
        help='Root containing gts/{scene_idx}/{token}/labels.npz. Defaults to data-root.')
    parser.add_argument(
        '--out-root',
        type=Path,
        default=Path('data/gausstr_chunks'),
        help='Output root. Defaults to data/gausstr_chunks and can be a symlink to a TOS mount path.')
    parser.add_argument(
        '--ratio',
        type=float,
        default=1.0,
        help='Scene-level subset ratio in (0, 1].')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional maximum number of selected samples.')
    parser.add_argument(
        '--seed',
        type=int,
        default=2026,
        help='Random seed for train subset and sample order.')
    parser.add_argument(
        '--target-chunk-size',
        default='100MB',
        help='Target chunk size, e.g. 100MB.')
    parser.add_argument(
        '--max-chunk-size',
        default='160MB',
        help='Hard chunk size limit. Chunks larger than this fail fast.')
    parser.add_argument(
        '--min-samples-per-chunk',
        type=int,
        default=1,
        help='Lower bound for inferred samples_per_chunk.')
    parser.add_argument(
        '--max-samples-per-chunk',
        type=int,
        default=64,
        help='Upper bound for inferred samples_per_chunk.')
    parser.add_argument(
        '--samples-per-chunk',
        type=int,
        default=None,
        help='Override inferred samples_per_chunk.')
    parser.add_argument(
        '--size-estimate-samples',
        type=int,
        default=4,
        help='Number of randomly selected samples used by sizing pass.')
    parser.add_argument(
        '--size-percentile',
        type=float,
        default=95,
        help='Sample-size percentile used to infer samples_per_chunk.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of chunk build workers. Use 1 for serial processing.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing chunks and manifests.')
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
        help='Skip torch.load sanity check after writing each chunk.')
    parser.add_argument(
        '--skip-file-sha256',
        action='store_true',
        help='Skip full-file sha256 computation for chunk files.')
    parser.add_argument(
        '--validate-sha256',
        action='store_true',
        help='Re-read chunk files at the end to validate sha256.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only run selection and sizing pass; do not write chunks.')
    return parser.parse_args()


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
            return int(float(text[:-len(unit)].strip()) * multiplier)
    return int(float(text))


def load_pickle(path: Path) -> Any:
    try:
        import mmengine

        return mmengine.load(path)
    except Exception:
        with path.open('rb') as f:
            return pickle.load(f)


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


def is_retryable_torch_load_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and any(
        message in str(exc) for message in RETRYABLE_TORCH_LOAD_MESSAGES)


def torch_load_once(path: Path) -> Any:
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def torch_load(path: Path,
               retries: int = 0,
               base_delay: float = 0.5) -> Any:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return torch_load_once(path)
        except OSError as exc:
            if exc.errno not in RETRYABLE_FS_ERRNOS or attempt >= retries:
                raise
            last_exc = exc
        except RuntimeError as exc:
            if not is_retryable_torch_load_error(exc) or attempt >= retries:
                raise
            last_exc = exc
        delay = min(base_delay * (1.5**attempt), 5.0)
        print(
            f'Retrying torch.load for {path} '
            f'({attempt + 1}/{retries}) after {last_exc!r}; '
            f'sleep={delay:.1f}s',
            flush=True)
        time.sleep(delay)
    raise RuntimeError(f'Unreachable torch_load retry state for {path}')


def replace_with_retry(src: Path,
                       dst: Path,
                       retries: int = 20,
                       base_delay: float = 0.2) -> None:
    for attempt in range(retries + 1):
        try:
            src.replace(dst)
            return
        except OSError as exc:
            if exc.errno not in RETRYABLE_FS_ERRNOS or attempt >= retries:
                raise
            time.sleep(min(base_delay * (1.5**attempt), 5.0))


def unlink_with_retry(path: Path,
                      retries: int = 20,
                      base_delay: float = 0.2) -> None:
    for attempt in range(retries + 1):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno not in RETRYABLE_FS_ERRNOS or attempt >= retries:
                raise
            time.sleep(min(base_delay * (1.5**attempt), 5.0))


def io_call_with_retry(description: str,
                       fn,
                       retries: int = 10,
                       base_delay: float = 0.2):
    for attempt in range(retries + 1):
        try:
            return fn()
        except OSError as exc:
            if exc.errno not in RETRYABLE_FS_ERRNOS or attempt >= retries:
                raise
            delay = min(base_delay * (1.5**attempt), 5.0)
            print(
                f'Retrying {description} ({attempt + 1}/{retries}) after '
                f'{exc!r}; sleep={delay:.1f}s',
                flush=True)
            time.sleep(delay)
    raise RuntimeError(f'Unreachable retry state for {description}')


def path_exists_with_retry(path: Path,
                           retries: int = 10,
                           base_delay: float = 0.2) -> bool:
    for attempt in range(retries + 1):
        try:
            path.stat()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            if exc.errno not in RETRYABLE_FS_ERRNOS or attempt >= retries:
                raise
            delay = min(base_delay * (1.5**attempt), 5.0)
            print(
                f'Retrying stat {path} ({attempt + 1}/{retries}) after '
                f'{exc!r}; sleep={delay:.1f}s',
                flush=True)
            time.sleep(delay)
    raise RuntimeError(f'Unreachable stat retry state for {path}')


def dump_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    replace_with_retry(tmp_path, path)


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path_exists_with_retry(path):
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


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


def validate_infos(
    infos: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    profile: Mapping[str, Optional[str]],
    split: str,
    allow_nonstandard_cameras: bool,
) -> None:
    require_occ = split in {'val', 'test'}
    for idx in selected_indices:
        info = infos[idx]
        ident = info.get('sample_idx', info.get('token', idx))
        for key in ('token', 'scene_idx', 'scene_token', 'images'):
            if key not in info:
                raise KeyError(f'Sample {ident} missing required key: {key}')
        for _, cam_item in iter_cameras(info, allow_nonstandard_cameras):
            for key in ('img_path', 'cam2ego', 'cam2img', 'lidar2cam'):
                if key not in cam_item:
                    raise KeyError(f'Sample {ident} camera missing {key}.')
        if require_occ:
            for key in ('scene_idx', 'token'):
                if key not in info:
                    raise KeyError(f'Sample {ident} missing occ key: {key}')


def group_by_scene(infos: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    scenes: Dict[str, List[int]] = {}
    for idx, info in enumerate(infos):
        scenes.setdefault(scene_identifier(info), []).append(idx)
    return scenes


def select_indices(infos: Sequence[Mapping[str, Any]], split: str, ratio: float,
                   max_samples: Optional[int], seed: int) -> List[int]:
    if ratio <= 0 or ratio > 1:
        raise ValueError('--ratio must be in (0, 1].')
    if max_samples is not None and max_samples <= 0:
        raise ValueError('--max-samples must be positive.')
    if ratio >= 1 and max_samples is None:
        indices = list(range(len(infos)))
        if split == 'train':
            rng = random.Random(seed)
            rng.shuffle(indices)
        return indices

    target = max(1, math.ceil(len(infos) * ratio))
    if max_samples is not None:
        target = min(target, max_samples)
    scenes = group_by_scene(infos)
    scene_ids = list(scenes)
    if split == 'train':
        rng = random.Random(seed)
        rng.shuffle(scene_ids)
    else:
        scene_ids.sort()

    selected: List[int] = []
    for scene_id in scene_ids:
        remaining = target - len(selected)
        if remaining <= 0:
            break
        selected.extend(scenes[scene_id][:remaining])
        if len(selected) >= target:
            break
    if split == 'train':
        rng = random.Random(seed)
        rng.shuffle(selected)
    else:
        selected.sort()
    return selected


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
        if path_exists_with_retry(candidate):
            return candidate
    return candidates[0]


def modality_file_path(args: argparse.Namespace, kind: str,
                       info: Mapping[str, Any],
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
    raise ValueError(f'Unsupported modality kind: {kind}')


def make_meta_sample(info: Mapping[str, Any]) -> Dict[str, Any]:
    meta = copy.deepcopy(dict(info))
    meta['occ_path'] = f"gts/{info.get('scene_idx')}/{info.get('token')}"
    return meta


def read_file_bytes(path: Path, allow_missing: bool) -> Optional[bytes]:
    try:
        return io_call_with_retry(f'read bytes {path}', path.read_bytes)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise


def profile_image_size(profile_name: str) -> Tuple[int, int]:
    return tuple(PROFILE_DEFS[profile_name]['image_size'])  # type: ignore


def profile_resize_scale(profile_name: str) -> float:
    return float(PROFILE_DEFS[profile_name]['resize_scale'])


def materialized_aug_mat(resize_scale: float,
                         crop_xy: Tuple[int, int] = (0, 0)) -> List[List[float]]:
    crop_x, crop_y = crop_xy
    mat = np.eye(4, dtype=np.float32)
    mat[0, 0] = resize_scale
    mat[1, 1] = resize_scale
    mat[0, 3] = -float(crop_x)
    mat[1, 3] = -float(crop_y)
    return mat.tolist()


def materialize_image_bytes(data: Optional[bytes],
                            target_hw: Tuple[int, int],
                            resize_scale: float,
                            allow_missing: bool) -> Tuple[Optional[bytes], Dict[str, Any]]:
    if data is None:
        if allow_missing:
            return None, {}
        raise FileNotFoundError('Cannot materialize missing image bytes.')

    target_h, target_w = target_hw
    image = Image.open(io.BytesIO(data)).convert('RGB')
    original_w, original_h = image.size
    resize_w = int(original_w * resize_scale)
    resize_h = int(original_h * resize_scale)
    if resize_w < target_w or resize_h < target_h:
        raise ValueError(
            f'Resized image {(resize_h, resize_w)} is smaller than target '
            f'{target_hw}. original={(original_h, original_w)} '
            f'resize_scale={resize_scale}.')
    crop_x = max(0, (resize_w - target_w) // 2)
    crop_y = max(0, resize_h - target_h)
    image = image.resize((resize_w, resize_h))
    image = image.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue(), {
        'materialized': True,
        'original_shape': [original_h, original_w],
        'shape': [target_h, target_w],
        'resize_scale': resize_scale,
        'crop_xy': [crop_x, crop_y],
        'img_aug_mat': materialized_aug_mat(resize_scale, (crop_x, crop_y)),
        'encoding': 'jpeg',
    }


def load_npy_tensor(path: Path, allow_missing: bool) -> Optional[torch.Tensor]:
    try:
        array = io_call_with_retry(f'np.load {path}', lambda: np.load(path))
    except FileNotFoundError:
        if allow_missing:
            return None
        raise
    return torch.from_numpy(np.asarray(array))


def resize_2d_tensor(tensor: Optional[torch.Tensor],
                     target_hw: Tuple[int, int],
                     mode: str,
                     resize_scale: float) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim != 2:
        raise ValueError(
            f'Expected a 2D tensor for materialized resize, got {tensor.shape}.')
    target_h, target_w = target_hw
    h, w = int(tensor.shape[0]), int(tensor.shape[1])
    resize_h = int(h * resize_scale)
    resize_w = int(w * resize_scale)
    if resize_w < target_w or resize_h < target_h:
        raise ValueError(
            f'Resized tensor {(resize_h, resize_w)} is smaller than target '
            f'{target_hw}. original={(h, w)} resize_scale={resize_scale}.')
    crop_x = max(0, (resize_w - target_w) // 2)
    crop_y = max(0, resize_h - target_h)
    orig_dtype = tensor.dtype
    value = tensor
    if not torch.is_floating_point(value):
        value = value.float()
    kwargs = {}
    if mode != 'nearest':
        kwargs['align_corners'] = False
    value = F.interpolate(
        value[None, None],
        size=(resize_h, resize_w),
        mode=mode,
        **kwargs).squeeze(0).squeeze(0)
    value = value[crop_y:crop_y + target_h, crop_x:crop_x + target_w]
    if not torch.is_floating_point(torch.empty((), dtype=orig_dtype)):
        value = value.round().to(orig_dtype)
    return value


def load_occ_npz(path: Path, allow_missing: bool) -> Optional[Dict[str, torch.Tensor]]:
    def load_arrays():
        with np.load(path) as occ:
            for key in ('semantics', 'mask_lidar', 'mask_camera'):
                if key not in occ:
                    raise KeyError(f'{path} missing key {key}.')
            return {
                'semantics': torch.from_numpy(np.asarray(occ['semantics'])),
                'mask_lidar': torch.from_numpy(np.asarray(occ['mask_lidar'])),
                'mask_camera': torch.from_numpy(np.asarray(occ['mask_camera'])),
            }

    try:
        return io_call_with_retry(f'np.load {path}', load_arrays)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise


def profile_fields(profile_name: str, split: str) -> Dict[str, Optional[str]]:
    fields = dict(PROFILE_DEFS[profile_name])
    if split in {'val', 'test'}:
        fields['sem_seg'] = None
        fields['occ_gt'] = 'occ_gt'
    else:
        fields['occ_gt'] = None
    return fields


def profile_payload(args: argparse.Namespace,
                    fields: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    source_roots = {
        'data_root': str(args.data_root),
        'image_prefix': str(args.image_prefix or args.data_root),
    }
    if fields.get('depth'):
        source_roots['depth_root'] = str(args.depth_root)
    if fields.get('feats'):
        source_roots['feat_root'] = str(args.feat_root)
    if fields.get('sem_seg'):
        source_roots['sem_seg_root'] = str(args.sem_seg_root)
    if fields.get('occ_gt'):
        source_roots['occ_root'] = str(args.occ_root or args.data_root)
    payload = {
        'schema_version': 1,
        'name': args.profile,
        'split': args.split,
        'source_roots': source_roots,
        'fields': {
            'raw': True,
            'depth': fields.get('depth'),
            'feats': fields.get('feats'),
            'sem_seg': fields.get('sem_seg'),
            'occ_gt': bool(fields.get('occ_gt')),
        },
        'materialization': {
            'image_size': list(profile_image_size(args.profile)),
            'resize_scale': profile_resize_scale(args.profile),
            'patch_size': int(PROFILE_DEFS[args.profile]['patch_size']),
            'raw_images': True,
            'depth': bool(fields.get('depth')),
            'sem_seg': bool(fields.get('sem_seg')),
            'feats': False,
            'occ_gt': False,
        },
        'camera_order': list(CAMERA_ORDER),
    }
    payload['profile_sha256'] = stable_json_hash(payload)
    return payload


def build_sample(args: argparse.Namespace, info: Mapping[str, Any],
                 source_index: int, global_offset: int,
                 fields: Mapping[str, Optional[str]],
                 is_padding: bool = False,
                 source_sample_idx: Optional[str] = None) -> Dict[str, Any]:
    image_prefix = args.image_prefix or args.data_root
    sample_idx = sample_identifier(info)
    target_hw = profile_image_size(args.profile)
    resize_scale = profile_resize_scale(args.profile)
    sample = {
        'sample_idx': sample_idx,
        'source_sample_idx': source_sample_idx or sample_idx,
        'source_index': int(source_index),
        'global_offset': int(global_offset),
        'is_padding': bool(is_padding),
        'scene_idx': scene_identifier(info),
        'token': str(info.get('token', '')),
        'meta': make_meta_sample(info),
        'images': copy.deepcopy(dict(info['images'])),
        'image_bytes': {},
    }

    for cam_name, cam_item in iter_cameras(
            info, args.allow_nonstandard_cameras):
        path = resolve_image_path(
            args.data_root, image_prefix, cam_name, str(cam_item['img_path']))
        raw_bytes = read_file_bytes(path, args.allow_missing)
        image_bytes, image_materialization = materialize_image_bytes(
            raw_bytes, target_hw, resize_scale, args.allow_missing)
        sample['image_bytes'][cam_name] = {
            'img_path': str(cam_item['img_path']),
            'source_path': str(path),
            'bytes': image_bytes,
            **image_materialization,
        }

    for kind in ('depth', 'feats', 'sem_seg'):
        if not fields.get(kind):
            continue
        views = {}
        for cam_name, cam_item in iter_cameras(
                info, args.allow_nonstandard_cameras):
            path = modality_file_path(args, kind, info, cam_item)
            tensor = load_npy_tensor(path, args.allow_missing)
            materialized = False
            if kind == 'depth':
                tensor = resize_2d_tensor(
                    tensor, target_hw, mode='bilinear',
                    resize_scale=resize_scale)
                materialized = True
            elif kind == 'sem_seg':
                tensor = resize_2d_tensor(
                    tensor, target_hw, mode='nearest',
                    resize_scale=resize_scale)
                materialized = True
            views[cam_name] = {
                'img_path': str(cam_item['img_path']),
                'source_path': str(path),
                'tensor': tensor,
                'materialized': materialized,
                'shape': list(tensor.shape) if tensor is not None else None,
            }
        sample[kind] = views

    if fields.get('occ_gt'):
        path = modality_file_path(args, 'occ_gt', info)
        sample['occ_gt'] = {
            'source_path': str(path),
            'arrays': load_occ_npz(path, args.allow_missing),
        }
    return sample


def serialized_size(obj: Any) -> int:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return len(buffer.getbuffer())


def percentile(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    return int(np.percentile(np.asarray(values, dtype=np.float64), q))


def estimate_sample_sizes(args: argparse.Namespace,
                          infos: Sequence[Mapping[str, Any]],
                          indices: Sequence[int],
                          fields: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    count = min(len(indices), max(1, int(args.size_estimate_samples)))
    rng = random.Random(args.seed + 17)
    sampled_indices = rng.sample(list(indices), count)
    values = []
    for offset, source_index in enumerate(sampled_indices):
        sample = build_sample(args, infos[source_index], source_index, offset,
                              fields)
        values.append(serialized_size({
            'metadata': {
                'schema_version': 1,
                'profile': args.profile,
                'split': args.split,
            },
            'samples': [sample],
        }))
        print(
            f'[size] {len(values)}/{count} sample_idx={sample["sample_idx"]} '
            f'bytes={values[-1]}',
            flush=True)

    stats = {
        'num_samples': len(values),
        'p50': percentile(values, 50),
        'p90': percentile(values, 90),
        'selected_percentile': percentile(values, args.size_percentile),
        'p95': percentile(values, 95),
        'p99': percentile(values, 99),
        'mean': int(np.mean(values)) if values else 0,
        'max': max(values) if values else 0,
        'values': values,
    }
    return stats


def infer_samples_per_chunk(args: argparse.Namespace,
                            size_stats: Mapping[str, Any]) -> int:
    if args.samples_per_chunk is not None:
        if args.samples_per_chunk <= 0:
            raise ValueError('--samples-per-chunk must be positive.')
        return int(args.samples_per_chunk)

    target_bytes = parse_size(args.target_chunk_size)
    sample_bytes = max(1, int(size_stats['selected_percentile']))
    raw_count = max(1, target_bytes // sample_bytes)
    return int(max(args.min_samples_per_chunk,
                   min(args.max_samples_per_chunk, raw_count)))


def build_chunk_plan(
    infos: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    samples_per_chunk: int,
) -> List[Dict[str, Any]]:
    chunks = []
    for chunk_index, start in enumerate(range(0, len(indices), samples_per_chunk)):
        valid_source_indices = list(indices[start:start + samples_per_chunk])
        stored_source_indices = list(valid_source_indices)
        num_padding = samples_per_chunk - len(stored_source_indices)
        if num_padding > 0:
            stored_source_indices.extend(
                valid_source_indices[i % len(valid_source_indices)]
                for i in range(num_padding))

        valid_infos = [infos[i] for i in valid_source_indices]
        stored_infos = [infos[i] for i in stored_source_indices]
        chunks.append({
            'chunk_id': f'{chunk_index:06d}',
            'global_start': start,
            'global_end': start + len(valid_source_indices),
            'valid_source_indices': valid_source_indices,
            'stored_source_indices': stored_source_indices,
            'valid_sample_indices': [
                sample_identifier(info) for info in valid_infos
            ],
            'stored_sample_indices': [
                sample_identifier(info) for info in stored_infos
            ],
            'num_valid_samples': len(valid_source_indices),
            'num_padding_samples': num_padding,
            'is_tail': num_padding > 0,
        })
    return chunks


def summarize_sample(sample: Mapping[str, Any],
                     fields: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    shape_counts: Dict[str, int] = {}
    dtype_counts: Dict[str, int] = {}
    missing = []

    def add_tensor(name: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            missing.append(name)
            return
        shape = 'x'.join(str(dim) for dim in tensor.shape)
        dtype = str(tensor.dtype).replace('torch.', '')
        shape_counts[f'{name}:{shape}'] = shape_counts.get(
            f'{name}:{shape}', 0) + 1
        dtype_counts[f'{name}:{dtype}'] = dtype_counts.get(
            f'{name}:{dtype}', 0) + 1

    for cam_name, item in sample['image_bytes'].items():
        if item['bytes'] is None:
            missing.append(f'image:{cam_name}')
    for kind in ('depth', 'feats', 'sem_seg'):
        if not fields.get(kind):
            continue
        for cam_name, item in sample[kind].items():
            add_tensor(f'{kind}:{cam_name}', item['tensor'])
    if fields.get('occ_gt'):
        arrays = sample['occ_gt']['arrays']
        if arrays is None:
            missing.append('occ_gt')
        else:
            for key, tensor in arrays.items():
                add_tensor(f'occ_gt:{key}', tensor)
    return {
        'shape_counts': shape_counts,
        'dtype_counts': dtype_counts,
        'missing': missing,
    }


def merge_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    shapes: Dict[str, int] = {}
    dtypes: Dict[str, int] = {}
    missing = []
    for row in rows:
        for key, value in row['shape_counts'].items():
            shapes[key] = shapes.get(key, 0) + value
        for key, value in row['dtype_counts'].items():
            dtypes[key] = dtypes.get(key, 0) + value
        missing.extend(row['missing'])
    return {
        'shape_summary': dict(sorted(shapes.items())),
        'dtype_summary': dict(sorted(dtypes.items())),
        'missing_count': len(missing),
        'missing': sorted(set(missing)),
    }


def make_chunk_payload(args: argparse.Namespace,
                       infos: Sequence[Mapping[str, Any]],
                       chunk: Mapping[str, Any],
                       fields: Mapping[str, Optional[str]],
                       profile: Mapping[str, Any],
                       samples_per_chunk: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    samples = []
    summaries = []
    valid_set = set(chunk['valid_source_indices'])
    for offset, source_index in enumerate(chunk['stored_source_indices']):
        is_padding = offset >= chunk['num_valid_samples']
        source_sample_idx = sample_identifier(infos[source_index])
        global_offset = chunk['global_start'] + offset
        sample = build_sample(
            args,
            infos[source_index],
            source_index,
            global_offset,
            fields,
            is_padding=is_padding,
            source_sample_idx=source_sample_idx)
        if is_padding:
            sample['sample_idx'] = f'__padding__:{chunk["chunk_id"]}:{offset}'
        if not is_padding and source_index in valid_set:
            summaries.append(summarize_sample(sample, fields))
        samples.append(sample)

    payload = {
        'schema_version': 1,
        'format': 'gausstr-training-chunk-v1',
        'metadata': {
            'schema_version': 1,
            'chunk_id': chunk['chunk_id'],
            'profile': args.profile,
            'profile_sha256': profile['profile_sha256'],
            'split': args.split,
            'sample_count': len(samples),
            'valid_sample_count': chunk['num_valid_samples'],
            'padding_sample_count': chunk['num_padding_samples'],
            'samples_per_chunk': samples_per_chunk,
            'camera_order': list(CAMERA_ORDER),
            'sources': profile['source_roots'],
        },
        'sample_idx': [str(sample['sample_idx']) for sample in samples],
        'valid_sample_idx': list(chunk['valid_sample_indices']),
        'stored_source_indices': list(chunk['stored_source_indices']),
        'valid_source_indices': list(chunk['valid_source_indices']),
        'samples': samples,
    }
    return payload, merge_summary(summaries)


def sanity_check_chunk(payload: Mapping[str, Any],
                       samples_per_chunk: int) -> None:
    samples = payload.get('samples', [])
    if len(samples) != samples_per_chunk:
        raise RuntimeError(
            f'Chunk {payload["metadata"]["chunk_id"]} has {len(samples)} '
            f'samples, expected {samples_per_chunk}.')
    for sample in samples:
        for key in ('sample_idx', 'token', 'scene_idx', 'meta', 'image_bytes'):
            if key not in sample:
                raise KeyError(
                    f'Chunk {payload["metadata"]["chunk_id"]} sample missing {key}.')


def torch_save_atomic(path: Path,
                      payload: Mapping[str, Any],
                      overwrite: bool,
                      sanity_load: bool,
                      compute_sha256: bool,
                      max_bytes: int,
                      samples_per_chunk: int) -> Dict[str, Any]:
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    if path_exists_with_retry(tmp_path):
        unlink_with_retry(tmp_path)
    if path_exists_with_retry(path) and overwrite:
        unlink_with_retry(path)
    if path_exists_with_retry(path) and not overwrite:
        raise FileExistsError(f'{path} exists and cannot be reused.')

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp_path)
    actual_bytes = tmp_path.stat().st_size
    if actual_bytes > max_bytes:
        unlink_with_retry(tmp_path)
        raise RuntimeError(
            f'Chunk {path} is {actual_bytes} bytes, exceeding '
            f'--max-chunk-size={max_bytes}. Lower --target-chunk-size or '
            '--max-samples-per-chunk.')
    if sanity_load:
        loaded = torch_load(tmp_path, retries=20, base_delay=0.5)
        sanity_check_chunk(loaded, samples_per_chunk)

    digest = file_sha256(tmp_path) if compute_sha256 else None
    replace_with_retry(tmp_path, path)
    return {
        'bytes': path.stat().st_size,
        'sha256': digest,
        'skipped': False,
    }


def existing_chunk_entry(path: Path,
                         manifest_entry: Optional[Mapping[str, Any]],
                         chunk: Mapping[str, Any],
                         args: argparse.Namespace,
                         profile: Mapping[str, Any],
                         samples_per_chunk: int) -> Optional[Dict[str, Any]]:
    if args.overwrite or not path_exists_with_retry(path):
        return None
    if manifest_entry is not None:
        expected_bytes = int(manifest_entry.get('bytes', -1))
        expected_samples = list(manifest_entry.get('sample_indices', []))
        expected_sources = list(manifest_entry.get('source_indices', []))
        expected_stored_sources = list(
            manifest_entry.get('stored_source_indices', []))
        if (expected_bytes == path.stat().st_size
                and expected_samples == list(chunk['valid_sample_indices'])
                and expected_sources == list(chunk['valid_source_indices'])
                and expected_stored_sources == list(
                    chunk['stored_source_indices'])):
            return dict(manifest_entry)

    payload = torch_load(path, retries=20, base_delay=0.5)
    metadata = payload.get('metadata', {})
    if metadata.get('profile_sha256') != profile['profile_sha256']:
        raise RuntimeError(f'Existing chunk profile mismatch: {path}')
    if metadata.get('samples_per_chunk') != samples_per_chunk:
        raise RuntimeError(f'Existing chunk samples_per_chunk mismatch: {path}')
    sanity_check_chunk(payload, samples_per_chunk)
    if payload.get('valid_sample_idx') != list(chunk['valid_sample_indices']):
        raise RuntimeError(f'Existing chunk sample order mismatch: {path}')
    digest = None if args.skip_file_sha256 else file_sha256(path)
    return make_manifest_entry(path, chunk, path.stat().st_size, digest,
                               skipped=True)


def make_manifest_entry(path: Path,
                        chunk: Mapping[str, Any],
                        num_bytes: int,
                        digest: Optional[str],
                        skipped: bool,
                        summary: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    entry = {
        'path': path.name,
        'bytes': int(num_bytes),
        'sha256': digest,
        'num_samples': len(chunk['stored_source_indices']),
        'num_valid_samples': chunk['num_valid_samples'],
        'num_padding_samples': chunk['num_padding_samples'],
        'sample_indices': list(chunk['valid_sample_indices']),
        'stored_sample_indices': list(chunk['stored_sample_indices']),
        'source_indices': list(chunk['valid_source_indices']),
        'stored_source_indices': list(chunk['stored_source_indices']),
        'is_tail': bool(chunk['is_tail']),
        'skipped': bool(skipped),
    }
    if summary:
        entry.update(summary)
    return entry


def write_one_chunk(args: argparse.Namespace,
                    infos: Sequence[Mapping[str, Any]],
                    chunk: Mapping[str, Any],
                    fields: Mapping[str, Optional[str]],
                    profile: Mapping[str, Any],
                    out_dir: Path,
                    samples_per_chunk: int,
                    max_chunk_bytes: int,
                    existing_manifest: Optional[Mapping[str, Any]]) -> Tuple[str, Dict[str, Any], bool]:
    chunk_id = str(chunk['chunk_id'])
    path = out_dir / f'{chunk_id}.torch'
    manifest_entry = None
    if existing_manifest is not None:
        manifest_entry = existing_manifest.get('chunks', {}).get(chunk_id)
    existing = existing_chunk_entry(path, manifest_entry, chunk, args, profile,
                                    samples_per_chunk)
    if existing is not None:
        return chunk_id, existing, True

    payload, summary = make_chunk_payload(args, infos, chunk, fields, profile,
                                          samples_per_chunk)
    write_info = torch_save_atomic(
        path,
        payload,
        overwrite=args.overwrite,
        sanity_load=not args.no_sanity_load,
        compute_sha256=not args.skip_file_sha256,
        max_bytes=max_chunk_bytes,
        samples_per_chunk=samples_per_chunk)
    entry = make_manifest_entry(
        path,
        chunk,
        write_info['bytes'],
        write_info['sha256'],
        skipped=False,
        summary=summary)
    return chunk_id, entry, False


def git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:
        return None


def producer_info(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        'script': 'tools/build_training_chunks.py',
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'numpy': np.__version__,
        'git_revision': git_revision(),
        'command': ' '.join(sys.argv),
        'num_workers': args.num_workers,
    }


def build_index(infos: Sequence[Mapping[str, Any]],
                chunks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    samples = []
    by_sample_idx = {}
    for chunk in chunks:
        for offset, source_index in enumerate(chunk['valid_source_indices']):
            info = infos[source_index]
            sample_idx = sample_identifier(info)
            item = {
                'sample_idx': sample_idx,
                'global_offset': chunk['global_start'] + offset,
                'chunk_id': chunk['chunk_id'],
                'offset': offset,
                'token': str(info.get('token', '')),
                'scene_idx': scene_identifier(info),
                'is_padding': False,
            }
            samples.append(item)
            by_sample_idx[sample_idx] = {
                'chunk_id': chunk['chunk_id'],
                'offset': offset,
                'global_offset': item['global_offset'],
            }
    return {
        'schema_version': 1,
        'num_samples': len(samples),
        'samples': samples,
        'by_sample_idx': by_sample_idx,
    }


def validate_output(out_dir: Path,
                    chunks: Sequence[Mapping[str, Any]],
                    manifest: Mapping[str, Any],
                    index: Mapping[str, Any],
                    validate_sha256: bool) -> Dict[str, Any]:
    validation = {
        'schema_version': 1,
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'ok': True,
        'num_chunks': len(chunks),
        'num_torch_files': 0,
        'num_tmp_files': 0,
        'num_valid_samples': index['num_samples'],
        'errors': [],
    }

    def add_error(message: str) -> None:
        validation['ok'] = False
        validation['errors'].append(message)

    seen = set()
    for item in index['samples']:
        sample_idx = item['sample_idx']
        if sample_idx in seen:
            add_error(f'duplicate sample in index: {sample_idx}')
        seen.add(sample_idx)

    for chunk in chunks:
        chunk_id = str(chunk['chunk_id'])
        entry = manifest['chunks'].get(chunk_id)
        if entry is None:
            add_error(f'missing manifest entry for chunk {chunk_id}')
            continue
        if list(entry.get('sample_indices', [])) != list(
                chunk['valid_sample_indices']):
            add_error(f'chunk sample order mismatch: {chunk_id}')
        if list(entry.get('source_indices', [])) != list(
                chunk['valid_source_indices']):
            add_error(f'chunk source index mismatch: {chunk_id}')
        if int(entry.get('num_valid_samples', -1)) != int(
                chunk['num_valid_samples']):
            add_error(f'chunk valid sample count mismatch: {chunk_id}')
        if int(entry.get('num_padding_samples', -1)) != int(
                chunk['num_padding_samples']):
            add_error(f'chunk padding sample count mismatch: {chunk_id}')
        path = out_dir / entry['path']
        if not path_exists_with_retry(path):
            add_error(f'missing chunk file: {path}')
            continue
        validation['num_torch_files'] += 1
        actual_bytes = path.stat().st_size
        if actual_bytes != int(entry['bytes']):
            add_error(
                f'chunk size mismatch: {path} expected={entry["bytes"]} actual={actual_bytes}')
        if validate_sha256 and entry.get('sha256'):
            actual_sha = file_sha256(path)
            if actual_sha != entry['sha256']:
                add_error(f'chunk sha256 mismatch: {path}')

    tmp_files = sorted(out_dir.rglob('*.tmp'))
    validation['num_tmp_files'] = len(tmp_files)
    for tmp_path in tmp_files[:20]:
        add_error(f'stale tmp file: {tmp_path}')
    if len(tmp_files) > 20:
        add_error(f'{len(tmp_files) - 20} more stale tmp files under {out_dir}')

    expected_valid = sum(int(chunk['num_valid_samples']) for chunk in chunks)
    if expected_valid != index['num_samples']:
        add_error(
            f'index num_samples mismatch: expected={expected_valid} actual={index["num_samples"]}')

    if not validation['ok']:
        preview = '\n'.join(validation['errors'][:20])
        raise RuntimeError(
            'Training chunk validation failed after preprocessing:\n'
            f'{preview}')
    print(
        'Validated training chunks: '
        f'{validation["num_chunks"]} chunks, '
        f'{validation["num_torch_files"]} .torch files, '
        f'{validation["num_valid_samples"]} valid samples.',
        flush=True)
    return validation


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

    target_chunk_bytes = parse_size(args.target_chunk_size)
    max_chunk_bytes = parse_size(args.max_chunk_size)
    if max_chunk_bytes < target_chunk_bytes:
        raise ValueError('--max-chunk-size must be >= --target-chunk-size.')

    fields = profile_fields(args.profile, args.split)
    out_dir = args.out_root / args.split / args.profile

    print(f'Loading annotation file: {args.ann_file}', flush=True)
    ann_data = load_pickle(args.ann_file)
    infos = ann_data['data_list']
    print(f'Loaded {len(infos)} samples.', flush=True)

    indices = select_indices(infos, args.split, args.ratio, args.max_samples,
                             args.seed)
    print(f'Selected {len(indices)} samples for split {args.split}.', flush=True)
    validate_infos(infos, indices, fields, args.split,
                   args.allow_nonstandard_cameras)

    profile = profile_payload(args, fields)
    size_stats = estimate_sample_sizes(args, infos, indices, fields)
    samples_per_chunk = infer_samples_per_chunk(args, size_stats)
    chunks = build_chunk_plan(infos, indices, samples_per_chunk)
    print(
        'Chunk sizing: '
        f'target={target_chunk_bytes} max={max_chunk_bytes} '
        f'p{args.size_percentile:g}={size_stats["selected_percentile"]} '
        f'samples_per_chunk={samples_per_chunk} '
        f'num_chunks={len(chunks)}',
        flush=True)

    profile['sizing'] = {
        'target_chunk_bytes': target_chunk_bytes,
        'max_chunk_bytes': max_chunk_bytes,
        'size_estimate_samples': args.size_estimate_samples,
        'size_percentile': args.size_percentile,
        'sample_size_stats': size_stats,
        'samples_per_chunk': samples_per_chunk,
    }
    profile['producer'] = producer_info(args)
    profile['profile_sha256'] = stable_json_hash({
        key: value
        for key, value in profile.items()
        if key not in {'profile_sha256', 'producer', 'sizing'}
    })

    if args.dry_run:
        print('Dry run requested; no chunks written.', flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = None if args.overwrite else load_json_if_exists(
        out_dir / 'chunk_manifest.json')
    if existing_manifest is not None:
        if existing_manifest.get('profile_sha256') != profile['profile_sha256']:
            raise RuntimeError(
                f'Existing manifest profile mismatch: {out_dir / "chunk_manifest.json"}. '
                'Use --overwrite or choose a different output directory.')
        if int(existing_manifest.get('samples_per_chunk',
                                     samples_per_chunk)) != samples_per_chunk:
            raise RuntimeError(
                'Existing manifest samples_per_chunk mismatch. Use --overwrite '
                'or choose a different output directory.')

    chunk_entries = {}
    if args.num_workers <= 1 or len(chunks) <= 1:
        for index, chunk in enumerate(chunks, 1):
            chunk_id, entry, skipped = write_one_chunk(
                args, infos, chunk, fields, profile, out_dir,
                samples_per_chunk, max_chunk_bytes, existing_manifest)
            chunk_entries[chunk_id] = entry
            action = 'skipped' if skipped else 'wrote'
            print(
                f'[{args.profile}] {index}/{len(chunks)} {action} '
                f'chunk {chunk_id} -> {out_dir / (chunk_id + ".torch")}',
                flush=True)
    else:
        max_workers = min(args.num_workers, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    write_one_chunk, args, infos, chunk, fields, profile,
                    out_dir, samples_per_chunk, max_chunk_bytes,
                    existing_manifest)
                for chunk in chunks
            ]
            for index, future in enumerate(as_completed(futures), 1):
                chunk_id, entry, skipped = future.result()
                chunk_entries[chunk_id] = entry
                action = 'skipped' if skipped else 'wrote'
                print(
                    f'[{args.profile}] {index}/{len(chunks)} {action} '
                    f'chunk {chunk_id} -> {out_dir / (chunk_id + ".torch")}',
                    flush=True)

    chunk_entries = dict(sorted(chunk_entries.items()))
    index = build_index(infos, chunks)
    manifest = {
        'schema_version': 1,
        'format': 'gausstr-training-chunks-v1',
        'split': args.split,
        'profile': args.profile,
        'profile_sha256': profile['profile_sha256'],
        'target_chunk_bytes': target_chunk_bytes,
        'max_chunk_bytes': max_chunk_bytes,
        'samples_per_chunk': samples_per_chunk,
        'num_chunks': len(chunks),
        'num_samples': len(indices),
        'num_stored_samples': len(chunks) * samples_per_chunk,
        'num_padding_samples': len(chunks) * samples_per_chunk - len(indices),
        'chunks': chunk_entries,
        'producer': producer_info(args),
    }
    validation = validate_output(out_dir, chunks, manifest, index,
                                 args.validate_sha256)
    manifest['validation'] = validation

    dump_json(out_dir / 'profile.json', profile)
    dump_json(out_dir / 'index.json', index)
    dump_json(out_dir / 'chunk_manifest.json', manifest)
    print(f'Done. Summary written to {out_dir / "chunk_manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
