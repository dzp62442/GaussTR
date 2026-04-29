import copy
import errno
import json
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from mmengine.dataset import Compose
from mmengine.dist import get_dist_info
from torch.utils.data import IterableDataset, get_worker_info

from mmdet3d.registry import DATASETS

from .nuscenes_occ import NuScenesOccDataset


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


def is_retryable_torch_load_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and any(
        message in str(exc) for message in RETRYABLE_TORCH_LOAD_MESSAGES)


def torch_load_once(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def torch_load(path: Path, retries: int = 20, base_delay: float = 0.5):
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
            f'[NuScenesOccChunkDataset] retry torch.load path={path} '
            f'attempt={attempt + 1}/{retries} sleep={delay:.1f}s '
            f'error={last_exc!r}',
            flush=True)
        time.sleep(delay)
    raise RuntimeError(f'Unreachable torch_load retry state for {path}')


@DATASETS.register_module()
class NuScenesOccChunkDataset(IterableDataset):
    """nuScenes occupancy dataset backed by fused training chunks.

    The dataset owns distributed/worker partitioning internally. Configs using
    it should not attach a PyTorch/MMEngine sampler.
    """

    METAINFO = NuScenesOccDataset.METAINFO

    def __init__(self,
                 chunk_root='data/gausstr_shards',
                 split='train',
                 profile='featup_metric3d_sam2',
                 pipeline=None,
                 metainfo=None,
                 chunk_shuffle=True,
                 sample_shuffle=True,
                 seed=2026,
                 pad_train_chunks=True,
                 skip_padding=True,
                 debug=False,
                 slow_log_threshold=1.0,
                 test_mode=False,
                 **kwargs):
        super().__init__()
        if metainfo is None:
            metainfo = self.METAINFO
        elif 'classes' not in metainfo:
            metainfo['classes'] = self.METAINFO['classes']
        metainfo['label2cat'] = {
            i: cat_name
            for i, cat_name in enumerate(metainfo['classes'])
        }
        self._metainfo = metainfo
        self.chunk_root = Path(chunk_root)
        self.split = split
        self.profile = profile
        self.profile_root = self.chunk_root / split / profile
        self.chunk_shuffle = bool(chunk_shuffle)
        self.sample_shuffle = bool(sample_shuffle)
        self.seed = int(seed)
        self.pad_train_chunks = bool(pad_train_chunks)
        self.skip_padding = bool(skip_padding)
        self.debug = bool(debug)
        self.slow_log_threshold = float(slow_log_threshold)
        self.test_mode = test_mode
        self._epoch = 0
        self._iter_epoch = 0
        self._shared_epoch = mp.Value('i', 0)
        self._shared_epoch_set = mp.Value('i', 0)

        self.pipeline = Compose(pipeline or [])
        self.manifest = self._load_json(self.profile_root /
                                        'chunk_manifest.json')
        self.index = self._load_json(self.profile_root / 'index.json')
        self.profile_meta = self._load_json(self.profile_root / 'profile.json')
        self.chunks = [
            dict(chunk_id=chunk_id, **entry)
            for chunk_id, entry in sorted(self.manifest['chunks'].items())
        ]
        self.num_valid_samples = int(self.manifest.get(
            'num_samples', self.index.get('num_samples', 0)))
        self.samples_per_chunk = int(self.manifest['samples_per_chunk'])
        self.index_samples = sorted(
            self.index.get('samples', []),
            key=lambda item: int(item['global_offset']))
        self.chunk_by_id = {str(chunk['chunk_id']): chunk for chunk in self.chunks}

    @staticmethod
    def _load_json(path: Path):
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)

    @property
    def metainfo(self):
        return copy.deepcopy(self._metainfo)

    def __len__(self):
        rank, world_size = get_dist_info()
        if self.split == 'train' and self.pad_train_chunks and self.chunks:
            chunks_per_rank = (len(self.chunks) + world_size - 1) // world_size
            return chunks_per_rank * self.samples_per_chunk
        return len(self._eval_items_for_rank(rank, world_size))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)
        with self._shared_epoch_set.get_lock():
            self._shared_epoch_set.value = 1

    def _ordered_chunks(self, epoch: int):
        chunks = list(self.chunks)
        if self.split == 'train' and self.chunk_shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(chunks)
        if self.split == 'train' and self.pad_train_chunks and chunks:
            _, world_size = get_dist_info()
            total_chunks = ((len(chunks) + world_size - 1) // world_size *
                            world_size)
            for i in range(total_chunks - len(chunks)):
                chunks.append(chunks[i % len(chunks)])
        return chunks

    def _partition_chunks(self, chunks):
        rank, world_size = get_dist_info()
        rank_chunks = chunks[rank::world_size]
        worker = get_worker_info()
        if worker is None:
            return rank_chunks
        return rank_chunks[worker.id::worker.num_workers]

    def _eval_items_for_rank(self, rank: int,
                             world_size: int) -> Sequence[Mapping]:
        return [
            item for item in self.index_samples
            if int(item['global_offset']) % world_size == rank
        ]

    def _partition_eval_chunks(self):
        rank, world_size = get_dist_info()
        worker = get_worker_info()
        rank_items = self._eval_items_for_rank(rank, world_size)
        if worker is not None:
            rank_items = [
                item for pos, item in enumerate(rank_items)
                if pos % worker.num_workers == worker.id
            ]

        grouped_offsets = {}
        for item in rank_items:
            chunk_id = str(item['chunk_id'])
            grouped_offsets.setdefault(chunk_id, set()).add(int(item['offset']))

        return [(self.chunk_by_id[chunk_id], offsets)
                for chunk_id, offsets in grouped_offsets.items()]

    def _chunk_path(self, entry: Mapping) -> Path:
        return self.profile_root / entry['path']

    def _sample_to_results(self, sample: Mapping) -> dict:
        results = copy.deepcopy(sample['meta'])
        results['sample_idx'] = sample['source_sample_idx']
        results['token'] = sample.get('token', results.get('token'))
        results['scene_idx'] = sample.get('scene_idx', results.get('scene_idx'))
        results['images'] = copy.deepcopy(sample['images'])
        results['_chunk_sample'] = sample
        return results

    def _iter_samples(self,
                      chunk_entry: Mapping,
                      epoch: int,
                      valid_offsets: Optional[set] = None):
        path = self._chunk_path(chunk_entry)
        started = time.monotonic()
        payload = torch_load(path)
        elapsed = time.monotonic() - started
        if self.debug or elapsed >= self.slow_log_threshold:
            print(
                f'[NuScenesOccChunkDataset] load chunk={chunk_entry["chunk_id"]} '
                f'bytes={chunk_entry.get("bytes")} elapsed={elapsed:.2f}s '
                f'path={path}',
                flush=True)

        samples = list(payload['samples'])
        if self.split == 'train' and self.sample_shuffle:
            rng = random.Random(self.seed + epoch * 1000003 +
                                int(chunk_entry['chunk_id']))
            rng.shuffle(samples)
        for offset, sample in enumerate(samples):
            if valid_offsets is not None and offset not in valid_offsets:
                continue
            if sample.get('is_padding', False) and self.skip_padding:
                continue
            data = self.pipeline(self._sample_to_results(sample))
            if data is not None:
                yield data

    def _current_epoch(self) -> int:
        with self._shared_epoch_set.get_lock():
            shared_epoch_set = bool(self._shared_epoch_set.value)
        if shared_epoch_set:
            with self._shared_epoch.get_lock():
                return int(self._shared_epoch.value)
        epoch = self._epoch + self._iter_epoch
        self._iter_epoch += 1
        return epoch

    def __iter__(self):
        epoch = self._current_epoch()
        if self.split == 'train':
            chunks = self._partition_chunks(self._ordered_chunks(epoch))
            for chunk in chunks:
                yield from self._iter_samples(chunk, epoch)
            return

        for chunk, offsets in self._partition_eval_chunks():
            yield from self._iter_samples(chunk, epoch, valid_offsets=offsets)
