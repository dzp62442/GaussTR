import errno
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


RETRYABLE_FS_ERRNOS = {
    errno.EAGAIN,
    errno.EBUSY,
    errno.EINTR,
    getattr(errno, 'ESTALE', errno.EIO),
}


def _torch_load_once(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def torch_load(path: Path,
               retries: int = 3,
               retry_wait: float = 1.0,
               debug_log=None):
    for attempt in range(retries + 1):
        try:
            return _torch_load_once(path)
        except OSError as exc:
            if exc.errno not in RETRYABLE_FS_ERRNOS or attempt >= retries:
                raise
            wait = retry_wait * (2**attempt)
            if debug_log is not None:
                debug_log(
                    f'torch.load retry {attempt + 1}/{retries} for {path}: '
                    f'{exc!r}; sleep={wait:.1f}s')
            time.sleep(wait)


class ShardMemoryStore:
    """Read GaussTR shard groups from a mounted path.

    `lazy` mode keeps an in-process LRU cache and optionally prefetches shards
    from the mounted TOS path. No local disk cache is used.
    """

    def __init__(self,
                 shard_root,
                 split: str,
                 groups: Iterable[str],
                 preload_mode: str = 'lazy',
                 require_success: bool = False,
                 max_cache_bytes: int = 24 * 1024**3,
                 prefetch_shards: int = 1,
                 prefetch_workers: int = 1,
                 raw_group: str = 'raw_nuscenes',
                 raw_shard_order: Optional[Sequence[str]] = None,
                 debug: bool = False,
                 debug_interval: int = 100,
                 torch_load_retries: int = 3,
                 torch_load_retry_wait: float = 1.0) -> None:
        self.shard_root = Path(shard_root)
        self.split = split
        self.split_root = self.shard_root / split
        self.groups = list(dict.fromkeys(groups))
        self.preload_mode = preload_mode
        self.require_success = require_success
        self.max_cache_bytes = int(max_cache_bytes)
        self.prefetch_shards = int(prefetch_shards)
        self.raw_group = raw_group
        self.raw_shard_order = list(raw_shard_order) if raw_shard_order else None
        self.debug = bool(debug)
        self.debug_interval = max(1, int(debug_interval))
        self.torch_load_retries = max(0, int(torch_load_retries))
        self.torch_load_retry_wait = max(0.0, float(torch_load_retry_wait))
        self.debug_prefix = (
            f'[ShardMemoryStore pid={os.getpid()} split={split}]')
        self.stats = dict(
            get_calls=0,
            cache_hits=0,
            cache_misses=0,
            shard_loads=0,
            prefetch_scheduled=0,
            prefetch_waits=0,
            prefetch_hits=0,
            evictions=0)

        if preload_mode not in {'all', 'lazy'}:
            raise ValueError(
                f"Unsupported preload_mode={preload_mode!r}; expected 'all' or 'lazy'."
            )

        self.index = self._load_json(self.split_root / 'index.json')
        self.samples = self.index['samples']
        self.group_manifests = {
            group: self._load_json(self.split_root / group /
                                   'group_manifest.json')
            for group in self.groups
        }
        self.group_samples: Dict[str, Dict[str, Mapping]] = {
            group: {}
            for group in self.groups
        }
        self.loaded_shards: Dict[str, Dict[str, Mapping]] = {
            group: {}
            for group in self.groups
        }
        self.shard_lru: OrderedDict[Tuple[str, str], int] = OrderedDict()
        self.current_cache_bytes = 0
        self.lock = threading.RLock()
        self.prefetch_executor = None
        self.prefetch_futures = {}
        if preload_mode == 'lazy' and prefetch_shards > 0 and prefetch_workers > 0:
            self.prefetch_executor = ThreadPoolExecutor(
                max_workers=prefetch_workers, thread_name_prefix='shard-prefetch')

        if preload_mode == 'all':
            self.preload_all()
        self._debug(
            'initialized '
            f'groups={self.groups} preload_mode={self.preload_mode} '
            f'max_cache_bytes={self.max_cache_bytes} '
            f'prefetch_shards={self.prefetch_shards} '
            f'prefetch_workers={prefetch_workers}')

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f'{self.debug_prefix} {message}', flush=True)

    def _debug_stats(self) -> None:
        if not self.debug:
            return
        if self.stats['get_calls'] % self.debug_interval != 0:
            return
        self._debug(
            'stats '
            f"get_calls={self.stats['get_calls']} "
            f"cache_hits={self.stats['cache_hits']} "
            f"cache_misses={self.stats['cache_misses']} "
            f"shard_loads={self.stats['shard_loads']} "
            f"prefetch_scheduled={self.stats['prefetch_scheduled']} "
            f"prefetch_waits={self.stats['prefetch_waits']} "
            f"prefetch_hits={self.stats['prefetch_hits']} "
            f"evictions={self.stats['evictions']} "
            f'cache_bytes={self.current_cache_bytes}/'
            f'{self.max_cache_bytes}')

    @staticmethod
    def _load_json(path: Path):
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)

    def _shard_path(self, group: str, shard_id: str) -> Path:
        manifest = self.group_manifests[group]
        entry = manifest['shards'][shard_id]
        return self.split_root / entry['path']

    def _manifest_shard_bytes(self, group: str, shard_id: str) -> int:
        return int(self.group_manifests[group]['shards'][shard_id].get(
            'bytes', 0))

    def _load_shard(self, group: str, shard_id: str) -> Mapping:
        key = (group, shard_id)
        with self.lock:
            if shard_id in self.loaded_shards[group]:
                self.shard_lru.move_to_end(key)
                return self.loaded_shards[group][shard_id]

        path = self._shard_path(group, shard_id)
        if self.require_success:
            success_path = path.with_suffix('.SUCCESS')
            if not success_path.exists():
                raise FileNotFoundError(
                    f'Shard success marker is missing: {success_path}')

        shard_bytes = self._manifest_shard_bytes(group, shard_id)
        self._debug(
            f'load start group={group} shard={shard_id} '
            f'bytes={shard_bytes} path={path}')
        started_at = time.monotonic()
        payload = torch_load(
            path,
            retries=self.torch_load_retries,
            retry_wait=self.torch_load_retry_wait,
            debug_log=self._debug)
        elapsed = time.monotonic() - started_at
        with self.lock:
            if shard_id in self.loaded_shards[group]:
                self.shard_lru.move_to_end(key)
                return self.loaded_shards[group][shard_id]
            self.loaded_shards[group][shard_id] = payload
            self.shard_lru[key] = shard_bytes
            self.current_cache_bytes += shard_bytes
            self.stats['shard_loads'] += 1
            for sample in payload['samples']:
                self.group_samples[group][str(sample['sample_idx'])] = sample
            if self.preload_mode == 'lazy':
                self._evict_if_needed(protected={key})
        self._debug(
            f'load done group={group} shard={shard_id} '
            f'samples={len(payload["samples"])} elapsed={elapsed:.2f}s '
            f'cache_bytes={self.current_cache_bytes}/{self.max_cache_bytes}')
        return payload

    def _wait_for_prefetch(self, group: str, shard_id: str) -> bool:
        key = (group, shard_id)
        with self.lock:
            future = self.prefetch_futures.get(key)
        if future is None:
            return False
        self.stats['prefetch_waits'] += 1
        self._debug(f'prefetch wait group={group} shard={shard_id}')
        future.result()
        self.stats['prefetch_hits'] += 1
        return True

    def _evict_if_needed(self, protected=None) -> None:
        if self.max_cache_bytes <= 0:
            return
        protected = protected or set()
        while self.current_cache_bytes > self.max_cache_bytes and self.shard_lru:
            evict_key, evict_bytes = next(iter(self.shard_lru.items()))
            if evict_key in protected and len(self.shard_lru) == 1:
                break
            if evict_key in protected:
                self.shard_lru.move_to_end(evict_key)
                continue
            group, shard_id = evict_key
            self.shard_lru.pop(evict_key)
            payload = self.loaded_shards[group].pop(shard_id, None)
            if payload is not None:
                for sample in payload['samples']:
                    self.group_samples[group].pop(
                        str(sample['sample_idx']), None)
            self.current_cache_bytes -= evict_bytes
            self.stats['evictions'] += 1
            self._debug(
                f'evict group={group} shard={shard_id} bytes={evict_bytes} '
                f'cache_bytes={self.current_cache_bytes}/{self.max_cache_bytes}')

    def preload_all(self) -> None:
        for group in self.groups:
            for shard_id in sorted(self.group_manifests[group]['shards']):
                self._load_shard(group, shard_id)

    def get(self, group: str, sample_idx: str) -> Mapping:
        sample_idx = str(sample_idx)
        self.stats['get_calls'] += 1
        with self.lock:
            sample = self.group_samples[group].get(sample_idx)
            if sample is not None:
                self.stats['cache_hits'] += 1
                sample_entry = self.index['by_sample_idx'][sample_idx]['groups'][
                    group]
                key = (group, sample_entry['shard_id'])
                if key in self.shard_lru:
                    self.shard_lru.move_to_end(key)
                self._schedule_prefetch(sample_idx, group)
                self._debug_stats()
                return sample

        sample_entry = self.index['by_sample_idx'][sample_idx]['groups'][group]
        self.stats['cache_misses'] += 1
        self._wait_for_prefetch(group, sample_entry['shard_id'])
        shard = self._load_shard(group, sample_entry['shard_id'])
        self._schedule_prefetch(sample_idx, group)
        self._debug_stats()
        return shard['samples'][sample_entry['offset']]

    def get_optional(self, group: Optional[str], sample_idx: str):
        if not group:
            return None
        return self.get(group, sample_idx)

    def _schedule_prefetch(self, sample_idx: str, accessed_group: str) -> None:
        if (self.preload_mode != 'lazy' or self.prefetch_executor is None
                or accessed_group != self.raw_group):
            return
        tasks = self._next_prefetch_tasks(sample_idx)
        for task in tasks:
            with self.lock:
                group, shard_id = task
                if shard_id in self.loaded_shards[group] or task in self.prefetch_futures:
                    continue
                future = self.prefetch_executor.submit(self._prefetch_one, task)
                self.prefetch_futures[task] = future
                self.stats['prefetch_scheduled'] += 1
                self._debug(f'prefetch scheduled group={group} shard={shard_id}')

    def _prefetch_one(self, task: Tuple[str, str]) -> None:
        group, shard_id = task
        try:
            self._debug(f'prefetch start group={group} shard={shard_id}')
            self._load_shard(group, shard_id)
            self._debug(f'prefetch done group={group} shard={shard_id}')
        finally:
            with self.lock:
                self.prefetch_futures.pop(task, None)

    def _next_prefetch_tasks(self, sample_idx: str) -> Sequence[Tuple[str, str]]:
        raw_entry = self.index['by_sample_idx'][sample_idx]['groups'].get(
            self.raw_group)
        if raw_entry is None:
            return []
        raw_shard_ids = sorted(self.group_manifests[self.raw_group]['shards'])
        if self.raw_shard_order:
            raw_shard_ids = self.raw_shard_order
        try:
            raw_pos = raw_shard_ids.index(raw_entry['shard_id'])
        except ValueError:
            return []

        tasks = []
        group_ranges = self.index.get('groups', {})
        raw_ids_to_prefetch = raw_shard_ids[raw_pos:raw_pos + 1 +
                                            self.prefetch_shards]
        for next_raw_id in raw_ids_to_prefetch:
            raw_range = group_ranges[self.raw_group][next_raw_id]
            start, end = raw_range['start'], raw_range['end']
            for group in self.groups:
                for shard_id, shard_range in group_ranges[group].items():
                    if shard_range['end'] <= start or shard_range['start'] >= end:
                        continue
                    tasks.append((group, shard_id))
        return tasks

    def close(self) -> None:
        if self.prefetch_executor is not None:
            self.prefetch_executor.shutdown(wait=False, cancel_futures=True)
            self.prefetch_executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
