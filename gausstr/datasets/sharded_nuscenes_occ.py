import copy
import json
from pathlib import Path
from typing import Dict, Optional

from mmengine.dataset import BaseDataset
from torch.utils.data import get_worker_info
from mmdet3d.registry import DATASETS

from .nuscenes_occ import NuScenesOccDataset
from .shard_store import ShardMemoryStore


@DATASETS.register_module()
class NuScenesOccShardedDataset(BaseDataset):
    """nuScenes occupancy dataset backed by GaussTR `.torch` shards."""

    METAINFO = NuScenesOccDataset.METAINFO

    def __init__(self,
                 shard_root='data/gausstr_shards',
                 split='train',
                 required_groups: Optional[Dict[str, Optional[str]]] = None,
                 preload_mode='lazy',
                 require_success=False,
                 max_cache_bytes=24 * 1024**3,
                 prefetch_shards=1,
                 prefetch_workers=1,
                 debug=False,
                 debug_interval=100,
                 torch_load_retries=3,
                 torch_load_retry_wait=1.0,
                 metainfo=None,
                 pipeline=None,
                 test_mode=False,
                 serialize_data=False,
                 **kwargs):
        if metainfo is None:
            metainfo = self.METAINFO
        elif 'classes' not in metainfo:
            metainfo['classes'] = self.METAINFO['classes']
        metainfo['label2cat'] = {
            i: cat_name
            for i, cat_name in enumerate(metainfo['classes'])
        }

        self.shard_root = Path(shard_root)
        self.split = split
        self.required_groups = required_groups or dict(
            raw='raw_nuscenes',
            depth='depth_metric3d',
            feats='feats_featup',
            sem_seg='sem_seg_grounded_sam2')
        self.preload_mode = preload_mode
        self.require_success = require_success
        self.max_cache_bytes = int(max_cache_bytes)
        self.prefetch_shards = int(prefetch_shards)
        self.prefetch_workers = int(prefetch_workers)
        self.debug = bool(debug)
        self.debug_interval = int(debug_interval)
        self.torch_load_retries = int(torch_load_retries)
        self.torch_load_retry_wait = float(torch_load_retry_wait)
        self.store: Optional[ShardMemoryStore] = None
        self.index = None
        self.raw_shard_indices = None
        self.raw_shard_order = None
        self.epoch = 0
        self._store_worker_id = None
        for legacy_key in ('ann_file', 'data_root', 'data_prefix', 'modality',
                           'filter_empty_gt'):
            kwargs.pop(legacy_key, None)

        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root='',
            data_prefix={},
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=serialize_data,
            **kwargs)

    def load_data_list(self):
        index_path = self.shard_root / self.split / 'index.json'
        with index_path.open('r', encoding='utf-8') as f:
            self.index = json.load(f)
        self.raw_shard_indices = self._build_raw_shard_indices(
            self.required_groups.get('raw', 'raw_nuscenes'))
        return [dict(item) for item in self.index['samples']]

    def _build_store(self) -> ShardMemoryStore:
        groups = [group for group in self.required_groups.values() if group]
        if 'raw_nuscenes' not in groups:
            groups.insert(0, 'raw_nuscenes')
        return ShardMemoryStore(
            self.shard_root,
            self.split,
            groups,
            preload_mode=self.preload_mode,
            require_success=self.require_success,
            max_cache_bytes=self.max_cache_bytes,
            prefetch_shards=self.prefetch_shards,
            prefetch_workers=self.prefetch_workers,
            raw_group=self.required_groups.get('raw', 'raw_nuscenes'),
            raw_shard_order=self.raw_shard_order,
            debug=self.debug,
            debug_interval=self.debug_interval,
            torch_load_retries=self.torch_load_retries,
            torch_load_retry_wait=self.torch_load_retry_wait)

    def _get_store(self) -> ShardMemoryStore:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        if self.store is None or self._store_worker_id != worker_id:
            if self.store is not None:
                self.store.close()
            self.store = self._build_store()
            self._store_worker_id = worker_id
        return self.store

    def _build_raw_shard_indices(self, raw_group: str):
        if self.index is None:
            return {}
        group_ranges = self.index['groups'][raw_group]
        result = {}
        for shard_id, shard_range in group_ranges.items():
            result[shard_id] = list(
                range(int(shard_range['start']), int(shard_range['end'])))
        return dict(sorted(result.items()))

    def get_raw_shard_indices(self, raw_group='raw_nuscenes'):
        if self.raw_shard_indices is None:
            self.raw_shard_indices = self._build_raw_shard_indices(raw_group)
        return self.raw_shard_indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_raw_shard_order(self, raw_shard_order):
        self.raw_shard_order = list(raw_shard_order)
        if self.store is not None:
            self.store.raw_shard_order = self.raw_shard_order

    def get_data_info(self, idx: int) -> dict:
        if self.index is None:
            self.full_init()
        store = self._get_store()

        item = self.data_list[idx]
        sample_idx = str(item['sample_idx'])
        raw_group = self.required_groups.get('raw', 'raw_nuscenes')
        raw_sample = store.get(raw_group, sample_idx)

        results = copy.deepcopy(raw_sample['meta'])
        results['sample_idx'] = sample_idx
        results['token'] = raw_sample.get('token', results.get('token'))
        results['scene_idx'] = raw_sample.get('scene_idx',
                                              results.get('scene_idx'))
        results['_sharded_raw'] = raw_sample
        results['_sharded_groups'] = {}

        for logical_name, group in self.required_groups.items():
            if logical_name == 'raw' or not group:
                continue
            results['_sharded_groups'][group] = store.get(group, sample_idx)

        return results
