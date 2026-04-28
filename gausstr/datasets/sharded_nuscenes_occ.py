import copy
from pathlib import Path
from typing import Dict, Optional

from mmengine.dataset import BaseDataset
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
                 preload_mode='all',
                 require_success=True,
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
        self.store: Optional[ShardMemoryStore] = None
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
        groups = [group for group in self.required_groups.values() if group]
        if 'raw_nuscenes' not in groups:
            groups.insert(0, 'raw_nuscenes')
        self.store = ShardMemoryStore(
            self.shard_root,
            self.split,
            groups,
            preload_mode=self.preload_mode,
            require_success=self.require_success)
        return [dict(item) for item in self.store.samples]

    def get_data_info(self, idx: int) -> dict:
        if self.store is None:
            self.full_init()
        assert self.store is not None

        item = self.data_list[idx]
        sample_idx = str(item['sample_idx'])
        raw_group = self.required_groups.get('raw', 'raw_nuscenes')
        raw_sample = self.store.get(raw_group, sample_idx)

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
            results['_sharded_groups'][group] = self.store.get(
                group, sample_idx)

        return results
