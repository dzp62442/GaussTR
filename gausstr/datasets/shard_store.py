import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch


def torch_load(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


class ShardMemoryStore:
    """Read GaussTR shard groups from a mounted path.

    The default mode loads all requested shards into process memory. This is
    deliberate for the current 1% sharded dataset: avoid local disk cache and
    avoid training-time TOS reads.
    """

    def __init__(self,
                 shard_root,
                 split: str,
                 groups: Iterable[str],
                 preload_mode: str = 'all',
                 require_success: bool = True) -> None:
        self.shard_root = Path(shard_root)
        self.split = split
        self.split_root = self.shard_root / split
        self.groups = list(dict.fromkeys(groups))
        self.preload_mode = preload_mode
        self.require_success = require_success

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

        if preload_mode == 'all':
            self.preload_all()

    @staticmethod
    def _load_json(path: Path):
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)

    def _shard_path(self, group: str, shard_id: str) -> Path:
        manifest = self.group_manifests[group]
        entry = manifest['shards'][shard_id]
        return self.split_root / entry['path']

    def _load_shard(self, group: str, shard_id: str) -> Mapping:
        if shard_id in self.loaded_shards[group]:
            return self.loaded_shards[group][shard_id]

        path = self._shard_path(group, shard_id)
        if self.require_success:
            success_path = path.with_suffix('.SUCCESS')
            if not success_path.exists():
                raise FileNotFoundError(
                    f'Shard success marker is missing: {success_path}')

        payload = torch_load(path)
        self.loaded_shards[group][shard_id] = payload
        for sample in payload['samples']:
            self.group_samples[group][str(sample['sample_idx'])] = sample
        return payload

    def preload_all(self) -> None:
        for group in self.groups:
            for shard_id in sorted(self.group_manifests[group]['shards']):
                self._load_shard(group, shard_id)

    def get(self, group: str, sample_idx: str) -> Mapping:
        sample_idx = str(sample_idx)
        sample = self.group_samples[group].get(sample_idx)
        if sample is not None:
            return sample

        sample_entry = self.index['by_sample_idx'][sample_idx]['groups'][group]
        shard = self._load_shard(group, sample_entry['shard_id'])
        return shard['samples'][sample_entry['offset']]

    def get_optional(self, group: Optional[str], sample_idx: str):
        if not group:
            return None
        return self.get(group, sample_idx)

