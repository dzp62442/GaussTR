import itertools
import math
from typing import Iterator, Optional

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet3d.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class ShardAwareSampler(Sampler):
    """Sampler that shuffles raw shards but keeps shard-local sample order.

    This preserves large-file locality for sharded datasets while still using
    MMEngine's map-style dataset and DDP-compatible sampling contract.
    """

    def __init__(self,
                 dataset,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True,
                 raw_group: str = 'raw_nuscenes',
                 num_workers: int = 1,
                 prefetch_shards: int = 1,
                 sample_shuffle_block_size: int = 1,
                 prefetch_samples: int = 16) -> None:
        rank, world_size = get_dist_info()
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = sync_random_seed() if seed is None else seed
        self.epoch = 0
        self.round_up = round_up
        self.raw_group = raw_group
        self.num_workers = max(1, int(num_workers))
        self.prefetch_shards = max(0, int(prefetch_shards))
        self.sample_shuffle_block_size = max(1, int(sample_shuffle_block_size))
        self.prefetch_samples = max(0, int(prefetch_samples))
        self.num_samples_per_rank = self._estimate_samples_per_rank()
        self.total_size = self.num_samples_per_rank * world_size

    def _estimate_samples_per_rank(self) -> int:
        """Return a conservative per-rank length for shard-level partitioning.

        `len(dataset) / world_size` is not enough here because ranks receive
        whole raw shards. If one rank owns one more raw shard than another,
        cropping to the sample-level ceil would silently drop samples from that
        rank. Using the worst-case shard count keeps all shard-owned samples and
        pads shorter ranks instead, which matches DDP's equal-length contract.
        """
        index = self.dataset.get_raw_shard_indices(self.raw_group)
        if not index:
            return 0
        max_shards_per_rank = int(math.ceil(len(index) / self.world_size))
        max_shard_len = max(len(indices) for indices in index.values())
        return max_shards_per_rank * max_shard_len

    def __iter__(self) -> Iterator:
        index = self.dataset.get_raw_shard_indices(self.raw_group)
        shard_ids = list(index)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(shard_ids), generator=generator).tolist()
            shard_ids = [shard_ids[i] for i in perm]
        if hasattr(self.dataset, 'set_raw_shard_order'):
            self.dataset.set_raw_shard_order(shard_ids)

        rank_shard_ids = shard_ids[self.rank::self.world_size]
        if self.round_up and not rank_shard_ids and shard_ids:
            rank_shard_ids = [shard_ids[self.rank % len(shard_ids)]]
        worker_shards = [[] for _ in range(self.num_workers)]
        for shard_pos, shard_id in enumerate(rank_shard_ids):
            shard_indices = list(index[shard_id])
            if self.shuffle:
                generator = torch.Generator()
                generator.manual_seed(self.seed + self.epoch +
                                      int(shard_id))
                if self.sample_shuffle_block_size <= 1:
                    perm = torch.randperm(
                        len(shard_indices), generator=generator).tolist()
                    shard_indices = [shard_indices[i] for i in perm]
                else:
                    block_size = self.sample_shuffle_block_size
                    blocks = [
                        shard_indices[start:start + block_size]
                        for start in range(0, len(shard_indices), block_size)
                    ]
                    perm = torch.randperm(
                        len(blocks), generator=generator).tolist()
                    shard_indices = [
                        sample_index for block_index in perm
                        for sample_index in blocks[block_index]
                    ]
            worker_shards[shard_pos % self.num_workers].append(
                (shard_id, shard_indices))

        worker_sample_streams = [[] for _ in range(self.num_workers)]
        for worker_id, shard_stream in enumerate(worker_shards):
            for _, shard_indices in shard_stream:
                worker_sample_streams[worker_id].extend(shard_indices)

        if self.round_up and sum(len(stream)
                                 for stream in worker_sample_streams
                                 ) < self.num_samples_per_rank:
            for worker_id, stream in enumerate(worker_sample_streams):
                if not stream:
                    continue
                target_len = (
                    self.num_samples_per_rank + self.num_workers - 1 -
                    worker_id) // self.num_workers
                if len(stream) < target_len:
                    source = tuple(stream)
                    stream.extend(
                        itertools.islice(
                            itertools.cycle(source),
                            target_len - len(stream)))

        worker_streams = []
        for stream in worker_sample_streams:
            worker_stream = []
            for pos, sample_index in enumerate(stream):
                next_sample_indices = tuple(stream[pos + 1:pos + 1 +
                                                   self.prefetch_samples])
                worker_stream.append((sample_index, next_sample_indices))
            worker_streams.append(worker_stream)

        indices = []
        max_stream_len = max((len(stream) for stream in worker_streams),
                             default=0)
        for offset in range(max_stream_len):
            for stream in worker_streams:
                if offset < len(stream):
                    indices.append(stream[offset])

        if self.round_up and len(indices) < self.num_samples_per_rank:
            if not indices:
                return iter(indices)
            repeats = math.ceil(self.num_samples_per_rank / len(indices))
            indices = list(itertools.chain.from_iterable(
                itertools.repeat(indices, repeats)))
        indices = indices[:self.num_samples_per_rank]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)
