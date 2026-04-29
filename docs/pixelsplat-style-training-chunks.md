# PixelSplat-style training chunk design

## 1. 背景

当前 `NuScenesOccShardedDataset` 使用按来源分组的 shard：

```text
raw_nuscenes/
depth_metric3d/
feats_featup/
sem_seg_grounded_sam2/
occ_gt/
```

这个组织方式利于管理资产来源：深度、特征、语义分割可以独立替换、复用和检查。但它把来源组合推迟到了训练热路径：

```text
sample -> raw shard + depth shard + feat shard + sem shard -> assemble
```

在 TOS/FUSE 路径上，这会导致一个训练样本或一段局部样本跨多个 100-260MiB `.torch` 包读取。4 卡训练时，即使关闭后台预取、每 rank 只保留 1 个 DataLoader worker，TOS 多进程读取仍会把单包带宽压到很低，GPU 长时间等待数据。

PixelSplat 的数据组织方式更接近训练消费模型：

```text
chunk -> torch.load(chunk) -> chunk 内多个完整 training examples
```

它不是按来源分组在训练时 join，而是把训练所需字段预先 materialize 到同一个 chunk 内。训练热路径只读一个 chunk，然后从内存中连续产出多个 sample。

## 2. 目标

新增一层 **training chunks**，把“资产管理”和“训练消费”分开：

```text
source group shards     = 可替换、可复用的中间资产
training fused chunks   = 某个训练 profile 的消费视图
```

目标：

- 保留现有分组资产，继续支持替换 depth/feat/sem 来源。
- 针对一次训练配置生成 fused chunks，训练时不再跨 group runtime join。
- chunk 大小控制在约 100-200MiB，避免单包过大。
- 一个 chunk 内包含多个完整 sample，覆盖多个 iteration。
- dataloader 的 IO 模式接近 PixelSplat：shuffle chunk，load chunk，chunk 内 shuffle sample。

非目标：

- 不要求所有来源组合都常驻一份 fused 数据。
- 不用 fused chunks 替代 source group shards 的资产管理职责。
- 不在训练时从 TOS 动态拼接 fused chunk。

## 3. 数据布局

建议新增：

```text
data/gausstr_training_chunks/
  train/
    metric3d_featup_sam2/
      index.json
      chunk_manifest.json
      000000.torch
      000001.torch
      ...
  val/
    metric3d_featup_sam2/
      index.json
      chunk_manifest.json
      000000.torch
      ...
```

`metric3d_featup_sam2` 是 training profile 名称，表达本次 materialized view 的来源组合。

profile 配置示例：

```json
{
  "raw": "raw_nuscenes",
  "depth": "depth_metric3d",
  "feats": "feats_featup",
  "sem_seg": "sem_seg_grounded_sam2",
  "occ_gt": "occ_gt"
}
```

## 4. Chunk 内容

每个 `.torch` chunk 存多个完整 sample：

```python
{
    "metadata": {
        "version": 1,
        "split": "train",
        "profile": "metric3d_featup_sam2",
        "sample_count": 12,
        "sources": {
            "raw": "raw_nuscenes",
            "depth": "depth_metric3d",
            "feats": "feats_featup",
            "sem_seg": "sem_seg_grounded_sam2",
        },
    },
    "samples": [
        {
            "sample_idx": "000000",
            "token": "...",
            "scene_idx": 12,
            "meta": {...},
            "raw": {
                "imgs": ...,
                "mask_camera": ...,
                "...": "fields needed by BEVLoadMultiViewImageFromShards",
            },
            "depth": ...,
            "feats": ...,
            "sem_seg": ...,
            "occ_gt": ...,
        },
        ...
    ],
}
```

sample 内部仍按字段/来源组织；改变的是这些字段被放在同一个 chunk 文件内。

训练 pipeline 可以继续使用清晰的 logical keys：

```text
img, depth, feats, sem_seg, occ_gt
```

但这些字段来自同一个 loaded chunk，而不是多个 group shard。

## 5. Chunk 大小策略

不要按固定 sample 数切 chunk。不同来源组合的 sample 大小差异很大，应按目标字节数切：

```text
target_chunk_bytes = 100-200 MiB
max_chunk_bytes    = 256 MiB
```

预处理过程逐个 sample assemble，估算当前 chunk 序列化后大小，达到 target 后 flush。

初始建议：

```text
train target_chunk_bytes = 128 MiB
val   target_chunk_bytes = 128 MiB
```

如果 TOS 单包 `torch.load` 稳定但文件打开开销大，可升到 200MiB；如果多卡仍出现单包长尾，可降到 64-100MiB。

## 6. Index 与 Manifest

`chunk_manifest.json`：

```json
{
  "profile": "metric3d_featup_sam2",
  "target_chunk_bytes": 134217728,
  "chunks": {
    "000000": {
      "path": "000000.torch",
      "bytes": 129883210,
      "num_samples": 12,
      "sample_indices": ["000000", "000001"]
    }
  }
}
```

`index.json`：

```json
{
  "samples": [
    {
      "sample_idx": "000000",
      "chunk_id": "000000",
      "offset": 0
    }
  ],
  "by_sample_idx": {
    "000000": {
      "chunk_id": "000000",
      "offset": 0
    }
  }
}
```

训练 dataloader 主要依赖 manifest 的 chunk 列表；按 sample 随机访问时才需要 `by_sample_idx`。

## 7. 预处理流程

输入仍然是现有 source group shards：

```text
raw_nuscenes + depth_metric3d + feats_featup + sem_seg_grounded_sam2
```

处理步骤：

1. 读取 source `index.json`，确定 split 的样本顺序。
2. 根据 profile 选择 required groups。
3. 对每个 sample，从 source group shards 读取对应 payload。
4. assemble 成完整 training sample。
5. 累积到当前 chunk。
6. 达到 target size 后写出 `.torch.tmp`。
7. 原子 rename 为 `.torch`。
8. 写 `chunk_manifest.json` 和 `index.json`。

预处理可以继续复用现有 `ShardMemoryStore` 或更轻量的 source reader。这里的 runtime join 发生在离线预处理阶段，不发生在训练热路径。

## 8. Dataloader 设计

推荐新增 `NuScenesOccChunkDataset`，优先实现成 `IterableDataset` 风格：

```text
每个 epoch:
  DDP rank 切分 chunk list
  DataLoader worker 切分 rank 内 chunk list
  shuffle chunk order
  for chunk in worker_chunks:
      payload = torch.load(chunk)
      optionally shuffle payload["samples"]
      yield sample
```

关键点：

- shuffle 的主要单位是 chunk。
- chunk 内 sample 可以 block shuffle 或全 shuffle，因为所有字段已经在内存中。
- 一个 worker 同一时刻只需要加载一个 chunk。
- 不需要按 group 预取。
- 不需要 runtime 计算 derived group shard 覆盖关系。

示意：

```python
class NuScenesOccChunkDataset(IterableDataset):
    def __iter__(self):
        chunks = self.partition_chunks_by_rank_and_worker()
        chunks = self.shuffle_chunks(chunks, self.epoch)
        for chunk_path in chunks:
            payload = torch.load(chunk_path, map_location="cpu")
            samples = payload["samples"]
            samples = self.shuffle_samples(samples, self.epoch, chunk_path)
            for sample in samples:
                yield self.pipeline(self.convert_sample(sample))
```

如果必须保持 map-style dataset，也应让 sampler 以 chunk-local 顺序输出 sample，避免跨 chunk 随机访问。

## 9. 与现有 pipeline 的关系

现有 pipeline 中的 shard loader：

```text
BEVLoadMultiViewImageFromShards
LoadShardedFeatMaps(depth)
LoadShardedFeatMaps(feats)
LoadShardedFeatMaps(sem_seg)
```

可以替换为 chunk-aware loader，或在 dataset `get_data_info` 中直接把字段准备好：

```text
_chunk_raw
_chunk_depth
_chunk_feats
_chunk_sem_seg
```

更简单的第一版：

- Dataset 从 chunk sample 中构造与现有 pipeline 期望一致的 `results`。
- 新增 loader 类只从 `results` 内存字段取值，不再访问磁盘。

## 10. 训练 IO 模式对比

当前 grouped 热路径：

```text
sample -> raw torch.load
       -> depth torch.load
       -> feat torch.load
       -> sem torch.load
       -> assemble
```

fused chunk 热路径：

```text
chunk torch.load -> many complete samples -> pipeline
```

这会减少：

- 每个 sample 触发的文件数量。
- 多 group shard 粒度不一致带来的随机 miss。
- DataLoader worker 与 shard prefetch 线程之间的 TOS 读竞争。
- DDP rank 因某个 group shard 长尾读而同步等待的概率。

## 11. 来源替换策略

分组资产仍然负责灵活替换：

```text
depth_metric3d
depth_metric3d_v2
feats_featup
feats_dino
sem_seg_grounded_sam2
```

training chunks 只 materialize 当前实验需要的组合：

```text
metric3d_featup_sam2
metric3d_v2_featup_sam2
metric3d_dino_sam2
```

如果只替换一个来源，只需要重新生成对应 profile 的 chunks，不需要改 dataloader 逻辑。

## 12. 风险与取舍

优点：

- 训练热路径简单，接近 PixelSplat。
- 每次 `torch.load` 产出多个完整 samples。
- 对 TOS/FUSE 更友好，减少多文件 join。
- 可以保留分组资产作为离线来源层。

代价：

- 需要为每个 training profile 额外占用一份 fused 数据。
- profile 改变后需要重新 materialize。
- 预处理时间增加。
- 若 profile 很多，存储管理需要明确生命周期。

## 13. 推荐实施顺序

1. 在 10% train/val 上实现 `build_training_chunks.py`。
2. 生成 `metric3d_featup_sam2` profile，target chunk 128MiB。
3. 实现 `NuScenesOccChunkDataset` 的最小训练路径。
4. 4 卡跑 10% 数据，比较：

```text
Checkpoints -> first 50 iter
steady-state time/data_time
slow load count
GPU utilization
```

5. 若明显优于 grouped loader，再扩展到全量。
6. 全量阶段只 materialize 实际要训练的 profile，避免生成过多组合。

## 14. 初始配置建议

```python
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=1,
    dataset=dict(
        type="NuScenesOccChunkDataset",
        chunk_root="data/gausstr_training_chunks",
        split="train",
        profile="metric3d_featup_sam2",
        chunk_shuffle=True,
        sample_shuffle=True,
        target_chunk_bytes=128 * 1024**2,
        pipeline=train_pipeline,
    ),
)
```

`num_workers` 应从 1 或 2 起步，根据 TOS slow-load 和 GPU 利用率逐步增加。

