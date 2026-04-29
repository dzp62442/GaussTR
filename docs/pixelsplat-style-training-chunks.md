# PixelSplat-style training chunk design

## 1. 背景

当前实验过的 `NuScenesOccShardedDataset` 使用按来源分组的 shard：

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

分组 shard 只用于前期 1%-10% 验证和问题定位。`dev-shunk` 分支后续不再以 grouped shard 作为训练数据格式，也不要求兼容旧 grouped shard runtime。

PixelSplat 的数据组织方式更接近训练消费模型：

```text
chunk -> torch.load(chunk) -> chunk 内多个完整 training examples
```

它不是按来源分组在训练时 join，而是把训练所需字段预先 materialize 到同一个 chunk 内。训练热路径只读一个 chunk，然后从内存中连续产出多个 sample。

## 2. 目标

新增一层 **training chunks**，直接面向训练消费：

```text
nuScenes annotation + 原始/派生数据目录
  -> offline materialize
  -> training fused chunks
  -> dataloader torch.load(chunk)
```

目标：

- 直接从原始 nuScenes 组织和派生数据目录生成 chunks，不依赖 grouped shard。
- 针对一次训练配置生成 fused chunks，训练时不再跨 group runtime join。
- chunk 大小控制在约 100MiB，避免单包过大。
- 同一 split/profile 的每个 chunk 使用统一 `samples_per_chunk`，便于 dataloader 按 chunk 流式加载。
- 一个 chunk 内包含多个完整 sample，覆盖多个 iteration。
- dataloader 的 IO 模式接近 PixelSplat：shuffle chunk，load chunk，chunk 内 shuffle sample。

非目标：

- 不要求所有来源组合都常驻一份 fused 数据。
- 不保留 grouped shard runtime 兼容路径。
- 不从 grouped shard 二次生成 chunks。
- 不在训练时从 TOS 动态拼接 sample 字段。

## 3. 数据布局

建议新增：

```text
data/gausstr_shards/
  train/
    featup_metric3d_sam2/
      profile.json
      index.json
      chunk_manifest.json
      000000.torch
      000001.torch
      ...
  val/
    featup_metric3d_sam2/
      profile.json
      index.json
      chunk_manifest.json
      000000.torch
      ...
```

`featup_metric3d_sam2` 是 training profile 名称，表达本次 materialized view 的来源组合。

实现时默认输出根目录使用：

```text
data/gausstr_shards/
```

因此实际目录是：

```text
data/gausstr_shards/
  train/
    featup_metric3d_sam2/
      profile.json
      index.json
      chunk_manifest.json
      000000.torch
      ...
  val/
    featup_metric3d_sam2/
      ...
```

`profile` 表示一次训练配置需要 materialize 哪些字段。它不是训练超参，而是数据视图名称：同一份 nuScenes 原始数据可以按不同 profile 生成不同 chunk 目录。

profile 配置表达本次 materialized view 从哪些原始目录取字段。示例：

```json
{
  "name": "featup_metric3d_sam2",
  "source_roots": {
    "data_root": "data/nuscenes",
    "image_prefix": "data/nuscenes",
    "depth_root": "data/nuscenes_metric3d",
    "feat_root": "data/nuscenes_featup",
    "sem_seg_root": "data/nuscenes_grounded_sam2",
    "occ_root": "data/nuscenes"
  },
  "fields": {
    "raw": true,
    "depth": "metric3d",
    "feats": "featup",
    "sem_seg": "grounded_sam2",
    "occ_gt": true
  }
}
```

第一版建议内置两个训练 profile：

```text
talk2dino_metric3d:
  train: raw + depth_metric3d
  val:   raw + depth_metric3d + occ_gt

featup_metric3d_sam2:
  train: raw + depth_metric3d + feats_featup + sem_seg_grounded_sam2
  val:   raw + depth_metric3d + feats_featup + occ_gt
```

profile 只做声明式字段选择，不做隐式 fallback。缺少任何必需源文件应 fail-fast，除非显式传入 debug/allow-missing 参数。

## 4. Chunk 内容

每个 `.torch` chunk 存多个完整 sample：

```python
{
    "metadata": {
        "version": 1,
        "split": "train",
        "profile": "featup_metric3d_sam2",
        "sample_count": 8,
        "samples_per_chunk": 8,
        "sources": {
            "data_root": "data/nuscenes",
            "depth_root": "data/nuscenes_metric3d",
            "feat_root": "data/nuscenes_featup",
            "sem_seg_root": "data/nuscenes_grounded_sam2"
        },
        "schema_version": 1
    },
    "samples": [
        {
            "sample_idx": "000000",
            "is_padding": false,
            "token": "...",
            "scene_idx": 12,
            "meta": {...},
            "images": {...},
            "image_bytes": {...},
            "depth": ...,
            "feats": ...,
            "sem_seg": ...,
            "occ_gt": ...,
        },
        ...
    ],
}
```

sample 内部仍按字段组织；改变的是这些字段被放在同一个 chunk 文件内。chunk sample 应尽量贴近当前 pipeline 需要的 `results` 字段，避免 dataloader 再从磁盘补字段。

训练 pipeline 可以继续使用清晰的 logical keys：

```text
img, depth, feats, sem_seg, occ_gt
```

但这些字段来自同一个 loaded chunk，而不是多个 group shard。

## 5. Chunk 大小策略

目标是让 chunk 文件接近 100MiB，同时保持同一 split/profile 的每个 chunk 拥有统一样本数。

不同 profile 的单样本体积差异很大，因此不能手写固定样本数；但训练阶段也不希望每个 chunk 的样本数大幅波动。预处理应先做 sizing pass：

```text
1. 按 split/profile 从已选样本中随机抽样 assemble 少量 sample，默认 4 个。
2. 用 torch.save 到临时文件或内存估算单样本序列化体积。
3. 使用 p90/p95 单样本字节数计算 samples_per_chunk。
4. samples_per_chunk 在整个 split/profile 内固定。
5. 后续正式写 chunk 时按固定 samples_per_chunk 切分。
```

推荐默认：

```text
target_chunk_bytes = 100 MiB
max_chunk_bytes    = 160 MiB
min_samples_per_chunk = 1
max_samples_per_chunk = 64
```

确定性计算规则：

```text
p95_sample_bytes = sizing pass 得到的 p95
raw_count = floor(target_chunk_bytes / p95_sample_bytes)
samples_per_chunk = clamp(raw_count, min_samples_per_chunk, max_samples_per_chunk)
```

如果写出后发现某个 chunk 超过 `max_chunk_bytes`，脚本应 fail-fast 并提示降低 `target_chunk_bytes` 或 `max_samples_per_chunk`，不应在同一次构建中混用不同 `samples_per_chunk`。

尾部 chunk 处理：

- 所有 chunk 文件的 `samples` 列表长度都等于 `samples_per_chunk`。
- 如果最后一个 chunk 的真实样本不足，预处理脚本写入 deterministic padding records 补齐。
- padding record 必须显式标记 `is_padding=true`，并记录 `source_sample_idx`。
- train 可以按配置消费 padding 样本或在 epoch 层重新 padding/drop，以保证 DDP rank 等长。
- val/test 必须跳过 padding records，不计入 evaluator，保证全集覆盖且无重复评估。

如果后续实测 TOS 对 100MiB 文件仍有较大打开开销，可在同一 sizing 规则下压测 64/100/128MiB 三档。第一版不建议默认超过 160MiB。

## 6. Index 与 Manifest

`chunk_manifest.json`：

```json
{
  "profile": "featup_metric3d_sam2",
  "profile_sha256": "...",
  "target_chunk_bytes": 104857600,
  "max_chunk_bytes": 167772160,
  "samples_per_chunk": 8,
  "chunks": {
    "000000": {
      "path": "000000.torch",
      "bytes": 101883210,
      "sha256": "...",
      "num_samples": 8,
      "num_valid_samples": 8,
      "num_padding_samples": 0,
      "sample_indices": ["000000", "000001"],
      "source_indices": [0, 1],
      "is_tail": false
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
      "global_offset": 0,
      "chunk_id": "000000",
      "offset": 0,
      "token": "...",
      "scene_idx": "...",
      "is_padding": false
    }
  ],
  "by_sample_idx": {
    "000000": {
      "chunk_id": "000000",
      "offset": 0,
      "global_offset": 0
    }
  }
}
```

训练 dataloader 主要依赖 manifest 的 chunk 列表；`index.json` 用于校验、调试、可视化或按 sample 定位问题。

## 7. 预处理流程

输入直接来自当前项目原始数据组织和派生产物目录：

```text
data/nuscenes/nuscenes_infos_{split}.pkl
data/nuscenes/samples/CAM_*
data/nuscenes/gts/{scene_idx}/{token}/labels.npz
data/nuscenes_metric3d/**/*.npy
data/nuscenes_featup/**/*.npy
data/nuscenes_grounded_sam2/**/*.npy
```

处理步骤：

1. 读取 `nuscenes_infos_{split}.pkl`，确定 split 样本列表。
2. 根据 `ratio/max_samples/seed` 选择样本。train 可以 scene-level subset 后 deterministic shuffle；val/test 保持稳定顺序。
3. 根据 profile 解析每个 sample 所需的 image/depth/feat/sem/occ 路径。
4. sizing pass：随机抽样 assemble sample，默认 4 个，估算 p90/p95 sample bytes，确定固定 `samples_per_chunk`。
5. build pass：按固定 `samples_per_chunk` 顺序 assemble 完整 samples。
6. 写 `000000.torch.tmp`。
7. sanity `torch.load`，校验 sample 数、必需字段、shape/dtype。
8. 计算 sha256，原子 rename 为 `.torch`。
9. 写 `profile.json`、`chunk_manifest.json` 和 `index.json`。
10. 结束时做全目录校验，确认无缺真实样本、无重复真实样本、padding 标记正确、无残留 `.tmp`。

这里的 join 发生在离线预处理阶段，不发生在训练热路径。由于输入是原始小文件和派生小文件，预处理阶段可以使用多线程/多进程并行读取；训练阶段只面对完整 chunk 文件。

TOS/FUSE 抖动处理沿用旧预处理脚本中的策略，并扩展到源文件读取：

- image bytes、`.npy`、`.npz` 读取遇到 `EAGAIN/EBUSY/EINTR/ESTALE` 时重试。
- 源文件存在性判断使用 `stat` retry，不依赖 `Path.exists()` 的静默失败结果。
- `.torch.tmp` 写完后做 sanity `torch.load`，对 PyTorch zip central directory 短暂不可见错误重试。
- `.tmp -> .torch` 使用原子 rename，并对可重试文件系统错误退避重试。
- manifest 使用临时 JSON 写入后 rename，避免半写入 manifest。

断点续跑策略：

- 已存在 `.torch` 且 manifest 记录匹配时跳过。
- `.tmp` 或 manifest 不一致时重建该 chunk。
- 不在训练时检查 `.SUCCESS` sidecar；完整性由预处理结束校验保证。
- 若用户删除 sha/sidecar，不影响训练；若要继续断点续跑，必须保留 manifest 和 `.torch`。

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
- 训练 split 按 chunk 分 rank/worker，优先保证大文件局部性。
- val/test 按 `index.json` 中的 `global_offset % world_size` 做 sample 级 rank 分片，再在 rank 内按 worker 分片；这样即使后续 `samples_per_chunk > 1`，评估也不会因 chunk 粒度不均而漏样本或重复样本。
- 不需要按 group 预取。
- 不需要 runtime 计算 derived group shard 覆盖关系。
- 配置中 `sampler=None`；rank/worker 分片由 dataset 内部完成。
- 训练 split 可以 shuffle；val/test 固定 chunk 顺序和 chunk 内顺序。
- 训练时 `torch.load(chunk)` 也要对 TOS/FUSE 临时错误重试；预处理结束校验保证完整性，dataloader 不做 sha256 校验。
- 由于没有外部 sampler，MMEngine epoch 需要通过 `ChunkDatasetEpochHook` 传给 dataset；dataset 使用共享 epoch 状态让 persistent workers 在恢复训练和跨 epoch 时使用正确 shuffle seed。

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

epoch length：

- 预处理 manifest 记录每个 chunk 的 `num_samples`、`num_valid_samples` 和全局真实样本数。
- 训练时 dataset 计算 per-rank 可产出的 sample 数，并保证各 rank 等长。
- 不足部分采用可复现 padding 或 drop；策略必须写入 config，避免 DDP hang。
- val/test 跳过 chunk 内 `is_padding=true` records，不 shuffle，按 MMEngine 分布式评估要求覆盖全集并汇总。
- val/test 的 `__len__` 返回当前 rank 实际负责的真实样本数；worker 只切分本 rank 内部样本，不改变 rank 级长度。

第一版不建议保持 map-style dataset。map-style 随机 sample 访问很容易退化成反复打开同一个 chunk 或跨 chunk 随机访问，违背本方案目标。

## 9. 与现有 pipeline 的关系

现有 pipeline 中的 shard loader：

```text
BEVLoadMultiViewImageFromShards
LoadShardedFeatMaps(depth)
LoadShardedFeatMaps(feats)
LoadShardedFeatMaps(sem_seg)
```

可以替换为 chunk-aware loader，或在 dataset `__iter__` 中直接把字段准备好：

```text
_chunk_raw
_chunk_depth
_chunk_feats
_chunk_sem_seg
```

更简单的第一版：

- Dataset 从 chunk sample 中构造与现有 pipeline 期望一致的 `results`。
- 新增 loader 类只从 `results` 内存字段取值，不再访问磁盘。
- 模型侧 `GaussTR.prepare_inputs()` 不应感知数据来自 chunk。

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

来源替换通过 profile 和输入目录完成：

```text
depth_root = data/nuscenes_metric3d
depth_root = data/nuscenes_metric3d_v2
feat_root  = data/nuscenes_featup
feat_root  = data/nuscenes_dinov2
sem_seg_root = data/nuscenes_grounded_sam2
```

training chunks 只 materialize 当前实验需要的组合。替换一个来源时，生成新的 profile 输出目录：

```text
featup_metric3d_sam2
featup_metric3d_v2_sam2
dinov2_metric3d_sam2
```

如果只替换一个来源，只需要重新生成对应 profile 的 chunks，不需要改 dataloader 逻辑。

## 12. 风险与取舍

优点：

- 训练热路径简单，接近 PixelSplat。
- 每次 `torch.load` 产出多个完整 samples。
- 对 TOS/FUSE 更友好，减少多文件 join。
- 预处理阶段一次性承受小文件读取成本，训练阶段只读 chunk。

代价：

- 需要为每个 training profile 额外占用一份 fused 数据。
- profile 改变后需要重新 materialize。
- 预处理时间增加，且会重新读取原始/派生小文件。
- 若 profile 很多，存储管理需要明确生命周期。

## 13. 推荐实施顺序

1. 在 10% train/val 上实现 `build_training_chunks.py`，直接从原始数据和派生数据目录读取。
2. 生成 `featup_metric3d_sam2` 与 `talk2dino_metric3d` profile，target chunk 100MiB。
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
        chunk_root="data/gausstr_shards",
        split="train",
        profile="featup_metric3d_sam2",
        chunk_shuffle=True,
        sample_shuffle=True,
        target_chunk_bytes=100 * 1024**2,
        pipeline=train_pipeline,
    ),
)

custom_hooks = [
    dict(type="AutoResumeHook"),
    dict(type="ChunkDatasetEpochHook"),
]
```

`num_workers` 应从 1 或 2 起步，根据 TOS slow-load 和 GPU 利用率逐步增加。
