# TOS Sharded Dataset 设计方案

本文设计 GaussTR 在 TOS 挂载路径上训练时的数据重组方案。目标是把训练期的大量小文件随机读取，转换为少量较大 `.torch` shard 的顺序读取，并保留后续替换深度、特征、伪标签方法的灵活性。

方案参考 pixelSplat / RealEstate10K 的 chunk 思路：离线把多个样本打包成 `.torch` 文件，训练时 `torch.load()` 整个 shard 到 CPU 内存，再在内存中取样。但 GaussTR 的数据类型更多，且中间产物可能替换，因此采用：

```text
统一 sample_order
+ raw_nuscenes 合并原始 nuScenes meta/images
+ 派生数据按类型和方法分组
+ 不同组允许不同 samples_per_shard
+ 所有组的 shard 边界按 base_block 对齐，保持包含关系
```

## 1. 当前约束

- 训练最多单机 8 卡，常用单机 4 卡，不考虑多机。
- 1 张 A100 约配 14 个 vCPU 和 245GB 内存，增加卡数时 CPU 和内存同步增加。
- TOS 已挂载到服务器文件系统，可以直接按本地路径访问。
- TOS 顺序读取大文件速度约 100MB/s 以上，但随机读取大量小文件很慢。
- TOS 空间近似无限，服务器内存充足，本地硬盘空间紧张。

因此 MVP 主路径是：

```text
TOS mount path -> torch.load(shard) -> CPU memory cache -> sample assemble
```

不把 “TOS -> 本地硬盘 cache -> 内存” 作为默认路径。

## 2. 核心分组

### 2.1 raw_nuscenes

`raw_nuscenes` 是稳定的原始 nuScenes 数据组，合并：

- sample meta
- scene/token/sample_idx
- 6 路 camera 标定与位姿
- 6 路 camera image bytes
- 原始 image path/source path 作为追踪信息

它不再拆成 `meta/` 和 `images/` 两个类型组。原因是训练时如果继续从原始 nuScenes 目录随机读取 images，仍然会遇到大量小文件 I/O 问题；但 meta 和 images 都来自稳定原始数据，后续替换深度或特征时不应重复生成。

### 2.2 派生数据组

派生数据按类型和方法分组：

```text
depth_metric3d
feats_featup
sem_seg_grounded_sam2
occ_gt
```

以后替换深度或特征方法时新增组即可，例如：

```text
depth_depthanything
feats_dinov2
```

不同方法互不覆盖，dataloader 通过配置选择需要的组。

## 3. 全局一致性

所有组共享同一个 `sample_order`。训练集 `sample_order` 由 seed 确定；验证/测试集保持稳定顺序。

每个样本有稳定位置：

```text
global_offset = position in sample_order
sample_id = sample_idx or token
```

每个数据组都有自己的 `samples_per_shard`，但必须满足：

```text
samples_per_shard(group) = base_block_size * integer
```

也就是说，各组 shard 边界都落在同一套 base block 边界上。小体积组的 shard 可以覆盖多个大体积组 shard。

示例：

```text
base_block_size = 8 samples

depth_metric3d:          8 samples/shard
feats_featup:           16 samples/shard
sem_seg_grounded_sam2:  32 samples/shard
raw_nuscenes:           64 samples/shard
```

则：

```text
raw_nuscenes/000000 covers sample offsets 0-63
depth_metric3d/000000 covers offsets 0-7
depth_metric3d/000001 covers offsets 8-15
...
depth_metric3d/000007 covers offsets 56-63
```

这个包含关系不是 dataloader 正确性的前提，但能让 cache、预取、调试和一致性校验更简单。

## 4. 目录结构

默认输出在项目内：

```text
data/gausstr_shards/
```

该路径可以是指向 TOS 挂载目录的软链接。

推荐结构：

```text
data/gausstr_shards/
├── manifest.json
└── train/
    ├── manifest.json
    ├── sample_manifest.pkl
    ├── sample_manifest.json
    ├── sample_order.json
    ├── index.json
    ├── stats.json
    ├── build_summary.json
    ├── raw_nuscenes/
    │   ├── group_manifest.json
    │   ├── shard_plan.json
    │   ├── 000000.torch
    │   └── 000001.torch
    ├── depth_metric3d/
    │   ├── group_manifest.json
    │   ├── shard_plan.json
    │   └── 000000.torch
    ├── feats_featup/
    ├── sem_seg_grounded_sam2/
    └── occ_gt/
```

根 `manifest.json` 描述数据集格式和全局信息。split `manifest.json` 描述当前 split 使用的样本集合、source roots、构建参数和已构建 group。

`sample_order.json` 是全局样本顺序，所有 group 的 shard plan 都只能按它切分。

`index.json` 是统一索引，格式为：

```json
{
  "schema_version": 2,
  "split": "train",
  "num_samples": 28130,
  "sample_order_sha256": "...",
  "samples": [
    {
      "global_offset": 0,
      "sample_idx": "...",
      "token": "...",
      "scene_idx": "..."
    }
  ],
  "groups": {
    "raw_nuscenes": {
      "000000": {"start": 0, "end": 64}
    },
    "depth_metric3d": {
      "000000": {"start": 0, "end": 8}
    }
  },
  "by_sample_idx": {
    "...": {
      "global_offset": 0,
      "groups": {
        "raw_nuscenes": {"shard_id": "000000", "offset": 0},
        "depth_metric3d": {"shard_id": "000000", "offset": 0}
      }
    }
  }
}
```

dataloader 只依赖 `sample_id -> group -> shard_id/offset`，不需要假设不同 group 的 shard_id 相同。

## 5. shard 内容

### 5.1 raw_nuscenes

建议保存 JPEG 原始 bytes，而不是解码后的 float tensor。

```python
{
    "schema_version": 2,
    "format": "gausstr-sharded-v2",
    "split": "train",
    "group": "raw_nuscenes",
    "shard_id": "000000",
    "sample_idx": [...],
    "global_offsets": [...],
    "sample_order_sha256": "...",
    "camera_order": [...],
    "samples": [
        {
            "sample_idx": str,
            "token": str,
            "scene_idx": str,
            "meta": dict,
            "images": {
                "CAM_FRONT": {
                    "img_path": str,
                    "source_path": str,
                    "bytes": bytes
                }
            }
        }
    ]
}
```

保留完整 `meta` 是为了尽量复用现有 GaussTR pipeline 的字段，不在预处理阶段重新定义一套复杂 schema。

### 5.2 depth_metric3d

```python
{
    "schema_version": 2,
    "format": "gausstr-sharded-v2",
    "group": "depth_metric3d",
    "kind": "depth",
    "method": "metric3d",
    "sample_idx": [...],
    "global_offsets": [...],
    "samples": [
        {
            "sample_idx": str,
            "depth": {
                "CAM_FRONT": {
                    "source_path": str,
                    "tensor": Tensor
                }
            }
        }
    ]
}
```

第一版不强制 dtype 转换，保持源 `.npy` 的 dtype。后续可增加 `--depth-dtype float16` 作为显式压缩选项。

### 5.3 feats_featup / sem_seg_grounded_sam2

结构同 depth，只是字段分别为 `feats` 和 `sem_seg`。

### 5.4 occ_gt

`occ_gt` 默认只在 `val/test` 自动加入。训练如需使用，可以通过 `--modalities occ_gt` 显式指定。

## 6. shard 大小策略

当前 10% 验证结果显示，64 samples/shard 时大致为：

```text
raw_nuscenes(meta + images): ~88MB/shard
depth_metric3d:             ~2GB/shard
feats_featup:               ~970MB/shard
sem_seg_grounded_sam2:      ~520MB/shard
```

因此不能让所有 group 共用 `64 samples/shard`。推荐按 base block 对齐：

```text
base_block_size: 8
raw_nuscenes: 64 samples/shard
depth_metric3d: 8 or 16 samples/shard
feats_featup: 16 or 32 samples/shard
sem_seg_grounded_sam2: 32 or 64 samples/shard
```

默认建议：

```text
--base-block-size 8
--raw-target-shard-size 100MB
--target-shard-size 256MB
--min-blocks-per-shard 1
--max-blocks-per-shard 8
```

自动推断时，对每个 group 单独用 P90 单样本大小估算：

```text
raw_nuscenes 用 raw target，默认约 100MB
派生组用 target_shard_size，默认约 256MB
```

然后把推断出的样本数向下对齐到 `base_block_size` 的整数倍。

最后一个 shard 允许不足一个完整 shard，但必须仍然是 sample_order 的连续后缀。

## 7. 随机性

随机性由 sample 级顺序控制，不由 shard 文件顺序决定。

构建阶段：

- train：按 scene 采样子集，再对 selected samples 做 deterministic shuffle。
- val/test：保持稳定顺序。

训练阶段：

- 每个 epoch 由 `ShardAwareSampler` shuffle raw shard 顺序。
- 每个 raw shard 内部再按 epoch seed shuffle 样本顺序。
- DDP 单机多卡按 raw shard 分配给各 rank，避免不同 rank 频繁竞争同一批 shard。
- 每个 rank 的 epoch 长度按“最多 raw shard 数 * 最大 raw shard 样本数”估算；短 rank 只补样本，不截断长 rank 已持有的 shard 样本。
- 如果小比例调试集的 raw shard 数少于 GPU 数，空 rank 会复用一个 raw shard，以保证 DDP 每个 rank 都有数据；全量训练 raw shard 数远大于 GPU 数时不会触发这个退化路径。
- dataloader 根据 sample_id 查各 group 的 shard/offset。

因此，训练随机性来自 epoch 级 shard shuffle 和 shard 内 sample shuffle；预处理时的 `sample_order` 只提供稳定全局索引，不要求 dataloader 再做完全 sample-level 随机访问。

### 7.1 worker-local shard stream

训练使用 map-style dataset，真正的取样顺序由主进程 sampler 产生，再由 PyTorch DataLoader 分发给 worker。为了保持大文件局部性，`ShardAwareSampler` 的输出必须满足：

```text
每个 rank 先获得一组 raw shard
每个 raw shard 内按 epoch seed shuffle 局部 sample block
raw shard 以 worker stream 为单位分配
sampler 按 worker round-robin 交错输出样本
```

在 `batch_size=1`、`sampler.num_workers == dataloader.num_workers` 时，PyTorch DataLoader 会把交错后的第 `i, i + num_workers, ...` 个样本发给同一个 worker，因此每个 worker 实际消费的是自己的 raw shard stream。

`num_workers` 不能只按 CPU 数量拉高。每个 rank 分到的 raw shard 数如果不能较均匀地分给 worker，epoch 尾部会只剩少数 worker stream 还有样本；这些尾部样本会被 PyTorch 继续轮询分发到多个 worker，造成未预取 shard 的同步加载、重复加载和 DDP straggler。对 TOS/FUSE 路径，过多 worker 还会把 200MiB 级 `.torch` 并发读打到极低带宽。10% train shard 的 4 卡稳定性验证配置使用 `num_workers=1`，先把每 rank 的前台读并发压到 1；如果 slow-load 日志显示单包带宽稳定，再逐步尝试 2 或 3 worker。

这个约束不是普通 sample-level 随机采样；它是 shard-local 随机采样。随机性来自：

- epoch 级 raw shard shuffle；
- shard 内局部 sample block shuffle；
- DDP rank 间 raw shard 分片。

不应对 raw shard 内全部样本做完全随机排列。当前 derived groups 的 shard 粒度不同，`depth_metric3d` 约 8 samples/shard、`feats_featup` 约 16 samples/shard、`sem_seg_grounded_sam2` 约 24 samples/shard。如果 raw shard 内样本完全随机，一个 worker 在进入 raw shard 的前十几个样本内就可能同步 miss 多个 depth/feat/sem 包，启动期会被放大成多卡多 worker 的 TOS 并发读风暴。训练配置使用 `sample_shuffle_block_size=16`，让 block 间随机、block 内保持连续，兼顾局部性和训练随机性。

不能在 worker 内再通过 sorted shard id 推断“下一个 raw shard”，也不能只按 raw shard 覆盖区间生成 derived group 预取任务。sampler 必须把 worker-local 的后续 sample 信息随样本索引一起传给 dataset/store，否则 worker 预取会沿全局 sorted 顺序或 offset range 走，和真实消费顺序不一致。

更进一步，预取不能只知道 next raw shard。因为 raw shard 内部还会按局部 block shuffle，derived group 的真实访问顺序由随机后的 sample 顺序决定。`ShardAwareSampler` 必须随当前 sample 传入同一个 worker stream 中随机打乱后的后续 `prefetch_samples` 个 sample index；`ShardMemoryStore` 再按这些 sample index 的真实顺序生成 `(group, shard_id)` 预取任务。这样预取顺序与实际读取顺序一致。

## 8. dataloader 设计

### 8.1 设计取舍

1% shard 只是为了降低预处理验证成本，最终目标是全量 train。全量数据中仅 `depth_metric3d` 就可能接近 1TB，单卡配套约 240GB 内存，且 DDP 下通常每张卡一个训练进程，因此 **不能把所有 shard 全部读入内存**。

正式训练默认采用：

```text
preload_mode = "lazy"
```

主路径是：

```text
TOS mount path -> torch.load(required shard) -> per-process memory LRU -> sample assemble
```

仍然不实现、不启用本地硬盘 cache。原因：

- TOS 已挂载为本地路径，大文件顺序读取速度可接受。
- 本地硬盘空间紧张，额外 cache 会引入容量和一致性问题。
- 内存充足，应该把可用资源用于 shard 级内存 LRU 和后台预取。

`preload_mode="all"` 只保留为 1% 小数据调试选项，不作为全量训练默认值。

### 8.2 内存 LRU

lazy 模式下，`ShardMemoryStore` 维护进程内 LRU cache：

```text
cache key = (group, shard_id)
cache value = torch.load(...) 后的 shard payload
```

每次 `store.get(group, sample_idx)`：

```text
1. 查 index: sample_idx -> group -> shard_id/offset
2. 如果 shard 在内存，直接返回 payload["samples"][offset]
3. 如果 shard 不在内存，从 TOS mount 路径 torch.load()
4. 加入 LRU，必要时逐出最久未访问 shard
```

训练默认不检查 `.SUCCESS`，也不读取 `.sha256`。完整性检查前移到预处理结束阶段，dataloader 热路径只需要读取 manifest/index 和实际 `.torch` shard，避免每个 shard 首次加载时额外访问 sidecar 文件。

cache 预算按 DataLoader worker 进程设置：

```text
max_cache_bytes = 24GB per DataLoader worker process
```

当前 10% sharded 稳定性验证配置默认 `num_workers=1`，因此理论上限是：

```text
1 GPU, 1 worker: 24GB/worker -> 24GB shard cache budget
4 GPU, 1 worker/rank: 24GB/worker -> 96GB shard cache budget
8 GPU, 1 worker/rank: 24GB/worker -> 192GB shard cache budget
```

这是上限，不是启动时立即占用；实际占用取决于每个 worker 当前访问过的 shard 和 LRU 逐出。这个 cache 是内存 cache，不落盘。

### 8.3 后台预取

为了减少 GPU 等数据，lazy 模式需要两层预取。

第一层依赖 PyTorch DataLoader：

```text
num_workers > 0
prefetch_factor = 1
persistent_workers = True
pin_memory = True
```

第二层是 `ShardMemoryStore` 的后台 shard 预取：

```text
当前 sample 访问了 raw_nuscenes/000123
sampler 随当前 sample 传入当前 worker stream 的后续 sample indices
根据随机打乱后的后续 sample index
按真实访问顺序找到 raw/depth/feat/sem shard
后台线程提前 torch.load 这些 shard 到 LRU
```

因为 group shard 边界按 `base_block_size` 对齐，一个 raw shard 覆盖的 offsets 可以稳定映射到若干 depth/feat/sem shard。预取不影响正确性，最多影响命中率。

推荐默认：

```text
prefetch_shards = 0
prefetch_samples = 16
prefetch_workers = 0
prefetch_max_tasks_per_call = 0
```

含义是当前稳定性验证阶段关闭 `ShardMemoryStore` 后台预取，只依赖 PyTorch DataLoader worker 的异步取样。这样可以避免前台 worker 与后台 prefetch 线程同时对 TOS/FUSE 发起大量 200MiB 级读取。`prefetch_samples` 保留在 sampler 输出中，便于后续重新打开 shard 预取时继续按真实 sample 顺序生成任务。

如果后续重新打开后台 shard 预取，由于一个 raw shard 可能覆盖多个 depth/feat/sem shard，不能在一次 sample 访问里把下一个 raw shard 覆盖的所有包全部提交到后台队列。应使用较小的 `prefetch_max_tasks_per_call`，把启动期和 shard 切换期的 TOS 读峰值摊平。

预取窗口不应包含当前 raw shard。当前样本所需的 raw 和派生 group 如果未命中，应该由前台同步加载；后台预取只负责下一段 worker-local raw shard，避免启动阶段把当前 shard 覆盖的所有派生 shard 都压入后台队列，造成 TOS 并发读放大和错误预取。

### 8.4 与 pixelSplat 的关系

pixelSplat 的 `DatasetRE10k` 是 `IterableDataset`：训练时收集 `.torch` chunks，shuffle chunk，`torch.load(chunk)` 后再 shuffle chunk 内样本。这个思路说明 chunk 级大文件读取是有效的，但 GaussTR 不能完全照搬：

- pixelSplat 单个 chunk 内已经包含一个样本所需的图像和相机。
- GaussTR 一个样本分布在 `raw_nuscenes`、`depth_metric3d`、`feats_featup`、`sem_seg_grounded_sam2` 等多个 group。
- pixelSplat 训练阶段没有严格按 worker 拆 chunk；GaussTR 若用 IterableDataset，必须额外处理 DDP rank/worker 分片，避免重复样本。

因此本项目选择更容易接入 MMEngine/MMDet3D 的 map-style dataset：

```text
ShardAwareSampler 负责 raw shard shuffle、shard 内 sample shuffle 和 rank 分片
NuScenesOccShardedDataset 负责从 shard store 组装样本
```

### 8.5 数据集类

新增：

```text
gausstr/datasets/shard_store.py
gausstr/datasets/sharded_nuscenes_occ.py
```

`NuScenesOccShardedDataset` 行为：

1. 读取 `{shard_root}/{split}/index.json`。
2. 读取需要的 group manifest。
3. `preload_mode="lazy"` 时，只在样本访问时加载需要的 shard。
4. `preload_mode="all"` 时，加载所有需要的 `.torch` shard，仅用于小数据调试。
5. `get_data_info(index)` 返回一个与原 `NuScenesOccDataset` pipeline 兼容的 results dict。

`ShardMemoryStore` 核心接口：

```python
store.get("raw_nuscenes", sample_idx)
store.get("depth_metric3d", sample_idx)
store.get("feats_featup", sample_idx)
store.get("sem_seg_grounded_sam2", sample_idx)
```

`preload_mode="lazy"` 下，`get()` 通过 LRU cache 避免重复加载同一个 shard。

### 8.6 sharded transforms

新增 transform：

```text
BEVLoadMultiViewImageFromShards
LoadShardedFeatMaps
LoadShardedOccFromArrays
```

保留现有：

```text
ImageAug3D
Pack3DDetInputs
```

训练 pipeline 对照：

```text
raw pipeline:
  BEVLoadMultiViewImageFromFiles
  ImageAug3D
  LoadFeatMaps(depth)
  LoadFeatMaps(feats)
  LoadFeatMaps(sem_seg)
  Pack3DDetInputs

sharded pipeline:
  BEVLoadMultiViewImageFromShards
  ImageAug3D
  LoadShardedFeatMaps(depth_metric3d -> depth)
  LoadShardedFeatMaps(feats_featup -> feats)
  LoadShardedFeatMaps(sem_seg_grounded_sam2 -> sem_seg)
  Pack3DDetInputs
```

`BEVLoadMultiViewImageFromShards` 与原 `BEVLoadMultiViewImageFromFiles` 输出相同字段：

```text
img
filename
img_path
cam2img
lidar2cam
cam2ego
ori_cam2img
img_shape
ori_shape
pad_shape
scale_factor
img_norm_cfg
num_views
```

区别只是图像 bytes 来自 `raw_nuscenes` shard，而不是 `mmengine.fileio.get(img_path)`。

### 8.7 DataLoader 参数

lazy 模式下，训练期主要开销包括：

- shard cache miss 时的 `torch.load`
- JPEG decode
- numpy/tensor 组装
- ImageAug3D
- CPU 到 GPU batch 传输

推荐从较稳的配置开始：

```python
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=1,
    sampler=dict(
        type='ShardAwareSampler',
        shuffle=True,
        num_workers=1,
        sample_shuffle_block_size=16,
        prefetch_samples=16),
    dataset=dict(
        type='NuScenesOccShardedDataset',
        shard_root='data/gausstr_shards',
        split='train',
        preload_mode='lazy',
        require_success=False,
        max_cache_bytes=24 * 1024**3,
        prefetch_shards=0,
        prefetch_workers=0,
        prefetch_max_tasks_per_call=0,
        debug=False,
        debug_interval=100,
        required_groups=dict(
            raw='raw_nuscenes',
            depth='depth_metric3d',
            feats='feats_featup',
            sem_seg='sem_seg_grounded_sam2')))
```

对于 1% 调试数据，可以临时把 `preload_mode` 改为 `all`；全量训练配置不应使用 `all`。

### 8.8 调试日志与容错

`NuScenesOccShardedDataset` 支持 debug 开关：

```python
dataset=dict(
    type='NuScenesOccShardedDataset',
    debug=True,
    debug_interval=20)
```

开启后，每个 DataLoader worker 会打印：

- shard load start/done、路径、样本数、耗时和 cache bytes。
- cache hit/miss、shard load、prefetch scheduled/wait/hit、eviction 统计。
- 预取线程的 start/done。
- TOS/mount 路径出现短暂 `EAGAIN/EBUSY/EINTR/ESTALE` 时的 `torch.load` 重试。

debug 日志可能很密，正式训练默认关闭；只在定位数据等待、cache 过小、TOS 抖动或 shard 损坏时打开。

## 9. 预处理脚本接口

脚本路径：

```text
tools/build_sharded_dataset.py
```

默认命令尽量短：

```bash
PYTHONPATH=. python tools/build_sharded_dataset.py --ratio 0.1
```

常用默认值：

```text
--preset all
--split train
--data-root data/nuscenes
--ann-file data/nuscenes/nuscenes_infos_{split}.pkl
--out-root data/gausstr_shards
--depth-root data/nuscenes_metric3d
--feat-root data/nuscenes_featup
--sem-seg-root data/nuscenes_grounded_sam2
--base-block-size 8
--target-shard-size 256MB
--raw-target-shard-size 100MB
--num-workers 4
--seed 2026
```

`--preset all` 默认准备 FeatUp 和 Talk2DINO 所需并集：

```text
raw_nuscenes + depth:metric3d + feats:featup + sem_seg:grounded_sam2
```

`--preset talk2dino`：

```text
raw_nuscenes + depth:metric3d
```

`--preset featup`：

```text
raw_nuscenes + depth:metric3d + feats:featup + sem_seg:grounded_sam2
```

`val/test` 默认额外加入 `occ_gt`。

## 10. 预处理阶段

### 阶段 A：读取 annotation

读取：

```text
data/nuscenes/nuscenes_infos_{split}.pkl
```

并校验 sample 必要字段、6 路相机和所需源文件。

### 阶段 B：生成 sample_order

`ratio < 1` 时按 scene 采样，保证子集更接近真实分布。train 再对 selected samples 做 seed 控制的 sample-level shuffle。

输出：

```text
sample_order.json
sample_manifest.pkl
sample_manifest.json
```

### 阶段 C：估算 group size

对最多 `--size-estimate-samples` 个样本估算每组 P50/P90/P99：

```text
raw_nuscenes
depth_metric3d
feats_featup
sem_seg_grounded_sam2
occ_gt
```

输出 `stats.json`。

### 阶段 D：生成 group shard_plan

每个 group 单独生成：

```text
{group}/shard_plan.json
```

但所有 shard 都只能覆盖 `sample_order` 的连续区间，且起点终点按 `base_block_size` 对齐，最后一个 shard 除外。

### 阶段 E：写 shard

按 group 顺序处理，同一个 group 内用线程池并行写 shard：

```text
000000.torch.tmp
  -> sanity torch.load with retry
  -> sha256
  -> atomic rename
  -> .SUCCESS
```

每个 group 写：

```text
group_manifest.json
```

如果用户中断后重跑，已存在且有 `.SUCCESS` 的 shard 会跳过。

TOS/FUSE 挂载路径上，`torch.save()` 返回后立刻 `torch.load()` 可能短暂读到未完全可见的 zip central directory。脚本会对这类 sanity-load 错误重试；如果重试后仍失败，不写 `.SUCCESS`，下次重跑会把不完整 shard 当作未完成文件重建。

### 阶段 F：结束校验

所有 group 写完后，脚本会按 `group_manifest.json` 对数据目录做一次完整性检查：

- `.torch` 文件存在。
- `.torch` 文件大小与 manifest 记录一致。
- `.SUCCESS` 文件存在。
- 如果 manifest 记录了 sha256，则 `.sha256` sidecar 存在且内容一致。
- 不存在残留 `.tmp` 文件。

校验结果写入：

```text
build_summary.json -> validation
```

校验通过后，训练 dataloader 默认 `require_success=False`，不会再检查 `.SUCCESS`。如果确认不需要在同一个目录继续断点续跑预处理，可以手动删除 `.SUCCESS` 和 `.sha256` sidecar，以减少 TOS 文件数量；删除后不要再对这个目录直接重跑预处理脚本，否则脚本会把“有 `.torch` 但无 `.SUCCESS`”视为未完成 shard。

## 11. 与当前 pipeline 的关系

现有 GaussTR 数据流大致为：

```text
NuScenesOccDataset
  -> BEVLoadMultiViewImageFromFiles
  -> ImageAug3D
  -> LoadFeatMaps
  -> LoadOccFromFile
  -> Pack3DDetInputs
```

后续新增 sharded dataloader 时，目标是让它输出与现有 pipeline 等价的字段：

```text
img
img_path
cam2img
cam2ego
ego2global
depth
feats
sem_seg
gt_semantic_seg
mask_camera
mask_lidar
sample_idx
num_views
```

模型侧 `GaussTR.prepare_inputs()` 不应关心数据来自原始小文件还是 shard。

### 11.1 训练与评估适配

sharded config 同时覆盖训练和验证：

```text
train_dataloader -> data/gausstr_shards/train
val_dataloader   -> data/gausstr_shards/val
test_dataloader  -> val_dataloader
```

原始 config 中 `train_cfg.val_interval=1`，每个 epoch 后的 eval 使用 `val_dataloader`。`test_dataloader = val_dataloader`，因此当前项目没有单独的 test split 流程；除非显式改配置，test 入口也会评估 val shard。

训练 split 默认准备：

```text
raw_nuscenes + depth_metric3d + feats_featup + sem_seg_grounded_sam2
```

val split 默认额外准备：

```text
occ_gt
```

因为评估 pipeline 需要 `gt_semantic_seg`、`mask_camera` 等占据真值。正式使用时至少需要分别构建：

```bash
PYTHONPATH=. python tools/build_sharded_dataset.py --split train --ratio 1
PYTHONPATH=. python tools/build_sharded_dataset.py --split val --ratio 1
```

如果只构建了 train shard，训练可以开始，但到 epoch 结束进入 val 时会因为缺少 `data/gausstr_shards/val/index.json` 或 `occ_gt` shard 失败。

## 12. 第一版实现边界

第一版只实现：

- `raw_nuscenes` 合并 meta/images。
- 派生组 `depth_metric3d`、`feats_featup`、`sem_seg_grounded_sam2`、`occ_gt`。
- group 独立 shard size，但按 base block 对齐。
- TOS 挂载路径 direct `torch.load()`。
- 中断后跳过已完成 shard。
- 预处理结束后的目录完整性检查。
- lazy LRU 内存 cache。
- DataLoader worker 内 shard 预取。
- 单机多卡 `ShardAwareSampler`。
- train/val/test 配置层同步接入 sharded dataloader，其中 test 默认复用 val。
- dataloader debug 日志和 `torch.load` 短暂文件系统错误重试。

暂不实现：

- 自动 dtype 压缩。
- 本地硬盘 cache。
- 多机 shard 分配协议。
- 多方法自动 fallback。
