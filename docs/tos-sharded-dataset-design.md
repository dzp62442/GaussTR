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

- 每个 epoch 对 `sample_order` 重新 shuffle。
- DDP 单机多卡用 `rank/world_size` 对 sample 序列切分。
- dataloader 根据 sample_id 查各 group 的 shard/offset。

因此，即使某个 batch 中样本来自多个 group 的不同 shard，也不会破坏训练随机性。

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

cache 预算按进程设置：

```text
max_cache_bytes = 24GB 到 64GB per DataLoader worker process
```

实际值需要结合 GPU 数量调整：

```text
1 GPU, 4 workers: 32GB/worker -> 128GB shard cache budget
4 GPU, 4 workers/rank: 32GB/worker -> 512GB shard cache budget
8 GPU, 4 workers/rank: 24GB-32GB/worker -> 768GB-1024GB shard cache budget
```

这个 cache 是内存 cache，不落盘。

### 8.3 后台预取

为了减少 GPU 等数据，lazy 模式需要两层预取。

第一层依赖 PyTorch DataLoader：

```text
num_workers > 0
prefetch_factor = 2 or 4
persistent_workers = True
pin_memory = True
```

第二层是 `ShardMemoryStore` 的后台 shard 预取：

```text
当前 sample 访问了 raw_nuscenes/000123
根据 raw shard 覆盖的 global offset 区间
找到下一段 raw shard 以及其覆盖的 depth/feat/sem shards
后台线程提前 torch.load 这些 shard 到 LRU
```

因为 group shard 边界按 `base_block_size` 对齐，一个 raw shard 覆盖的 offsets 可以稳定映射到若干 depth/feat/sem shard。预取不影响正确性，最多影响命中率。

推荐默认：

```text
prefetch_shards = 1
prefetch_workers = 1
```

含义是访问当前 raw shard 时，后台预取下一个 raw shard 及其覆盖的 required group shards。预取线程只读 TOS mount，不写本地盘。

### 8.4 与 pixelSplat 的关系

pixelSplat 的 `DatasetRE10k` 是 `IterableDataset`：训练时收集 `.torch` chunks，shuffle chunk，`torch.load(chunk)` 后再 shuffle chunk 内样本。这个思路说明 chunk 级大文件读取是有效的，但 GaussTR 不能完全照搬：

- pixelSplat 单个 chunk 内已经包含一个样本所需的图像和相机。
- GaussTR 一个样本分布在 `raw_nuscenes`、`depth_metric3d`、`feats_featup`、`sem_seg_grounded_sam2` 等多个 group。
- pixelSplat 训练阶段没有严格按 worker 拆 chunk；GaussTR 若用 IterableDataset，必须额外处理 DDP rank/worker 分片，避免重复样本。

因此本项目选择更容易接入 MMEngine/MMDet3D 的 map-style dataset：

```text
DefaultSampler/DistributedSampler 负责 sample shuffle 和 rank 分片
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
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NuScenesOccShardedDataset',
        shard_root='data/gausstr_shards',
        split='train',
        preload_mode='lazy',
        max_cache_bytes=32 * 1024**3,
        prefetch_shards=1,
        required_groups=dict(
            raw='raw_nuscenes',
            depth='depth_metric3d',
            feats='feats_featup',
	    sem_seg='sem_seg_grounded_sam2')))
```

对于 1% 调试数据，可以临时把 `preload_mode` 改为 `all`；全量训练配置不应使用 `all`。

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
000000.torch.tmp -> sanity torch.load -> sha256 -> rename -> .SUCCESS
```

每个 group 写：

```text
group_manifest.json
```

如果用户中断后重跑，已存在且有 `.SUCCESS` 的 shard 会跳过。

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

## 12. 第一版实现边界

第一版只实现：

- `raw_nuscenes` 合并 meta/images。
- 派生组 `depth_metric3d`、`feats_featup`、`sem_seg_grounded_sam2`、`occ_gt`。
- group 独立 shard size，但按 base block 对齐。
- TOS 挂载路径 direct `torch.load()`。
- 中断后跳过已完成 shard。

暂不实现：

- 自动 dtype 压缩。
- 本地硬盘 cache。
- 异步预取。
- 多机 shard 分配协议。
- 多方法自动 fallback。
