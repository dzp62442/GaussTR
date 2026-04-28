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

当前已经构建出的 shard 是 1% 数据集，四个训练 group 合计约 18GB。训练服务器内存非常充足，本地硬盘空间反而紧张，因此第一版 dataloader 采用 **内存优先** 策略：

```text
TOS mount path -> torch.load(all required shards) -> process memory -> training
```

不实现、不启用本地硬盘 cache。只要进程内存够，训练期间不应再触发 TOS shard 读取，从而最大限度避免 GPU 等待数据。

第一版默认：

```text
preload_mode = "all"
```

含义是 dataset 初始化时把配置需要的所有 group shard 全量读入内存，并建立：

```python
group -> sample_idx -> sample_payload
```

这样 `__getitem__` 不再读文件，只做内存查表、图像解码、特征取出和数据增强。

后续如果处理全量数据后发现单进程全量预加载不合适，再增加：

```text
preload_mode = "lazy"
prefetch_shards = 1 or 2
```

lazy 模式仍然只从 TOS mount 直接 `torch.load()`，不落本地磁盘。

### 8.2 与 pixelSplat 的关系

pixelSplat 的 `DatasetRE10k` 是 `IterableDataset`：训练时收集 `.torch` chunks，shuffle chunk，`torch.load(chunk)` 后再 shuffle chunk 内样本。这个思路说明 chunk 级大文件读取是有效的，但 GaussTR 不能完全照搬：

- pixelSplat 单个 chunk 内已经包含一个样本所需的图像和相机。
- GaussTR 一个样本分布在 `raw_nuscenes`、`depth_metric3d`、`feats_featup`、`sem_seg_grounded_sam2` 等多个 group。
- pixelSplat 训练阶段没有严格按 worker 拆 chunk；GaussTR 若用 IterableDataset，必须额外处理 DDP rank/worker 分片，避免重复样本。

因此第一版选择更容易接入 MMEngine/MMDet3D 的 map-style dataset：

```text
DefaultSampler/DistributedSampler 负责 sample shuffle 和 rank 分片
NuScenesOccShardedDataset 负责从内存中的 shard payload 组装样本
```

### 8.3 数据集类

新增：

```text
gausstr/datasets/shard_store.py
gausstr/datasets/sharded_nuscenes_occ.py
```

`NuScenesOccShardedDataset` 行为：

1. 读取 `{shard_root}/{split}/index.json`。
2. 读取需要的 group manifest。
3. `preload_mode="all"` 时，加载所有需要的 `.torch` shard。
4. 建立 `sample_idx -> group payload` 内存索引。
5. `get_data_info(index)` 返回一个与原 `NuScenesOccDataset` pipeline 兼容的 results dict。

`ShardMemoryStore` 核心接口：

```python
store.get("raw_nuscenes", sample_idx)
store.get("depth_metric3d", sample_idx)
store.get("feats_featup", sample_idx)
store.get("sem_seg_grounded_sam2", sample_idx)
```

`preload_mode="all"` 下，`get()` 是纯内存查表。

### 8.4 sharded transforms

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

### 8.5 DataLoader 参数

因为 shard 已全量进入内存，训练时主要开销变成：

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
        preload_mode='all',
        required_groups=dict(
            raw='raw_nuscenes',
            depth='depth_metric3d',
            feats='feats_featup',
            sem_seg='sem_seg_grounded_sam2')))
```

当前 1% shard 可以全量读取。若单进程内存和 fork worker 的复制开销仍然可接受，再把 `num_workers` 增到 8 或 16。

### 8.6 预取策略

当前默认 `preload_mode="all"` 时，**没有下一个 shard 需要在训练中预取**：所有 shard 已经在训练开始前进入内存。GPU 等数据的问题主要交给 PyTorch DataLoader 的 worker 预取解决：

```text
num_workers > 0
prefetch_factor = 2 or 4
persistent_workers = True
pin_memory = True
```

如果未来启用 `preload_mode="lazy"`，再实现后台线程预取：

```text
当前 sample 所在 raw shard 正在消费
后台预取下一个 raw shard 覆盖区间内需要的 depth/feat/sem shards
```

但该模式仍然是内存 cache，不写本地磁盘。

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
