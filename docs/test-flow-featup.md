# GaussTR 代码调用流程说明

本文以如下测试命令为例，系统化说明本项目从命令入口到最终评估输出的代码调用流程：

```bash
PYTHONPATH=. mim test mmdet3d configs/gausstr_featup.py -C ckpts/gausstr_featup_e24_miou11.70.pth -G 1
```

本文只覆盖 `GaussTR-FeatUp` 测试链路，不展开训练流程，也不展开 `Talk2DINO` 分支。

## 1. 一句话总览

这条命令会完成如下流程：

1. `mim test mmdet3d` 调用 `mmdetection3d` 的测试入口。
2. 读取 `configs/gausstr_featup.py`，并通过 `custom_imports` 注册本项目自定义模块。
3. 构建 `Runner`、测试数据集、模型和评估器。
4. 从 `nuscenes_infos_val.pkl` 逐条读取样本，经过测试 pipeline 生成模型输入。
5. `GaussTR` 使用离线 `FeatUp` 特征和 `Metric3D` 深度，预测一组 3D Gaussians。
6. 将 3D Gaussians 体素化为 occupancy 语义网格。
7. `OccMetric` 使用 `mask_camera` 过滤无效体素，累计 IoU / mIoU。

## 2. 顶层调用链

可以把主链路概括为：

```text
mim test mmdet3d
  -> third_party/mmdetection3d/tools/test.py
  -> Config.fromfile(configs/gausstr_featup.py)
  -> Runner.from_cfg(cfg)
  -> runner.test()
  -> test loop
  -> dataloader(test_dataloader)
  -> model.predict()
  -> evaluator.process()
  -> evaluator.compute_metrics()
```

本项目内真正参与这条链路的核心文件是：

- `configs/gausstr_featup.py`
- `gausstr/datasets/nuscenes_occ.py`
- `gausstr/datasets/transforms.py`
- `gausstr/models/gausstr.py`
- `gausstr/models/vitdet_fpn.py`
- `gausstr/models/gausstr_decoder.py`
- `gausstr/models/gausstr_head.py`
- `gausstr/models/gaussian_voxelizer.py`
- `gausstr/evaluation/occ_metric.py`

## 3. 命令入口层

`mim test mmdet3d ...` 本质上是在调用 `mmdetection3d` 的测试脚本 `third_party/mmdetection3d/tools/test.py`。

这个入口脚本主要做四件事：

1. 解析命令行参数。
2. 读取配置文件。
3. 将 `-C ckpts/...pth` 写入 `cfg.load_from`。
4. 构建 `Runner` 并执行 `runner.test()`。

`PYTHONPATH=.` 的作用是让 Python 能够导入当前仓库下的 `gausstr` 包。否则配置里的 `custom_imports = dict(imports=['gausstr'])` 无法生效，本项目自定义的模型、数据集、变换、评估器都不会被注册到 OpenMMLab registry。

## 4. 配置解析层

测试命令使用的配置文件是 `configs/gausstr_featup.py`。它定义了本次测试的三类核心对象：

- 模型：`GaussTR`
- 测试数据：`NuScenesOccDataset + test_pipeline`
- 评估器：`OccMetric`

其中几个最关键的配置点如下。

### 4.1 模型配置

- `type='GaussTR'`
- `num_queries=300`
- `neck='ViTDetFPN'`
- `decoder='GaussTRDecoder'`
- `gauss_head='GaussTRHead'`
- 没有 `backbone`

这说明 `FeatUp` 分支测试时不在线提取视觉主特征，而是直接读取磁盘上的离线特征 `data/nuscenes_featup/*.npy`。

### 4.2 测试数据配置

测试集使用：

- `dataset_type='NuScenesOccDataset'`
- `ann_file='nuscenes_infos_val.pkl'`
- `batch_size=1`
- `num_views=6`

测试 pipeline 为：

1. `BEVLoadMultiViewImageFromFiles`
2. `LoadOccFromFile`
3. `ImageAug3D`
4. `LoadFeatMaps(key='depth')`
5. `LoadFeatMaps(key='feats')`
6. `Pack3DDetInputs`

### 4.3 评估配置

评估器使用：

- `type='OccMetric'`
- `num_classes=18`
- `use_image_mask=True`

这意味着评估时只在相机可见体素上统计指标。

## 5. 自定义模块注册层

配置被加载时，会先 import `gausstr`。`gausstr/__init__.py` 会继续导入：

- `gausstr.datasets`
- `gausstr.evaluation`
- `gausstr.hooks`
- `gausstr.models`

这些模块内部通过 `@DATASETS.register_module()`、`@TRANSFORMS.register_module()`、`@MODELS.register_module()`、`@METRICS.register_module()` 完成注册。

因此，配置文件中的如下类型名才能被正确实例化：

- `NuScenesOccDataset`
- `BEVLoadMultiViewImageFromFiles`
- `LoadOccFromFile`
- `ImageAug3D`
- `LoadFeatMaps`
- `GaussTR`
- `ViTDetFPN`
- `GaussTRDecoder`
- `GaussTRHead`
- `GaussianVoxelizer`
- `OccMetric`

## 6. 数据读取与测试 pipeline

### 6.1 数据集对象

测试集类是 `gausstr/datasets/nuscenes_occ.py` 中的 `NuScenesOccDataset`。

它继承自 `NuScenesDataset`，在标准 nuScenes 样本信息基础上，额外补充了 occupancy ground truth 路径：

```text
data/nuscenes/gts/{scene_idx}/{token}
```

因此 `tools/update_data.py` 预先写入 `scene_idx` 是必须的。

### 6.2 Pipeline 第 1 步：读取多视角图像和相机参数

`BEVLoadMultiViewImageFromFiles` 完成：

- 读取 6 个相机图像
- 组织 `img`
- 组织 `cam2img`
- 组织 `cam2ego`
- 保存 `img_path`
- 保存 `num_views`

这一步提供了后续反投影到 3D 所需的几何信息。

### 6.3 Pipeline 第 2 步：读取 occupancy GT

`LoadOccFromFile` 从 `labels.npz` 中读取：

- `gt_semantic_seg`
- `mask_lidar`
- `mask_camera`

其中：

- `gt_semantic_seg` 用于评估
- `mask_camera` 用于评估时过滤不可见体素

### 6.4 Pipeline 第 3 步：测试时图像增强

`ImageAug3D` 在测试时不会做随机增强，而是做固定的：

- resize
- center crop

同时记录增强变换矩阵 `img_aug_mat`。这个矩阵不是只给图像预处理用，后面 2D query 反投影回 3D 时也会使用。

### 6.5 Pipeline 第 4 步：读取离线深度

`LoadFeatMaps(data_root='data/nuscenes_metric3d', key='depth')` 会按图像文件名加载对应 `.npy` 深度图。

如果 `apply_aug=True`，还会对深度图做与图像一致的 resize / crop，对齐当前视图。

### 6.6 Pipeline 第 5 步：读取离线 FeatUp 特征

`LoadFeatMaps(data_root='data/nuscenes_featup', key='feats')` 会按图像文件名加载对应 `.npy` 特征图。

同样地，这些特征也会应用和图像一致的几何变化。

### 6.7 Pipeline 输出结果

经过 `Pack3DDetInputs` 后，送进模型的核心内容可以概括为：

- `inputs['imgs']`：6 视角 RGB 图像
- `data_samples[*].depth`：6 视角深度图
- `data_samples[*].feats`：6 视角 FeatUp 特征图
- `data_samples[*].cam2img / cam2ego / img_aug_mat`
- `data_samples[*].gt_pts_seg.semantic_seg`
- `data_samples[*].mask_camera`

## 7. 模型前向主流程

模型主类是 `gausstr/models/gausstr.py` 中的 `GaussTR`。

测试时走的是：

```text
GaussTR.forward(..., mode='predict')
```

整体可以拆成 6 个阶段：

1. 整理输入张量
2. 获取视觉特征
3. 通过 FPN 构造多尺度特征
4. 通过 decoder 预测 2D queries
5. 通过深度将 2D queries 抬升成 3D Gaussians
6. 将 3D Gaussians 体素化为 occupancy 预测

### 7.1 `prepare_inputs`

`prepare_inputs` 会把 `Det3DDataSample` 中的元信息统一整理成模型更容易消费的 tensor，包括：

- `cam2img`
- `cam2ego`
- `ego2global`
- `img_aug_mat`
- `depth`
- `feats`

### 7.2 特征输入来源

`FeatUp` 分支没有配置在线 backbone，因此模型直接执行：

```python
x = data_samples['feats'].flatten(0, 1)
```

即：

- 直接使用离线特征
- 不再从 RGB 图像在线提取主特征

这一点是 `FeatUp` 分支与 `Talk2DINO` 分支最本质的区别。

### 7.3 FPN 多尺度特征构造

`x` 会送入 `ViTDetFPN`。

`ViTDetFPN` 的作用是：

- 对单尺度输入做上采样、保持、下采样
- 生成 4 个尺度的特征图
- 每个尺度都投影到统一的 `embed_dims=256`

输出是一个长度为 4 的特征列表，用于 deformable attention。

### 7.4 `pre_transformer`

`pre_transformer` 会把多尺度特征展平成一个长序列，并生成：

- `memory`
- `spatial_shapes`
- `level_start_index`
- `valid_ratios`

由于当前配置没有 encoder，因此不会进入额外的 transformer encoder，展平后的多尺度特征序列会直接作为 decoder 的 memory。

### 7.5 `pre_decoder`

`pre_decoder` 会初始化：

- `query`：300 个可学习 query embedding
- `reference_points`：300 个随机二维参考点

这里的 `reference_points` 位于图像归一化平面中，代表 query 初始关注的位置。

### 7.6 `GaussTRDecoder`

`GaussTRDecoder` 包含 3 层 decoder，每层结构为：

1. multi-scale deformable cross-attention
2. self-attention
3. FFN

每层结束后，还会通过 `reg_branches` 更新 `reference_points`。因此 query 不仅在聚合图像信息，也在逐层细化自己的二维位置。

到 decoder 结束时，模型得到：

- 最后一层 query feature
- 对应的 refined 2D `reference_points`

## 8. `GaussTRHead`：从 2D query 到 3D occupancy

`GaussTRHead` 是测试推理的核心。

测试时只使用最后一层 decoder 的输出。

### 8.1 从 2D reference point 采样深度

先用 `regress_head` 预测 3 维残差：

- 前 2 维用于修正 2D 位置
- 第 3 维用于修正采样深度

然后在离线深度图 `depth` 上使用 `grid_sample` 采样，得到每个 query 对应的 metric depth。

因此，`data/nuscenes_metric3d` 在测试时不是辅助监督，而是 2D 到 3D 抬升的直接输入。

### 8.2 从像素坐标反投影为 3D 点

有了 `(u, v, d)` 后，调用 `cam2world`：

- 撤销图像增强 `img_aug_mat`
- 使用 `cam2img` 反投影到相机坐标
- 使用 `cam2ego` 转到 3D 世界坐标

得到每个 query 的 3D 中心 `means3d`。

### 8.3 预测 Gaussian 属性

每个 query 同时预测：

- `opacity`
- `feature`
- `scale`

随后结合 `cam2ego` 生成：

- `covariance`
- `rotation`

至此，每个 query 都变成一个参数化的 3D Gaussian。

### 8.4 文本原型对齐

测试分支会将 Gaussian feature 与 `text_proto_embeds_clip.pth` 中的文本原型相乘，得到语义相似度表示。

这一步将“视觉特征”映射到“语义类别空间”。

### 8.5 Gaussian 体素化

`GaussianVoxelizer` 会：

1. 构建固定三维体素网格
2. 遍历每个 Gaussian
3. 只在其局部 `3σ` 范围内计算高斯密度贡献
4. 以密度为权重聚合语义特征

输出：

- `density`
- `grid_feats`

### 8.6 语义后处理

在体素网格上还会做三步后处理：

1. `prompt_denoising`
   作用：去掉低置信文本 prompt 响应。
2. `merge_probs`
   作用：将文本 prompt 级别的类别合并为 Occ3D 定义的语义组。
3. density threshold
   作用：若体素密度过低，则判为 `free`。

最终输出就是 occupancy 预测张量 `preds`。

## 9. 评估流程

评估器是 `gausstr/evaluation/occ_metric.py` 中的 `OccMetric`。

### 9.1 `process`

对每个 batch：

1. 接收模型预测 `preds`
2. 提取 GT `gt_pts_seg.semantic_seg`
3. 根据 `mask_camera` 过滤相机不可见体素
4. 累计 confusion matrix `hist`

### 9.2 `compute_metrics`

所有样本测试完成后：

1. 根据 `hist` 计算每类 IoU
2. 对除 `free` 类外的类别取平均，得到 `mIoU`
3. 额外计算 occupancy-level 的 `IoU`

最终测试日志中的表格就是在这里生成的。

## 10. 关键设计理解

从代码实现角度，`GaussTR-FeatUp` 测试链路最关键的三点是：

### 10.1 主视觉特征来自离线特征，而不是在线 backbone

测试时真正进入 transformer 的主特征是 `data/nuscenes_featup/*.npy`，而不是模型现场从 RGB 图像中提取的特征。

### 10.2 深度是推理输入，不只是训练监督

`data/nuscenes_metric3d/*.npy` 在测试时直接决定 query 的 3D 位置，是推理链路的核心组成部分。

### 10.3 最终预测不是“直接分类每个体素”

模型先预测一组 3D Gaussians，再将它们 splat / voxelize 到体素网格中，最后才得到 occupancy 语义结果。

## 11. 简化版调用图

```text
命令行
  -> mmdet3d/tools/test.py
  -> 读取 configs/gausstr_featup.py
  -> import gausstr 并注册自定义模块
  -> Runner.from_cfg(cfg)
  -> runner.test()
  -> NuScenesOccDataset.__getitem__()
  -> test_pipeline
       -> 读图像
       -> 读 occupancy GT
       -> 图像几何变换
       -> 读 depth
       -> 读 FeatUp 特征
  -> GaussTR.forward(mode='predict')
       -> prepare_inputs
       -> ViTDetFPN
       -> GaussTRDecoder
       -> GaussTRHead
           -> 深度采样
           -> 2D -> 3D 反投影
           -> 预测 3D Gaussians
           -> GaussianVoxelizer
           -> 输出 occupancy preds
  -> OccMetric.process()
  -> OccMetric.compute_metrics()
  -> 输出 mIoU / IoU
```

## 12. 阅读代码的推荐顺序

如果要继续深入源码，建议按下面顺序阅读：

1. `configs/gausstr_featup.py`
2. `gausstr/datasets/nuscenes_occ.py`
3. `gausstr/datasets/transforms.py`
4. `gausstr/models/gausstr.py`
5. `gausstr/models/gausstr_decoder.py`
6. `gausstr/models/gausstr_head.py`
7. `gausstr/models/gaussian_voxelizer.py`
8. `gausstr/evaluation/occ_metric.py`

这个顺序和实际运行顺序基本一致。
